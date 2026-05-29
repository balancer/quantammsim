"""Optuna tuning of the fantasticlamm_sign pool, maximising annualised return
over (deposit) HODL.

Tunes the trend-band trigger + inner-decay params on a train window, then
validates the winner on a held-out OOS window with a fresh deposit (the honest
evaluation the QuantAMM docs recommend). The tunable params are passed through
the pool's ``params`` dict (a traced pytree), so optuna varies them without
recompiling the JIT kernel each trial; price data is preloaded once.

Run:  python experiments/tune_fantasticlamm_sign.py --n-trials 150
"""

import os
import sys
import copy
import math
import argparse

# Ensure the worktree copy of quantammsim is imported (not an installed one).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp
import optuna

import quantammsim
from pathlib import Path
from quantammsim.runners.jax_runners import do_run_on_historic_data
from quantammsim.utils.data_processing.historic_data_utils import (
    get_historic_parquet_data,
)

DATA_ROOT = Path(quantammsim.__file__).parent / "data"
TOKENS = ["AAVE", "ETH"]
SECONDS_PER_STEP = 60.0  # arb_frequency (1) * 60

# Train on 2yr, validate on the 6mo OOS window studied earlier.
TRAIN_START = "2023-11-20 00:00:00"
TRAIN_END = "2025-11-20 00:00:00"
OOS_START = "2025-11-20 00:00:00"
OOS_END = "2026-05-20 00:00:00"

OBJECTIVE = "annualised_returns_over_hodl"
FEE = 0.003
INITIAL = 10_000.0


def base_fingerprint(start, end):
    return {
        "rule": "fantasticlamm_sign",
        "tokens": TOKENS,
        "startDateString": start,
        "endDateString": end,
        "initial_pool_value": INITIAL,
        "do_arb": True,
        "fees": FEE,
        "gas_cost": 0.0,
        "arb_fees": 0.0,
        "chunk_period": 60,
        "weight_interpolation_period": 60,
        "return_val": OBJECTIVE,
        "max_memory_days": 365.0,
    }


def trial_params(t):
    """Sample the 8-dim search space and return a fantasticlamm params dict."""
    ratio_base = t.suggest_float("ratio_base", 1.01, 20.0, log=True)
    ratio_span = t.suggest_float("ratio_span", 1.5, 300.0, log=True)
    ratio_max = ratio_base * ratio_span
    centeredness_margin = t.suggest_float("centeredness_margin", 0.01, 0.99)
    shift_exponent = t.suggest_float("shift_exponent", 1e-3, 125.0, log=True)
    deadband = t.suggest_float("deadband", 0.0, 0.9)
    sharpness = t.suggest_float("sharpness", 0.3, 5.0)
    trigger_alpha = t.suggest_float("trigger_alpha", 1e-4, 0.3, log=True)
    reconcentration_hours = t.suggest_float("reconcentration_hours", 1.0, 240.0, log=True)

    # Per-step log-ratio cap that unwinds ratio_max -> ratio_base over the
    # chosen re-concentration time.
    max_narrow_log_step = (
        math.log(ratio_max / ratio_base) * SECONDS_PER_STEP
        / (reconcentration_hours * 3600.0)
    )
    return {
        "price_ratio": jnp.array(ratio_base),
        "centeredness_margin": jnp.array(centeredness_margin),
        "shift_exponent": jnp.array(shift_exponent),
        "ratio_max": jnp.array(ratio_max),
        "deadband": jnp.array(deadband),
        "sharpness": jnp.array(sharpness),
        "trigger_alpha": jnp.array(trigger_alpha),
        "max_narrow_log_step": jnp.array(max_narrow_log_step),
    }


def run(fp, params, price_df):
    return do_run_on_historic_data(
        run_fingerprint=copy.deepcopy(fp), params=params,
        root=DATA_ROOT, price_data=price_df, verbose=False,
    )


def annualised_returns_over_hodl(result):
    """Exact replica of forward_pass's metric (minute-resolution value series).

        (value[-1] / (initial_reserves . prices[-1])) ** (mins_per_year/(T-1)) - 1
    """
    value = np.asarray(result["value"]).reshape(-1)
    prices = np.asarray(result["prices"])
    res0 = np.asarray(result["reserves"])[0]
    deposit_hodl = (res0 * prices[-1]).sum()
    T = value.shape[0]
    return float((value[-1] / deposit_hodl) ** (365.0 * 24 * 60 / (T - 1)) - 1.0)


def metrics(result):
    """Resolution-independent HODL metrics + fees from a do_run result."""
    value = np.asarray(result["value"]).reshape(-1)
    prices = np.asarray(result["prices"])
    res0 = np.asarray(result["reserves"])[0]
    fees = float(np.asarray(result["fee_revenue"]).sum())
    deposit_hodl = float((res0 * prices[-1]).sum())
    units = (INITIAL / 2.0) / prices[0]
    uniform_hodl = float((prices * units).sum(axis=1)[-1])
    return {
        "final_value": float(result["final_value"]),
        "returns_over_hodl_pct": 100.0 * (value[-1] / deposit_hodl - 1.0),
        "vs_uniform_hodl_pct": 100.0 * (value[-1] / uniform_hodl - 1.0),
        "fees": fees,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Preloading {TOKENS} price data from {DATA_ROOT} ...", flush=True)
    price_df = get_historic_parquet_data(list(TOKENS), ["close"], str(DATA_ROOT) + "/")
    print(f"  loaded {price_df.shape[0]:,} rows", flush=True)

    fp_train = base_fingerprint(TRAIN_START, TRAIN_END)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(t):
        try:
            result = run(fp_train, trial_params(t), price_df)
            obj = annualised_returns_over_hodl(result)
        except Exception as e:  # noqa: BLE001 - reject bad trials
            print(f"  trial {t.number}: FAILED ({type(e).__name__}: {e})", flush=True)
            return -1e9
        if not np.isfinite(obj):
            return -1e9
        return obj

    def log_cb(study, trial):
        dur = trial.duration.total_seconds() if trial.duration else float("nan")
        print(f"  trial {trial.number:3d}: obj={trial.value:+.4f}  "
              f"best={study.best_value:+.4f}  ({dur:.1f}s)", flush=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    print(f"Tuning fantasticlamm_sign on {TRAIN_START} -> {TRAIN_END} "
          f"({OBJECTIVE}), {args.n_trials} trials ...", flush=True)
    study.optimize(objective, n_trials=args.n_trials, callbacks=[log_cb])

    bp = study.best_params
    ratio_max = bp["ratio_base"] * bp["ratio_span"]
    print("\n=== BEST (train) ===")
    print(f"  train {OBJECTIVE} = {study.best_value:+.4f}")
    print(f"  ratio_base            = {bp['ratio_base']:.4f}")
    print(f"  ratio_max             = {ratio_max:.4f}  (span {bp['ratio_span']:.2f})")
    print(f"  centeredness_margin   = {bp['centeredness_margin']:.4f}")
    print(f"  shift_exponent        = {bp['shift_exponent']:.5f}")
    print(f"  deadband              = {bp['deadband']:.4f}")
    print(f"  sharpness             = {bp['sharpness']:.4f}")
    print(f"  trigger_alpha         = {bp['trigger_alpha']:.5f}")
    print(f"  reconcentration_hours = {bp['reconcentration_hours']:.2f}")

    # Rebuild best params and validate OOS on a fresh deposit.
    class _Fixed:
        def __init__(self, p):
            self.p = p
        def suggest_float(self, name, *a, **k):
            return self.p[name]
        number = -1
    best_params = trial_params(_Fixed(bp))

    fp_oos = base_fingerprint(OOS_START, OOS_END)
    oos_result = run(fp_oos, best_params, price_df)
    oos = metrics(oos_result)
    oos_obj = annualised_returns_over_hodl(oos_result)
    print(f"\n=== OOS validation (fresh $ {INITIAL:,.0f}, {OOS_START} -> {OOS_END}) ===")
    print(f"  {OBJECTIVE} = {oos_obj:+.4f}")
    print(f"  final value          = {oos['final_value']:,.2f}")
    print(f"  return vs depositHODL = {oos['returns_over_hodl_pct']:+.2f}%")
    print(f"  return vs uniformHODL = {oos['vs_uniform_hodl_pct']:+.2f}%")
    print(f"  fees collected        = {oos['fees']:,.2f}")


if __name__ == "__main__":
    main()
