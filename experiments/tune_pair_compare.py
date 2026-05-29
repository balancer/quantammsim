"""Tune reCLAMM and fantasticlamm_sign on a token pair (via optuna), then
compare their OOS fresh-deposit performance vs HODL.

For each rule:
- Optuna maximises annualised_returns_over_hodl on a train window.
- Best config is validated on a held-out OOS window with a fresh $10k deposit.

A single orchestrator process tunes both pools (sharing the preloaded price
data) and prints a final HODL-relative comparison table.

Run:  python experiments/tune_pair_compare.py --tokens ETH USDC --n-trials 150
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
SECONDS_PER_STEP = 60.0  # arb_frequency (1) * 60

# Defaults (matching the AAVE/ETH study for reproducibility).
TRAIN_START = "2023-11-20 00:00:00"
TRAIN_END = "2025-11-20 00:00:00"
OOS_START = "2025-11-20 00:00:00"
OOS_END = "2026-05-20 00:00:00"

OBJECTIVE = "annualised_returns_over_hodl"
FEE = 0.003
INITIAL = 10_000.0
RULES = ["reclamm", "fantasticlamm_sign"]


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

def reclamm_params(t):
    """3-param search matching the QuantAMM Docs PARAMETER_CONFIG."""
    return {
        "price_ratio": jnp.array(
            t.suggest_float("price_ratio", 1.01, 200.0, log=True)
        ),
        "centeredness_margin": jnp.array(
            t.suggest_float("centeredness_margin", 0.01, 0.99)
        ),
        "shift_exponent": jnp.array(
            t.suggest_float("shift_exponent", 1e-3, 125.0, log=True)
        ),
    }


def fantasticlamm_sign_params(t):
    """8-param search for fantasticlamm_sign."""
    ratio_base = t.suggest_float("ratio_base", 1.01, 20.0, log=True)
    ratio_span = t.suggest_float("ratio_span", 1.5, 300.0, log=True)
    ratio_max = ratio_base * ratio_span
    reconcentration_hours = t.suggest_float(
        "reconcentration_hours", 1.0, 240.0, log=True
    )
    max_narrow_log_step = (
        math.log(ratio_max / ratio_base) * SECONDS_PER_STEP
        / (reconcentration_hours * 3600.0)
    )
    return {
        "price_ratio": jnp.array(ratio_base),
        "centeredness_margin": jnp.array(
            t.suggest_float("centeredness_margin", 0.01, 0.99)
        ),
        "shift_exponent": jnp.array(
            t.suggest_float("shift_exponent", 1e-3, 125.0, log=True)
        ),
        "ratio_max": jnp.array(ratio_max),
        "deadband": jnp.array(t.suggest_float("deadband", 0.0, 0.9)),
        "sharpness": jnp.array(t.suggest_float("sharpness", 0.3, 5.0)),
        "trigger_alpha": jnp.array(
            t.suggest_float("trigger_alpha", 1e-4, 0.3, log=True)
        ),
        "max_narrow_log_step": jnp.array(max_narrow_log_step),
    }


SAMPLERS = {
    "reclamm": reclamm_params,
    "fantasticlamm_sign": fantasticlamm_sign_params,
}


def reconstruct_best(rule, bp):
    """Rebuild a params dict from optuna's best_params (no trial object)."""
    class _Fixed:
        number = -1
        def __init__(self, p): self.p = p
        def suggest_float(self, name, *a, **k): return self.p[name]
    return SAMPLERS[rule](_Fixed(bp))


# ---------------------------------------------------------------------------
# Simulation and metrics
# ---------------------------------------------------------------------------

def base_fingerprint(rule, tokens, start, end):
    return {
        "rule": rule,
        "tokens": list(tokens),
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


def run(fp, params, price_df):
    return do_run_on_historic_data(
        run_fingerprint=copy.deepcopy(fp), params=params,
        root=DATA_ROOT, price_data=price_df, verbose=False,
    )


def annualised_returns_over_hodl(result):
    value = np.asarray(result["value"]).reshape(-1)
    prices = np.asarray(result["prices"])
    res0 = np.asarray(result["reserves"])[0]
    deposit_hodl = (res0 * prices[-1]).sum()
    T = value.shape[0]
    return float((value[-1] / deposit_hodl) ** (365.0 * 24 * 60 / (T - 1)) - 1.0)


def metrics(result):
    value = np.asarray(result["value"]).reshape(-1)
    prices = np.asarray(result["prices"])
    res0 = np.asarray(result["reserves"])[0]
    deposit_hodl = float((res0 * prices[-1]).sum())
    units = (INITIAL / 2.0) / prices[0]
    uniform_hodl = float((prices * units).sum(axis=1)[-1])
    fees = float(np.asarray(result["fee_revenue"]).sum())
    return {
        "final_value": float(result["final_value"]),
        "deposit_hodl": deposit_hodl,
        "uniform_hodl": uniform_hodl,
        "vs_deposit_hodl_pct": 100.0 * (value[-1] / deposit_hodl - 1.0),
        "vs_uniform_hodl_pct": 100.0 * (value[-1] / uniform_hodl - 1.0),
        "fees": fees,
    }


# ---------------------------------------------------------------------------
# Tune one pool
# ---------------------------------------------------------------------------

def tune(rule, tokens, n_trials, seed, price_df):
    print(f"\n=== TUNING {rule} ({tokens[0]}/{tokens[1]}) on "
          f"{TRAIN_START} -> {TRAIN_END}, {n_trials} trials ===", flush=True)
    fp_train = base_fingerprint(rule, tokens, TRAIN_START, TRAIN_END)
    sampler = SAMPLERS[rule]

    def objective(t):
        try:
            result = run(fp_train, sampler(t), price_df)
            obj = annualised_returns_over_hodl(result)
        except Exception as e:  # noqa: BLE001
            print(f"  [{rule}] trial {t.number}: FAILED "
                  f"({type(e).__name__}: {e})", flush=True)
            return -1e9
        return obj if np.isfinite(obj) else -1e9

    def log_cb(study, trial):
        dur = trial.duration.total_seconds() if trial.duration else float("nan")
        print(f"  [{rule}] trial {trial.number:3d}: obj={trial.value:+.4f}  "
              f"best={study.best_value:+.4f}  ({dur:.1f}s)", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[log_cb])

    print(f"\n--- BEST {rule} (train annualised_returns_over_hodl = "
          f"{study.best_value:+.4f}) ---", flush=True)
    for k, v in study.best_params.items():
        print(f"    {k:22s} = {v:.6f}", flush=True)

    fp_oos = base_fingerprint(rule, tokens, OOS_START, OOS_END)
    best_params = reconstruct_best(rule, study.best_params)
    oos_result = run(fp_oos, best_params, price_df)
    oos = metrics(oos_result)
    oos["annualised_vs_hodl"] = annualised_returns_over_hodl(oos_result)
    return {
        "rule": rule, "best_params": dict(study.best_params),
        "train_obj": float(study.best_value), **oos,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs=2, default=["ETH", "USDC"])
    parser.add_argument("--n-trials", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokens = args.tokens
    print(f"Preloading {tokens} price data from {DATA_ROOT} ...", flush=True)
    price_df = get_historic_parquet_data(
        sorted(tokens), ["close"], str(DATA_ROOT) + "/"
    )
    print(f"  loaded {price_df.shape[0]:,} rows", flush=True)

    results = [tune(r, tokens, args.n_trials, args.seed, price_df) for r in RULES]

    # Both runs share the same OOS window/price series, so any result's
    # uniform_hodl is the common yardstick.
    uniform_hodl = results[0]["uniform_hodl"]

    print(f"\n=========== OOS comparison ({tokens[0]}/{tokens[1]}, "
          f"fresh ${INITIAL:,.0f}, {OOS_START} -> {OOS_END}) ===========")
    print(f"{'pool':24s}{'annual vs HODL %':>18s}{'final $':>12s}"
          f"{'vs depHODL %':>14s}{'vs unifHODL %':>15s}{'fees $':>12s}")
    print("-" * 95)
    for r in results:
        print(f"{r['rule']+' (tuned)':24s}"
              f"{100.0*r['annualised_vs_hodl']:>18.2f}"
              f"{r['final_value']:>12,.2f}"
              f"{r['vs_deposit_hodl_pct']:>14.2f}"
              f"{r['vs_uniform_hodl_pct']:>15.2f}"
              f"{r['fees']:>12,.2f}")
    print("-" * 95)
    print(f"{'uniform 50/50 HODL':24s}{'':>18s}{uniform_hodl:>12,.2f}")


if __name__ == "__main__":
    main()
