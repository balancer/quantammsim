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
RULES = [
    "reclamm",
    "fantasticlamm_sign",
    "fantasticlamm_sign_ema_mag",
]


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

def reclamm_params(t):
    """3-param search matching the QuantAMM Docs PARAMETER_CONFIG."""
    p = {
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
    return p, {}


def _fantasticlamm_base_params(t):
    """The 8 core params shared by all fantasticlamm variants."""
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


def fantasticlamm_sign_params(t):
    return _fantasticlamm_base_params(t), {}


def fantasticlamm_sign_ema_params(t):
    """sign_ema (binary EMA-cross): adds trigger_beta."""
    p = _fantasticlamm_base_params(t)
    p["trigger_beta"] = jnp.array(
        t.suggest_float("trigger_beta", 1e-5, 0.1, log=True)
    )
    return p, {}


def fantasticlamm_sign_ema_mag_params(t):
    """sign_ema_mag: trigger_beta + magnitude_k."""
    p = _fantasticlamm_base_params(t)
    p["trigger_beta"] = jnp.array(
        t.suggest_float("trigger_beta", 1e-5, 0.1, log=True)
    )
    p["magnitude_k"] = jnp.array(
        t.suggest_float("magnitude_k", 1e-3, 0.5, log=True)
    )
    return p, {}


def fantasticlamm_sign_hybrid_params(t):
    """sign_hybrid: blends sign + sign_ema_mag with weight w_static in [0, 1]."""
    p = _fantasticlamm_base_params(t)
    p["trigger_beta"] = jnp.array(
        t.suggest_float("trigger_beta", 1e-5, 0.1, log=True)
    )
    p["magnitude_k"] = jnp.array(
        t.suggest_float("magnitude_k", 1e-3, 0.5, log=True)
    )
    p["w_static"] = jnp.array(t.suggest_float("w_static", 0.0, 1.0))
    return p, {}


# ER window candidates (minutes). Each unique value triggers one JIT recompile.
ER_WINDOWS = [240, 1440, 4320, 10080, 20160]  # 4h, 1d, 3d, 1w, 2w


def fantasticlamm_er_params(t):
    """ER: same base params (no trigger_beta/magnitude_k), plus categorical window."""
    p = _fantasticlamm_base_params(t)
    # trigger_alpha is unused by ER but harmless to leave at the sampled value.
    window = t.suggest_categorical("window", ER_WINDOWS)
    return p, {"fantasticlamm_window": int(window)}


SAMPLERS = {
    "reclamm": reclamm_params,
    "fantasticlamm_sign": fantasticlamm_sign_params,
    "fantasticlamm_sign_ema": fantasticlamm_sign_ema_params,
    "fantasticlamm_sign_ema_mag": fantasticlamm_sign_ema_mag_params,
    "fantasticlamm_sign_hybrid": fantasticlamm_sign_hybrid_params,
    "fantasticlamm_er": fantasticlamm_er_params,
}


def reconstruct_best(rule, bp):
    """Rebuild (params, fp_overrides) from optuna's best_params (no trial object)."""
    class _Fixed:
        number = -1
        def __init__(self, p): self.p = p
        def suggest_float(self, name, *a, **k): return self.p[name]
        def suggest_categorical(self, name, choices): return self.p[name]
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
            params, fp_overrides = sampler(t)
            fp_trial = {**fp_train, **fp_overrides}
            result = run(fp_trial, params, price_df)
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
        v_str = f"{v:.6f}" if isinstance(v, (int, float)) else str(v)
        print(f"    {k:22s} = {v_str}", flush=True)

    fp_oos = base_fingerprint(rule, tokens, OOS_START, OOS_END)
    best_params, best_fp_overrides = reconstruct_best(rule, study.best_params)
    fp_oos = {**fp_oos, **best_fp_overrides}
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
    global TRAIN_START, TRAIN_END, OOS_START, OOS_END
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs=2, default=["ETH", "USDC"])
    parser.add_argument("--n-trials", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-start", default=TRAIN_START)
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--oos-start", default=OOS_START)
    parser.add_argument("--oos-end", default=OOS_END)
    args = parser.parse_args()

    TRAIN_START = args.train_start
    TRAIN_END = args.train_end
    OOS_START = args.oos_start
    OOS_END = args.oos_end

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
