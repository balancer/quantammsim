"""Compare reCLAMM vs the three fantasticlamm variants on AAVE/ETH.

Test #4: a fresh-deposit backtest on the trending OOS window
(2025-11-20 -> 2026-05-20), 0.3% fee, $10k. The reCLAMM baseline uses the
optuna-tuned config from the QuantAMM Simulation Docs (price_ratio 162.9,
centeredness_margin 0.964, shift_exponent 0.135). The fantasticlamm pools share
those inner-decay params but start concentrated (ratio_base 1.1) and widen the
band toward ratio_max 200 when the market trends, re-concentrating over ~24h.

Reports, per pool: final value from $10k, total return, return vs uniform
50/50 HODL (the common yardstick), return vs its own deposit-HODL, and total
LP fees collected.

Run inside the venv:  python scripts/compare_fantasticlamm.py
"""

import copy
import math
from pathlib import Path

import numpy as np
import jax.numpy as jnp

import quantammsim
from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
from quantammsim.runners.jax_runners import do_run_on_historic_data

# Data lives next to the imported quantammsim package (download_data.py writes
# <TICKER>_USD.parquet there). Pin root so the run uses these files regardless
# of which quantammsim copy is on sys.path.
DATA_ROOT = Path(quantammsim.__file__).parent / "data"


def to_daily_price_shift_base(exp):
    return 1.0 - exp / 124649.0


TOKENS = ["AAVE", "ETH"]
START = "2025-11-20 00:00:00"
END = "2026-05-20 00:00:00"
FEE = 0.003
INITIAL = 10_000.0

RATIO_BASE = 1.1
RATIO_MAX = 200.0
SECONDS_PER_STEP = 60.0  # arb_frequency (1) * 60
RECONCENTRATION_HOURS = 24.0

# Per-step log-ratio cap that unwinds RATIO_MAX -> RATIO_BASE over 24h.
MAX_NARROW_LOG_STEP = (
    math.log(RATIO_MAX / RATIO_BASE) * SECONDS_PER_STEP / (RECONCENTRATION_HOURS * 3600.0)
)

# Shared inner-decay params (the reCLAMM tuned winner for this pair).
SHARED = dict(
    centeredness_margin=0.964,
    daily_price_shift_base=to_daily_price_shift_base(0.135),
)


def base_fingerprint(rule):
    fp = copy.deepcopy(run_fingerprint_defaults)
    fp["tokens"] = TOKENS
    fp["rule"] = rule
    fp["startDateString"] = START
    fp["endDateString"] = END
    fp["chunk_period"] = 60
    fp["weight_interpolation_period"] = 60
    fp["initial_pool_value"] = INITIAL
    fp["do_arb"] = True
    fp["fees"] = FEE
    fp["gas_cost"] = 0.0
    fp["arb_fees"] = 0.0
    if rule.startswith("fantasticlamm"):
        fp["fantasticlamm_ratio_max"] = RATIO_MAX
        fp["fantasticlamm_deadband"] = 0.3
        fp["fantasticlamm_sharpness"] = 1.0
        fp["fantasticlamm_window"] = 120
        fp["fantasticlamm_trigger_alpha"] = 0.02
        fp["fantasticlamm_max_narrow_log_step"] = MAX_NARROW_LOG_STEP
    return fp


def params_for(rule):
    price_ratio = RATIO_BASE if rule.startswith("fantasticlamm") else 162.9
    return {
        "price_ratio": jnp.array(price_ratio),
        "centeredness_margin": jnp.array(SHARED["centeredness_margin"]),
        "daily_price_shift_base": jnp.array(SHARED["daily_price_shift_base"]),
    }


def summarise(rule):
    result = do_run_on_historic_data(
        run_fingerprint=base_fingerprint(rule), params=params_for(rule),
        root=DATA_ROOT, verbose=False,
    )
    value = np.asarray(result["value"]).reshape(-1)
    prices = np.asarray(result["prices"])
    reserves0 = np.asarray(result["reserves"])[0]
    final_value = float(result["final_value"])
    fees = float(np.asarray(result["fee_revenue"]).sum())

    # Deposit-HODL: hold the basket this pool actually deposited at t=0.
    deposit_hodl_final = float((reserves0 * prices[-1]).sum())
    # Uniform 50/50 HODL: split capital evenly at t=0, never trade.
    units = (INITIAL / 2.0) / prices[0]
    uniform_hodl_final = float((prices * units).sum(axis=1)[-1])

    return {
        "rule": rule,
        "final_value": final_value,
        "return_pct": 100.0 * (final_value / INITIAL - 1.0),
        "vs_uniform_hodl_pct": 100.0 * (final_value / uniform_hodl_final - 1.0),
        "vs_deposit_hodl_pct": 100.0 * (final_value / deposit_hodl_final - 1.0),
        "fees": fees,
        "uniform_hodl_final": uniform_hodl_final,
    }


def main():
    print(f"AAVE/ETH  {START} -> {END}   fee={FEE:.3%}   deposit=${INITIAL:,.0f}")
    print(f"fantasticlamm: ratio_base={RATIO_BASE}, ratio_max={RATIO_MAX}, "
          f"max_narrow_log_step={MAX_NARROW_LOG_STEP:.6f} (~{RECONCENTRATION_HOURS:.0f}h unwind)")
    print()

    rules = ["reclamm", "fantasticlamm_er", "fantasticlamm_ewma", "fantasticlamm_sign"]
    rows = [summarise(r) for r in rules]

    uniform = rows[0]["uniform_hodl_final"]
    print(f"{'pool':22s}{'final $':>12s}{'return %':>10s}"
          f"{'vs HODL50 %':>13s}{'vs depHODL %':>14s}{'fees $':>12s}")
    print("-" * 83)
    for r in rows:
        print(f"{r['rule']:22s}{r['final_value']:>12,.2f}{r['return_pct']:>10.2f}"
              f"{r['vs_uniform_hodl_pct']:>13.2f}{r['vs_deposit_hodl_pct']:>14.2f}"
              f"{r['fees']:>12,.2f}")
    print("-" * 83)
    print(f"{'uniform 50/50 HODL':22s}{uniform:>12,.2f}")


if __name__ == "__main__":
    main()
