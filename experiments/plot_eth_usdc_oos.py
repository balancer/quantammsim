"""Plot tuned reCLAMM vs fantasticlamm_sign on the ETH/USDC OOS window.

Three charts (all on the OOS fresh-deposit run, 2025-11-20 -> 2026-05-20,
$10k, fee 0.3%):

  1. USD value of each pool over time, vs uniform 50/50 HODL.
  2. fantasticlamm price ratio over time, overlaid with the ETH/USDC market
     price (twin y-axis).
  3. Same value lines as chart 1 with translucent shaded bands marking the
     periods each pool is rebalancing (centeredness < centeredness_margin).

PNGs are saved under ``results/eth_usdc_oos/``.

Run:  python experiments/plot_eth_usdc_oos.py
"""

import os
import sys

# Ensure the worktree copy of quantammsim is imported (not an installed one),
# and make sibling modules in this directory importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import math
import numpy as np
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product

import quantammsim
from quantammsim.pools.reCLAMM.reclamm_reserves import (
    initialise_reclamm_reserves,
    compute_centeredness,
    compute_price_ratio,
)
from quantammsim.utils.data_processing.historic_data_utils import (
    get_historic_parquet_data,
)

from _diagnostics import reclamm_full_state, fantasticlamm_full_state


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path(quantammsim.__file__).parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "eth_usdc_oos"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOKENS = ["ETH", "USDC"]  # alphabetical -> matches get_historic_parquet_data order
OOS_START = "2025-11-20 00:00:00"
OOS_END = "2026-05-20 00:00:00"
INITIAL = 10_000.0
FEE = 0.003

ALL_SIG_VARIATIONS_2 = jnp.array([
    list(s) for s in product([1, 0, -1], repeat=2)
    if sum(1 for x in s if x == 1) == 1 and sum(1 for x in s if x == -1) == 1
])
SHIFT_DIVISOR = 124649.0

# ---- Best tuned params from the 150-trial optuna run on 2023-11 -> 2025-11.
RECLAMM_BEST = dict(
    price_ratio=9.353684,
    centeredness_margin=0.065530,
    shift_exponent=0.016790,
)
FANT_BEST = dict(
    ratio_base=1.014837,
    ratio_span=6.961556,
    reconcentration_hours=134.510169,
    centeredness_margin=0.010854,
    shift_exponent=0.067607,
    deadband=0.356527,
    sharpness=2.317129,
    trigger_alpha=0.014983,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_oos_prices():
    """Return (prices_array (T,2), dates DatetimeIndex)."""
    df = get_historic_parquet_data(
        sorted(TOKENS), ["close"], str(DATA_ROOT) + "/",
    )
    start_ms = int(pd.Timestamp(OOS_START).timestamp() * 1000)
    end_ms = int(pd.Timestamp(OOS_END).timestamp() * 1000)
    mask = (df.index >= start_ms) & (df.index <= end_ms)
    df = df.loc[mask, [f"close_{t}" for t in sorted(TOKENS)]]
    prices = jnp.array(df.to_numpy(dtype=np.float64))
    dates = pd.to_datetime(df.index, unit="ms")
    return prices, dates


# ---------------------------------------------------------------------------
# Run both pools on the OOS window
# ---------------------------------------------------------------------------

def run_pools(prices):
    rc = RECLAMM_BEST
    fl = FANT_BEST

    rc_init_res, rc_Va, rc_Vb = initialise_reclamm_reserves(
        INITIAL, prices[0], jnp.array(rc["price_ratio"]),
    )
    fl_init_res, fl_Va, fl_Vb = initialise_reclamm_reserves(
        INITIAL, prices[0], jnp.array(fl["ratio_base"]),
    )

    rc_dpsb = 1.0 - rc["shift_exponent"] / SHIFT_DIVISOR
    fl_dpsb = 1.0 - fl["shift_exponent"] / SHIFT_DIVISOR
    fl_ratio_max = fl["ratio_base"] * fl["ratio_span"]
    fl_mnls = (
        math.log(fl_ratio_max / fl["ratio_base"]) * 60.0
        / (fl["reconcentration_hours"] * 3600.0)
    )

    rc_reserves, rc_fee, rc_Va_h, rc_Vb_h = reclamm_full_state(
        rc_init_res, rc_Va, rc_Vb, prices,
        centeredness_margin=jnp.float64(rc["centeredness_margin"]),
        daily_price_shift_base=jnp.float64(rc_dpsb),
        fees=jnp.float64(FEE),
        all_sig_variations=ALL_SIG_VARIATIONS_2,
    )
    fl_reserves, fl_fee, fl_Va_h, fl_Vb_h = fantasticlamm_full_state(
        fl_init_res, fl_Va, fl_Vb, prices,
        centeredness_margin=jnp.float64(fl["centeredness_margin"]),
        daily_price_shift_base=jnp.float64(fl_dpsb),
        fees=jnp.float64(FEE),
        all_sig_variations=ALL_SIG_VARIATIONS_2,
        ratio_base=jnp.float64(fl["ratio_base"]),
        ratio_max=jnp.float64(fl_ratio_max),
        deadband=jnp.float64(fl["deadband"]),
        sharpness=jnp.float64(fl["sharpness"]),
        trigger_alpha=jnp.float64(fl["trigger_alpha"]),
        max_narrow_log_step=jnp.float64(fl_mnls),
    )
    return {
        "reclamm": {
            "reserves": np.asarray(rc_reserves), "Va": np.asarray(rc_Va_h),
            "Vb": np.asarray(rc_Vb_h), "fee": np.asarray(rc_fee),
            "centeredness_margin": rc["centeredness_margin"],
        },
        "fantasticlamm_sign": {
            "reserves": np.asarray(fl_reserves), "Va": np.asarray(fl_Va_h),
            "Vb": np.asarray(fl_Vb_h), "fee": np.asarray(fl_fee),
            "centeredness_margin": fl["centeredness_margin"],
        },
        "init_reserves": np.asarray(rc_init_res),  # both = INITIAL/2 each side
    }


# ---------------------------------------------------------------------------
# Derived trajectories
# ---------------------------------------------------------------------------

def derive(p, prices_np):
    """USD value, centeredness, price_ratio, is_rebalancing per-step."""
    Ra = p["reserves"][:, 0]; Rb = p["reserves"][:, 1]
    Va = p["Va"]; Vb = p["Vb"]
    value = Ra * prices_np[:, 0] + Rb * prices_np[:, 1]
    num = Ra * Vb; den = Va * Rb
    is_above = num > den
    centeredness = np.where(
        is_above,
        den / np.maximum(num, 1e-30),
        num / np.maximum(den, 1e-30),
    )
    # max_price / min_price = ((Ra+Va)(Rb+Vb))^2 / (Va*Vb)^2
    L = (Ra + Va) * (Rb + Vb)
    price_ratio = (L / (Va * Vb)) ** 2
    is_rebalancing = centeredness < p["centeredness_margin"]
    return value, centeredness, price_ratio, is_rebalancing


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_value(dates, rc_value, fl_value, dep_hodl, unif_hodl, path):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(dates, rc_value, label="reCLAMM (tuned)", color="#cc3333", lw=1.6)
    ax.plot(dates, fl_value, label="fantasticlamm_sign (tuned)", color="#2e7d32", lw=1.6)
    ax.plot(dates, dep_hodl, label="deposit-HODL", color="#888888",
            lw=1.2, ls="--")
    ax.plot(dates, unif_hodl, label="uniform 50/50 HODL", color="#333333",
            lw=1.2, ls=":")
    ax.set_title("Pool USD value over the OOS window (fresh $10k, ETH/USDC)")
    ax.set_xlabel("date"); ax.set_ylabel("USD value")
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_ratio_vs_price(dates, fl_ratio, eth_usdc, path):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(dates, fl_ratio, label="fantasticlamm price ratio",
            color="#2e7d32", lw=1.5)
    ax.set_yscale("log")
    ax.set_ylabel("price_ratio (log)", color="#2e7d32")
    ax.tick_params(axis="y", labelcolor="#2e7d32")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(dates, eth_usdc, label="ETH/USDC", color="#1f3a8a", lw=1.0, alpha=0.85)
    ax2.set_ylabel("ETH price (USDC)", color="#1f3a8a")
    ax2.tick_params(axis="y", labelcolor="#1f3a8a")

    ax.set_title("fantasticlamm price ratio vs ETH/USDC market price")
    ax.set_xlabel("date")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_value_with_rebalance(dates, rc_value, fl_value, dep_hodl,
                              rc_reb, fl_reb, path):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ymin, ymax = ax.get_ylim()  # placeholder

    ax.plot(dates, rc_value, label="reCLAMM (tuned)", color="#cc3333", lw=1.6, zorder=3)
    ax.plot(dates, fl_value, label="fantasticlamm_sign (tuned)", color="#2e7d32", lw=1.6, zorder=3)
    ax.plot(dates, dep_hodl, label="deposit-HODL", color="#888888", lw=1.0, ls="--", zorder=2)

    ax.set_title("Pool USD value with rebalance periods "
                 "(shaded = centeredness < margin)")
    ax.set_xlabel("date"); ax.set_ylabel("USD value")
    ax.grid(alpha=0.3)

    # fill_between is computed against the y-limits derived from data
    ymin = min(rc_value.min(), fl_value.min(), dep_hodl.min()) * 0.98
    ymax = max(rc_value.max(), fl_value.max(), dep_hodl.max()) * 1.02
    ax.set_ylim(ymin, ymax)
    ax.fill_between(dates, ymin, ymax, where=rc_reb, color="#cc3333",
                    alpha=0.10, step="post", linewidth=0,
                    label="reCLAMM rebalancing", zorder=1)
    ax.fill_between(dates, ymin, ymax, where=fl_reb, color="#2e7d32",
                    alpha=0.10, step="post", linewidth=0,
                    label="fantasticlamm rebalancing", zorder=1)

    ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading OOS prices from {DATA_ROOT} ...", flush=True)
    prices, dates = load_oos_prices()
    prices_np = np.asarray(prices)
    print(f"  {prices.shape[0]:,} minute samples  "
          f"{dates[0]} -> {dates[-1]}", flush=True)

    print("Running tuned reCLAMM and fantasticlamm_sign on OOS ...", flush=True)
    pools = run_pools(prices)

    rc_value, rc_cent, _, rc_reb = derive(pools["reclamm"], prices_np)
    fl_value, fl_cent, fl_ratio, fl_reb = derive(
        pools["fantasticlamm_sign"], prices_np,
    )

    # HODL baselines
    dep_hodl = (pools["init_reserves"] * prices_np).sum(axis=1)
    units = (INITIAL / 2.0) / prices_np[0]
    unif_hodl = (prices_np * units).sum(axis=1)

    # Decimate to hourly to keep plots responsive (260k -> ~4400 points).
    step = 60
    d = dates[::step]
    rc_v = rc_value[::step]; fl_v = fl_value[::step]
    dep = dep_hodl[::step]; unif = unif_hodl[::step]
    rc_r = rc_reb[::step]; fl_r = fl_reb[::step]
    fl_ratio_d = fl_ratio[::step]
    eth_usdc = (prices_np[:, 0] / prices_np[:, 1])[::step]

    p1 = RESULTS_DIR / "01_value_over_time.png"
    p2 = RESULTS_DIR / "02_price_ratio_vs_eth_usdc.png"
    p3 = RESULTS_DIR / "03_value_with_rebalance.png"

    plot_value(d, rc_v, fl_v, dep, unif, p1)
    plot_ratio_vs_price(d, fl_ratio_d, eth_usdc, p2)
    plot_value_with_rebalance(d, rc_v, fl_v, dep, rc_r, fl_r, p3)

    print(f"\nFinal values @ {dates[-1]}:")
    print(f"  reCLAMM           ${rc_value[-1]:,.2f}")
    print(f"  fantasticlamm     ${fl_value[-1]:,.2f}")
    print(f"  deposit-HODL      ${dep_hodl[-1]:,.2f}")
    print(f"  uniform-HODL      ${unif_hodl[-1]:,.2f}")
    print(f"  reCLAMM fees      ${pools['reclamm']['fee'].sum():,.2f}")
    print(f"  fantasticlamm fees ${pools['fantasticlamm_sign']['fee'].sum():,.2f}")
    print(f"  reCLAMM rebalancing fraction      {rc_reb.mean():.1%}")
    print(f"  fantasticlamm rebalancing fraction {fl_reb.mean():.1%}")
    print("\nCharts saved:")
    for p in (p1, p2, p3):
        print(f"  {p}")


if __name__ == "__main__":
    main()
