"""Diagnostic kernels: with-fees scan that also outputs (Va, Vb) histories.

The production kernels return only reserves (+ fee revenue). For plotting the
price-ratio trajectory and rebalance periods we also need the virtual balances
at every step. These thin wrappers reuse the proven scan steps and augment the
``ys`` output with Va/Vb from the carry.

2-asset, ratio noise model, no protocol fees, no noise traders (matches the
fingerprint used by the tuning runs).
"""

import jax.numpy as jnp
from jax import jit
from jax.lax import scan
from jax.tree_util import Partial
from functools import partial

from quantammsim.pools.G3M.optimal_n_pool_arb import (
    precalc_shared_values_for_all_signatures,
    precalc_components_of_optimal_trade_across_prices,
)
from quantammsim.pools.reCLAMM.reclamm_reserves import (
    _reclamm_scan_step_with_fees_and_revenue,
)
from quantammsim.pools.fantasticlamm.fantasticlamm_reserves import (
    _fantasticlamm_scan_step_with_fees_and_revenue,
    _compute_desired_ratio_array,
)


# ---------------------------------------------------------------------------
# Scan-step wrappers — emit Va/Vb alongside reserves and fee revenue.
# ---------------------------------------------------------------------------

def _reclamm_step_full(carry, input_list, **kwargs):
    new_carry, (reserves, fee_rev) = _reclamm_scan_step_with_fees_and_revenue(
        carry, input_list, **kwargs,
    )
    return new_carry, (reserves, fee_rev, new_carry[1], new_carry[2])


def _fantasticlamm_step_full(carry, input_list, **kwargs):
    new_carry, (reserves, fee_rev) = _fantasticlamm_scan_step_with_fees_and_revenue(
        carry, input_list, **kwargs,
    )
    return new_carry, (reserves, fee_rev, new_carry[1], new_carry[2])


# ---------------------------------------------------------------------------
# Reclamm diagnostic
# ---------------------------------------------------------------------------

@jit
def reclamm_full_state(
    initial_reserves, initial_Va, initial_Vb, prices,
    centeredness_margin, daily_price_shift_base, fees,
    all_sig_variations, seconds_per_step=60.0,
):
    n_assets = 2
    weights = jnp.array([0.5, 0.5])
    gamma = 1.0 - fees

    _, atd, ttd, loo = precalc_shared_values_for_all_signatures(
        all_sig_variations, n_assets,
    )
    aiw, par, aoar = precalc_components_of_optimal_trade_across_prices(
        weights, prices, gamma, ttd, atd, loo,
    )

    T = prices.shape[0]
    gamma_a = jnp.full(T, gamma)
    zero_a = jnp.zeros(T)
    pru = jnp.zeros((T, 4), dtype=prices.dtype).at[:, 3].set(jnp.nan)
    lp = jnp.full(T, 1.0)

    scan_fn = Partial(
        _reclamm_step_full,
        weights=weights, tokens_to_drop=ttd,
        active_trade_directions=atd, n=n_assets,
        centeredness_margin=centeredness_margin,
        daily_price_shift_base=daily_price_shift_base,
        seconds_per_step=seconds_per_step,
        arc_length_speed=0.0, centeredness_scaling=False,
        protocol_fee_split=0.0, noise_trader_ratio=0.0,
        noise_model="ratio", noise_params={},
    )

    scan_inputs = [prices, aiw, par, aoar, gamma_a, zero_a, zero_a, pru, lp]
    carry_init = [
        initial_reserves, initial_Va, initial_Vb, lp[0],
        jnp.float64(0.0),  # step_idx
        jnp.float64(0.0), jnp.float64(0.0),  # active_start_ratio, target
        jnp.float64(0.0), jnp.float64(0.0),  # start_step, end_step
        jnp.array(False),  # active_enabled
    ]
    _, (reserves, fee_rev, Va_h, Vb_h) = scan(scan_fn, carry_init, scan_inputs)
    return reserves, fee_rev, Va_h, Vb_h


# ---------------------------------------------------------------------------
# Fantasticlamm diagnostic
# ---------------------------------------------------------------------------

@partial(jit, static_argnames=("trigger_mode", "window"))
def fantasticlamm_full_state(
    initial_reserves, initial_Va, initial_Vb, prices,
    centeredness_margin, daily_price_shift_base, fees,
    all_sig_variations,
    ratio_base, ratio_max, deadband, sharpness, trigger_alpha,
    max_narrow_log_step,
    trigger_beta=0.001, magnitude_k=0.05, w_static=0.5,
    trigger_mode="sign", window=60, seconds_per_step=60.0,
):
    n_assets = 2
    weights = jnp.array([0.5, 0.5])
    gamma = 1.0 - fees

    _, atd, ttd, loo = precalc_shared_values_for_all_signatures(
        all_sig_variations, n_assets,
    )
    aiw, par, aoar = precalc_components_of_optimal_trade_across_prices(
        weights, prices, gamma, ttd, atd, loo,
    )

    T = prices.shape[0]
    gamma_a = jnp.full(T, gamma)
    zero_a = jnp.zeros(T)
    pru = jnp.zeros((T, 4), dtype=prices.dtype).at[:, 3].set(jnp.nan)
    lp = jnp.full(T, 1.0)

    desired = _compute_desired_ratio_array(
        prices, trigger_mode, window, trigger_alpha,
        ratio_base, ratio_max, deadband, sharpness,
    )

    scan_fn = Partial(
        _fantasticlamm_step_full,
        weights=weights, tokens_to_drop=ttd,
        active_trade_directions=atd, n=n_assets,
        centeredness_margin=centeredness_margin,
        daily_price_shift_base=daily_price_shift_base,
        seconds_per_step=seconds_per_step,
        arc_length_speed=0.0, centeredness_scaling=False,
        protocol_fee_split=0.0, noise_trader_ratio=0.0,
        noise_model="ratio", noise_params={},
        trigger_mode=trigger_mode,
        ratio_base=ratio_base, ratio_max=ratio_max,
        deadband=deadband, sharpness=sharpness,
        trigger_alpha=trigger_alpha,
        trigger_beta=trigger_beta,
        magnitude_k=magnitude_k,
        w_static=w_static,
        max_narrow_log_step=max_narrow_log_step,
    )

    ema_slow_init = (initial_Vb + initial_reserves[1]) / jnp.maximum(
        initial_Va + initial_reserves[0], 1e-30,
    )
    scan_inputs = [
        prices, aiw, par, aoar, gamma_a, zero_a, zero_a, pru, lp, desired,
    ]
    carry_init = [
        initial_reserves, initial_Va, initial_Vb, lp[0],
        jnp.float64(0.0),
        jnp.float64(0.0), jnp.float64(0.0),
        jnp.float64(0.0), jnp.float64(0.0),
        jnp.array(False),
        jnp.float64(0.0),  # sign_drift
        ema_slow_init,     # ema_slow_spot
    ]
    _, (reserves, fee_rev, Va_h, Vb_h) = scan(scan_fn, carry_init, scan_inputs)
    return reserves, fee_rev, Va_h, Vb_h
