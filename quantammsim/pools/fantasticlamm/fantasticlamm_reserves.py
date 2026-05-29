"""fantasticlamm reserve math — a reClAMM variant.

Differences from reClAMM (all orthogonal to the underlying arb/fee machinery,
which is reused from ``reclamm_reserves``):

1. Price-ratio retargeting holds the *spot price* constant (reClAMM holds
   centeredness constant). Retargeting therefore never opens an arb against LPs.
2. Retargeting is *instant* when the new ratio is wider (deconcentration), and
   *windowed* (capped per-step) when narrower (concentration). A widen at any
   step preempts an in-progress narrow.
3. The target ratio is driven by a volatility/trend trigger (see ``triggers/``)
   that widens the band when the market trends monotonically, protecting LPs.

The per-step transform here only sets the band *width* (price ratio). reClAMM's
existing virtual-balance decay still tracks price *within* the band, so these
kernels compose the transform with reClAMM's proven scan steps.
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
    compute_price_ratio,
    compute_centeredness,
    _reclamm_scan_step_zero_fees,
    _reclamm_scan_step_with_fees_and_revenue,
)

from quantammsim.pools.fantasticlamm.triggers.efficiency_ratio import (
    efficiency_ratio_signal,
)
from quantammsim.pools.fantasticlamm.triggers.ewma_efficiency import (
    ewma_efficiency_signal,
)
from quantammsim.pools.fantasticlamm.triggers.sign_ewma import sign_ewma_update


def retarget_constant_spot(Ra, Rb, Va, Vb, target_price_ratio):
    """Retarget virtual balances to a price ratio while holding spot constant.

    Holds real balances fixed and solves for the virtual balances that achieve
    ``target_price_ratio`` while keeping the marginal price P = (Rb+Vb)/(Ra+Va)
    unchanged. Because the spot price does not move, the retarget itself opens
    no arbitrage against LPs and is therefore safe to apply at any size.

    With q = sqrt(target_price_ratio) and current spot P, the effective reserve
    Ea' = Ra + Va' solves the quadratic

        P*(1 - 1/q)*Ea'^2  -  (Rb + P*Ra)*Ea'  +  Ra*Rb  =  0

    whose larger root is the physical one (Va', Vb' > 0). The discriminant is
    non-negative for all positive reserves (AM-GM on Rb and P*Ra).

    Parameters
    ----------
    Ra, Rb : float
        Real balances (held fixed).
    Va, Vb : float
        Current virtual balances.
    target_price_ratio : float
        Desired max_price / min_price.

    Returns
    -------
    Va_new, Vb_new : float
        Retargeted virtual balances.
    """
    safe_ratio = jnp.maximum(target_price_ratio, 1.0 + 1e-12)
    q = jnp.sqrt(safe_ratio)

    Ea = Ra + Va
    Eb = Rb + Vb
    spot = Eb / jnp.maximum(Ea, 1e-30)

    a = spot * (1.0 - 1.0 / q)
    b = -(Rb + spot * Ra)
    c = Ra * Rb

    disc = jnp.maximum(b * b - 4.0 * a * c, 0.0)
    sqrt_disc = jnp.sqrt(disc)

    # Larger root; when a -> 0 (target ratio -> 1) virtuals diverge, so keep
    # the current effective reserve as a degenerate fallback.
    Ea_new = jnp.where(a > 1e-30, (-b + sqrt_disc) / (2.0 * a), Ea)

    Va_new = Ea_new - Ra
    Vb_new = spot * Ea_new - Rb
    return Va_new, Vb_new


def apply_windowed_ratio_update(
    Ra, Rb, Va, Vb, desired_price_ratio, max_narrow_log_step
):
    """Move the price ratio toward ``desired_price_ratio``, instant up / windowed down.

    Widening (desired >= current) is applied instantly. Narrowing is capped to a
    maximum downward step of ``max_narrow_log_step`` in log-ratio space, so a
    full concentration spans roughly ``log(span) / max_narrow_log_step`` steps.
    Because the comparison is against the *current* ratio every step, a widen
    automatically preempts an in-progress narrow.

    Parameters
    ----------
    Ra, Rb, Va, Vb : float
        Current real and virtual balances.
    desired_price_ratio : float
        Target ratio requested by the trigger this step.
    max_narrow_log_step : float
        Maximum allowed decrease in log(price_ratio) per step (>= 0).

    Returns
    -------
    Va_new, Vb_new : float
        Virtual balances after the (possibly rate-limited) retarget.
    """
    current_ratio = compute_price_ratio(Ra, Rb, Va, Vb)
    log_current = jnp.log(jnp.maximum(current_ratio, 1.0 + 1e-12))
    log_desired = jnp.log(jnp.maximum(desired_price_ratio, 1.0 + 1e-12))

    widen = log_desired >= log_current
    log_next = jnp.where(
        widen,
        log_desired,
        jnp.maximum(log_desired, log_current - max_narrow_log_step),
    )
    next_ratio = jnp.exp(log_next)
    return retarget_constant_spot(Ra, Rb, Va, Vb, next_ratio)


def response_curve(efficiency, ratio_base, ratio_max, deadband, sharpness=1.0):
    """Map a trend-efficiency signal in [0, 1] to a target price ratio.

    Geometric interpolation between ``ratio_base`` (tight, concentrated band for
    a choppy/ranging market) and ``ratio_max`` (wide, deconcentrated band for a
    monotone trend). Signal below ``deadband`` keeps the band at ``ratio_base``.

    Parameters
    ----------
    efficiency : float
        Trend strength in [0, 1]; 1 = perfectly monotone, 0 = pure chop.
    ratio_base, ratio_max : float
        Price-ratio endpoints (ratio_base <= ratio_max).
    deadband : float
        Efficiency threshold below which no deconcentration occurs.
    sharpness : float
        Exponent on the normalized signal; >1 reacts only to strong trends.

    Returns
    -------
    target_price_ratio : float
    """
    span = jnp.maximum(1.0 - deadband, 1e-12)
    frac = jnp.clip((efficiency - deadband) / span, 0.0, 1.0) ** sharpness
    log_base = jnp.log(jnp.maximum(ratio_base, 1.0 + 1e-12))
    log_max = jnp.log(jnp.maximum(ratio_max, 1.0 + 1e-12))
    return jnp.exp(log_base + frac * (log_max - log_base))


# ---------------------------------------------------------------------------
# Trigger -> desired-ratio precompute
# ---------------------------------------------------------------------------

def _compute_desired_ratio_array(
    prices, trigger_mode, window, alpha, ratio_base, ratio_max, deadband, sharpness
):
    """Precompute the per-step desired price ratio for exogenous triggers.

    For the exogenous price-series triggers ("er", "ewma") the efficiency
    signal — and therefore the desired ratio — depends only on the price path,
    so it is computed once outside the scan. The endogenous "sign" trigger
    depends on path-dependent pool state, so it is computed in-scan; this
    returns a placeholder array for that mode.
    """
    market_price = prices[:, 0] / prices[:, 1]
    if trigger_mode == "er":
        efficiency = efficiency_ratio_signal(market_price, window)
        return response_curve(efficiency, ratio_base, ratio_max, deadband, sharpness)
    if trigger_mode == "ewma":
        efficiency = ewma_efficiency_signal(market_price, alpha)
        return response_curve(efficiency, ratio_base, ratio_max, deadband, sharpness)
    return jnp.zeros(prices.shape[0], dtype=prices.dtype)


# ---------------------------------------------------------------------------
# Scan steps — retarget pre-transform composed with reClAMM inner steps
# ---------------------------------------------------------------------------

def _fantasticlamm_scan_step_zero_fees(
    carry_list,
    input_list,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    arc_length_speed,
    centeredness_scaling,
    trigger_mode,
    ratio_base,
    ratio_max,
    deadband,
    sharpness,
    trigger_alpha,
    max_narrow_log_step,
):
    """Zero-fee fantasticlamm step: retarget the band, then run reClAMM's step.

    Carry: [reserves (2,), Va, Vb, prev_lp_supply, sign_drift]
    Input: [prices (2,), lp_supply, desired_price_ratio]
    """
    reserves = carry_list[0]
    Va = carry_list[1]
    Vb = carry_list[2]
    drift_prev = carry_list[4]
    Ra = reserves[0]
    Rb = reserves[1]

    prices = input_list[0]
    lp_supply = input_list[1]
    desired_input = input_list[2]

    if trigger_mode == "sign":
        _, is_above = compute_centeredness(Ra, Rb, Va, Vb)
        drift_new, efficiency = sign_ewma_update(drift_prev, is_above, trigger_alpha)
        desired = response_curve(
            efficiency, ratio_base, ratio_max, deadband, sharpness
        )
    else:
        drift_new = drift_prev
        desired = desired_input

    Va_rt, Vb_rt = apply_windowed_ratio_update(
        Ra, Rb, Va, Vb, desired, max_narrow_log_step
    )

    inner_carry = [reserves, Va_rt, Vb_rt, carry_list[3]]
    new_inner_carry, new_reserves = _reclamm_scan_step_zero_fees(
        inner_carry,
        [prices, lp_supply],
        centeredness_margin=centeredness_margin,
        daily_price_shift_base=daily_price_shift_base,
        seconds_per_step=seconds_per_step,
        arc_length_speed=arc_length_speed,
        centeredness_scaling=centeredness_scaling,
    )

    new_carry = [
        new_reserves,
        new_inner_carry[1],
        new_inner_carry[2],
        new_inner_carry[3],
        drift_new,
    ]
    return new_carry, new_reserves


def _fantasticlamm_scan_step_with_fees_and_revenue(
    carry_list,
    input_list,
    weights,
    tokens_to_drop,
    active_trade_directions,
    n,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    arc_length_speed,
    centeredness_scaling,
    protocol_fee_split,
    noise_trader_ratio,
    noise_model,
    noise_params,
    trigger_mode,
    ratio_base,
    ratio_max,
    deadband,
    sharpness,
    trigger_alpha,
    max_narrow_log_step,
):
    """With-fees fantasticlamm step: retarget the band, then run reClAMM's step.

    Carry: reClAMM with-fees carry (10 elements) + [sign_drift].
    Input: reClAMM with-fees inputs + [desired_price_ratio] (appended last).
    """
    reserves = carry_list[0]
    Va = carry_list[1]
    Vb = carry_list[2]
    drift_prev = carry_list[10]
    Ra = reserves[0]
    Rb = reserves[1]

    desired_input = input_list[-1]

    if trigger_mode == "sign":
        _, is_above = compute_centeredness(Ra, Rb, Va, Vb)
        drift_new, efficiency = sign_ewma_update(drift_prev, is_above, trigger_alpha)
        desired = response_curve(
            efficiency, ratio_base, ratio_max, deadband, sharpness
        )
    else:
        drift_new = drift_prev
        desired = desired_input

    Va_rt, Vb_rt = apply_windowed_ratio_update(
        Ra, Rb, Va, Vb, desired, max_narrow_log_step
    )

    inner_carry = [
        reserves, Va_rt, Vb_rt,
        carry_list[3], carry_list[4], carry_list[5],
        carry_list[6], carry_list[7], carry_list[8], carry_list[9],
    ]
    new_inner_carry, (new_reserves, fee_revenue) = (
        _reclamm_scan_step_with_fees_and_revenue(
            inner_carry,
            input_list[:-1],
            weights=weights,
            tokens_to_drop=tokens_to_drop,
            active_trade_directions=active_trade_directions,
            n=n,
            centeredness_margin=centeredness_margin,
            daily_price_shift_base=daily_price_shift_base,
            seconds_per_step=seconds_per_step,
            arc_length_speed=arc_length_speed,
            centeredness_scaling=centeredness_scaling,
            protocol_fee_split=protocol_fee_split,
            noise_trader_ratio=noise_trader_ratio,
            noise_model=noise_model,
            noise_params=noise_params,
        )
    )

    new_carry = list(new_inner_carry) + [drift_new]
    return new_carry, (new_reserves, fee_revenue)


# ---------------------------------------------------------------------------
# JIT kernels
# ---------------------------------------------------------------------------

@partial(jit, static_argnames=("trigger_mode", "window"))
def _jax_calc_fantasticlamm_reserves_zero_fees(
    initial_reserves,
    initial_Va,
    initial_Vb,
    prices,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    arc_length_speed=0.0,
    centeredness_scaling=False,
    lp_supply_array=None,
    trigger_mode="sign",
    window=60,
    trigger_alpha=0.05,
    ratio_base=1.5,
    ratio_max=16.0,
    deadband=0.3,
    sharpness=1.0,
    max_narrow_log_step=0.01,
):
    """Calculate fantasticlamm reserves over time with zero fees."""
    if lp_supply_array is None:
        lp_supply_array = jnp.array(1.0)
    lp_supply_array = jnp.where(
        lp_supply_array.size == 1,
        jnp.full(prices.shape[0], lp_supply_array),
        lp_supply_array,
    )

    desired = _compute_desired_ratio_array(
        prices, trigger_mode, window, trigger_alpha,
        ratio_base, ratio_max, deadband, sharpness,
    )

    scan_fn = Partial(
        _fantasticlamm_scan_step_zero_fees,
        centeredness_margin=centeredness_margin,
        daily_price_shift_base=daily_price_shift_base,
        seconds_per_step=seconds_per_step,
        arc_length_speed=arc_length_speed,
        centeredness_scaling=centeredness_scaling,
        trigger_mode=trigger_mode,
        ratio_base=ratio_base,
        ratio_max=ratio_max,
        deadband=deadband,
        sharpness=sharpness,
        trigger_alpha=trigger_alpha,
        max_narrow_log_step=max_narrow_log_step,
    )

    carry_init = [
        initial_reserves, initial_Va, initial_Vb,
        lp_supply_array[0], jnp.float64(0.0),
    ]
    _, reserves = scan(scan_fn, carry_init, [prices, lp_supply_array, desired])
    return reserves


@partial(jit, static_argnames=("noise_model", "trigger_mode", "window"))
def _jax_calc_fantasticlamm_reserves_and_fee_revenue_with_fees(
    initial_reserves,
    initial_Va,
    initial_Vb,
    prices,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    fees=0.003,
    arb_thresh=0.0,
    arb_fees=0.0,
    all_sig_variations=None,
    arc_length_speed=0.0,
    centeredness_scaling=False,
    protocol_fee_split=0.0,
    noise_trader_ratio=0.0,
    lp_supply_array=None,
    noise_model="ratio",
    noise_params=None,
    volatility_array=None,
    dow_sin_array=None,
    dow_cos_array=None,
    noise_base_array=None,
    noise_tvl_coeff_array=None,
    competitor_tvl_array=None,
    trigger_mode="sign",
    window=60,
    trigger_alpha=0.05,
    ratio_base=1.5,
    ratio_max=16.0,
    deadband=0.3,
    sharpness=1.0,
    max_narrow_log_step=0.01,
):
    """Calculate fantasticlamm reserves and LP fee revenue over time with fees."""
    if lp_supply_array is None:
        lp_supply_array = jnp.array(1.0)
    lp_supply_array = jnp.where(
        lp_supply_array.size == 1,
        jnp.full(prices.shape[0], lp_supply_array),
        lp_supply_array,
    )

    n_assets = 2
    weights = jnp.array([0.5, 0.5])
    gamma = 1.0 - fees

    _, active_trade_directions, tokens_to_drop, leave_one_out_idxs = (
        precalc_shared_values_for_all_signatures(all_sig_variations, n_assets)
    )
    active_initial_weights, per_asset_ratios, all_other_assets_ratios = (
        precalc_components_of_optimal_trade_across_prices(
            weights, prices, gamma, tokens_to_drop,
            active_trade_directions, leave_one_out_idxs,
        )
    )

    gamma_array = jnp.full(prices.shape[0], gamma)
    arb_thresh_array = jnp.full(prices.shape[0], arb_thresh)
    arb_fees_array = jnp.full(prices.shape[0], arb_fees)
    price_ratio_updates = jnp.zeros((prices.shape[0], 4), dtype=prices.dtype)
    price_ratio_updates = price_ratio_updates.at[:, 3].set(jnp.nan)

    desired = _compute_desired_ratio_array(
        prices, trigger_mode, window, trigger_alpha,
        ratio_base, ratio_max, deadband, sharpness,
    )

    scan_fn = Partial(
        _fantasticlamm_scan_step_with_fees_and_revenue,
        weights=weights,
        tokens_to_drop=tokens_to_drop,
        active_trade_directions=active_trade_directions,
        n=n_assets,
        centeredness_margin=centeredness_margin,
        daily_price_shift_base=daily_price_shift_base,
        seconds_per_step=seconds_per_step,
        arc_length_speed=arc_length_speed,
        centeredness_scaling=centeredness_scaling,
        protocol_fee_split=protocol_fee_split,
        noise_trader_ratio=noise_trader_ratio,
        noise_model=noise_model,
        noise_params=noise_params if noise_params is not None else {},
        trigger_mode=trigger_mode,
        ratio_base=ratio_base,
        ratio_max=ratio_max,
        deadband=deadband,
        sharpness=sharpness,
        trigger_alpha=trigger_alpha,
        max_narrow_log_step=max_narrow_log_step,
    )

    scan_inputs = [
        prices, active_initial_weights, per_asset_ratios, all_other_assets_ratios,
        gamma_array, arb_thresh_array, arb_fees_array, price_ratio_updates,
        lp_supply_array,
    ]
    if noise_model in ("tsoukalas_sqrt", "tsoukalas_log", "loglinear"):
        scan_inputs.append(volatility_array)
    elif noise_model == "calibrated":
        scan_inputs.append(volatility_array)
        scan_inputs.append(dow_sin_array)
        scan_inputs.append(dow_cos_array)
    elif noise_model == "market_linear":
        scan_inputs.append(noise_base_array)
        scan_inputs.append(noise_tvl_coeff_array)
    elif noise_model == "mm_observed":
        scan_inputs.append(noise_base_array)
        scan_inputs.append(competitor_tvl_array)
    scan_inputs.append(desired)

    carry_init = [
        initial_reserves, initial_Va, initial_Vb,
        lp_supply_array[0],
        jnp.float64(0.0),  # step_idx
        jnp.float64(0.0),  # active_start_ratio
        jnp.float64(0.0),  # active_target_ratio
        jnp.float64(0.0),  # active_start_step
        jnp.float64(0.0),  # active_end_step
        jnp.array(False),  # active_enabled
        jnp.float64(0.0),  # sign_drift
    ]
    _, (reserves, fee_revenue) = scan(scan_fn, carry_init, scan_inputs)
    return reserves, fee_revenue


@partial(jit, static_argnames=("noise_model", "trigger_mode", "window"))
def _jax_calc_fantasticlamm_reserves_with_fees(
    initial_reserves,
    initial_Va,
    initial_Vb,
    prices,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    fees=0.003,
    arb_thresh=0.0,
    arb_fees=0.0,
    all_sig_variations=None,
    arc_length_speed=0.0,
    centeredness_scaling=False,
    protocol_fee_split=0.0,
    noise_trader_ratio=0.0,
    lp_supply_array=None,
    noise_model="ratio",
    noise_params=None,
    volatility_array=None,
    dow_sin_array=None,
    dow_cos_array=None,
    noise_base_array=None,
    noise_tvl_coeff_array=None,
    competitor_tvl_array=None,
    trigger_mode="sign",
    window=60,
    trigger_alpha=0.05,
    ratio_base=1.5,
    ratio_max=16.0,
    deadband=0.3,
    sharpness=1.0,
    max_narrow_log_step=0.01,
):
    """Calculate fantasticlamm reserves over time with fees (reserves only)."""
    reserves, _fee_revenue = _jax_calc_fantasticlamm_reserves_and_fee_revenue_with_fees(
        initial_reserves, initial_Va, initial_Vb, prices,
        centeredness_margin, daily_price_shift_base, seconds_per_step,
        fees=fees, arb_thresh=arb_thresh, arb_fees=arb_fees,
        all_sig_variations=all_sig_variations,
        arc_length_speed=arc_length_speed,
        centeredness_scaling=centeredness_scaling,
        protocol_fee_split=protocol_fee_split,
        noise_trader_ratio=noise_trader_ratio,
        lp_supply_array=lp_supply_array,
        noise_model=noise_model,
        noise_params=noise_params,
        volatility_array=volatility_array,
        dow_sin_array=dow_sin_array,
        dow_cos_array=dow_cos_array,
        noise_base_array=noise_base_array,
        noise_tvl_coeff_array=noise_tvl_coeff_array,
        competitor_tvl_array=competitor_tvl_array,
        trigger_mode=trigger_mode,
        window=window,
        trigger_alpha=trigger_alpha,
        ratio_base=ratio_base,
        ratio_max=ratio_max,
        deadband=deadband,
        sharpness=sharpness,
        max_narrow_log_step=max_narrow_log_step,
    )
    return reserves
