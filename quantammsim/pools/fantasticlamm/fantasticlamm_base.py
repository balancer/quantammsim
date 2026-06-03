"""fantasticlamm base pool — a reClAMM variant with a trend-aware price band.

Reuses ``ReClammPool`` for parameter handling, noise arrays and fee resolution,
overriding only the reserve kernels (which apply the spot-constant /
instant-up-windowed-down retarget) and pool-state setup (which reads the trigger
configuration). Concrete pools set ``_TRIGGER_MODE`` to select the trend signal.
"""

from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit
from jax.lax import dynamic_slice
from functools import partial
from typing import Dict, Any, Optional, NamedTuple

from quantammsim.pools.reCLAMM.reclamm import (
    ReClammPool,
    _prepare_dynamic_array,
    SHIFT_EXPONENT_DIVISOR,
)
from quantammsim.pools.reCLAMM.reclamm_reserves import initialise_reclamm_reserves
from quantammsim.pools.fantasticlamm.fantasticlamm_reserves import (
    _jax_calc_fantasticlamm_reserves_zero_fees,
    _jax_calc_fantasticlamm_reserves_with_fees,
    _jax_calc_fantasticlamm_reserves_and_fee_revenue_with_fees,
)


def _trigger_value(params, run_fingerprint, param_key, fp_key, default):
    """Resolve a trigger parameter, preferring learnable ``params``.

    Reading from ``params`` (a traced pytree) lets optuna vary the value across
    trials without recompiling the JIT kernel. Falls back to the static
    ``run_fingerprint`` key, then a default, for non-tuned forward runs.
    """
    if param_key in params:
        return jnp.squeeze(params[param_key])
    return jnp.float64(run_fingerprint.get(fp_key, default))


class _FLPoolState(NamedTuple):
    """Intermediate state produced by ``_init_pool_state``."""
    local_prices: jnp.ndarray
    arb_prices: jnp.ndarray
    initial_reserves: jnp.ndarray
    Va: jnp.ndarray
    Vb: jnp.ndarray
    centeredness_margin: jnp.ndarray
    daily_price_shift_base: jnp.ndarray
    seconds_per_step: float
    centeredness_scaling: bool
    window: int
    trigger_alpha: jnp.ndarray
    trigger_beta: jnp.ndarray
    magnitude_k: jnp.ndarray
    w_static: jnp.ndarray
    ratio_base: jnp.ndarray
    ratio_max: jnp.ndarray
    deadband: jnp.ndarray
    sharpness: jnp.ndarray
    max_narrow_log_step: jnp.ndarray


class FantasticLammBasePool(ReClammPool):
    """Base class for fantasticlamm pools; subclasses set ``_TRIGGER_MODE``.

    The initial ``price_ratio`` parameter is the *base* (concentrated) ratio.
    The trend trigger widens the ratio toward ``fantasticlamm_ratio_max`` when
    the market trends, instantly; it re-concentrates gradually (rate-limited by
    ``fantasticlamm_max_narrow_log_step``) when the trend subsides.
    """

    _TRIGGER_MODE = "sign"

    def _init_pool_state(self, params, run_fingerprint, prices, start_index):
        assert run_fingerprint["n_assets"] == 2

        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        local_prices = dynamic_slice(
            prices, start_index, (bout_length - 1, n_assets)
        )
        if run_fingerprint["arb_frequency"] != 1:
            arb_prices = local_prices[:: run_fingerprint["arb_frequency"]]
        else:
            arb_prices = local_prices

        ratio_base = jnp.squeeze(params["price_ratio"])
        centeredness_margin = jnp.squeeze(params["centeredness_margin"])
        if "shift_exponent" in params:
            daily_price_shift_base = (
                1.0 - jnp.squeeze(params["shift_exponent"]) / SHIFT_EXPONENT_DIVISOR
            )
        else:
            daily_price_shift_base = jnp.squeeze(params["daily_price_shift_base"])

        seconds_per_step = run_fingerprint["arb_frequency"] * 60.0

        onchain = run_fingerprint.get("reclamm_initial_state", None)
        if onchain is not None:
            initial_reserves = jnp.array(
                [onchain["Ra"], onchain["Rb"]], dtype=jnp.float64,
            )
            Va = jnp.float64(onchain["Va"])
            Vb = jnp.float64(onchain["Vb"])
        else:
            initial_pool_value = run_fingerprint["initial_pool_value"]
            initial_reserves, Va, Vb = initialise_reclamm_reserves(
                initial_pool_value, local_prices[0], ratio_base
            )

        centeredness_scaling = run_fingerprint.get(
            "reclamm_centeredness_scaling", False
        )

        return _FLPoolState(
            local_prices=local_prices,
            arb_prices=arb_prices,
            initial_reserves=initial_reserves,
            Va=Va,
            Vb=Vb,
            centeredness_margin=centeredness_margin,
            daily_price_shift_base=daily_price_shift_base,
            seconds_per_step=seconds_per_step,
            centeredness_scaling=centeredness_scaling,
            window=int(run_fingerprint.get("fantasticlamm_window", 60)),
            trigger_alpha=_trigger_value(
                params, run_fingerprint, "trigger_alpha",
                "fantasticlamm_trigger_alpha", 0.05,
            ),
            trigger_beta=_trigger_value(
                params, run_fingerprint, "trigger_beta",
                "fantasticlamm_trigger_beta", 0.001,
            ),
            magnitude_k=_trigger_value(
                params, run_fingerprint, "magnitude_k",
                "fantasticlamm_magnitude_k", 0.05,
            ),
            w_static=_trigger_value(
                params, run_fingerprint, "w_static",
                "fantasticlamm_w_static", 0.5,
            ),
            ratio_base=ratio_base,
            ratio_max=_trigger_value(
                params, run_fingerprint, "ratio_max",
                "fantasticlamm_ratio_max", 16.0,
            ),
            deadband=_trigger_value(
                params, run_fingerprint, "deadband",
                "fantasticlamm_deadband", 0.3,
            ),
            sharpness=_trigger_value(
                params, run_fingerprint, "sharpness",
                "fantasticlamm_sharpness", 1.0,
            ),
            max_narrow_log_step=_trigger_value(
                params, run_fingerprint, "max_narrow_log_step",
                "fantasticlamm_max_narrow_log_step", 0.005,
            ),
        )

    def _trigger_kwargs(self, s):
        """Common trigger keyword arguments forwarded to the kernels."""
        return dict(
            trigger_mode=self._TRIGGER_MODE,
            window=s.window,
            trigger_alpha=s.trigger_alpha,
            trigger_beta=s.trigger_beta,
            magnitude_k=s.magnitude_k,
            w_static=s.w_static,
            ratio_base=s.ratio_base,
            ratio_max=s.ratio_max,
            deadband=s.deadband,
            sharpness=s.sharpness,
            max_narrow_log_step=s.max_narrow_log_step,
        )

    @partial(jit, static_argnums=(2,))
    def calculate_reserves_and_fee_revenue_with_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
        lp_supply_array: Optional[jnp.ndarray] = None,
    ):
        s = self._init_pool_state(params, run_fingerprint, prices, start_index)

        bout_length = run_fingerprint["bout_length"]
        arb_freq = run_fingerprint["arb_frequency"]
        lp_prepared = (
            _prepare_dynamic_array(
                lp_supply_array, start_index, bout_length,
                arb_freq, s.arb_prices.shape[0],
            )
            if lp_supply_array is not None else None
        )

        noise_model = run_fingerprint.get("noise_model", "ratio")
        noise_params = run_fingerprint.get("reclamm_noise_params", None)
        if noise_params is not None and type(noise_params) is not dict:
            noise_params = dict(noise_params)

        _na = self._prepare_noise_arrays(
            prices, run_fingerprint, start_index,
            bout_length, arb_freq, s.arb_prices.shape[0],
        )

        if run_fingerprint["do_arb"]:
            return _jax_calc_fantasticlamm_reserves_and_fee_revenue_with_fees(
                s.initial_reserves, s.Va, s.Vb,
                s.arb_prices,
                s.centeredness_margin,
                s.daily_price_shift_base,
                s.seconds_per_step,
                fees=self._resolve_fees(params, run_fingerprint),
                arb_thresh=run_fingerprint["gas_cost"],
                arb_fees=run_fingerprint["arb_fees"],
                all_sig_variations=jnp.array(run_fingerprint["all_sig_variations"]),
                arc_length_speed=0.0,
                centeredness_scaling=s.centeredness_scaling,
                protocol_fee_split=run_fingerprint.get("protocol_fee_split", 0.0),
                noise_trader_ratio=run_fingerprint.get("noise_trader_ratio", 0.0),
                lp_supply_array=lp_prepared,
                noise_model=noise_model,
                noise_params=noise_params,
                volatility_array=_na["volatility"],
                dow_sin_array=_na["dow_sin"],
                dow_cos_array=_na["dow_cos"],
                noise_base_array=_na["noise_base"],
                noise_tvl_coeff_array=_na["noise_tvl_coeff"],
                competitor_tvl_array=_na.get("competitor_tvl"),
                **self._trigger_kwargs(s),
            )
        return (
            jnp.broadcast_to(s.initial_reserves, s.arb_prices.shape),
            jnp.zeros(s.arb_prices.shape[0]),
        )

    @partial(jit, static_argnums=(2,))
    def calculate_reserves_with_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
        lp_supply_array: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        s = self._init_pool_state(params, run_fingerprint, prices, start_index)

        bout_length = run_fingerprint["bout_length"]
        arb_freq = run_fingerprint["arb_frequency"]
        lp_prepared = (
            _prepare_dynamic_array(
                lp_supply_array, start_index, bout_length,
                arb_freq, s.arb_prices.shape[0],
            )
            if lp_supply_array is not None else None
        )

        noise_model = run_fingerprint.get("noise_model", "ratio")
        noise_params = run_fingerprint.get("reclamm_noise_params", None)
        if noise_params is not None and type(noise_params) is not dict:
            noise_params = dict(noise_params)

        _na = self._prepare_noise_arrays(
            prices, run_fingerprint, start_index,
            bout_length, arb_freq, s.arb_prices.shape[0],
        )

        if run_fingerprint["do_arb"]:
            return _jax_calc_fantasticlamm_reserves_with_fees(
                s.initial_reserves, s.Va, s.Vb,
                s.arb_prices,
                s.centeredness_margin,
                s.daily_price_shift_base,
                s.seconds_per_step,
                fees=self._resolve_fees(params, run_fingerprint),
                arb_thresh=run_fingerprint["gas_cost"],
                arb_fees=run_fingerprint["arb_fees"],
                all_sig_variations=jnp.array(run_fingerprint["all_sig_variations"]),
                arc_length_speed=0.0,
                centeredness_scaling=s.centeredness_scaling,
                protocol_fee_split=run_fingerprint.get("protocol_fee_split", 0.0),
                noise_trader_ratio=run_fingerprint.get("noise_trader_ratio", 0.0),
                lp_supply_array=lp_prepared,
                noise_model=noise_model,
                noise_params=noise_params,
                volatility_array=_na["volatility"],
                dow_sin_array=_na["dow_sin"],
                dow_cos_array=_na["dow_cos"],
                noise_base_array=_na["noise_base"],
                noise_tvl_coeff_array=_na["noise_tvl_coeff"],
                competitor_tvl_array=_na.get("competitor_tvl"),
                **self._trigger_kwargs(s),
            )
        return jnp.broadcast_to(s.initial_reserves, s.arb_prices.shape)

    @partial(jit, static_argnums=(2,))
    def _calculate_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        s = self._init_pool_state(params, run_fingerprint, prices, start_index)
        if run_fingerprint["do_arb"]:
            return _jax_calc_fantasticlamm_reserves_zero_fees(
                s.initial_reserves, s.Va, s.Vb,
                s.arb_prices,
                s.centeredness_margin,
                s.daily_price_shift_base,
                s.seconds_per_step,
                arc_length_speed=0.0,
                centeredness_scaling=s.centeredness_scaling,
                lp_supply_array=None,
                **self._trigger_kwargs(s),
            )
        return jnp.broadcast_to(s.initial_reserves, s.arb_prices.shape)

    def calculate_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        return self._calculate_reserves_zero_fees(
            params, run_fingerprint, prices, start_index, additional_oracle_input
        )

    def calculate_weights(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        s = self._init_pool_state(params, run_fingerprint, prices, start_index)
        reserves = self._calculate_reserves_zero_fees(
            params, run_fingerprint, prices, start_index, additional_oracle_input
        )
        value = reserves * s.arb_prices
        return value / jnp.sum(value, axis=-1, keepdims=True)

    def calculate_reserves_with_dynamic_inputs(self, *args, **kwargs):
        raise NotImplementedError(
            "fantasticlamm does not support dynamic-input runs yet "
            "(the trend trigger drives the price ratio internally)."
        )

    def calculate_reserves_and_fee_revenue_with_dynamic_inputs(self, *args, **kwargs):
        raise NotImplementedError(
            "fantasticlamm does not support dynamic-input runs yet "
            "(the trend trigger drives the price ratio internally)."
        )

    def is_trainable(self):
        return False
