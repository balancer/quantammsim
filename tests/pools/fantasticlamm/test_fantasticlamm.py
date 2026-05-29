"""Tests for the fantasticlamm pool (reClAMM variant).

Covers the three behavioural changes vs reClAMM:
1. spot-constant price-ratio retargeting,
2. instant-up / windowed-down ratio updates,
3. the three trend triggers (exact ER, EWMA-efficiency, sign-EWMA),
plus kernel composition and pool registration.
"""

import copy
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

import quantammsim
from quantammsim.pools.creator import create_pool
from quantammsim.pools.reCLAMM.reclamm_reserves import (
    initialise_reclamm_reserves,
    compute_price_ratio,
    _jax_calc_reclamm_reserves_zero_fees,
)
from quantammsim.pools.fantasticlamm.fantasticlamm_reserves import (
    retarget_constant_spot,
    apply_windowed_ratio_update,
    response_curve,
    _jax_calc_fantasticlamm_reserves_zero_fees,
    _jax_calc_fantasticlamm_reserves_with_fees,
)
from quantammsim.pools.fantasticlamm.fantasticlamm_er import (
    FantasticLammEfficiencyRatioPool,
)
from quantammsim.pools.fantasticlamm.fantasticlamm_ewma import (
    FantasticLammEwmaEfficiencyPool,
)
from quantammsim.pools.fantasticlamm.fantasticlamm_sign import (
    FantasticLammSignEwmaPool,
)
from quantammsim.pools.fantasticlamm.triggers.efficiency_ratio import (
    efficiency_ratio_signal,
)
from quantammsim.pools.fantasticlamm.triggers.ewma_efficiency import (
    ewma_efficiency_signal,
)
from quantammsim.pools.fantasticlamm.triggers.sign_ewma import (
    sign_ewma_init,
    sign_ewma_update,
)

ALL_SIG_VARIATIONS_2 = jnp.array([[1, -1], [-1, 1]])
TEST_DATA_DIR = Path(quantammsim.__file__).parent.parent / "tests" / "data"


def _spot(Ra, Rb, Va, Vb):
    return (Rb + Vb) / (Ra + Va)


# ---------------------------------------------------------------------------
# #1 spot-constant retarget
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "Ra,Rb,Va,Vb,target",
    [
        (100.0, 100.0, 200.0, 200.0, 4.0),    # narrow, symmetric
        (100.0, 100.0, 200.0, 200.0, 9.0),    # widen, symmetric
        (100.0, 457.98, 157.98, 315.96, 2.0),  # narrow, asymmetric
        (50.0, 900.0, 80.0, 1200.0, 12.0),     # widen, asymmetric
    ],
)
def test_retarget_hits_target_ratio_and_preserves_spot(Ra, Rb, Va, Vb, target):
    spot0 = _spot(Ra, Rb, Va, Vb)
    Va2, Vb2 = retarget_constant_spot(Ra, Rb, Va, Vb, target)
    Va2, Vb2 = float(Va2), float(Vb2)

    assert Va2 > 0 and Vb2 > 0
    npt.assert_allclose(
        float(compute_price_ratio(Ra, Rb, Va2, Vb2)), target, rtol=1e-9
    )
    npt.assert_allclose(_spot(Ra, Rb, Va2, Vb2), spot0, rtol=1e-9)


# ---------------------------------------------------------------------------
# #2 instant-up / windowed-down
# ---------------------------------------------------------------------------

def test_widen_is_instant():
    Ra, Rb, Va, Vb = 100.0, 100.0, 200.0, 200.0
    Va2, Vb2 = apply_windowed_ratio_update(
        Ra, Rb, Va, Vb, desired_price_ratio=20.0, max_narrow_log_step=0.01
    )
    npt.assert_allclose(
        float(compute_price_ratio(Ra, Rb, float(Va2), float(Vb2))), 20.0, rtol=1e-9
    )
    npt.assert_allclose(_spot(Ra, Rb, float(Va2), float(Vb2)), 1.0, rtol=1e-9)


def test_narrow_is_rate_limited():
    Ra, Rb, Va, Vb = 100.0, 100.0, 200.0, 200.0
    q0 = float(compute_price_ratio(Ra, Rb, Va, Vb))
    step = 0.01
    Va2, Vb2 = apply_windowed_ratio_update(
        Ra, Rb, Va, Vb, desired_price_ratio=1.5, max_narrow_log_step=step
    )
    q1 = float(compute_price_ratio(Ra, Rb, float(Va2), float(Vb2)))
    # Ratio decreased, but by no more than one log-step.
    assert q1 < q0
    npt.assert_allclose(q1, np.exp(np.log(q0) - step), rtol=1e-9)
    npt.assert_allclose(_spot(Ra, Rb, float(Va2), float(Vb2)), 1.0, rtol=1e-9)


def test_widen_preempts_in_progress_narrow():
    # Even mid-narrow, a widen request larger than current applies instantly.
    Ra, Rb, Va, Vb = 100.0, 100.0, 200.0, 200.0
    q0 = float(compute_price_ratio(Ra, Rb, Va, Vb))
    Va2, Vb2 = apply_windowed_ratio_update(
        Ra, Rb, Va, Vb, desired_price_ratio=q0 * 2.0, max_narrow_log_step=0.001
    )
    npt.assert_allclose(
        float(compute_price_ratio(Ra, Rb, float(Va2), float(Vb2))),
        q0 * 2.0, rtol=1e-9,
    )


# ---------------------------------------------------------------------------
# response curve
# ---------------------------------------------------------------------------

def test_response_curve_endpoints_and_monotonicity():
    base, mx, dead = 1.5, 16.0, 0.3
    # Below deadband -> base.
    npt.assert_allclose(float(response_curve(0.0, base, mx, dead)), base, rtol=1e-9)
    npt.assert_allclose(float(response_curve(0.3, base, mx, dead)), base, rtol=1e-9)
    # Fully trending -> max.
    npt.assert_allclose(float(response_curve(1.0, base, mx, dead)), mx, rtol=1e-9)
    # Monotone non-decreasing in efficiency.
    grid = np.linspace(0.0, 1.0, 50)
    vals = np.array([float(response_curve(e, base, mx, dead)) for e in grid])
    assert np.all(np.diff(vals) >= -1e-9)


# ---------------------------------------------------------------------------
# #3 triggers — separate trend from chop
# ---------------------------------------------------------------------------

def _trend_path(T=400):
    return jnp.array(100.0 * np.exp(np.cumsum(np.full(T, 0.002))))


def _chop_path(T=400):
    return jnp.array(100.0 * (1.0 + 0.05 * np.sin(np.arange(T) * 0.5)))


def test_efficiency_ratio_separates_regimes():
    er_trend = efficiency_ratio_signal(_trend_path(), window=60)
    er_chop = efficiency_ratio_signal(_chop_path(), window=60)
    assert float(er_trend[-100:].mean()) > 0.9
    assert float(er_chop[-100:].mean()) < 0.2


def test_ewma_efficiency_separates_regimes():
    ew_trend = ewma_efficiency_signal(_trend_path(), alpha=0.05)
    ew_chop = ewma_efficiency_signal(_chop_path(), alpha=0.05)
    assert float(ew_trend[-100:].mean()) > 0.9
    assert float(ew_chop[-100:].mean()) < 0.3


def test_sign_ewma_separates_regimes():
    alpha = 0.05
    drift = sign_ewma_init()
    for _ in range(300):
        drift, eff_trend = sign_ewma_update(drift, True, alpha)
    drift = sign_ewma_init()
    for k in range(300):
        drift, eff_chop = sign_ewma_update(drift, bool(k % 2 == 0), alpha)
    assert float(eff_trend) > 0.9
    assert float(eff_chop) < 0.2


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,kw", [
    ("er", dict(window=120)),
    ("ewma", dict(trigger_alpha=0.02)),
    ("sign", dict(trigger_alpha=0.02)),
])
def test_zero_fee_kernel_runs_finite(mode, kw):
    T = 500
    A = 100.0 * np.exp(np.cumsum(np.full(T, 0.001)))
    prices = jnp.array(np.stack([A, np.ones(T)], axis=1))
    init_res, Va, Vb = initialise_reclamm_reserves(10000.0, prices[0], jnp.array(1.5))
    reserves = _jax_calc_fantasticlamm_reserves_zero_fees(
        init_res, Va, Vb, prices,
        jnp.float64(0.5), jnp.float64(1.0 - 1.0 / 124000.0), 60.0,
        trigger_mode=mode, ratio_base=1.5, ratio_max=16.0, deadband=0.3,
        sharpness=1.0, max_narrow_log_step=0.005, **kw,
    )
    assert reserves.shape == (T, 2)
    assert bool(np.all(np.isfinite(np.array(reserves))))


@pytest.mark.parametrize("mode", ["er", "ewma", "sign"])
def test_fantasticlamm_protects_value_in_strong_trend(mode):
    # Zero-fee: under a strong monotone uptrend, deconcentration should preserve
    # more LP value than a static reClAMM band.
    T = 2000
    A = 100.0 * np.exp(np.cumsum(np.full(T, 0.0008)))
    prices = jnp.array(np.stack([A, np.ones(T)], axis=1))
    cm, dpsb = jnp.float64(0.5), jnp.float64(1.0 - 1.0 / 124000.0)
    init_res, Va, Vb = initialise_reclamm_reserves(10000.0, prices[0], jnp.array(1.5))

    rc = _jax_calc_reclamm_reserves_zero_fees(init_res, Va, Vb, prices, cm, dpsb, 60.0)
    fl = _jax_calc_fantasticlamm_reserves_zero_fees(
        init_res, Va, Vb, prices, cm, dpsb, 60.0,
        trigger_mode=mode, window=120, trigger_alpha=0.02,
        ratio_base=1.5, ratio_max=16.0, deadband=0.3, sharpness=1.0,
        max_narrow_log_step=0.005,
    )
    rc_value = float(rc[-1] @ prices[-1])
    fl_value = float(fl[-1] @ prices[-1])
    assert fl_value > rc_value


def test_with_fees_kernel_runs_finite():
    T = 400
    A = 100.0 * np.exp(np.cumsum(np.full(T, 0.0005)))
    prices = jnp.array(np.stack([A, np.ones(T)], axis=1))
    init_res, Va, Vb = initialise_reclamm_reserves(10000.0, prices[0], jnp.array(1.5))
    reserves = _jax_calc_fantasticlamm_reserves_with_fees(
        init_res, Va, Vb, prices,
        jnp.float64(0.5), jnp.float64(1.0 - 1.0 / 124000.0), 60.0,
        fees=0.0025, all_sig_variations=ALL_SIG_VARIATIONS_2,
        trigger_mode="sign", trigger_alpha=0.02,
        ratio_base=1.5, ratio_max=16.0, deadband=0.3, sharpness=1.0,
        max_narrow_log_step=0.005,
    )
    assert reserves.shape == (T, 2)
    assert bool(np.all(np.isfinite(np.array(reserves))))


# ---------------------------------------------------------------------------
# pool registration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule,cls,mode", [
    ("fantasticlamm_er", FantasticLammEfficiencyRatioPool, "er"),
    ("fantasticlamm_ewma", FantasticLammEwmaEfficiencyPool, "ewma"),
    ("fantasticlamm_sign", FantasticLammSignEwmaPool, "sign"),
])
def test_create_pool_registration(rule, cls, mode):
    pool = create_pool(rule)
    assert isinstance(pool, cls)
    assert pool._TRIGGER_MODE == mode
    assert pool.is_trainable() is False


# ---------------------------------------------------------------------------
# end-to-end through the runner
# ---------------------------------------------------------------------------

@pytest.mark.requires_data
@pytest.mark.parametrize("rule", [
    "reclamm", "fantasticlamm_er", "fantasticlamm_ewma", "fantasticlamm_sign",
])
def test_end_to_end_run(rule):
    from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
    from quantammsim.runners.jax_runners import do_run_on_historic_data

    fp = copy.deepcopy(run_fingerprint_defaults)
    fp["chunk_period"] = 1440
    fp["weight_interpolation_period"] = 1440
    fp["startDateString"] = "2023-01-01 00:00:00"
    fp["endDateString"] = "2023-03-01 00:00:00"
    fp["tokens"] = ["BTC", "ETH"]
    fp["rule"] = rule
    fp["fees"] = 0.0025
    fp["gas_cost"] = 0.0
    fp["arb_fees"] = 0.0
    fp["initial_pool_value"] = 10000.0
    fp["max_memory_days"] = 365.0
    fp["do_arb"] = True
    fp["fantasticlamm_ratio_max"] = 16.0
    fp["fantasticlamm_deadband"] = 0.3
    fp["fantasticlamm_window"] = 120
    fp["fantasticlamm_trigger_alpha"] = 0.02
    fp["fantasticlamm_max_narrow_log_step"] = 0.005

    params = {
        "price_ratio": jnp.array(1.5),
        "centeredness_margin": jnp.array(0.5),
        "daily_price_shift_base": jnp.array(1.0 - 1.0 / 124000.0),
    }
    result = do_run_on_historic_data(
        run_fingerprint=fp, params=params, root=TEST_DATA_DIR, verbose=False
    )
    assert result["final_value"] > 0
    assert not np.any(np.isnan(np.array(result["reserves"])))
