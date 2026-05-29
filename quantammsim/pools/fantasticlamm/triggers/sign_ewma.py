"""Endogenous sign-EWMA trend signal — cheapest and most manipulation-resistant.

Tracks a single EWMA of the pool's signed center displacement s in {-1, +1}
(above / below center), which the pool already computes each step:

    D_t = (1-a)*D_{t-1} + a*s_t
    E_t = |D_t|                        in [0, 1]

A sustained trend keeps the pool pinned to one edge, so s holds its sign and
|D| -> 1; a ranging market flips s and |D| -> 0. Keeping the pool pinned to one
edge requires sustained directional order flow (real arbitrage capital), so a
single manipulated tick barely moves D. Needs no external price feed and only
one scalar of carried state, so it is the cheapest of the three triggers.
"""

import jax.numpy as jnp


def sign_ewma_init():
    """Initial drift state (centered)."""
    return jnp.float64(0.0)


def sign_ewma_update(drift_prev, is_above_center, alpha):
    """Advance the sign-EWMA one step and return the trend efficiency.

    Parameters
    ----------
    drift_prev : float
        Previous EWMA of the signed displacement.
    is_above_center : bool
        Whether the pool is above center this step (token A undervalued).
    alpha : float
        EWMA smoothing factor in (0, 1].

    Returns
    -------
    drift_new : float
        Updated EWMA of the signed displacement.
    efficiency : float
        Trend strength |drift_new| in [0, 1].
    """
    sign = jnp.where(is_above_center, 1.0, -1.0)
    drift_new = (1.0 - alpha) * drift_prev + alpha * sign
    return drift_new, jnp.clip(jnp.abs(drift_new), 0.0, 1.0)
