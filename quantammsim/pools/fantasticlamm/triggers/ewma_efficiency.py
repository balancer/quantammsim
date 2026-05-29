"""Exogenous EWMA efficiency signal — the deployable O(1)-state approximation.

Maintains two exponential moving averages of the market log-return r:

    D_t = (1-a)*D_{t-1} + a*r_t        (signed drift)
    V_t = (1-a)*V_{t-1} + a*|r_t|      (activity / path length)
    E_t = |D_t| / V_t                  in [0, 1]

By the triangle inequality |D| <= V, so E is bounded in [0, 1] with the same
trend/chop semantics as the exact efficiency ratio: all moves same sign -> 1,
moves cancel -> 0. Unlike the fixed window, this needs only two scalars of state
and one update per step, so it is cheap to maintain on-chain.
"""

import jax.numpy as jnp
from jax.lax import scan


def ewma_efficiency_signal(market_prices, alpha):
    """Per-step EWMA efficiency over market log-returns.

    Parameters
    ----------
    market_prices : jnp.ndarray, shape (T,)
        Market price (token A in terms of B) at each step.
    alpha : float
        EWMA smoothing factor in (0, 1]; larger reacts faster.

    Returns
    -------
    efficiency : jnp.ndarray, shape (T,)
        Efficiency in [0, 1]; starts at 0 (state initialised to zero).
    """
    log_price = jnp.log(jnp.maximum(market_prices, 1e-30))
    returns = jnp.diff(log_price, prepend=log_price[:1])

    def step(carry, r):
        drift, activity = carry
        drift = (1.0 - alpha) * drift + alpha * r
        activity = (1.0 - alpha) * activity + alpha * jnp.abs(r)
        efficiency = jnp.abs(drift) / jnp.maximum(activity, 1e-30)
        return (drift, activity), efficiency

    _, efficiency = scan(step, (0.0, 0.0), returns)
    return jnp.clip(efficiency, 0.0, 1.0)
