"""Exact Kaufman Efficiency Ratio trend signal (exogenous, idealized reference).

    ER_t = |P_t - P_{t-n}|  /  sum_{i=t-n+1..t} |P_i - P_{i-1}|     in [0, 1]

1 = a perfectly monotone move over the window (trend), 0 = pure chop with no net
displacement. This is the true fixed-window statistic; on-chain it would cost
O(n) storage, so it is not deployable. In simulation it serves as the upper
bound against which the cheap EWMA approximations are measured.
"""

import jax.numpy as jnp


def efficiency_ratio_signal(market_prices, window):
    """Per-step efficiency ratio over a trailing window.

    Parameters
    ----------
    market_prices : jnp.ndarray, shape (T,)
        Market price (token A in terms of B) at each step.
    window : int
        Lookback length n (static).

    Returns
    -------
    efficiency : jnp.ndarray, shape (T,)
        Efficiency ratio in [0, 1]; zero until a full window of history exists.
    """
    P = market_prices
    T = P.shape[0]
    n = int(window)

    # Per-step absolute moves (moves[0] = 0 via prepend of P[0]).
    moves = jnp.abs(jnp.diff(P, prepend=P[:1]))
    cumulative = jnp.cumsum(moves)
    if n < T:
        shifted = jnp.concatenate([jnp.zeros(n, dtype=cumulative.dtype), cumulative[:-n]])
    else:
        shifted = jnp.zeros(T, dtype=cumulative.dtype)
    path_length = cumulative - shifted

    idx = jnp.arange(T)
    ref_idx = jnp.maximum(idx - n, 0)
    net_move = jnp.abs(P - P[ref_idx])

    efficiency = jnp.clip(net_move / jnp.maximum(path_length, 1e-30), 0.0, 1.0)
    return jnp.where(idx < n, 0.0, efficiency)
