"""fantasticlamm pool driven by the magnitude-scaled sign-EMA-cross trigger.

Like ``fantasticlamm_sign_ema`` but the per-block "sign" is a continuous value
in [-1, +1] scaled by how far spot has departed from EMA_slow as a fraction:

    deviation = (spot - EMA_slow) / EMA_slow
    sign      = clip(deviation / magnitude_k, -1, +1)

Slow drifts give a small contribution to the trend EWMA; sharp moves saturate
the signal faster. Captures the "sharp move / fast depletion" intuition.
"""

from jax import tree_util

from quantammsim.pools.fantasticlamm.fantasticlamm_base import FantasticLammBasePool


class FantasticLammSignEmaMagPool(FantasticLammBasePool):
    """fantasticlamm with the magnitude-scaled EMA-cross trend trigger."""

    _TRIGGER_MODE = "sign_ema_mag"


tree_util.register_pytree_node(
    FantasticLammSignEmaMagPool,
    FantasticLammSignEmaMagPool._tree_flatten,
    FantasticLammSignEmaMagPool._tree_unflatten,
)
