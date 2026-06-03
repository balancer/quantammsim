"""fantasticlamm pool driven by the hybrid sign trigger.

Combines the original sign trigger (static range center) with the magnitude
sign_ema_mag trigger (slow EMA reference), weighted by a tunable ``w_static``
in [0, 1]:

    sign = w_static * sign_above_center  +  (1 - w_static) * sign_ema_magnitude

``w_static = 1`` reproduces ``fantasticlamm_sign``; ``w_static = 0`` reproduces
``fantasticlamm_sign_ema_mag``; values in between blend the two. Optuna picks
the balance from the data.
"""

from jax import tree_util

from quantammsim.pools.fantasticlamm.fantasticlamm_base import FantasticLammBasePool


class FantasticLammSignHybridPool(FantasticLammBasePool):
    """fantasticlamm with the hybrid (static + EMA-cross magnitude) trigger."""

    _TRIGGER_MODE = "sign_hybrid"


tree_util.register_pytree_node(
    FantasticLammSignHybridPool,
    FantasticLammSignHybridPool._tree_flatten,
    FantasticLammSignHybridPool._tree_unflatten,
)
