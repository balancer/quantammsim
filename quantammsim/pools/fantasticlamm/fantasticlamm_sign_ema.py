"""fantasticlamm pool driven by the binary sign-EMA-cross trigger.

Variant of the endogenous sign-EWMA trigger that uses a *slow EMA of the
pool's spot price* as the trend reference, not the static geometric center of
the price range. This fixes the failure mode where a price sitting sideways at
any non-center value would permanently saturate the trend signal.

Sign each block: +1 if spot > EMA_slow(spot), -1 otherwise. The same EWMA of
the sign and the same response curve as ``fantasticlamm_sign``.
"""

from jax import tree_util

from quantammsim.pools.fantasticlamm.fantasticlamm_base import FantasticLammBasePool


class FantasticLammSignEmaPool(FantasticLammBasePool):
    """fantasticlamm with the binary EMA-cross trend trigger."""

    _TRIGGER_MODE = "sign_ema"


tree_util.register_pytree_node(
    FantasticLammSignEmaPool,
    FantasticLammSignEmaPool._tree_flatten,
    FantasticLammSignEmaPool._tree_unflatten,
)
