"""fantasticlamm pool driven by the endogenous sign-EWMA trigger.

Cheapest, most manipulation-resistant variant: the trend signal is |EWMA(s)|
where s is the pool's own signed center displacement. Needs no external price
feed and one scalar of state. See ``triggers/sign_ewma.py``.
"""

from jax import tree_util

from quantammsim.pools.fantasticlamm.fantasticlamm_base import FantasticLammBasePool


class FantasticLammSignEwmaPool(FantasticLammBasePool):
    """fantasticlamm with the endogenous sign-EWMA trend trigger."""

    _TRIGGER_MODE = "sign"


tree_util.register_pytree_node(
    FantasticLammSignEwmaPool,
    FantasticLammSignEwmaPool._tree_flatten,
    FantasticLammSignEwmaPool._tree_unflatten,
)
