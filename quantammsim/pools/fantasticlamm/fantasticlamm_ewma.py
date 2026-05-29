"""fantasticlamm pool driven by the exogenous EWMA-efficiency trigger.

Deployable variant: the trend signal is |EWMA(return)| / EWMA(|return|), an
O(1)-state approximation of the efficiency ratio. See
``triggers/ewma_efficiency.py``.
"""

from jax import tree_util

from quantammsim.pools.fantasticlamm.fantasticlamm_base import FantasticLammBasePool


class FantasticLammEwmaEfficiencyPool(FantasticLammBasePool):
    """fantasticlamm with the exogenous EWMA-efficiency trend trigger."""

    _TRIGGER_MODE = "ewma"


tree_util.register_pytree_node(
    FantasticLammEwmaEfficiencyPool,
    FantasticLammEwmaEfficiencyPool._tree_flatten,
    FantasticLammEwmaEfficiencyPool._tree_unflatten,
)
