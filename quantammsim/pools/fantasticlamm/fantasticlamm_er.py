"""fantasticlamm pool driven by the exact Kaufman Efficiency Ratio trigger.

Idealized (non-deployable) reference variant: the trend signal is the true
fixed-window efficiency ratio. See ``triggers/efficiency_ratio.py``.
"""

from jax import tree_util

from quantammsim.pools.fantasticlamm.fantasticlamm_base import FantasticLammBasePool


class FantasticLammEfficiencyRatioPool(FantasticLammBasePool):
    """fantasticlamm with the exact efficiency-ratio trend trigger."""

    _TRIGGER_MODE = "er"


tree_util.register_pytree_node(
    FantasticLammEfficiencyRatioPool,
    FantasticLammEfficiencyRatioPool._tree_flatten,
    FantasticLammEfficiencyRatioPool._tree_unflatten,
)
