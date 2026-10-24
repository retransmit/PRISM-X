"""Invertible flow module (DEPRECATED).

This module has been relocated to `prism_generator.experimental.invertible_flow`.
Please import from `prism_generator.experimental` directly.
"""

from .experimental.invertible_flow import (
    AffineCouplingLayer,
    ConditionalRealNVPFlow,
    LatentAumannShapleyAttributor,
    TransactionFlowFeatureCodec,
)

__all__ = [
    "AffineCouplingLayer",
    "ConditionalRealNVPFlow",
    "LatentAumannShapleyAttributor",
    "TransactionFlowFeatureCodec",
]
