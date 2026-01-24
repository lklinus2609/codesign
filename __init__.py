# Hybrid Co-Design Module
# Implements Algorithm 1: Hybrid PPO Control + Differentiable Physics Morphology

from .parametric_g1 import ParametricG1Model
from .hybrid_agent import HybridAMPAgent
from .diff_rollout import DifferentiableRollout

__all__ = [
    "ParametricG1Model",
    "HybridAMPAgent",
    "DifferentiableRollout",
]
