"""
Environments package for PGHC algorithm verification.

Contains environments for testing the algorithm at different complexity levels:
- pendulum: Simple pendulum (early experiments)
- cartpole: Differentiable cart-pole with PyTorch physics (Level 1)
- cartpole_newton: Vectorized cart-pole with Newton/Warp (Level 1.5)
- ant: Parametric Ant with Newton/Warp (Level 2)
"""

from .pendulum import PendulumEnv, ParametricPendulum
from .cartpole import CartPoleEnv, ParametricCartPole

__all__ = [
    "PendulumEnv", "ParametricPendulum",
    "CartPoleEnv", "ParametricCartPole",
]
