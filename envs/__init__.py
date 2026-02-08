"""
Environments package for PGHC algorithm verification.

Contains environments for testing the algorithm at different complexity levels:
- cartpole: Differentiable cart-pole with PyTorch physics (Level 1)
- cartpole_newton: Vectorized cart-pole with Newton/Warp (Level 1.5)
- ant: Parametric Ant with Newton/Warp (Level 2)
"""

from .cartpole import CartPoleEnv, ParametricCartPole

__all__ = [
    "CartPoleEnv", "ParametricCartPole",
]
