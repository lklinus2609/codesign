"""
Cart-Pole environment using Newton physics engine.

Level 1.5: Verify differentiable simulation with Newton before Level 2 (Ant).
"""

from .cartpole_newton_env import CartPoleNewtonEnv, ParametricCartPoleNewton

__all__ = ["CartPoleNewtonEnv", "ParametricCartPoleNewton"]
