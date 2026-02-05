"""
Cart-Pole environment using Newton physics engine.

Level 1.5: Verify differentiable simulation with Newton before Level 2 (Ant).
"""

from .cartpole_newton_vec_env import CartPoleNewtonVecEnv, ParametricCartPoleNewton

__all__ = ["CartPoleNewtonVecEnv", "ParametricCartPoleNewton"]
