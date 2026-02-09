"""
Walker2D environment using Newton physics engine.

Level 2.1: Test PGHC on 2D bipedal locomotion with morphology optimization.
"""

from .walker2d_env import Walker2DEnv, ParametricWalker2D
from .walker2d_vec_env import Walker2DVecEnv

__all__ = ["Walker2DEnv", "ParametricWalker2D", "Walker2DVecEnv"]
