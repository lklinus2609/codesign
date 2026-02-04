"""
Environments package for PGHC algorithm verification.

Contains simplified environments for testing the algorithm at different complexity levels.
"""

from .pendulum import PendulumEnv, ParametricPendulum

__all__ = ["PendulumEnv", "ParametricPendulum"]
