"""
Pendulum environment for Level 1 PGHC verification.

This module provides:
- ParametricPendulum: Parametric model for pole length
- PendulumEnv: Gym-style environment with differentiable physics
- MinimalCoDesignAgent: Simplified agent for testing outer loop logic

Key features:
- Pure PyTorch implementation (no Warp/Newton dependency)
- Differentiable physics for gradient-based co-design
- Configurable pole length as the design parameter
"""

from .pendulum_env import PendulumEnv, ParametricPendulum

__all__ = ["PendulumEnv", "ParametricPendulum"]
