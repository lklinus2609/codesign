"""
Pendulum Environment with Differentiable Physics

This module provides a simple inverted pendulum environment for Level 1 PGHC verification.
The environment is fully differentiable through PyTorch, enabling gradient-based co-design
of the pole length parameter.

Physics Model:
    θ̈ = (g/L)*sin(θ) + τ/(m*L²)

    where:
    - θ: angle from upright (θ=0 is upright, π is hanging down)
    - L: pole length (design parameter)
    - m: pole mass
    - g: gravity (9.81 m/s²)
    - τ: applied torque (control input)

Analytical Optimal Length (for swing-up with limited torque):
    L* ≈ (3 * τ_max) / (m * g)

    For τ_max = 2 Nm, m = 0.5 kg:
    L* ≈ 1.22 m

Reference: VERIFICATION_PLAN.md Level 1
"""

import math
import numpy as np
import torch
from typing import Tuple, Dict, Optional


class ParametricPendulum:
    """
    Parametric model for pendulum pole length.

    Manages the pole length parameter with:
    - Gradient tracking for co-design
    - Bounds projection
    - State serialization

    Attributes:
        L: Current pole length (torch.Tensor with grad)
        L_min: Minimum pole length
        L_max: Maximum pole length
        mass: Pole mass (fixed)
    """

    def __init__(
        self,
        L_init: float = 0.5,
        L_min: float = 0.4,
        L_max: float = 1.5,
        mass: float = 0.5,
        device: str = "cpu",
    ):
        """
        Initialize the parametric pendulum model.

        Args:
            L_init: Initial pole length (meters)
            L_min: Minimum allowed pole length
            L_max: Maximum allowed pole length
            mass: Pole mass (kg)
            device: Computation device
        """
        self.L_min = L_min
        self.L_max = L_max
        self.mass = mass
        self.device = device

        # Design parameter with gradient tracking
        self.L = torch.tensor(
            [L_init], dtype=torch.float64, device=device, requires_grad=True
        )

    def get_L(self) -> float:
        """Get current pole length as scalar."""
        return self.L.item()

    def set_L(self, value: float):
        """Set pole length, maintaining gradient tracking."""
        with torch.no_grad():
            self.L.fill_(value)
            self.L.clamp_(self.L_min, self.L_max)

    def project_bounds(self, L: torch.Tensor) -> torch.Tensor:
        """Project L to valid bounds using differentiable clamping."""
        return torch.clamp(L, self.L_min, self.L_max)

    def get_state_dict(self) -> dict:
        """Get state for serialization."""
        return {
            "L": self.get_L(),
            "L_min": self.L_min,
            "L_max": self.L_max,
            "mass": self.mass,
        }

    def load_state_dict(self, state: dict):
        """Load state from dict."""
        self.set_L(state["L"])

    def compute_optimal_length(self, torque_max: float, gravity: float = 9.81) -> float:
        """
        Compute analytically optimal pole length for swing-up.

        Based on the approximation: L* ≈ (3 * τ_max) / (m * g)

        Args:
            torque_max: Maximum available torque (Nm)
            gravity: Gravitational acceleration (m/s²)

        Returns:
            Optimal pole length (meters)
        """
        return (3.0 * torque_max) / (self.mass * gravity)


class PendulumEnv:
    """
    Differentiable pendulum environment for PGHC verification.

    This environment simulates an inverted pendulum with:
    - Differentiable physics (pure PyTorch)
    - Configurable pole length as design parameter
    - Gym-style interface (reset, step)

    State Space (2D):
        - θ: Angle from upright (rad), wrapped to [-π, π]
        - θ̇: Angular velocity (rad/s)

    Action Space (1D):
        - Continuous torque τ ∈ [-torque_max, torque_max]

    Reward:
        r = cos(θ) - 0.1*θ̇² - 0.001*τ²
        (maximized when upright with low velocity and low effort)

    Attributes:
        parametric_model: ParametricPendulum instance
        gravity: Gravitational acceleration
        dt: Simulation timestep
        torque_max: Maximum torque
    """

    def __init__(
        self,
        parametric_model: Optional[ParametricPendulum] = None,
        gravity: float = 9.81,
        dt: float = 0.05,
        torque_max: float = 2.0,
        device: str = "cpu",
    ):
        """
        Initialize the pendulum environment.

        Args:
            parametric_model: Parametric model (created if None)
            gravity: Gravitational acceleration (m/s²)
            dt: Simulation timestep (seconds)
            torque_max: Maximum torque (Nm)
            device: Computation device
        """
        self.device = device
        self.gravity = gravity
        self.dt = dt
        self.torque_max = torque_max

        # Create or use provided parametric model
        if parametric_model is None:
            self.parametric_model = ParametricPendulum(device=device)
        else:
            self.parametric_model = parametric_model

        # State: [theta, theta_dot]
        self.state = torch.zeros(2, dtype=torch.float64, device=device)

        # Episode tracking
        self.steps = 0
        self.max_steps = 200

        # Observation and action dimensions
        self.obs_dim = 3  # [cos(θ), sin(θ), θ̇]
        self.act_dim = 1

    @property
    def L(self) -> torch.Tensor:
        """Get current pole length tensor."""
        return self.parametric_model.L

    @property
    def mass(self) -> float:
        """Get pole mass."""
        return self.parametric_model.mass

    def reset(self, theta_init: Optional[float] = None) -> torch.Tensor:
        """
        Reset the environment to initial state.

        Args:
            theta_init: Initial angle (radians). If None, starts hanging down (π).

        Returns:
            Initial observation
        """
        self.steps = 0

        if theta_init is None:
            # Start hanging down (swing-up task)
            theta = math.pi
        else:
            theta = theta_init

        self.state = torch.tensor(
            [theta, 0.0], dtype=torch.float64, device=self.device
        )

        return self._get_obs()

    def _get_obs(self) -> torch.Tensor:
        """
        Get observation from current state.

        Observation: [cos(θ), sin(θ), θ̇]
        Using cos/sin representation avoids angle wrapping issues.
        """
        theta = self.state[0]
        theta_dot = self.state[1]

        return torch.stack([
            torch.cos(theta),
            torch.sin(theta),
            theta_dot / 8.0,  # Normalize velocity
        ])

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool, dict]:
        """
        Take a step in the environment (differentiable).

        Args:
            action: Torque command (will be clipped to [-torque_max, torque_max])

        Returns:
            obs: Observation after step
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        # Clip action to valid range
        torque = torch.clamp(action.squeeze(), -self.torque_max, self.torque_max)

        # Current state
        theta = self.state[0]
        theta_dot = self.state[1]

        # Get pole length (with gradient tracking)
        L = self.parametric_model.L.squeeze()
        m = self.mass
        g = self.gravity

        # Physics: θ̈ = (g/L)*sin(θ) + τ/(m*L²)
        # Note: We use sin(θ) directly - when θ=0 (upright), sin(θ)=0, no gravity torque
        # When θ=π (hanging), sin(θ)=0, unstable equilibrium
        # For swing-up, we want the pendulum to go from θ=π to θ=0

        # Actually, for standard pendulum where θ=0 is upright:
        # The gravity torque is m*g*L*sin(θ) / (m*L²) = g*sin(θ)/L
        # This accelerates toward θ=0 when θ>0 (for small angles)
        # Wait, that's wrong. Let me reconsider.

        # Standard inverted pendulum:
        # - θ=0: upright (unstable equilibrium)
        # - θ=π: hanging down (stable equilibrium)
        # Equation of motion: m*L²*θ̈ = m*g*L*sin(θ) + τ
        # => θ̈ = (g/L)*sin(θ) + τ/(m*L²)

        # When θ is small and positive, sin(θ)>0, so θ̈>0, angle increases (falls away)
        # This is correct for inverted pendulum - it's unstable at θ=0

        theta_ddot = (g / L) * torch.sin(theta) + torque / (m * L * L)

        # Semi-implicit Euler integration
        new_theta_dot = theta_dot + theta_ddot * self.dt
        new_theta = theta + new_theta_dot * self.dt

        # Wrap angle to [-π, π]
        new_theta = self._wrap_angle(new_theta)

        # Clamp velocity
        max_vel = 8.0
        new_theta_dot = torch.clamp(new_theta_dot, -max_vel, max_vel)

        # Update state
        self.state = torch.stack([new_theta, new_theta_dot])
        self.steps += 1

        # Compute reward
        # We want θ close to 0 (upright), low velocity, low torque
        # cos(θ) = 1 when θ=0, -1 when θ=π
        reward = (
            torch.cos(theta)  # Upright bonus
            - 0.1 * theta_dot ** 2  # Velocity penalty
            - 0.001 * torque ** 2  # Effort penalty
        )

        done = self.steps >= self.max_steps

        info = {
            "theta": new_theta.item(),
            "theta_dot": new_theta_dot.item(),
            "torque": torque.item(),
            "L": L.item(),
        }

        return self._get_obs(), reward, done, info

    def _wrap_angle(self, theta: torch.Tensor) -> torch.Tensor:
        """Wrap angle to [-π, π] in a differentiable way."""
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    def compute_episode_return(
        self,
        policy_fn,
        horizon: Optional[int] = None,
        theta_init: Optional[float] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Compute total return for an episode (differentiable through L).

        This is used for outer loop gradient computation.

        Args:
            policy_fn: Policy function that maps obs -> action
            horizon: Number of steps (uses max_steps if None)
            theta_init: Initial angle

        Returns:
            total_return: Sum of rewards (differentiable w.r.t. L)
            trajectory: List of (obs, action, reward) tuples
        """
        if horizon is None:
            horizon = self.max_steps

        obs = self.reset(theta_init)
        total_return = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        trajectory = []

        for _ in range(horizon):
            # Get action from policy (no gradient through policy for outer loop)
            with torch.no_grad():
                action = policy_fn(obs)

            # Step environment (differentiable through L)
            next_obs, reward, done, info = self.step(action)

            trajectory.append((obs.clone(), action.clone(), reward.clone()))
            total_return = total_return + reward
            obs = next_obs

            if done:
                break

        return total_return, trajectory

    def rollout_differentiable(
        self,
        policy_fn,
        horizon: int,
        theta_init: float = math.pi,
    ) -> dict:
        """
        Perform a differentiable rollout for gradient computation.

        This method:
        1. Runs the policy for `horizon` steps
        2. Computes the total return
        3. Returns the return and gradient w.r.t. L

        Args:
            policy_fn: Policy function (frozen - no gradients through it)
            horizon: Rollout horizon
            theta_init: Initial angle

        Returns:
            Dictionary with:
                - total_return: Scalar return value
                - L_grad: Gradient of return w.r.t. pole length L
                - trajectory: List of step info
        """
        # Reset environment
        obs = self.reset(theta_init)

        # Ensure L has gradient
        self.parametric_model.L.requires_grad_(True)

        # Accumulate return (differentiable)
        total_return = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        trajectory = []

        for t in range(horizon):
            # Get action from policy (frozen)
            with torch.no_grad():
                action = policy_fn(obs)

            # Convert action to tensor if needed
            if not isinstance(action, torch.Tensor):
                action = torch.tensor([action], dtype=torch.float64, device=self.device)

            # Step (differentiable through L)
            next_obs, reward, done, info = self.step(action)

            total_return = total_return + reward
            trajectory.append(info)
            obs = next_obs

            if done:
                break

        # Compute gradient of return w.r.t. L
        if total_return.requires_grad:
            total_return.backward(retain_graph=True)
            L_grad = self.parametric_model.L.grad.item() if self.parametric_model.L.grad is not None else 0.0
            self.parametric_model.L.grad = None  # Clear gradient
        else:
            L_grad = 0.0

        return {
            "total_return": total_return.item(),
            "L_grad": L_grad,
            "trajectory": trajectory,
            "final_theta": trajectory[-1]["theta"] if trajectory else 0.0,
        }


class SimplePendulumPolicy:
    """
    Simple energy-based swing-up policy for testing.

    This policy uses an energy-shaping approach:
    - If energy < target: pump energy into the system
    - If energy ≈ target and near top: balance with PD control

    IMPORTANT: For envelope theorem validity, the policy uses a FIXED reference
    L value for energy calculations, not the current L. This ensures the policy
    is truly frozen when computing gradients w.r.t. L.
    """

    def __init__(
        self,
        env: PendulumEnv,
        Kp: float = 10.0,
        Kd: float = 2.0,
        L_reference: Optional[float] = None,
    ):
        """
        Initialize the policy.

        Args:
            env: Pendulum environment
            Kp: Proportional gain for balance
            Kd: Derivative gain for balance
            L_reference: Fixed L for energy calculations. If None, uses current L
                        at policy creation time. This ensures policy doesn't adapt
                        when L changes during gradient computation.
        """
        self.env = env
        self.Kp = Kp
        self.Kd = Kd
        self.device = env.device
        # CRITICAL: Store L at policy creation time, don't query it dynamically
        self.L_ref = L_reference if L_reference is not None else env.parametric_model.get_L()
        self.m = env.mass
        self.g = env.gravity
        self.torque_max = env.torque_max

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute action given observation.

        Args:
            obs: Observation [cos(theta), sin(theta), theta_dot_normalized]

        Returns:
            Action (torque)
        """
        # Extract state from observation
        cos_theta = obs[0].item()
        sin_theta = obs[1].item()
        theta_dot = obs[2].item() * 8.0  # Unnormalize

        theta = math.atan2(sin_theta, cos_theta)

        # Compute mechanical energy using FIXED reference L
        # This ensures policy is truly frozen w.r.t. L changes
        L = self.L_ref  # Use fixed reference, NOT current env L
        m = self.m
        g = self.g

        KE = 0.5 * m * L * L * theta_dot * theta_dot
        PE = m * g * L * (1 + cos_theta)  # 2*m*g*L at top, 0 at bottom
        E = KE + PE
        E_target = 2 * m * g * L  # Energy at top with zero velocity

        # Energy error
        E_error = E - E_target

        # If near the top and low velocity, use PD balance control
        if cos_theta > 0.9 and abs(theta_dot) < 1.0:
            # Balance mode: PD control to keep upright
            torque = -self.Kp * theta - self.Kd * theta_dot
        else:
            # Swing-up mode: energy pumping
            # Pump energy when moving in the direction that increases E
            # torque = k * sign(theta_dot) * sign(E_error)
            # This adds energy when E < E_target and removes when E > E_target
            k = 1.5  # Pump gain
            if E_error < 0:
                # Need more energy: accelerate in direction of motion
                torque = k * (1.0 if theta_dot >= 0 else -1.0)
            else:
                # Too much energy: decelerate
                torque = -k * (1.0 if theta_dot >= 0 else -1.0)

        # Clamp to torque limits
        torque = max(-self.torque_max, min(self.torque_max, torque))

        return torch.tensor([torque], dtype=torch.float64, device=self.device)


if __name__ == "__main__":
    # Quick test
    print("Testing PendulumEnv...")

    env = PendulumEnv()
    policy = SimplePendulumPolicy(env)

    print(f"Initial L: {env.parametric_model.get_L():.3f} m")
    print(f"Optimal L (for τ_max={env.torque_max} Nm): {env.parametric_model.compute_optimal_length(env.torque_max):.3f} m")

    # Run an episode
    obs = env.reset()
    total_reward = 0

    for step in range(200):
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward.item()

        if step % 50 == 0:
            print(f"Step {step}: θ={info['theta']:.2f}, θ̇={info['theta_dot']:.2f}, r={reward.item():.3f}")

        if done:
            break

    print(f"Total reward: {total_reward:.2f}")

    # Test differentiable rollout
    print("\nTesting differentiable rollout...")
    result = env.rollout_differentiable(policy, horizon=50)
    print(f"Total return: {result['total_return']:.2f}")
    print(f"dReturn/dL: {result['L_grad']:.4f}")
