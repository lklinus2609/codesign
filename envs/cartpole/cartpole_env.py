"""
Differentiable Cart-Pole Environment

Following Gymnasium/MuJoCo InvertedPendulum specification:
- Action: Continuous force [-10, 10] N applied to cart (adjusted for controllability)
- State: [x, theta, x_dot, theta_dot]
- Reward: +1 for each timestep pole is upright (|theta| < 0.2 rad)
- Termination: |theta| > 0.2 rad

Design Parameter: Pole length L

Physics (Lagrangian mechanics for uniform rod):
    Cart mass: M (fixed)
    Pole linear density: rho (kg/m, fixed)
    Pole half-length: L (design parameter)
    Pole mass: m = rho * 2L (depends on L)
    Moment of inertia about pivot: I = (4/3) * m * L^2
    Gravity: g
    Force on cart: F

Equations of motion:
    theta_ddot = (g*sin(theta) - cos(theta)*temp) / (L * (4/3 - m*cos^2(theta)/(M+m)))
    x_ddot = temp - m*L*theta_ddot*cos(theta) / (M + m)
    where temp = (F + m*L*theta_dot^2*sin(theta)) / (M + m)

The 4/3 factor comes from moment of inertia of uniform rod about pivot.

Reference: Barto, Sutton, Anderson (1983)
"""

import math
import numpy as np
import torch
from typing import Tuple, Optional


class ParametricCartPole:
    """
    Parametric model for cart-pole with adjustable pole length.

    Design parameter: L (pole half-length, distance from pivot to CoM)

    Full physics model for uniform rod:
    - Total pole length = 2L
    - Pole mass scales with L: m = linear_density * 2L
    - Moment of inertia about pivot: I = (4/3) * m * L^2

    This creates richer optimization landscape where L affects:
    - Gravitational torque (longer = more torque)
    - Rotational inertia (longer = harder to rotate)
    - Total mass (longer = heavier)
    """

    def __init__(
        self,
        L_init: float = 0.6,
        L_min: float = 0.3,
        L_max: float = 1.2,
        cart_mass: float = 1.0,
        pole_linear_density: float = 0.1,  # kg/m - mass per unit length
        device: str = "cpu",
    ):
        """
        Initialize parametric cart-pole.

        Args:
            L_init: Initial pole half-length (meters) - distance from pivot to CoM
            L_min: Minimum pole half-length
            L_max: Maximum pole half-length
            cart_mass: Cart mass M (kg)
            pole_linear_density: Pole linear density (kg/m)
            device: Computation device
        """
        self.L_min = L_min
        self.L_max = L_max
        self.cart_mass = cart_mass  # M
        self.pole_linear_density = pole_linear_density  # rho (kg/m)
        self.device = device

        # Design parameter with gradient tracking
        self.L = torch.tensor(
            [L_init], dtype=torch.float64, device=device, requires_grad=True
        )

    def get_L(self) -> float:
        """Get current pole half-length."""
        return self.L.item()

    def set_L(self, value: float):
        """Set pole half-length."""
        with torch.no_grad():
            self.L.fill_(value)
            self.L.clamp_(self.L_min, self.L_max)

    def get_pole_mass(self, L: torch.Tensor = None) -> torch.Tensor:
        """
        Compute pole mass from length.
        m = rho * 2L (total length = 2L for half-length L)
        """
        if L is None:
            L = self.L
        return self.pole_linear_density * 2.0 * L

    def get_moment_of_inertia(self, L: torch.Tensor = None) -> torch.Tensor:
        """
        Compute moment of inertia about pivot.
        For uniform rod: I_pivot = (4/3) * m * L^2
        """
        if L is None:
            L = self.L
        m = self.get_pole_mass(L)
        return (4.0 / 3.0) * m * L ** 2

    @property
    def pole_mass(self) -> float:
        """Current pole mass (for compatibility)."""
        return self.get_pole_mass().item()

    def get_state_dict(self) -> dict:
        return {
            "L": self.get_L(),
            "cart_mass": self.cart_mass,
            "pole_linear_density": self.pole_linear_density,
            "pole_mass": self.pole_mass,
        }

    def load_state_dict(self, state: dict):
        self.set_L(state["L"])


class CartPoleEnv:
    """
    Differentiable Cart-Pole environment for PGHC verification.

    State: [x, theta, x_dot, theta_dot]
        - x: cart position (m)
        - theta: pole angle from upright (rad), 0 = upright
        - x_dot: cart velocity (m/s)
        - theta_dot: pole angular velocity (rad/s)

    Action: Force on cart [-force_max, force_max] (N)

    Reward: +1 if |theta| < 0.2 rad, else 0

    Termination: |theta| > 0.2 rad (pole fell)
    """

    def __init__(
        self,
        parametric_model: Optional[ParametricCartPole] = None,
        gravity: float = 9.81,
        dt: float = 0.02,  # 50 Hz like MuJoCo
        force_max: float = 10.0,  # Increased from 3.0 for controllability
        theta_threshold: float = 0.2,  # ~11.5 degrees
        x_threshold: float = 2.4,  # Cart position limit
        shaped_reward: bool = False,  # Use shaped reward for gradient flow
        device: str = "cpu",
    ):
        """
        Initialize cart-pole environment.

        Args:
            parametric_model: Parametric model (created if None)
            gravity: Gravitational acceleration (m/s²)
            dt: Simulation timestep (seconds)
            force_max: Maximum force on cart (N)
            theta_threshold: Angle threshold for termination (rad)
            x_threshold: Cart position threshold (m)
            device: Computation device
        """
        self.device = device
        self.gravity = gravity
        self.dt = dt
        self.force_max = force_max
        self.theta_threshold = theta_threshold
        self.x_threshold = x_threshold
        self.shaped_reward = shaped_reward

        if parametric_model is None:
            self.parametric_model = ParametricCartPole(device=device)
        else:
            self.parametric_model = parametric_model

        # State: [x, theta, x_dot, theta_dot]
        self.state = torch.zeros(4, dtype=torch.float64, device=device)

        # Episode tracking
        self.steps = 0
        self.max_steps = 1000  # As per Gymnasium spec
        self.terminated = False

        # Dimensions
        self.obs_dim = 4
        self.act_dim = 1

    @property
    def L(self) -> torch.Tensor:
        """Pole length."""
        return self.parametric_model.L

    @property
    def M(self) -> float:
        """Cart mass."""
        return self.parametric_model.cart_mass

    @property
    def m(self) -> float:
        """Pole mass."""
        return self.parametric_model.pole_mass

    @property
    def total_mass(self) -> float:
        """Total mass."""
        return self.parametric_model.total_mass

    def reset(self, noise_scale: float = 0.01) -> torch.Tensor:
        """
        Reset environment with small random initial state.

        Following Gymnasium spec: uniform random in [-0.01, 0.01]

        Args:
            noise_scale: Scale of initial state noise

        Returns:
            Initial observation
        """
        self.steps = 0
        self.terminated = False

        # Small random initial state (pole nearly upright)
        self.state = torch.tensor([
            np.random.uniform(-noise_scale, noise_scale),  # x
            np.random.uniform(-noise_scale, noise_scale),  # theta
            np.random.uniform(-noise_scale, noise_scale),  # x_dot
            np.random.uniform(-noise_scale, noise_scale),  # theta_dot
        ], dtype=torch.float64, device=self.device)

        return self._get_obs()

    def _get_obs(self) -> torch.Tensor:
        """Get observation (same as state for cart-pole)."""
        return self.state.clone()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action: Force on cart (will be clipped to [-force_max, force_max])

        Returns:
            obs: Observation after step
            reward: +1 if pole upright, 0 otherwise
            terminated: True if pole fell
            truncated: True if max steps reached
            info: Additional information
        """
        if self.terminated:
            # Environment already terminated, return zero reward
            return self._get_obs(), torch.tensor(0.0), True, False, {}

        # Clip action
        force = torch.clamp(action.squeeze(), -self.force_max, self.force_max)

        # Current state
        x = self.state[0]
        theta = self.state[1]
        x_dot = self.state[2]
        theta_dot = self.state[3]

        # Get parameters (L and m are differentiable through L)
        L = self.parametric_model.L.squeeze()
        M = self.M  # Cart mass (fixed)
        m = self.parametric_model.get_pole_mass(L)  # Pole mass = rho * 2L (differentiable)
        g = self.gravity
        total_mass = M + m

        # Physics computation (differentiable through L and m)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Intermediate calculation
        # temp = (F + m*L*theta_dot^2*sin(theta)) / (M + m)
        temp = (force + m * L * theta_dot ** 2 * sin_theta) / total_mass

        # Angular acceleration for uniform rod
        # The 4/3 factor comes from I_pivot / (m*L^2) + 1 = 1/3 + 1 = 4/3
        # theta_ddot = (g*sin - cos*temp) / (L * (4/3 - m*cos^2/(M+m)))
        denominator = L * (4.0 / 3.0 - m * cos_theta ** 2 / total_mass)
        theta_ddot = (g * sin_theta - cos_theta * temp) / denominator

        # Linear acceleration
        x_ddot = temp - m * L * theta_ddot * cos_theta / total_mass

        # Semi-implicit Euler integration
        new_x_dot = x_dot + x_ddot * self.dt
        new_theta_dot = theta_dot + theta_ddot * self.dt
        new_x = x + new_x_dot * self.dt
        new_theta = theta + new_theta_dot * self.dt

        # Update state
        self.state = torch.stack([new_x, new_theta, new_x_dot, new_theta_dot])
        self.steps += 1

        # Check termination (pole fell)
        terminated = torch.abs(new_theta) > self.theta_threshold
        self.terminated = terminated.item() if isinstance(terminated, torch.Tensor) else terminated

        # Check truncation (max steps)
        truncated = self.steps >= self.max_steps

        # Reward computation
        if self.shaped_reward:
            # Shaped reward: cos(theta) - provides gradient through theta -> L
            # Peaks at theta=0 (upright), differentiable
            reward = torch.cos(new_theta)
            # Also penalize cart position slightly
            reward = reward - 0.01 * new_x ** 2
        else:
            # Sparse reward: +1 if pole is upright (no gradient)
            if not self.terminated:
                reward = torch.tensor(1.0, dtype=torch.float64, device=self.device)
            else:
                reward = torch.tensor(0.0, dtype=torch.float64, device=self.device)

        info = {
            "x": new_x.item() if isinstance(new_x, torch.Tensor) else new_x,
            "theta": new_theta.item() if isinstance(new_theta, torch.Tensor) else new_theta,
            "theta_deg": math.degrees(new_theta.item() if isinstance(new_theta, torch.Tensor) else new_theta),
            "x_dot": new_x_dot.item() if isinstance(new_x_dot, torch.Tensor) else new_x_dot,
            "theta_dot": new_theta_dot.item() if isinstance(new_theta_dot, torch.Tensor) else new_theta_dot,
            "force": force.item() if isinstance(force, torch.Tensor) else force,
            "L": L.item() if isinstance(L, torch.Tensor) else L,
            "m": m.item() if isinstance(m, torch.Tensor) else m,  # Pole mass (depends on L)
            "terminated": self.terminated,
            "steps": self.steps,
        }

        return self._get_obs(), reward, self.terminated, truncated, info

    def rollout_differentiable(
        self,
        policy_fn,
        horizon: int,
        noise_scale: float = 0.01,
    ) -> dict:
        """
        Perform differentiable rollout for gradient computation.

        Args:
            policy_fn: Policy function (frozen)
            horizon: Maximum rollout steps
            noise_scale: Initial state noise

        Returns:
            Dictionary with total_return, L_grad, trajectory info
        """
        obs = self.reset(noise_scale)

        # Ensure L has gradient
        self.parametric_model.L.requires_grad_(True)

        total_return = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        trajectory = []

        for t in range(horizon):
            # Get action from policy (frozen)
            with torch.no_grad():
                action = policy_fn(obs)

            if not isinstance(action, torch.Tensor):
                action = torch.tensor([action], dtype=torch.float64, device=self.device)

            # Step (differentiable through L)
            next_obs, reward, terminated, truncated, info = self.step(action)

            total_return = total_return + reward
            trajectory.append(info)
            obs = next_obs

            if terminated or truncated:
                break

        # Compute gradient
        if total_return.requires_grad:
            total_return.backward(retain_graph=True)
            L_grad = self.parametric_model.L.grad.item() if self.parametric_model.L.grad is not None else 0.0
            self.parametric_model.L.grad = None
        else:
            L_grad = 0.0

        return {
            "total_return": total_return.item(),
            "L_grad": L_grad,
            "trajectory": trajectory,
            "episode_length": len(trajectory),
            "terminated": self.terminated,
            "final_theta": trajectory[-1]["theta"] if trajectory else 0.0,
        }


class PDCartPolePolicy:
    """
    Simple PD controller for cart-pole balancing.

    Uses state feedback to compute force:
        F = -Kp_x * x - Kd_x * x_dot - Kp_theta * theta - Kd_theta * theta_dot
    """

    def __init__(
        self,
        env: CartPoleEnv,
        Kp_x: float = 1.0,
        Kd_x: float = 2.0,
        Kp_theta: float = 50.0,
        Kd_theta: float = 10.0,
    ):
        """
        Initialize PD controller.

        Args:
            env: Cart-pole environment
            Kp_x: Proportional gain for cart position
            Kd_x: Derivative gain for cart velocity
            Kp_theta: Proportional gain for pole angle
            Kd_theta: Derivative gain for pole angular velocity
        """
        self.env = env
        self.Kp_x = Kp_x
        self.Kd_x = Kd_x
        self.Kp_theta = Kp_theta
        self.Kd_theta = Kd_theta
        self.device = env.device

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute control force.

        Args:
            obs: State [x, theta, x_dot, theta_dot]

        Returns:
            Force to apply to cart
        """
        x = obs[0].item()
        theta = obs[1].item()
        x_dot = obs[2].item()
        theta_dot = obs[3].item()

        # PD control law
        # When theta > 0 (leaning right), push cart right (F > 0) to catch pole
        # When x > 0 (cart right of center), push left (F < 0) to recenter
        force = (
            -self.Kp_x * x
            - self.Kd_x * x_dot
            + self.Kp_theta * theta  # Positive: push in direction of lean
            + self.Kd_theta * theta_dot
        )

        # Clip to force limits
        force = max(-self.env.force_max, min(self.env.force_max, force))

        return torch.tensor([force], dtype=torch.float64, device=self.device)


class EnergySwingUpPolicy:
    """
    Energy-based swing-up + PD balance policy for cart-pole.

    Uses energy pumping to swing up, then switches to PD control near upright.
    """

    def __init__(
        self,
        env: CartPoleEnv,
        L_reference: Optional[float] = None,
        Kp_theta: float = 30.0,
        Kd_theta: float = 8.0,
        Kp_x: float = 0.3,
        Kd_x: float = 0.8,
        swing_gain: float = 2.0,
    ):
        """
        Initialize swing-up policy.

        Args:
            env: Cart-pole environment
            L_reference: Fixed pole length for energy calculation (envelope theorem)
            Kp_theta, Kd_theta: PD gains for pole angle
            Kp_x, Kd_x: PD gains for cart position
            swing_gain: Gain for energy pumping
        """
        self.env = env
        self.L_ref = L_reference if L_reference is not None else env.parametric_model.get_L()
        self.Kp_theta = Kp_theta
        self.Kd_theta = Kd_theta
        self.Kp_x = Kp_x
        self.Kd_x = Kd_x
        self.swing_gain = swing_gain
        self.device = env.device

        # Store fixed parameters
        self.m = env.m
        self.g = env.gravity

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute control force."""
        x = obs[0].item()
        theta = obs[1].item()
        x_dot = obs[2].item()
        theta_dot = obs[3].item()

        # Near upright: use PD control
        if abs(theta) < 0.3:  # ~17 degrees
            force = (
                -self.Kp_theta * theta
                - self.Kd_theta * theta_dot
                - self.Kp_x * x
                - self.Kd_x * x_dot
            )
        else:
            # Swing-up: energy pumping
            # Current energy (using reference L)
            L = self.L_ref
            E = 0.5 * self.m * L**2 * theta_dot**2 + self.m * self.g * L * (1 - math.cos(theta))
            E_target = 2 * self.m * self.g * L  # Energy at top

            # Energy error
            E_err = E - E_target

            # Pump energy: accelerate cart in direction of pole motion
            force = self.swing_gain * theta_dot * math.cos(theta)
            if E_err > 0:
                force = -force  # Remove energy if too much

            # Add weak position control to keep cart centered
            force -= 0.1 * x + 0.2 * x_dot

        # Clip
        force = max(-self.env.force_max, min(self.env.force_max, force))

        return torch.tensor([force], dtype=torch.float64, device=self.device)


if __name__ == "__main__":
    print("Testing CartPoleEnv...")

    env = CartPoleEnv()
    policy = PDCartPolePolicy(env)

    print(f"Pole length L: {env.parametric_model.get_L():.2f} m")
    print(f"Cart mass M: {env.M:.2f} kg")
    print(f"Pole mass m: {env.m:.2f} kg")

    # Test episode
    obs = env.reset()
    total_reward = 0

    for step in range(500):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward.item()

        if step % 100 == 0:
            print(f"Step {step}: x={info['x']:.3f}, theta={info['theta_deg']:.1f}deg, reward={total_reward:.0f}")

        if terminated or truncated:
            print(f"Episode ended at step {step}: terminated={terminated}, truncated={truncated}")
            break

    print(f"\nTotal reward: {total_reward:.0f}")

    # Test differentiable rollout
    print("\nTesting differentiable rollout...")
    env2 = CartPoleEnv()
    policy2 = PDCartPolePolicy(env2)
    result = env2.rollout_differentiable(policy2, horizon=200)
    print(f"Return: {result['total_return']:.0f}, Episode length: {result['episode_length']}")
    print(f"dReturn/dL: {result['L_grad']:.4f}")
