"""
Newton-based Differentiable Cart-Pole Environment

Level 1.5: Test Newton physics engine integration before Level 2.

This uses Newton's differentiable physics (Warp-based) instead of
our hand-coded PyTorch physics from Level 1.

Key differences from Level 1:
- Physics computed by Newton's SolverSemiImplicit
- Gradients via Warp's wp.Tape() instead of PyTorch autograd
- GPU-accelerated simulation
"""

import numpy as np

try:
    import warp as wp
    import newton
    NEWTON_AVAILABLE = True
except ImportError:
    NEWTON_AVAILABLE = False
    print("Warning: Newton/Warp not available. Install with: pip install -e path/to/newton")


class ParametricCartPoleNewton:
    """
    Parametric cart-pole model for Newton.

    Design parameter: Pole length L

    Note: Modifying pole length requires rebuilding the Newton model,
    which is more expensive than our PyTorch version.
    """

    def __init__(
        self,
        L_init: float = 0.6,
        L_min: float = 0.3,
        L_max: float = 1.2,
        cart_mass: float = 1.0,
        pole_linear_density: float = 0.1,
    ):
        self.L = L_init
        self.L_min = L_min
        self.L_max = L_max
        self.cart_mass = cart_mass
        self.pole_linear_density = pole_linear_density

    def get_L(self) -> float:
        return self.L

    def set_L(self, value: float):
        self.L = np.clip(value, self.L_min, self.L_max)

    @property
    def pole_mass(self) -> float:
        """Pole mass = density * 2L (full pole length)"""
        return self.pole_linear_density * 2.0 * self.L


class CartPoleNewtonEnv:
    """
    Cart-Pole environment using Newton physics engine.

    State: [x, theta, x_dot, theta_dot]
    Action: Force on cart

    Uses Newton's differentiable simulation for gradient computation.
    """

    def __init__(
        self,
        parametric_model: ParametricCartPoleNewton = None,
        dt: float = 0.02,
        force_max: float = 10.0,
        theta_threshold: float = 0.2,
        num_substeps: int = 4,
        device: str = "cuda",
    ):
        if not NEWTON_AVAILABLE:
            raise ImportError("Newton/Warp required. Install with: pip install -e path/to/newton")

        self.device = device
        self.dt = dt
        self.force_max = force_max
        self.theta_threshold = theta_threshold
        self.num_substeps = num_substeps
        self.sub_dt = dt / num_substeps

        if parametric_model is None:
            self.parametric_model = ParametricCartPoleNewton()
        else:
            self.parametric_model = parametric_model

        # Build Newton model
        self._build_model()

        # Dimensions
        self.obs_dim = 4
        self.act_dim = 1

        # Episode state
        self.steps = 0
        self.max_steps = 500
        self.terminated = False

    def _build_model(self):
        """Build Newton cart-pole model with current parameters."""
        wp.init()

        L = self.parametric_model.L
        cart_mass = self.parametric_model.cart_mass
        pole_mass = self.parametric_model.pole_mass

        builder = newton.ModelBuilder()

        # Set default density (will be overridden by shape dimensions)
        cart_width = 0.3
        cart_height = 0.1
        cart_depth = 0.2
        pole_radius = 0.02

        # Calculate densities to achieve desired masses
        cart_volume = cart_width * cart_depth * cart_height
        pole_volume = np.pi * pole_radius**2 * 2 * L
        cart_density = cart_mass / cart_volume if cart_volume > 0 else 1000.0
        pole_density = pole_mass / pole_volume if pole_volume > 0 else 1000.0

        # Set default shape density
        builder.default_shape_cfg.density = cart_density

        # Cart link
        cart_link = builder.add_link()
        builder.add_shape_box(cart_link, hx=cart_width/2, hy=cart_depth/2, hz=cart_height/2)

        # Pole link
        builder.default_shape_cfg.density = pole_density
        pole_link = builder.add_link()
        builder.add_shape_capsule(pole_link, radius=pole_radius, half_height=L)

        # Prismatic joint for cart (moves along x-axis)
        j0 = builder.add_joint_prismatic(
            parent=-1,  # World
            child=cart_link,
            axis=wp.vec3(1.0, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, cart_height/2), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            limit_lower=-2.4,
            limit_upper=2.4,
        )

        # Revolute joint for pole (rotates around y-axis)
        j1 = builder.add_joint_revolute(
            parent=cart_link,
            child=pole_link,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, cart_height/2), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -L), wp.quat_identity()),
        )

        # Create articulation
        builder.add_articulation([j0, j1], key="cartpole")

        # Finalize with gradients enabled
        self.model = builder.finalize(requires_grad=True)

        # Use XPBD solver (works well for articulations)
        self.solver = newton.solvers.SolverXPBD(self.model)

        # Allocate states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Forward kinematics
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # No contacts needed for cart-pole
        self.contacts = None

    def reset(self, theta_init: float = None) -> np.ndarray:
        """Reset environment."""
        self.steps = 0
        self.terminated = False

        # Reset joint positions/velocities
        joint_q = self.model.joint_q.numpy()
        joint_qd = self.model.joint_qd.numpy()

        # Small random initial state
        if theta_init is None:
            theta_init = np.random.uniform(-0.05, 0.05)

        joint_q[0] = np.random.uniform(-0.05, 0.05)  # Cart position
        joint_q[1] = theta_init  # Pole angle
        joint_qd[0] = np.random.uniform(-0.05, 0.05)  # Cart velocity
        joint_qd[1] = np.random.uniform(-0.05, 0.05)  # Pole angular velocity

        self.model.joint_q.assign(joint_q)
        self.model.joint_qd.assign(joint_qd)

        # Evaluate forward kinematics
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Get observation [x, theta, x_dot, theta_dot]."""
        joint_q = self.model.joint_q.numpy()
        joint_qd = self.model.joint_qd.numpy()

        return np.array([
            joint_q[0],   # x
            joint_q[1],   # theta
            joint_qd[0],  # x_dot
            joint_qd[1],  # theta_dot
        ], dtype=np.float32)

    def step(self, action: float):
        """
        Take a step in the environment.

        Args:
            action: Force on cart

        Returns:
            obs, reward, terminated, truncated, info
        """
        if self.terminated:
            return self._get_obs(), 0.0, True, False, {}

        # Clip action
        force = np.clip(action, -self.force_max, self.force_max)

        # Apply force to cart joint
        joint_act = self.model.joint_act.numpy()
        joint_act[0] = force
        self.model.joint_act.assign(joint_act)

        # Simulate substeps
        for _ in range(self.num_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.steps += 1

        # Get new state
        obs = self._get_obs()
        theta = obs[1]

        # Check termination
        self.terminated = abs(theta) > self.theta_threshold
        truncated = self.steps >= self.max_steps

        # Shaped reward (for gradient flow)
        reward = float(np.cos(theta))

        info = {
            "x": obs[0],
            "theta": theta,
            "theta_deg": np.degrees(theta),
            "x_dot": obs[2],
            "theta_dot": obs[3],
            "force": force,
            "L": self.parametric_model.L,
        }

        return obs, reward, self.terminated, truncated, info

    def compute_gradient_wrt_L(
        self,
        policy_fn,
        horizon: int = 100,
        n_rollouts: int = 5,
    ) -> tuple:
        """
        Compute gradient of return w.r.t. pole length L using Newton's tape.

        This requires rebuilding the model for each L perturbation (expensive).
        For production, would need custom Warp kernel to make L differentiable.

        Args:
            policy_fn: Frozen policy function
            horizon: Rollout length
            n_rollouts: Number of rollouts for averaging

        Returns:
            (mean_return, gradient_estimate)
        """
        # Use finite difference since L isn't directly in the Warp computation graph
        # (would need custom implementation to make L a Warp variable)

        eps = 0.01
        L_current = self.parametric_model.L

        # Evaluate at L - eps
        self.parametric_model.set_L(L_current - eps)
        self._build_model()
        returns_minus = []
        for _ in range(n_rollouts):
            obs = self.reset()
            total_return = 0.0
            for _ in range(horizon):
                action = policy_fn(obs)
                obs, reward, terminated, truncated, _ = self.step(action)
                total_return += reward
                if terminated or truncated:
                    break
            returns_minus.append(total_return)

        # Evaluate at L + eps
        self.parametric_model.set_L(L_current + eps)
        self._build_model()
        returns_plus = []
        for _ in range(n_rollouts):
            obs = self.reset()
            total_return = 0.0
            for _ in range(horizon):
                action = policy_fn(obs)
                obs, reward, terminated, truncated, _ = self.step(action)
                total_return += reward
                if terminated or truncated:
                    break
            returns_plus.append(total_return)

        # Restore L
        self.parametric_model.set_L(L_current)
        self._build_model()

        # Finite difference gradient
        mean_return = (np.mean(returns_plus) + np.mean(returns_minus)) / 2
        gradient = (np.mean(returns_plus) - np.mean(returns_minus)) / (2 * eps)

        return mean_return, gradient


def test_newton_cartpole():
    """Test Newton cart-pole environment."""
    print("Testing Newton Cart-Pole Environment")
    print("=" * 50)

    env = CartPoleNewtonEnv()
    print(f"Pole length L: {env.parametric_model.L:.2f} m")
    print(f"Pole mass: {env.parametric_model.pole_mass:.3f} kg")

    # Random policy test
    obs = env.reset()
    total_reward = 0

    for step in range(200):
        action = np.random.uniform(-1, 1) * env.force_max
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 50 == 0:
            print(f"Step {step}: x={info['x']:.3f}, theta={info['theta_deg']:.1f}deg")

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    print(f"Total reward: {total_reward:.1f}")
    print("\nNewton cart-pole test complete!")


if __name__ == "__main__":
    test_newton_cartpole()
