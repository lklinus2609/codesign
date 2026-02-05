"""
Vectorized Newton-based Cart-Pole Environment

Runs N cart-poles in parallel on GPU using Newton's world replication.
This provides massive speedup for PPO training.

Usage:
    env = CartPoleNewtonVecEnv(num_worlds=64)
    obs = env.reset()  # Shape: (64, 4)
    actions = policy(obs)  # Shape: (64, 1)
    obs, rewards, dones, infos = env.step(actions)
"""

import numpy as np

try:
    import warp as wp
    import newton
    NEWTON_AVAILABLE = True
except ImportError:
    NEWTON_AVAILABLE = False
    print("Warning: Newton/Warp not available.")


# Warp kernel for computing rewards in parallel
@wp.kernel
def compute_rewards_kernel(
    joint_q: wp.array(dtype=float),
    forces: wp.array(dtype=float),
    force_max: float,
    ctrl_cost_weight: float,
    num_joints_per_world: int,
    rewards: wp.array(dtype=float),
):
    """Compute reward = cos(theta) - ctrl_cost for each world."""
    tid = wp.tid()

    # Get theta (pole angle) for this world
    # joint_q layout: [x0, theta0, x1, theta1, ...] (2 joints per world)
    theta_idx = tid * num_joints_per_world + 1
    theta = joint_q[theta_idx]

    # Balance reward
    balance_reward = wp.cos(theta)

    # Control cost
    force = forces[tid]
    normalized_force = force / force_max
    ctrl_cost = ctrl_cost_weight * normalized_force * normalized_force

    rewards[tid] = balance_reward - ctrl_cost


@wp.kernel
def check_termination_kernel(
    joint_q: wp.array(dtype=float),
    x_limit: float,
    num_joints_per_world: int,
    terminated: wp.array(dtype=int),
):
    """Check if each world has terminated (cart out of bounds)."""
    tid = wp.tid()

    x_idx = tid * num_joints_per_world
    x = joint_q[x_idx]

    # Only terminate if cart goes out of bounds (no theta limit for swing-up)
    if wp.abs(x) > x_limit:
        terminated[tid] = 1
    else:
        terminated[tid] = 0


@wp.kernel
def apply_forces_kernel(
    forces: wp.array(dtype=float),
    body_f: wp.array(dtype=wp.spatial_vector),
    num_bodies_per_world: int,
):
    """Apply cart forces to body_f array for all worlds."""
    tid = wp.tid()

    # Cart is body 0 in each world
    cart_body_idx = tid * num_bodies_per_world

    # Force in x direction (spatial vector: [torque, force])
    force = forces[tid]
    body_f[cart_body_idx] = wp.spatial_vector(0.0, 0.0, 0.0, force, 0.0, 0.0)


@wp.kernel
def reset_worlds_kernel(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    terminated: wp.array(dtype=int),
    num_joints_per_world: int,
    rng_seed: int,
):
    """Reset terminated worlds - pole starts nearly upright (theta = 0.1)."""
    tid = wp.tid()

    if terminated[tid] == 1:
        x_idx = tid * num_joints_per_world
        theta_idx = tid * num_joints_per_world + 1

        # Position: cart at center, pole nearly upright
        joint_q[x_idx] = 0.0
        joint_q[theta_idx] = 0.1  # Nearly upright, will fall

        # Velocity: zero
        joint_qd[x_idx] = 0.0
        joint_qd[theta_idx] = 0.0


class ParametricCartPoleNewton:
    """Parametric cart-pole model for Newton."""

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
        self.L = float(np.clip(value, self.L_min, self.L_max))

    @property
    def pole_mass(self) -> float:
        return self.pole_linear_density * 2.0 * self.L


class CartPoleNewtonVecEnv:
    """
    Vectorized Cart-Pole environment using Newton physics engine.

    Runs num_worlds cart-poles in parallel on GPU.

    Observations: (num_worlds, 4) - [x, theta, x_dot, theta_dot] per world
    Actions: (num_worlds,) or (num_worlds, 1) - force per world
    Rewards: (num_worlds,) - shaped reward per world
    """

    def __init__(
        self,
        num_worlds: int = 64,
        parametric_model: ParametricCartPoleNewton = None,
        dt: float = 0.02,
        force_max: float = 10.0,
        theta_threshold: float = 0.2,
        num_substeps: int = 4,
        ctrl_cost_weight: float = 0.5,
        device: str = "cuda:0",
    ):
        if not NEWTON_AVAILABLE:
            raise ImportError("Newton/Warp required.")

        self.num_worlds = num_worlds
        self.device = device
        self.dt = dt
        self.force_max = force_max
        self.theta_threshold = theta_threshold
        self.num_substeps = num_substeps
        self.sub_dt = dt / num_substeps
        self.ctrl_cost_weight = ctrl_cost_weight

        if parametric_model is None:
            self.parametric_model = ParametricCartPoleNewton()
        else:
            self.parametric_model = parametric_model

        # Build Newton model with replicated worlds
        self._build_model()

        # Dimensions
        self.obs_dim = 4
        self.act_dim = 1

        # Episode tracking per world
        self.max_steps = 500
        self.steps = np.zeros(num_worlds, dtype=np.int32)

        # Allocate GPU arrays for intermediate computations
        self._alloc_buffers()

        # Step counter for random seed
        self._step_count = 0

    def _build_single_cartpole(self) -> newton.ModelBuilder:
        """Build a single cart-pole articulation."""
        L = float(self.parametric_model.L)
        cart_mass = float(self.parametric_model.cart_mass)
        pole_mass = float(self.parametric_model.pole_mass)

        builder = newton.ModelBuilder()

        # Register MuJoCo solver attributes and set armature for stability
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg.armature = 0.1
        builder.default_body_armature = 0.1

        cart_width = 0.3
        cart_height = 0.1
        cart_depth = 0.2
        pole_radius = 0.02

        cart_volume = cart_width * cart_depth * cart_height
        pole_volume = np.pi * pole_radius**2 * 2 * L
        cart_density = float(cart_mass / cart_volume) if cart_volume > 0 else 1000.0
        pole_density = float(pole_mass / pole_volume) if pole_volume > 0 else 1000.0

        builder.default_shape_cfg.density = cart_density

        # Cart link
        cart_link = builder.add_link()
        builder.add_shape_box(cart_link, hx=cart_width/2, hy=cart_depth/2, hz=cart_height/2)

        # Pole link
        builder.default_shape_cfg.density = pole_density
        pole_link = builder.add_link()
        builder.add_shape_capsule(pole_link, radius=pole_radius, half_height=L)

        # Prismatic joint for cart
        j0 = builder.add_joint_prismatic(
            parent=-1,
            child=cart_link,
            axis=wp.vec3(1.0, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, cart_height/2), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            limit_lower=-2.4,
            limit_upper=2.4,
        )

        # Revolute joint for pole
        j1 = builder.add_joint_revolute(
            parent=cart_link,
            child=pole_link,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, cart_height/2), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -L), wp.quat_identity()),
        )

        builder.add_articulation([j0, j1], key="cartpole")

        # Set initial joint positions BEFORE finalize (like the example)
        # joint_q layout: [x, theta] for cart-pole
        # Start near upright (theta=0.1) so it falls under gravity
        builder.joint_q[-2:] = [0.0, 0.1]  # x=0, theta=0.1 (nearly upright)

        return builder

    def _build_model(self):
        """Build Newton model with replicated cart-poles."""
        wp.init()

        # Build single cart-pole template
        single_cartpole = self._build_single_cartpole()

        # Replicate across worlds
        builder = newton.ModelBuilder()
        builder.replicate(single_cartpole, self.num_worlds, spacing=(3.0, 0.0, 0.0))

        # Finalize
        self.model = builder.finalize()

        # Store counts per world
        self.num_bodies_per_world = self.model.body_count // self.num_worlds
        self.num_joints_per_world = 2  # x and theta

        # Use MuJoCo solver (reduced coordinates - updates joint_q directly)
        self.solver = newton.solvers.SolverMuJoCo(self.model)

        # Allocate states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        # Forward kinematics
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

    def _alloc_buffers(self):
        """Allocate GPU buffers for parallel computation."""
        self.forces_wp = wp.zeros(self.num_worlds, dtype=float, device=self.device)
        self.rewards_wp = wp.zeros(self.num_worlds, dtype=float, device=self.device)
        self.terminated_wp = wp.zeros(self.num_worlds, dtype=int, device=self.device)

    def reset(self) -> np.ndarray:
        """Reset all environments. Returns observations (num_worlds, 4)."""
        self.steps[:] = 0
        self._step_count = 0

        # Initial state: pole nearly upright (theta = 0.1)
        joint_q = self.model.joint_q.numpy()
        joint_qd = self.model.joint_qd.numpy()

        for i in range(self.num_worlds):
            x_idx = i * self.num_joints_per_world
            theta_idx = i * self.num_joints_per_world + 1

            # Cart at center, pole nearly upright (theta = 0.1)
            joint_q[x_idx] = 0.0
            joint_q[theta_idx] = 0.1  # Nearly upright, will fall
            joint_qd[x_idx] = 0.0
            joint_qd[theta_idx] = 0.0

        self.model.joint_q.assign(joint_q)
        self.model.joint_qd.assign(joint_qd)

        # Forward kinematics
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.synchronize()

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Get observations for all worlds. Returns (num_worlds, 4)."""
        joint_q = self.model.joint_q.numpy()
        joint_qd = self.model.joint_qd.numpy()

        # Debug: verify theta is changing (remove after testing)
        if self._step_count <= 5:
            print(f"[DEBUG] step={self._step_count}, theta[0]={joint_q[1]:.4f}")

        obs = np.zeros((self.num_worlds, self.obs_dim), dtype=np.float32)
        for i in range(self.num_worlds):
            x_idx = i * self.num_joints_per_world
            theta_idx = i * self.num_joints_per_world + 1

            obs[i, 0] = joint_q[x_idx]      # x
            obs[i, 1] = joint_q[theta_idx]   # theta
            obs[i, 2] = joint_qd[x_idx]      # x_dot
            obs[i, 3] = joint_qd[theta_idx]  # theta_dot

        return obs

    def step(self, actions: np.ndarray):
        """
        Step all environments in parallel.

        Args:
            actions: (num_worlds,) or (num_worlds, 1) forces

        Returns:
            obs: (num_worlds, 4)
            rewards: (num_worlds,)
            dones: (num_worlds,) bool
            infos: dict
        """
        # Flatten actions if needed
        actions = np.asarray(actions, dtype=np.float32).flatten()
        assert len(actions) == self.num_worlds

        # Clip actions
        forces = np.clip(actions * self.force_max, -self.force_max, self.force_max)
        self.forces_wp.assign(forces)

        # Simulate substeps (like the example - no force application for now)
        for _ in range(self.num_substeps):
            self.state_0.clear_forces()

            # TODO: Apply cart forces via actuator/control, not body_f
            # For now, test pure gravity simulation like the example
            # wp.launch(
            #     apply_forces_kernel,
            #     dim=self.num_worlds,
            #     inputs=[self.forces_wp, self.state_0.body_f, self.num_bodies_per_world],
            #     device=self.device,
            # )

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        # MuJoCo solver updates joint_q directly (reduced coordinates)
        wp.synchronize()

        self.steps += 1
        self._step_count += 1

        # Compute rewards using kernel
        wp.launch(
            compute_rewards_kernel,
            dim=self.num_worlds,
            inputs=[
                self.model.joint_q,
                self.forces_wp,
                self.force_max,
                self.ctrl_cost_weight,
                self.num_joints_per_world,
            ],
            outputs=[self.rewards_wp],
            device=self.device,
        )

        # Check termination using kernel (cart position limit, not theta)
        wp.launch(
            check_termination_kernel,
            dim=self.num_worlds,
            inputs=[
                self.model.joint_q,
                2.4,  # x_limit - cart position bounds
                self.num_joints_per_world,
            ],
            outputs=[self.terminated_wp],
            device=self.device,
        )

        # Get results
        rewards = self.rewards_wp.numpy().copy()
        terminated = self.terminated_wp.numpy().copy()

        # Check truncation (max steps)
        truncated = self.steps >= self.max_steps
        dones = (terminated == 1) | truncated

        # Auto-reset terminated worlds
        reset_mask = dones
        if np.any(reset_mask):
            self._reset_worlds(reset_mask)

        # Get observations
        obs = self._get_obs()

        infos = {
            "L": self.parametric_model.L,
            "mean_reward": np.mean(rewards),
            "num_dones": np.sum(dones),
        }

        return obs, rewards, dones, infos

    def _reset_worlds(self, reset_mask: np.ndarray):
        """Reset specific worlds that terminated - pole starts pointing down."""
        joint_q = self.model.joint_q.numpy()
        joint_qd = self.model.joint_qd.numpy()

        for i in range(self.num_worlds):
            if reset_mask[i]:
                x_idx = i * self.num_joints_per_world
                theta_idx = i * self.num_joints_per_world + 1

                # Cart at center, pole nearly upright
                joint_q[x_idx] = 0.0
                joint_q[theta_idx] = 0.1  # Nearly upright, will fall
                joint_qd[x_idx] = 0.0
                joint_qd[theta_idx] = 0.0

                self.steps[i] = 0

        self.model.joint_q.assign(joint_q)
        self.model.joint_qd.assign(joint_qd)

        # Update FK for reset worlds
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)


def test_vec_env():
    """Test vectorized environment."""
    print("Testing Vectorized Newton Cart-Pole")
    print("=" * 50)

    num_worlds = 64
    env = CartPoleNewtonVecEnv(num_worlds=num_worlds)

    print(f"Num worlds: {num_worlds}")
    print(f"Obs shape: ({num_worlds}, {env.obs_dim})")
    print(f"Act shape: ({num_worlds},)")

    obs = env.reset()
    print(f"\nInitial obs shape: {obs.shape}")
    print(f"Initial obs mean: {obs.mean(axis=0)}")

    # Random actions
    total_rewards = np.zeros(num_worlds)
    n_steps = 200

    import time
    start = time.time()

    for step in range(n_steps):
        actions = np.random.uniform(-1, 1, size=num_worlds)
        obs, rewards, dones, infos = env.step(actions)
        total_rewards += rewards

        if step % 50 == 0:
            print(f"Step {step}: mean_reward={infos['mean_reward']:.3f}, dones={infos['num_dones']}")

    elapsed = time.time() - start
    fps = (n_steps * num_worlds) / elapsed

    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Throughput: {fps:.0f} steps/sec ({num_worlds} worlds x {n_steps} steps)")
    print(f"Mean total reward: {total_rewards.mean():.1f}")
    print("\nVectorized environment test complete!")


if __name__ == "__main__":
    test_vec_env()
