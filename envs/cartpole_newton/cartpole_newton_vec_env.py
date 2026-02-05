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
    from newton.selection import ArticulationView
    NEWTON_AVAILABLE = True
except ImportError:
    NEWTON_AVAILABLE = False
    print("Warning: Newton/Warp not available.")


# Kernel to extract joint positions from body transforms
# MuJoCo solver updates body_q but not model.joint_q
@wp.kernel
def extract_joint_state_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    num_bodies_per_world: int,
    joint_q_out: wp.array(dtype=float),
    joint_qd_out: wp.array(dtype=float),
    num_joints_per_world: int,
    num_worlds: int,
):
    """Extract joint positions/velocities from body transforms."""
    tid = wp.tid()  # world index

    # Body indices for this world
    cart_idx = tid * num_bodies_per_world + 0
    pole_idx = tid * num_bodies_per_world + 1

    # Get body transforms
    cart_tf = body_q[cart_idx]
    pole_tf = body_q[pole_idx]

    # Get body velocities (spatial vectors: [angular, linear])
    cart_vel = body_qd[cart_idx]
    pole_vel = body_qd[pole_idx]

    # Cart position: x component of cart body position
    cart_pos = wp.transform_get_translation(cart_tf)
    # Subtract world offset (worlds are centered around origin)
    spacing = 3.0
    first_world_x = -float(num_worlds - 1) * spacing / 2.0
    world_offset = first_world_x + float(tid) * spacing
    x = cart_pos[0] - world_offset

    # Pole angle: extract from pole body quaternion
    # For rotation around y-axis: qy = sin(θ/2), qw = cos(θ/2)
    pole_quat = wp.transform_get_rotation(pole_tf)
    qy = pole_quat[1]  # y component
    qw = pole_quat[3]  # w component
    theta = 2.0 * wp.atan2(qy, qw)

    # Velocities
    # Cart x velocity: linear velocity x component
    x_dot = wp.spatial_bottom(cart_vel)[0]

    # Pole angular velocity around y-axis
    theta_dot = wp.spatial_top(pole_vel)[1]

    # Write to output arrays
    x_idx = tid * num_joints_per_world
    theta_idx = tid * num_joints_per_world + 1

    joint_q_out[x_idx] = x
    joint_q_out[theta_idx] = theta
    joint_qd_out[x_idx] = x_dot
    joint_qd_out[theta_idx] = theta_dot


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
    forces: wp.array(dtype=wp.float32),
    joint_f: wp.array(dtype=wp.float32),
    num_dofs_per_world: int,
):
    """Apply cart forces to joint_f array for all worlds.

    Uses control.joint_f which directly applies generalized forces to joint DOFs.
    For prismatic joint (cart), this is a force in the joint axis direction.
    """
    tid = wp.tid()

    # Cart is DOF 0 in each world (prismatic joint)
    # joint_f layout: [cart_x_0, pole_theta_0, cart_x_1, pole_theta_1, ...]
    cart_dof_idx = tid * num_dofs_per_world

    # Apply force to cart prismatic joint
    force = forces[tid]
    joint_f[cart_dof_idx] = force


@wp.kernel
def reset_worlds_kernel(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    terminated: wp.array(dtype=int),
    num_joints_per_world: int,
    rng_seed: int,
):
    """Reset terminated worlds - pole starts pointing down (theta = π) for swing-up."""
    tid = wp.tid()

    if terminated[tid] == 1:
        x_idx = tid * num_joints_per_world
        theta_idx = tid * num_joints_per_world + 1

        # Position: cart at center, pole pointing down for swing-up
        joint_q[x_idx] = 0.0
        joint_q[theta_idx] = 3.14159265359  # π - pointing down

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
        force_max: float = 30.0,  # 30N on 1kg cart, reasonable for balance/swing-up
        theta_threshold: float = 0.2,
        num_substeps: int = 4,
        ctrl_cost_weight: float = 0.01,
        x_limit: float = 3.0,  # IsaacLab uses (-3.0, 3.0)
        start_near_upright: bool = True,  # Start with balance task first
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
        self.x_limit = x_limit
        self.start_near_upright = start_near_upright

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

        # Prismatic joint for cart (limits slightly beyond termination for physics)
        j0 = builder.add_joint_prismatic(
            parent=-1,
            child=cart_link,
            axis=wp.vec3(1.0, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, cart_height/2), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            limit_lower=-4.0,
            limit_upper=4.0,
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

        # Set initial joint positions BEFORE finalize
        # These will be overwritten by reset() anyway
        builder.joint_q[-2:] = [0.0, 0.0]  # x=0, theta=0 (upright)

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
        self.num_dofs_per_world = 2  # prismatic (x) + revolute (theta)

        # Use MuJoCo solver (reduced coordinates - updates joint_q directly)
        # disable_contacts=True since cart-pole doesn't need collisions
        self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=True)

        # Allocate states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        # Create ArticulationView for batch operations (like Newton's official examples)
        self.cartpoles = ArticulationView(self.model, "*/cartpole", verbose=False)

        # Forward kinematics
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

    def _alloc_buffers(self):
        """Allocate GPU buffers for parallel computation."""
        self.forces_wp = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.rewards_wp = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.terminated_wp = wp.zeros(self.num_worlds, dtype=int, device=self.device)

    def reset(self) -> np.ndarray:
        """Reset all environments using Newton's official ArticulationView pattern."""
        self.steps[:] = 0
        self._step_count = 0

        # Get current joint state via ArticulationView (returns (num_worlds, num_dofs) array)
        joint_q = self.cartpoles.get_attribute("joint_q", self.state_0)
        joint_qd = self.cartpoles.get_attribute("joint_qd", self.state_0)

        # Convert to numpy for modification
        joint_q_np = joint_q.numpy()
        joint_qd_np = joint_qd.numpy()

        for i in range(self.num_worlds):
            if self.start_near_upright:
                # IsaacLab style: random start near upright for BALANCE task
                joint_q_np[i, 0] = np.random.uniform(-1.0, 1.0)      # cart x
                joint_q_np[i, 1] = np.random.uniform(-0.25 * np.pi, 0.25 * np.pi)  # pole angle
                joint_qd_np[i, 0] = np.random.uniform(-0.5, 0.5)     # cart velocity
                joint_qd_np[i, 1] = np.random.uniform(-0.25 * np.pi, 0.25 * np.pi)  # pole angular vel
            else:
                # Swing-up task: pole starts pointing down
                joint_q_np[i, 0] = 0.0
                joint_q_np[i, 1] = np.pi
                joint_qd_np[i, 0] = 0.0
                joint_qd_np[i, 1] = 0.0

        # Set state via ArticulationView (like Newton's official examples)
        joint_q_wp = wp.array(joint_q_np, dtype=wp.float32, device=self.device)
        joint_qd_wp = wp.array(joint_qd_np, dtype=wp.float32, device=self.device)
        self.cartpoles.set_attribute("joint_q", self.state_0, joint_q_wp)
        self.cartpoles.set_attribute("joint_qd", self.state_0, joint_qd_wp)

        # Also set on state_1 to avoid stale data
        self.cartpoles.set_attribute("joint_q", self.state_1, joint_q_wp)
        self.cartpoles.set_attribute("joint_qd", self.state_1, joint_qd_wp)

        # Propagate reset state by running one simulate step (like test_anymal_reset.py)
        # This ensures MuJoCo's internal state is properly initialized
        self._propagate_state()
        wp.synchronize()

        obs = self._get_obs()
        return obs

    def _propagate_state(self):
        """Run one simulation step to propagate state through solver (Newton pattern)."""
        # Use zero control for propagation
        joint_f = self.cartpoles.get_attribute("joint_f", self.control)
        joint_f_np = joint_f.numpy()
        joint_f_np[:] = 0.0
        joint_f_wp = wp.array(joint_f_np, dtype=wp.float32, device=self.device)
        self.cartpoles.set_attribute("joint_f", self.control, joint_f_wp)

        # Run one substep to propagate
        self.state_0.clear_forces()
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sub_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def _get_obs(self) -> np.ndarray:
        """Get observations for all worlds using ArticulationView (Newton's pattern)."""
        # Get joint state via ArticulationView (handles mapping automatically)
        # Returns (num_worlds, num_dofs) where num_dofs=2: [cart_x, pole_theta]
        joint_q = self.cartpoles.get_attribute("joint_q", self.state_0)
        joint_qd = self.cartpoles.get_attribute("joint_qd", self.state_0)

        joint_q_np = joint_q.numpy()
        joint_qd_np = joint_qd.numpy()

        obs = np.zeros((self.num_worlds, self.obs_dim), dtype=np.float32)
        obs[:, 0] = joint_q_np[:, 0]   # cart x
        obs[:, 1] = joint_q_np[:, 1]   # pole theta
        obs[:, 2] = joint_qd_np[:, 0]  # cart x_dot
        obs[:, 3] = joint_qd_np[:, 1]  # pole theta_dot

        return obs

    def step(self, actions: np.ndarray):
        """
        Step all environments in parallel using Newton's official pattern.

        Args:
            actions: (num_worlds,) or (num_worlds, 1) forces in [-1, 1]

        Returns:
            obs: (num_worlds, 4)
            rewards: (num_worlds,)
            dones: (num_worlds,) bool
            infos: dict
        """
        # Flatten actions if needed
        actions = np.asarray(actions, dtype=np.float32).flatten()
        assert len(actions) == self.num_worlds

        # Scale actions to forces (actions should already be in [-1, 1] from tanh policy)
        forces = actions * self.force_max

        # Set forces via ArticulationView (like selection_cartpole.py)
        # joint_f shape: (num_worlds, num_dofs_per_world)
        joint_f = self.cartpoles.get_attribute("joint_f", self.control)
        joint_f_np = joint_f.numpy()
        joint_f_np[:, 0] = forces  # Apply force to cart DOF only (index 0)
        joint_f_np[:, 1] = 0.0     # No torque on pole joint
        joint_f_wp = wp.array(joint_f_np, dtype=wp.float32, device=self.device)
        self.cartpoles.set_attribute("joint_f", self.control, joint_f_wp)

        # Simulate substeps using Newton's official pattern:
        # clear_forces() INSIDE the loop, BEFORE each solver.step()
        for _ in range(self.num_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        # Synchronize to prevent warp state corruption
        wp.synchronize()

        self.steps += 1
        self._step_count += 1

        # Get observations (computed from body state)
        obs = self._get_obs()

        # Compute rewards from observations
        theta = obs[:, 1]
        x = obs[:, 0]
        x_dot = obs[:, 2]
        theta_dot = obs[:, 3]

        # Check for NaN/Inf (physics instability)
        invalid = np.isnan(x) | np.isinf(x) | np.isnan(theta) | np.isinf(theta)
        if np.any(invalid):
            x = np.where(invalid, 0.0, x)
            theta = np.where(invalid, np.pi, theta)
            x_dot = np.where(invalid, 0.0, x_dot)
            theta_dot = np.where(invalid, 0.0, theta_dot)

        # IsaacLab-style reward structure:
        # (1) Alive bonus: +1.0 per step
        alive_reward = 1.0

        # (2) Pole position: penalize angle from upright (theta=0)
        # IsaacLab uses -1.0 * L2, we use cos which is similar
        # cos(theta) = 1 at upright, -1 at down
        pole_pos_reward = np.cos(theta)

        # (3) Cart velocity penalty (small) - IsaacLab uses -0.01 * L1
        cart_vel_penalty = 0.01 * np.abs(x_dot)

        # (4) Pole angular velocity penalty (small) - IsaacLab uses -0.005 * L1
        pole_vel_penalty = 0.005 * np.abs(theta_dot)

        rewards = alive_reward + pole_pos_reward - cart_vel_penalty - pole_vel_penalty

        # Check termination (cart position limit or invalid state)
        terminated = (np.abs(x) > self.x_limit) | invalid

        # Termination penalty: -2.0 (same as IsaacLab)
        rewards[terminated] -= 2.0

        # Check truncation (max steps)
        truncated = self.steps >= self.max_steps
        dones = terminated | truncated

        # Auto-reset terminated worlds
        reset_mask = dones
        if np.any(reset_mask):
            self._reset_worlds(reset_mask)
            # Re-get observations for reset worlds
            obs = self._get_obs()

        infos = {
            "L": self.parametric_model.L,
            "mean_reward": np.mean(rewards),
            "num_dones": np.sum(dones),
        }

        return obs, rewards, dones, infos

    def _reset_worlds(self, reset_mask: np.ndarray):
        """Reset specific worlds that terminated using ArticulationView pattern."""
        num_reset = np.sum(reset_mask)
        if num_reset == 0:
            return

        # Get current joint state via ArticulationView
        joint_q = self.cartpoles.get_attribute("joint_q", self.state_0)
        joint_qd = self.cartpoles.get_attribute("joint_qd", self.state_0)

        joint_q_np = joint_q.numpy()
        joint_qd_np = joint_qd.numpy()

        for i in range(self.num_worlds):
            if reset_mask[i]:
                if self.start_near_upright:
                    # IsaacLab style: random start near upright
                    joint_q_np[i, 0] = np.random.uniform(-1.0, 1.0)
                    joint_q_np[i, 1] = np.random.uniform(-0.25 * np.pi, 0.25 * np.pi)
                    joint_qd_np[i, 0] = np.random.uniform(-0.5, 0.5)
                    joint_qd_np[i, 1] = np.random.uniform(-0.25 * np.pi, 0.25 * np.pi)
                else:
                    # Swing-up: pole starts pointing down
                    joint_q_np[i, 0] = 0.0
                    joint_q_np[i, 1] = np.pi
                    joint_qd_np[i, 0] = 0.0
                    joint_qd_np[i, 1] = 0.0

                self.steps[i] = 0

        # Set state via ArticulationView (like Newton's official examples)
        joint_q_wp = wp.array(joint_q_np, dtype=wp.float32, device=self.device)
        joint_qd_wp = wp.array(joint_qd_np, dtype=wp.float32, device=self.device)
        self.cartpoles.set_attribute("joint_q", self.state_0, joint_q_wp)
        self.cartpoles.set_attribute("joint_qd", self.state_0, joint_qd_wp)
        self.cartpoles.set_attribute("joint_q", self.state_1, joint_q_wp)
        self.cartpoles.set_attribute("joint_qd", self.state_1, joint_qd_wp)

        # Propagate reset state (like test_anymal_reset.py)
        self._propagate_state()
        wp.synchronize()


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
