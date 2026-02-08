"""
Vectorized Newton-based Cart-Pole Environment

Runs N cart-poles in parallel on GPU using Newton's world replication.
All per-step computation (forces, obs, rewards, resets) stays GPU-resident
via Warp kernels — matching Newton's official ArticulationView pattern.

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


# ---------------------------------------------------------------------------
# Warp kernels — all per-step work stays on GPU
# ---------------------------------------------------------------------------

@wp.kernel
def set_joint_forces_kernel(
    actions: wp.array(dtype=float),
    force_max: float,
    joint_f: wp.array2d(dtype=float),
):
    """Scale actions [-1,1] to forces and write to cart DOF (index 0)."""
    tid = wp.tid()
    joint_f[tid, 0] = actions[tid] * force_max
    joint_f[tid, 1] = 0.0


@wp.kernel
def extract_obs_kernel(
    joint_q: wp.array2d(dtype=float),
    joint_qd: wp.array2d(dtype=float),
    obs: wp.array2d(dtype=float),
):
    """Extract [x, theta, x_dot, theta_dot] from joint state."""
    tid = wp.tid()
    obs[tid, 0] = joint_q[tid, 0]
    obs[tid, 1] = joint_q[tid, 1]
    obs[tid, 2] = joint_qd[tid, 0]
    obs[tid, 3] = joint_qd[tid, 1]


@wp.kernel
def step_rewards_done_kernel(
    obs: wp.array2d(dtype=float),
    steps: wp.array(dtype=int),
    x_limit: float,
    max_steps: int,
    rewards: wp.array(dtype=float),
    dones: wp.array(dtype=int),
):
    """Increment steps, compute IsaacLab-style rewards, check termination/truncation."""
    tid = wp.tid()

    steps[tid] = steps[tid] + 1

    x = obs[tid, 0]
    theta = obs[tid, 1]
    x_dot = obs[tid, 2]
    theta_dot = obs[tid, 3]

    # Termination: cart out of bounds only (allow pole to swing freely)
    terminated = 0
    if wp.abs(x) > x_limit:
        terminated = 1

    # Reward: cos(θ) clamped to [0,1] + termination penalty + velocity penalties
    terminated_f = float(terminated)
    angle_reward = wp.max(wp.cos(theta), 0.0)
    r = angle_reward - 100.0 * terminated_f - 0.01 * wp.abs(x_dot) - 0.005 * wp.abs(theta_dot)

    # Truncation: max steps reached
    truncated = 0
    if steps[tid] >= max_steps:
        truncated = 1

    done = 0
    if terminated == 1 or truncated == 1:
        done = 1

    rewards[tid] = r
    dones[tid] = done


@wp.kernel
def reset_done_worlds_kernel(
    joint_q: wp.array2d(dtype=float),
    joint_qd: wp.array2d(dtype=float),
    dones: wp.array(dtype=int),
    steps: wp.array(dtype=int),
    seed: int,
    start_near_upright: int,
):
    """Reset worlds where dones==1. Randomizes state on GPU."""
    tid = wp.tid()

    if dones[tid] == 1:
        rng = wp.rand_init(seed, tid)

        if start_near_upright == 1:
            # IsaacLab style: only randomize pole angle, cart centered, zero velocities
            joint_q[tid, 0] = 0.0
            joint_q[tid, 1] = -wp.pi / 4.0 + wp.pi / 2.0 * wp.randf(rng)  # ±45°
            joint_qd[tid, 0] = 0.0
            joint_qd[tid, 1] = 0.0
        else:
            joint_q[tid, 0] = 0.0
            joint_q[tid, 1] = wp.pi
            joint_qd[tid, 0] = 0.0
            joint_qd[tid, 1] = 0.0

        steps[tid] = 0


@wp.kernel
def init_all_worlds_kernel(
    joint_q: wp.array2d(dtype=float),
    joint_qd: wp.array2d(dtype=float),
    seed: int,
    start_near_upright: int,
):
    """Initialize all worlds for full reset."""
    tid = wp.tid()
    rng = wp.rand_init(seed, tid)

    if start_near_upright == 1:
        # IsaacLab style: only randomize pole angle, cart centered, zero velocities
        joint_q[tid, 0] = 0.0
        joint_q[tid, 1] = -wp.pi / 4.0 + wp.pi / 2.0 * wp.randf(rng)  # ±45°
        joint_qd[tid, 0] = 0.0
        joint_qd[tid, 1] = 0.0
    else:
        joint_q[tid, 0] = 0.0
        joint_q[tid, 1] = wp.pi
        joint_qd[tid, 0] = 0.0
        joint_qd[tid, 1] = 0.0


# ---------------------------------------------------------------------------
# Parametric model (unchanged)
# ---------------------------------------------------------------------------

class ParametricCartPoleNewton:
    """Parametric cart-pole model for Newton."""

    def __init__(
        self,
        L_init: float = 0.6,
        L_min: float = 0.3,
        L_max: float = 1.2,
        cart_mass: float = 1.0,
        pole_linear_density: float = 0.2,
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


# ---------------------------------------------------------------------------
# Vectorized environment
# ---------------------------------------------------------------------------

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

        self.max_steps = 250  # 5 seconds at dt=0.02 (matches IsaacLab)

        # Allocate GPU arrays for all per-step computation
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
        # Use the key we set in add_articulation()
        self.cartpoles = ArticulationView(self.model, "cartpole", verbose=False)

        # Forward kinematics
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

    def _alloc_buffers(self):
        """Allocate GPU buffers for all per-step computation."""
        self.obs_wp = wp.zeros((self.num_worlds, self.obs_dim), dtype=float, device=self.device)
        self.rewards_wp = wp.zeros(self.num_worlds, dtype=float, device=self.device)
        self.dones_wp = wp.zeros(self.num_worlds, dtype=int, device=self.device)
        self.steps_wp = wp.zeros(self.num_worlds, dtype=int, device=self.device)

    @property
    def steps(self):
        """Step counts per world (GPU->CPU transfer on access)."""
        return self.steps_wp.numpy()

    def _extract_obs_to_gpu(self):
        """Extract observations into self.obs_wp on GPU (no CPU transfer)."""
        joint_q = self.cartpoles.get_attribute("joint_q", self.state_0)
        joint_qd = self.cartpoles.get_attribute("joint_qd", self.state_0)
        wp.launch(extract_obs_kernel, dim=self.num_worlds,
                  inputs=[joint_q, joint_qd, self.obs_wp])

    def _get_obs(self) -> np.ndarray:
        """Get observations as numpy (extracts on GPU, then transfers once)."""
        self._extract_obs_to_gpu()
        return self.obs_wp.numpy()

    def reset(self) -> np.ndarray:
        """Reset all environments using Warp kernels (GPU-resident)."""
        wp.synchronize()

        self.steps_wp = wp.zeros(self.num_worlds, dtype=int, device=self.device)
        self._step_count = 0

        # Get joint state arrays via ArticulationView
        joint_q = self.cartpoles.get_attribute("joint_q", self.state_0)
        joint_qd = self.cartpoles.get_attribute("joint_qd", self.state_0)

        # Initialize all worlds via kernel (no CPU round-trip)
        wp.launch(init_all_worlds_kernel, dim=self.num_worlds,
                  inputs=[joint_q, joint_qd, 42, int(self.start_near_upright)])

        # Write back to both states
        self.cartpoles.set_attribute("joint_q", self.state_0, joint_q)
        self.cartpoles.set_attribute("joint_qd", self.state_0, joint_qd)
        self.cartpoles.set_attribute("joint_q", self.state_1, joint_q)
        self.cartpoles.set_attribute("joint_qd", self.state_1, joint_qd)

        # Propagate reset state through MuJoCo solver internals
        self._propagate_state()
        wp.synchronize()

        return self._get_obs()

    def _propagate_state(self):
        """Run one simulation step to propagate state through solver (Newton pattern)."""
        wp.synchronize()

        # Zero forces via a fresh array (no CPU round-trip)
        zero_f = wp.zeros((self.num_worlds, self.num_dofs_per_world), dtype=float, device=self.device)
        self.cartpoles.set_attribute("joint_f", self.control, zero_f)

        # Run one substep to propagate
        self.state_0.clear_forces()
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sub_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

        wp.synchronize()

    def step(self, actions: np.ndarray):
        """
        Step all environments in parallel.

        All per-step computation (forces, physics, obs, rewards, resets)
        runs on GPU via Warp kernels. Only one CPU->GPU transfer (actions)
        and one GPU->CPU transfer (obs + rewards + dones) per step.

        Args:
            actions: (num_worlds,) or (num_worlds, 1) forces in [-1, 1]

        Returns:
            obs: (num_worlds, 4)
            rewards: (num_worlds,)
            dones: (num_worlds,) bool
            infos: dict
        """
        # --- Single CPU -> GPU transfer: actions ---
        actions = np.asarray(actions, dtype=np.float32).flatten()
        assert len(actions) == self.num_worlds
        actions_wp = wp.array(actions, dtype=float, device=self.device)

        # --- GPU: set forces via kernel ---
        joint_f = self.cartpoles.get_attribute("joint_f", self.control)
        wp.launch(set_joint_forces_kernel, dim=self.num_worlds,
                  inputs=[actions_wp, self.force_max, joint_f])
        self.cartpoles.set_attribute("joint_f", self.control, joint_f)

        # --- GPU: physics substeps ---
        for _ in range(self.num_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        # --- GPU: extract observations ---
        self._extract_obs_to_gpu()

        # --- GPU: compute rewards, increment steps, check done ---
        wp.launch(step_rewards_done_kernel, dim=self.num_worlds,
                  inputs=[self.obs_wp, self.steps_wp, self.x_limit, self.max_steps,
                          self.rewards_wp, self.dones_wp])

        self._step_count += 1

        # --- GPU: auto-reset done worlds ---
        joint_q = self.cartpoles.get_attribute("joint_q", self.state_0)
        joint_qd = self.cartpoles.get_attribute("joint_qd", self.state_0)
        wp.launch(reset_done_worlds_kernel, dim=self.num_worlds,
                  inputs=[joint_q, joint_qd, self.dones_wp, self.steps_wp,
                          self._step_count, int(self.start_near_upright)])
        self.cartpoles.set_attribute("joint_q", self.state_0, joint_q)
        self.cartpoles.set_attribute("joint_qd", self.state_0, joint_qd)
        self.cartpoles.set_attribute("joint_q", self.state_1, joint_q)
        self.cartpoles.set_attribute("joint_qd", self.state_1, joint_qd)

        # --- GPU: re-extract obs (reset worlds now have fresh initial state) ---
        self._extract_obs_to_gpu()

        # --- Single GPU -> CPU transfer: obs, rewards, dones ---
        wp.synchronize()
        obs_np = self.obs_wp.numpy()
        rewards_np = self.rewards_wp.numpy()
        dones_np = self.dones_wp.numpy().astype(bool)

        infos = {
            "L": self.parametric_model.L,
            "mean_reward": float(np.mean(rewards_np)),
            "num_dones": int(np.sum(dones_np)),
        }

        return obs_np, rewards_np, dones_np, infos


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
