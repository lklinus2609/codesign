"""
Vectorized Newton-based Ant Environment

Runs N ants in parallel on GPU using Newton's world replication.
All per-step computation (torques, obs, rewards, resets) stays GPU-resident
via Warp kernels -- matching Newton's official ArticulationView pattern.

Usage:
    env = AntVecEnv(num_worlds=64)
    obs = env.reset()  # Shape: (64, 27)
    actions = policy(obs)  # Shape: (64, 8)
    obs, rewards, dones, infos = env.step(actions)
"""

import numpy as np
import tempfile
import os

try:
    import warp as wp
    import newton
    from newton.selection import ArticulationView
    NEWTON_AVAILABLE = True
    wp.init()
except ImportError:
    NEWTON_AVAILABLE = False
    print("Warning: Newton/Warp not available.")

from .ant_env import ParametricAnt


# ---------------------------------------------------------------------------
# Warp kernels -- all per-step work stays on GPU
# ---------------------------------------------------------------------------

@wp.kernel
def set_joint_torques_kernel(
    actions: wp.array2d(dtype=float),
    gear_ratio: float,
    joint_f: wp.array2d(dtype=float),
    num_act_dofs: int,
    num_free_dofs: int,
):
    """Map 8 actions [-1,1] to joint torques scaled by gear ratio.

    DOF layout per world: 6 free-root DOFs + 8 actuated DOFs = 14 total.
    Free root DOFs get 0 force; actuated DOFs get action * gear_ratio.
    """
    tid = wp.tid()
    # Zero free-root DOFs (indices 0..5)
    for i in range(num_free_dofs):
        joint_f[tid, i] = 0.0
    # Set actuated DOFs (indices 6..13)
    for i in range(num_act_dofs):
        joint_f[tid, num_free_dofs + i] = actions[tid, i] * gear_ratio


@wp.kernel
def extract_obs_kernel(
    joint_q: wp.array2d(dtype=float),
    joint_qd: wp.array2d(dtype=float),
    obs: wp.array2d(dtype=float),
):
    """Extract 27D observation per world.

    Uses joint_q/joint_qd (always flat float) instead of body_q/body_qd.
    For a free-base robot, joint_q[0:7] = [x,y,z,qw,qx,qy,qz] matches
    body_q[torso], and joint_qd[0:6] matches body_qd[torso].

    Obs layout:
      [0]     torso z position (1)        -- joint_q[2]
      [1:5]   torso orientation quat (4)   -- joint_q[3:7]
      [5:13]  joint angles (8)             -- joint_q[7:15]
      [13:19] torso velocity (6)           -- joint_qd[0:6]
      [19:27] joint velocities (8)         -- joint_qd[6:14]

    joint_q: 7 free-root + 8 hinge = 15 per world
    joint_qd: 6 free-root + 8 hinge = 14 per world
    """
    tid = wp.tid()

    # Torso z position (joint_q[2] = z)
    obs[tid, 0] = joint_q[tid, 2]

    # Torso orientation quaternion (joint_q[3:7] = qw, qx, qy, qz)
    obs[tid, 1] = joint_q[tid, 3]
    obs[tid, 2] = joint_q[tid, 4]
    obs[tid, 3] = joint_q[tid, 5]
    obs[tid, 4] = joint_q[tid, 6]

    # Joint angles (8 actuated joints, skip 7 free-root q-coords)
    for i in range(8):
        obs[tid, 5 + i] = joint_q[tid, 7 + i]

    # Torso velocity (6D: angular+linear from joint_qd[0:6])
    for i in range(6):
        obs[tid, 13 + i] = joint_qd[tid, i]

    # Joint velocities (8 actuated, skip 6 free-root qd-coords)
    for i in range(8):
        obs[tid, 19 + i] = joint_qd[tid, 6 + i]


@wp.kernel
def step_rewards_done_kernel(
    joint_q: wp.array2d(dtype=float),
    prev_x: wp.array(dtype=float),
    actions: wp.array2d(dtype=float),
    steps: wp.array(dtype=int),
    dt: float,
    forward_weight: float,
    ctrl_weight: float,
    healthy_bonus: float,
    z_min: float,
    z_max: float,
    max_steps: int,
    num_act_dofs: int,
    rewards: wp.array(dtype=float),
    dones: wp.array(dtype=int),
):
    """Compute reward, increment steps, check termination/truncation."""
    tid = wp.tid()

    steps[tid] = steps[tid] + 1

    # Torso x and z (joint_q: [x, y, z, qw, qx, qy, qz, ...] per world)
    torso_x = joint_q[tid, 0]
    torso_z = joint_q[tid, 2]

    # Forward velocity reward
    x_before = prev_x[tid]
    forward_reward = forward_weight * (torso_x - x_before) / dt

    # Update prev_x for next step
    prev_x[tid] = torso_x

    # Control cost
    ctrl_cost = 0.0
    for i in range(num_act_dofs):
        ctrl_cost = ctrl_cost + actions[tid, i] * actions[tid, i]
    ctrl_cost = ctrl_weight * ctrl_cost

    # Healthy check
    healthy = 1.0
    if torso_z < z_min or torso_z > z_max:
        healthy = 0.0

    reward = forward_reward + healthy_bonus * healthy - ctrl_cost

    # Termination: unhealthy or max steps
    terminated = 0
    if healthy < 0.5:
        terminated = 1
    truncated = 0
    if steps[tid] >= max_steps:
        truncated = 1

    done = 0
    if terminated == 1 or truncated == 1:
        done = 1

    rewards[tid] = reward
    dones[tid] = done


@wp.kernel
def reset_done_worlds_kernel(
    joint_q: wp.array2d(dtype=float),
    joint_qd: wp.array2d(dtype=float),
    prev_x: wp.array(dtype=float),
    dones: wp.array(dtype=int),
    steps: wp.array(dtype=int),
    seed: int,
    init_z: float,
    num_joint_q: int,
    num_joint_qd: int,
):
    """Reset worlds where dones==1. Small noise on actuated joints."""
    tid = wp.tid()

    if dones[tid] == 1:
        rng = wp.rand_init(seed, tid)

        # Free root joint q: [x, y, z, qw, qx, qy, qz]
        joint_q[tid, 0] = 0.0  # x
        joint_q[tid, 1] = 0.0  # y
        joint_q[tid, 2] = init_z  # z
        joint_q[tid, 3] = 1.0  # qw (identity)
        joint_q[tid, 4] = 0.0  # qx
        joint_q[tid, 5] = 0.0  # qy
        joint_q[tid, 6] = 0.0  # qz

        # Actuated joint angles with small noise
        for i in range(7, num_joint_q):
            joint_q[tid, i] = -0.1 + 0.2 * wp.randf(rng)

        # Zero all velocities with small noise
        for i in range(num_joint_qd):
            joint_qd[tid, i] = -0.05 + 0.1 * wp.randf(rng)

        # Reset prev_x
        prev_x[tid] = 0.0

        steps[tid] = 0


@wp.kernel
def init_all_worlds_kernel(
    joint_q: wp.array2d(dtype=float),
    joint_qd: wp.array2d(dtype=float),
    prev_x: wp.array(dtype=float),
    seed: int,
    init_z: float,
    num_joint_q: int,
    num_joint_qd: int,
):
    """Initialize all worlds for full reset."""
    tid = wp.tid()
    rng = wp.rand_init(seed, tid)

    # Free root joint q: [x, y, z, qw, qx, qy, qz]
    joint_q[tid, 0] = 0.0
    joint_q[tid, 1] = 0.0
    joint_q[tid, 2] = init_z
    joint_q[tid, 3] = 1.0
    joint_q[tid, 4] = 0.0
    joint_q[tid, 5] = 0.0
    joint_q[tid, 6] = 0.0

    # Actuated joints with small noise
    for i in range(7, num_joint_q):
        joint_q[tid, i] = -0.1 + 0.2 * wp.randf(rng)

    # Velocities with small noise
    for i in range(num_joint_qd):
        joint_qd[tid, i] = -0.05 + 0.1 * wp.randf(rng)

    # prev_x
    prev_x[tid] = 0.0


# ---------------------------------------------------------------------------
# Vectorized Ant Environment
# ---------------------------------------------------------------------------

class AntVecEnv:
    """
    Vectorized Ant environment using Newton physics engine.

    Runs num_worlds ants in parallel on GPU.

    Observations: (num_worlds, 27) per world
    Actions: (num_worlds, 8) joint torques in [-1, 1]
    Rewards: (num_worlds,) shaped reward per world
    """

    def __init__(
        self,
        num_worlds: int = 64,
        parametric_model: ParametricAnt = None,
        dt: float = 0.05,
        num_substeps: int = 5,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.5,
        healthy_reward: float = 1.0,
        healthy_z_range: tuple = (0.2, 1.0),
        gear_ratio: float = 15.0,
        device: str = "cuda:0",
    ):
        if not NEWTON_AVAILABLE:
            raise ImportError("Newton/Warp required.")

        self.num_worlds = num_worlds
        self.device = device
        self.dt = dt
        self.num_substeps = num_substeps
        self.sub_dt = dt / num_substeps
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_reward = healthy_reward
        self.healthy_z_range = healthy_z_range
        self.gear_ratio = gear_ratio

        if parametric_model is None:
            self.parametric_model = ParametricAnt()
        else:
            self.parametric_model = parametric_model

        # Build Newton model with replicated worlds
        self._build_model()

        # Dimensions
        self.obs_dim = 27
        self.act_dim = 8
        self.max_steps = 1000  # 20s at dt=0.05

        # Initial torso z from MJCF (body starts at z=0.75)
        self.init_z = 0.75

        # Allocate GPU arrays
        self._alloc_buffers()

        # Step counter for random seeds
        self._step_count = 0

    def _build_model(self):
        """Build Newton model from MJCF with replicated worlds."""
        # Generate MJCF and save to temp file
        mjcf_str = self.parametric_model.generate_mjcf()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(mjcf_str)
            mjcf_path = f.name

        try:
            # Build single-world template
            single_builder = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(single_builder)

            single_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
                limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
            )
            single_builder.default_shape_cfg.ke = 5.0e4
            single_builder.default_shape_cfg.kd = 5.0e2
            single_builder.default_shape_cfg.kf = 1.0e3
            single_builder.default_shape_cfg.mu = 0.75

            single_builder.add_mjcf(
                mjcf_path,
                ignore_names=["floor", "ground"],
                xform=wp.transform(wp.vec3(0, 0, 0.75)),
            )

            # Set joint target stiffness/damping
            for i in range(len(single_builder.joint_target_ke)):
                single_builder.joint_target_ke[i] = 150
                single_builder.joint_target_kd[i] = 5

            # Add ground plane to single builder
            single_builder.add_ground_plane()

            # Replicate across worlds
            builder = newton.ModelBuilder()
            builder.replicate(single_builder, self.num_worlds, spacing=(5.0, 5.0, 0.0))

            # Finalize
            self.model = builder.finalize(requires_grad=True)

            # Store DOF counts per world
            total_joint_q = self.model.joint_q.numpy().shape[0]
            total_joint_qd = self.model.joint_qd.numpy().shape[0]
            self.num_joint_q_per_world = total_joint_q // self.num_worlds   # 15 (7 free + 8 hinge)
            self.num_joint_qd_per_world = total_joint_qd // self.num_worlds  # 14 (6 free + 8 hinge)
            self.num_bodies_per_world = self.model.body_count // self.num_worlds

            # Create solver
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                njmax=100 * self.num_worlds,
                nconmax=50 * self.num_worlds,
            )

            # Allocate states
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.control = self.model.control()

            # Forward kinematics
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

            # Collision pipeline
            self.contacts = self.model.collide(self.state_0)

            # Create ArticulationView -- find the key auto-generated from MJCF
            # Newton creates articulations from MJCF automatically.
            # Try matching with wildcard or the model name from MJCF.
            try:
                self.artic_view = ArticulationView(self.model, "parametric_ant", verbose=False)
            except Exception:
                try:
                    self.artic_view = ArticulationView(self.model, "*", verbose=False)
                except Exception:
                    # Fallback: don't use ArticulationView, work with raw arrays
                    self.artic_view = None

        finally:
            os.unlink(mjcf_path)

    def _alloc_buffers(self):
        """Allocate GPU buffers for all per-step computation.

        Pre-allocated once and reused every step() to avoid per-step GPU
        allocations that accumulate and cause OOM/segfault.
        """
        self.obs_wp = wp.zeros((self.num_worlds, self.obs_dim), dtype=float, device=self.device)
        self.rewards_wp = wp.zeros(self.num_worlds, dtype=float, device=self.device)
        self.dones_wp = wp.zeros(self.num_worlds, dtype=int, device=self.device)
        self.steps_wp = wp.zeros(self.num_worlds, dtype=int, device=self.device)
        self.prev_x_wp = wp.zeros(self.num_worlds, dtype=float, device=self.device)

        # Pre-allocated transfer buffers for actions
        self._actions_cpu = wp.zeros((self.num_worlds, self.act_dim), dtype=float, device="cpu")
        self._actions_gpu = wp.zeros((self.num_worlds, self.act_dim), dtype=float, device=self.device)

        # Cache ArticulationView attribute views for both states
        if self.artic_view is not None:
            s0, s1 = self.state_0, self.state_1
            self._joint_q = {
                id(s0): self.artic_view.get_attribute("joint_q", s0),
                id(s1): self.artic_view.get_attribute("joint_q", s1),
            }
            self._joint_qd = {
                id(s0): self.artic_view.get_attribute("joint_qd", s0),
                id(s1): self.artic_view.get_attribute("joint_qd", s1),
            }
            self._joint_f = self.artic_view.get_attribute("joint_f", self.control)
        else:
            # Fallback: reshape raw model arrays manually
            self._cache_raw_views()

        self._gc_counter = 0

    def _cache_raw_views(self):
        """Fallback: create reshaped views of raw model arrays when ArticulationView unavailable."""
        # We'll handle this in _get_joint_q etc. methods
        self._joint_q = None
        self._joint_qd = None
        self._joint_f = None

    def _get_joint_q(self, state):
        """Get (num_worlds, num_joint_q_per_world) view of joint_q."""
        if self._joint_q is not None:
            return self._joint_q[id(state)]
        # Fallback: reshape from flat model array
        return state.joint_q.reshape((self.num_worlds, self.num_joint_q_per_world))

    def _get_joint_qd(self, state):
        """Get (num_worlds, num_joint_qd_per_world) view of joint_qd."""
        if self._joint_qd is not None:
            return self._joint_qd[id(state)]
        return state.joint_qd.reshape((self.num_worlds, self.num_joint_qd_per_world))

    def _get_joint_f(self):
        """Get (num_worlds, num_joint_qd_per_world) view of joint_f."""
        if self._joint_f is not None:
            return self._joint_f
        return self.control.joint_f.reshape((self.num_worlds, self.num_joint_qd_per_world))

    def cleanup(self):
        """Free GPU resources before rebuilding."""
        for attr in ('_joint_q', '_joint_qd', '_joint_f',
                     'state_0', 'state_1', 'control', 'solver', 'artic_view',
                     'model', 'contacts', 'obs_wp', 'rewards_wp', 'dones_wp',
                     'steps_wp', 'prev_x_wp',
                     '_actions_cpu', '_actions_gpu'):
            if hasattr(self, attr):
                delattr(self, attr)
        import gc
        gc.collect()
        wp.synchronize()

    def _extract_obs_to_gpu(self):
        """Extract observations into self.obs_wp on GPU (no CPU transfer)."""
        joint_q = self._get_joint_q(self.state_0)
        joint_qd = self._get_joint_qd(self.state_0)

        wp.launch(extract_obs_kernel, dim=self.num_worlds,
                  inputs=[joint_q, joint_qd, self.obs_wp])

    def _get_obs(self) -> np.ndarray:
        """Get observations as numpy."""
        self._extract_obs_to_gpu()
        return self.obs_wp.numpy()

    def reset(self) -> np.ndarray:
        """Reset all environments using Warp kernels (GPU-resident)."""
        wp.synchronize()

        self.steps_wp.zero_()
        self._step_count = 0

        joint_q = self._get_joint_q(self.state_0)
        joint_qd = self._get_joint_qd(self.state_0)

        # Initialize all worlds
        wp.launch(init_all_worlds_kernel, dim=self.num_worlds,
                  inputs=[joint_q, joint_qd, self.prev_x_wp, 42,
                          self.init_z, self.num_joint_q_per_world,
                          self.num_joint_qd_per_world])

        # Copy to state_1
        joint_q_1 = self._get_joint_q(self.state_1)
        joint_qd_1 = self._get_joint_qd(self.state_1)
        wp.copy(joint_q_1, joint_q)
        wp.copy(joint_qd_1, joint_qd)

        # Propagate state through solver
        self._propagate_state()
        wp.synchronize()

        return self._get_obs()

    def _propagate_state(self):
        """Run one simulation step to propagate state through solver."""
        wp.synchronize()

        # Zero forces
        joint_f = self._get_joint_f()
        joint_f.zero_()

        # Run one substep
        self.state_0.clear_forces()
        self.contacts = self.model.collide(self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sub_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

        wp.synchronize()

    def step(self, actions: np.ndarray):
        """
        Step all environments in parallel.

        Args:
            actions: (num_worlds, 8) torques in [-1, 1]

        Returns:
            obs: (num_worlds, 27)
            rewards: (num_worlds,)
            dones: (num_worlds,) bool
            infos: dict
        """
        # --- CPU -> GPU transfer: actions ---
        actions = np.asarray(actions, dtype=np.float64).reshape(self.num_worlds, self.act_dim)
        self._actions_cpu.numpy()[:] = actions
        wp.copy(self._actions_gpu, self._actions_cpu)

        # --- GPU: set joint torques ---
        joint_f = self._get_joint_f()
        wp.launch(set_joint_torques_kernel, dim=self.num_worlds,
                  inputs=[self._actions_gpu, self.gear_ratio, joint_f,
                          self.act_dim, 6])  # 6 free-root DOFs

        # --- GPU: physics substeps ---
        for _ in range(self.num_substeps):
            self.state_0.clear_forces()
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        # --- GPU: compute rewards, increment steps, check done ---
        joint_q = self._get_joint_q(self.state_0)
        wp.launch(step_rewards_done_kernel, dim=self.num_worlds,
                  inputs=[joint_q, self.prev_x_wp, self._actions_gpu,
                          self.steps_wp, self.dt,
                          self.forward_reward_weight, self.ctrl_cost_weight,
                          self.healthy_reward,
                          self.healthy_z_range[0], self.healthy_z_range[1],
                          self.max_steps, self.act_dim,
                          self.rewards_wp, self.dones_wp])

        self._step_count += 1

        # --- GPU: auto-reset done worlds ---
        joint_q = self._get_joint_q(self.state_0)
        joint_qd = self._get_joint_qd(self.state_0)
        wp.launch(reset_done_worlds_kernel, dim=self.num_worlds,
                  inputs=[joint_q, joint_qd, self.prev_x_wp, self.dones_wp,
                          self.steps_wp, self._step_count,
                          self.init_z, self.num_joint_q_per_world,
                          self.num_joint_qd_per_world])
        # Copy reset state to state_1
        wp.copy(self._get_joint_q(self.state_1), joint_q)
        wp.copy(self._get_joint_qd(self.state_1), joint_qd)

        # --- GPU: extract observations ---
        self._extract_obs_to_gpu()

        # --- GPU -> CPU transfer ---
        wp.synchronize()
        obs_np = self.obs_wp.numpy()
        rewards_np = self.rewards_wp.numpy()
        dones_np = self.dones_wp.numpy().astype(bool)

        infos = {
            "mean_reward": float(np.mean(rewards_np)),
            "num_dones": int(np.sum(dones_np)),
            "leg_length": self.parametric_model.leg_length,
            "foot_length": self.parametric_model.foot_length,
            "torso_radius": self.parametric_model.torso_radius,
        }

        # Periodic GC
        self._gc_counter += 1
        if self._gc_counter % 100 == 0:
            import gc
            gc.collect()

        return obs_np, rewards_np, dones_np, infos


def test_vec_env():
    """Test vectorized Ant environment."""
    print("Testing Vectorized Newton Ant")
    print("=" * 50)

    num_worlds = 64
    env = AntVecEnv(num_worlds=num_worlds)

    print(f"Num worlds: {num_worlds}")
    print(f"Obs dim: {env.obs_dim}, Act dim: {env.act_dim}")
    print(f"Joint Q per world: {env.num_joint_q_per_world}")
    print(f"Joint Qd per world: {env.num_joint_qd_per_world}")
    print(f"Bodies per world: {env.num_bodies_per_world}")

    obs = env.reset()
    print(f"\nInitial obs shape: {obs.shape}")
    print(f"Initial obs mean: {obs.mean(axis=0)[:5]}...")

    total_rewards = np.zeros(num_worlds)
    n_steps = 200

    import time
    start = time.time()

    for step in range(n_steps):
        actions = np.random.uniform(-1, 1, size=(num_worlds, 8))
        obs, rewards, dones, infos = env.step(actions)
        total_rewards += rewards

        if step % 50 == 0:
            print(f"Step {step}: mean_reward={infos['mean_reward']:.3f}, dones={infos['num_dones']}")

    elapsed = time.time() - start
    fps = (n_steps * num_worlds) / elapsed

    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Throughput: {fps:.0f} steps/sec ({num_worlds} worlds x {n_steps} steps)")
    print(f"Mean total reward: {total_rewards.mean():.1f}")
    print("\nVectorized Ant environment test complete!")

    env.cleanup()


if __name__ == "__main__":
    test_vec_env()
