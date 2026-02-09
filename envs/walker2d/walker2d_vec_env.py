"""
Vectorized Newton-based Walker2D Environment

Runs N Walker2D bipeds in parallel on GPU using Newton's world replication.
All per-step computation stays GPU-resident via Warp kernels.

Walker2D DOF layout per world:
- Root: rootx(slide) + rootz(slide) + rooty(hinge) = 3 joint_q, 3 joint_qd
- Actuated: 6 hinge joints = 6 joint_q, 6 joint_qd
- Total: 9 joint_q, 9 joint_qd per world

Observation (17D): joint_q[1:9] + joint_qd[0:9]
Actions (6D): joint torques for 6 actuated joints

Usage:
    env = Walker2DVecEnv(num_worlds=64)
    obs = env.reset()  # Shape: (64, 17)
    actions = policy(obs)  # Shape: (64, 6)
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

from .walker2d_env import ParametricWalker2D


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------

@wp.kernel
def set_joint_torques_kernel(
    actions: wp.array2d(dtype=float),
    gear_ratio: float,
    joint_f: wp.array2d(dtype=float),
    num_act_dofs: int,
    num_root_dofs: int,
):
    """Map 6 actions [-1,1] to joint torques. Zero root DOFs."""
    tid = wp.tid()
    for i in range(num_root_dofs):
        joint_f[tid, i] = 0.0
    for i in range(num_act_dofs):
        joint_f[tid, num_root_dofs + i] = actions[tid, i] * gear_ratio


@wp.kernel
def extract_obs_kernel(
    joint_q: wp.array2d(dtype=float),
    joint_qd: wp.array2d(dtype=float),
    obs: wp.array2d(dtype=float),
):
    """Extract 17D observation per world.

    Obs layout:
      [0:8]  joint_q[1:9] = rootz, rooty, 6 joint angles (skip rootx)
      [8:17] joint_qd[0:9] = all 9 velocities
    """
    tid = wp.tid()
    # qpos: skip rootx (index 0), take indices 1..8
    for i in range(8):
        obs[tid, i] = joint_q[tid, 1 + i]
    # qvel: all 9
    for i in range(9):
        obs[tid, 8 + i] = joint_qd[tid, i]


@wp.kernel
def step_rewards_done_kernel(
    joint_q: wp.array2d(dtype=float),
    body_q: wp.array2d(dtype=float),
    prev_rootx: wp.array(dtype=float),
    actions: wp.array2d(dtype=float),
    steps: wp.array(dtype=int),
    dt: float,
    forward_weight: float,
    ctrl_weight: float,
    healthy_bonus: float,
    z_min: float,
    z_max: float,
    angle_max: float,
    max_steps: int,
    num_act_dofs: int,
    rewards: wp.array(dtype=float),
    dones: wp.array(dtype=int),
):
    """Compute reward, increment steps, check termination."""
    tid = wp.tid()
    steps[tid] = steps[tid] + 1

    # Forward velocity from rootx displacement
    rootx = joint_q[tid, 0]
    forward_reward = forward_weight * (rootx - prev_rootx[tid]) / dt
    prev_rootx[tid] = rootx

    # Healthy: absolute z from body_q, angle from joint_q
    torso_z = body_q[tid, 2]  # body_q[torso, z] (body 0, offset 2)
    rooty = joint_q[tid, 2]

    healthy = 1.0
    if torso_z < z_min or torso_z > z_max:
        healthy = 0.0
    if wp.abs(rooty) > angle_max:
        healthy = 0.0

    # Control cost
    ctrl_cost = 0.0
    for i in range(num_act_dofs):
        ctrl_cost = ctrl_cost + actions[tid, i] * actions[tid, i]
    ctrl_cost = ctrl_weight * ctrl_cost

    reward = forward_reward + healthy_bonus * healthy - ctrl_cost

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
    prev_rootx: wp.array(dtype=float),
    dones: wp.array(dtype=int),
    steps: wp.array(dtype=int),
    seed: int,
    num_joint_q: int,
    num_joint_qd: int,
):
    """Reset worlds where dones==1."""
    tid = wp.tid()
    if dones[tid] == 1:
        rng = wp.rand_init(seed, tid)
        noise = 5.0e-3

        # Root: rootx=0, rootz=0 (displacement), rooty=noise
        joint_q[tid, 0] = 0.0
        joint_q[tid, 1] = 0.0
        joint_q[tid, 2] = -noise + 2.0 * noise * wp.randf(rng)

        # Actuated joints
        for i in range(3, num_joint_q):
            joint_q[tid, i] = -noise + 2.0 * noise * wp.randf(rng)

        # Velocities
        for i in range(num_joint_qd):
            joint_qd[tid, i] = -noise + 2.0 * noise * wp.randf(rng)

        prev_rootx[tid] = 0.0
        steps[tid] = 0


@wp.kernel
def init_all_worlds_kernel(
    joint_q: wp.array2d(dtype=float),
    joint_qd: wp.array2d(dtype=float),
    prev_rootx: wp.array(dtype=float),
    seed: int,
    num_joint_q: int,
    num_joint_qd: int,
):
    """Initialize all worlds for full reset."""
    tid = wp.tid()
    rng = wp.rand_init(seed, tid)
    noise = 5.0e-3

    joint_q[tid, 0] = 0.0
    joint_q[tid, 1] = 0.0
    joint_q[tid, 2] = -noise + 2.0 * noise * wp.randf(rng)

    for i in range(3, num_joint_q):
        joint_q[tid, i] = -noise + 2.0 * noise * wp.randf(rng)

    for i in range(num_joint_qd):
        joint_qd[tid, i] = -noise + 2.0 * noise * wp.randf(rng)

    prev_rootx[tid] = 0.0


# ---------------------------------------------------------------------------
# Vectorized Walker2D Environment
# ---------------------------------------------------------------------------

class Walker2DVecEnv:
    """
    Vectorized Walker2D environment using Newton physics engine.

    Runs num_worlds bipeds in parallel on GPU.
    """

    def __init__(
        self,
        num_worlds: int = 64,
        parametric_model: ParametricWalker2D = None,
        dt: float = 0.05,
        num_substeps: int = 5,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        healthy_z_range: tuple = (0.8, 2.0),
        healthy_angle_range: float = 1.0,
        gear_ratio: float = 100.0,
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
        self.healthy_angle_range = healthy_angle_range
        self.gear_ratio = gear_ratio

        if parametric_model is None:
            self.parametric_model = ParametricWalker2D()
        else:
            self.parametric_model = parametric_model

        self._build_model()

        self.obs_dim = 17
        self.act_dim = 6
        self.max_steps = 1000

        self._alloc_buffers()
        self._step_count = 0

    def _build_model(self):
        """Build Newton model from MJCF with replicated worlds."""
        mjcf_str = self.parametric_model.generate_mjcf()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(mjcf_str)
            mjcf_path = f.name

        try:
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
            )

            for i in range(len(single_builder.joint_target_ke)):
                single_builder.joint_target_ke[i] = 150
                single_builder.joint_target_kd[i] = 5

            single_builder.add_ground_plane()

            # Replicate
            builder = newton.ModelBuilder()
            builder.replicate(single_builder, self.num_worlds, spacing=(5.0, 5.0, 0.0))

            self.model = builder.finalize(requires_grad=True)

            # DOF counts per world
            total_joint_q = self.model.joint_q.numpy().shape[0]
            total_joint_qd = self.model.joint_qd.numpy().shape[0]
            self.num_joint_q_per_world = total_joint_q // self.num_worlds   # 9
            self.num_joint_qd_per_world = total_joint_qd // self.num_worlds  # 9
            self.num_bodies_per_world = self.model.body_count // self.num_worlds  # 7

            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                njmax=100 * self.num_worlds,
                nconmax=50 * self.num_worlds,
            )

            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.control = self.model.control()

            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
            self.contacts = self.model.collide(self.state_0)

            # ArticulationView
            try:
                self.artic_view = ArticulationView(self.model, "parametric_walker2d", verbose=False)
            except Exception:
                try:
                    self.artic_view = ArticulationView(self.model, "*", verbose=False)
                except Exception:
                    self.artic_view = None

        finally:
            os.unlink(mjcf_path)

    def _alloc_buffers(self):
        """Pre-allocate all GPU buffers."""
        self.obs_wp = wp.zeros((self.num_worlds, self.obs_dim), dtype=float, device=self.device)
        self.rewards_wp = wp.zeros(self.num_worlds, dtype=float, device=self.device)
        self.dones_wp = wp.zeros(self.num_worlds, dtype=int, device=self.device)
        self.steps_wp = wp.zeros(self.num_worlds, dtype=int, device=self.device)
        self.prev_rootx_wp = wp.zeros(self.num_worlds, dtype=float, device=self.device)

        self._actions_cpu = wp.zeros((self.num_worlds, self.act_dim), dtype=float, device="cpu")
        self._actions_gpu = wp.zeros((self.num_worlds, self.act_dim), dtype=float, device=self.device)

        # Cache ArticulationView attributes
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
            self._body_q = {
                id(s0): self.artic_view.get_attribute("body_q", s0),
                id(s1): self.artic_view.get_attribute("body_q", s1),
            }
        else:
            self._joint_q = None
            self._joint_qd = None
            self._joint_f = None
            self._body_q = None

        self._gc_counter = 0

    def _get_joint_q(self, state):
        if self._joint_q is not None:
            return self._joint_q[id(state)]
        return state.joint_q.reshape((self.num_worlds, self.num_joint_q_per_world))

    def _get_joint_qd(self, state):
        if self._joint_qd is not None:
            return self._joint_qd[id(state)]
        return state.joint_qd.reshape((self.num_worlds, self.num_joint_qd_per_world))

    def _get_joint_f(self):
        if self._joint_f is not None:
            return self._joint_f
        return self.control.joint_f.reshape((self.num_worlds, self.num_joint_qd_per_world))

    def _get_body_q(self, state):
        if self._body_q is not None:
            return self._body_q[id(state)]
        return state.body_q.reshape((self.num_worlds, self.num_bodies_per_world * 7))

    def cleanup(self):
        """Free GPU resources."""
        for attr in ('_joint_q', '_joint_qd', '_joint_f', '_body_q',
                     'state_0', 'state_1', 'control', 'solver', 'artic_view',
                     'model', 'contacts', 'obs_wp', 'rewards_wp', 'dones_wp',
                     'steps_wp', 'prev_rootx_wp',
                     '_actions_cpu', '_actions_gpu'):
            if hasattr(self, attr):
                delattr(self, attr)
        import gc
        gc.collect()
        wp.synchronize()

    def _extract_obs_to_gpu(self):
        joint_q = self._get_joint_q(self.state_0)
        joint_qd = self._get_joint_qd(self.state_0)
        wp.launch(extract_obs_kernel, dim=self.num_worlds,
                  inputs=[joint_q, joint_qd, self.obs_wp])

    def _get_obs(self) -> np.ndarray:
        self._extract_obs_to_gpu()
        return self.obs_wp.numpy()

    def reset(self) -> np.ndarray:
        wp.synchronize()

        self.steps_wp.zero_()
        self._step_count = 0

        joint_q = self._get_joint_q(self.state_0)
        joint_qd = self._get_joint_qd(self.state_0)

        wp.launch(init_all_worlds_kernel, dim=self.num_worlds,
                  inputs=[joint_q, joint_qd, self.prev_rootx_wp, 42,
                          self.num_joint_q_per_world, self.num_joint_qd_per_world])

        # Copy to state_1
        joint_q_1 = self._get_joint_q(self.state_1)
        joint_qd_1 = self._get_joint_qd(self.state_1)
        wp.copy(joint_q_1, joint_q)
        wp.copy(joint_qd_1, joint_qd)

        self._propagate_state()
        wp.synchronize()

        return self._get_obs()

    def _propagate_state(self):
        wp.synchronize()
        joint_f = self._get_joint_f()
        joint_f.zero_()
        self.state_0.clear_forces()
        self.contacts = self.model.collide(self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sub_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        wp.synchronize()

    def step(self, actions: np.ndarray):
        """Step all environments in parallel."""
        actions = np.asarray(actions, dtype=np.float64).reshape(self.num_worlds, self.act_dim)
        self._actions_cpu.numpy()[:] = actions
        wp.copy(self._actions_gpu, self._actions_cpu)

        # Set torques
        joint_f = self._get_joint_f()
        wp.launch(set_joint_torques_kernel, dim=self.num_worlds,
                  inputs=[self._actions_gpu, self.gear_ratio, joint_f,
                          self.act_dim, 3])  # 3 root DOFs

        # Physics substeps
        for _ in range(self.num_substeps):
            self.state_0.clear_forces()
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        # Rewards and dones
        joint_q = self._get_joint_q(self.state_0)
        body_q = self._get_body_q(self.state_0)
        wp.launch(step_rewards_done_kernel, dim=self.num_worlds,
                  inputs=[joint_q, body_q, self.prev_rootx_wp, self._actions_gpu,
                          self.steps_wp, self.dt,
                          self.forward_reward_weight, self.ctrl_cost_weight,
                          self.healthy_reward,
                          self.healthy_z_range[0], self.healthy_z_range[1],
                          self.healthy_angle_range,
                          self.max_steps, self.act_dim,
                          self.rewards_wp, self.dones_wp])

        self._step_count += 1

        # Auto-reset done worlds
        joint_qd = self._get_joint_qd(self.state_0)
        wp.launch(reset_done_worlds_kernel, dim=self.num_worlds,
                  inputs=[joint_q, joint_qd, self.prev_rootx_wp, self.dones_wp,
                          self.steps_wp, self._step_count,
                          self.num_joint_q_per_world, self.num_joint_qd_per_world])
        wp.copy(self._get_joint_q(self.state_1), joint_q)
        wp.copy(self._get_joint_qd(self.state_1), joint_qd)

        # Extract obs
        self._extract_obs_to_gpu()

        # Transfer to CPU
        wp.synchronize()
        obs_np = self.obs_wp.numpy()
        rewards_np = self.rewards_wp.numpy()
        dones_np = self.dones_wp.numpy().astype(bool)

        infos = {
            "mean_reward": float(np.mean(rewards_np)),
            "num_dones": int(np.sum(dones_np)),
            "thigh_length": self.parametric_model.thigh_length,
            "leg_length": self.parametric_model.leg_length,
            "foot_length": self.parametric_model.foot_length,
        }

        self._gc_counter += 1
        if self._gc_counter % 100 == 0:
            import gc
            gc.collect()

        return obs_np, rewards_np, dones_np, infos


def test_vec_env():
    """Test vectorized Walker2D environment."""
    print("Testing Vectorized Newton Walker2D")
    print("=" * 50)

    num_worlds = 64
    env = Walker2DVecEnv(num_worlds=num_worlds)

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
        actions = np.random.uniform(-1, 1, size=(num_worlds, 6))
        obs, rewards, dones, infos = env.step(actions)
        total_rewards += rewards

        if step % 50 == 0:
            print(f"Step {step}: mean_reward={infos['mean_reward']:.3f}, dones={infos['num_dones']}")

    elapsed = time.time() - start
    fps = (n_steps * num_worlds) / elapsed

    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Throughput: {fps:.0f} steps/sec ({num_worlds} worlds x {n_steps} steps)")
    print(f"Mean total reward: {total_rewards.mean():.1f}")
    print("\nVectorized Walker2D environment test complete!")

    env.cleanup()


if __name__ == "__main__":
    test_vec_env()
