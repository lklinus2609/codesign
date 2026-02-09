"""
Newton-based Walker2D Environment

Level 2.1: 2D bipedal locomotion with parametric morphology.

Walker2D is a planar biped with:
- 3 root DOFs (rootx slide, rootz slide, rooty hinge)
- 6 actuated hinge joints (thigh, leg, foot x2 sides)
- 17D observations (8 qpos excl. rootx + 9 qvel)
- 6D actions (joint torques)

Morphology parameters:
- thigh_length: Length of upper leg segments
- leg_length: Length of lower leg (shin) segments
- foot_length: Length of foot segments
"""

import numpy as np
import tempfile
import os

try:
    import warp as wp
    import newton
    NEWTON_AVAILABLE = True
except ImportError:
    NEWTON_AVAILABLE = False
    print("Warning: Newton/Warp not available")


class ParametricWalker2D:
    """
    Parametric Walker2D morphology model.

    Design parameters (symmetric for both legs):
    - thigh_length: Upper leg segment length [0.20, 0.70] m
    - leg_length: Lower leg (shin) segment length [0.25, 0.75] m
    - foot_length: Foot segment length [0.10, 0.40] m
    """

    def __init__(
        self,
        thigh_length: float = 0.45,
        leg_length: float = 0.50,
        foot_length: float = 0.20,
        torso_length: float = 0.40,  # Fixed, not optimized
    ):
        self.thigh_length = thigh_length
        self.leg_length = leg_length
        self.foot_length = foot_length
        self.torso_length = torso_length

        # Bounds
        self.thigh_length_min, self.thigh_length_max = 0.20, 0.70
        self.leg_length_min, self.leg_length_max = 0.25, 0.75
        self.foot_length_min, self.foot_length_max = 0.10, 0.40

    @property
    def init_z(self):
        """Initial torso center height = clearance + leg + thigh + half torso."""
        return 0.1 + self.leg_length + self.thigh_length + self.torso_length / 2.0

    def get_params(self) -> dict:
        return {
            "thigh_length": self.thigh_length,
            "leg_length": self.leg_length,
            "foot_length": self.foot_length,
        }

    def set_params(self, thigh_length=None, leg_length=None, foot_length=None):
        if thigh_length is not None:
            self.thigh_length = np.clip(thigh_length, self.thigh_length_min, self.thigh_length_max)
        if leg_length is not None:
            self.leg_length = np.clip(leg_length, self.leg_length_min, self.leg_length_max)
        if foot_length is not None:
            self.foot_length = np.clip(foot_length, self.foot_length_min, self.foot_length_max)

    def generate_mjcf(self) -> str:
        """Generate MJCF XML string with current parameters.

        Uses body-local coordinates: each body's pos is relative to parent,
        geom fromto is in the body's local frame.
        """
        ht = self.torso_length / 2.0  # half torso
        tl = self.thigh_length
        ll = self.leg_length
        fl = self.foot_length
        iz = self.init_z

        mjcf = f'''<mujoco model="parametric_walker2d">
  <compiler angle="degree" inertiafromgeom="true"/>

  <default>
    <joint armature="0.01" damping="0.1" limited="true"/>
    <geom condim="3" density="1000" friction="0.7 0.1 0.1" rgba="0.8 0.6 0.4 1"/>
  </default>

  <option timestep="0.016" iterations="50" solver="Newton"/>

  <worldbody>
    <body name="torso" pos="0 0 {iz:.4f}">
      <!-- 3 DOF planar root -->
      <joint name="rootx" type="slide" axis="1 0 0" armature="0" damping="0" limited="false" stiffness="0"/>
      <joint name="rootz" type="slide" axis="0 0 1" armature="0" damping="0" limited="false" stiffness="0"/>
      <joint name="rooty" type="hinge" axis="0 1 0" armature="0" damping="0" limited="false" stiffness="0"/>

      <geom name="torso_geom" type="capsule" fromto="0 0 {ht:.4f} 0 0 {-ht:.4f}" size="0.05" friction="0.9"/>

      <!-- Right leg -->
      <body name="thigh" pos="0 0 {-ht:.4f}">
        <joint name="thigh_joint" type="hinge" axis="0 -1 0" range="-150 0"/>
        <geom name="thigh_geom" type="capsule" fromto="0 0 0 0 0 {-tl:.4f}" size="0.05"/>
        <body name="leg" pos="0 0 {-tl:.4f}">
          <joint name="leg_joint" type="hinge" axis="0 -1 0" range="-150 0"/>
          <geom name="leg_geom" type="capsule" fromto="0 0 0 0 0 {-ll:.4f}" size="0.04"/>
          <body name="foot" pos="0 0 {-ll:.4f}">
            <joint name="foot_joint" type="hinge" axis="0 -1 0" range="-45 45"/>
            <geom name="foot_geom" type="capsule" fromto="0 0 0 {fl:.4f} 0 0" size="0.06"/>
          </body>
        </body>
      </body>

      <!-- Left leg -->
      <body name="thigh_left" pos="0 0 {-ht:.4f}">
        <joint name="thigh_left_joint" type="hinge" axis="0 -1 0" range="-150 0"/>
        <geom name="thigh_left_geom" type="capsule" fromto="0 0 0 0 0 {-tl:.4f}" size="0.05"/>
        <body name="leg_left" pos="0 0 {-tl:.4f}">
          <joint name="leg_left_joint" type="hinge" axis="0 -1 0" range="-150 0"/>
          <geom name="leg_left_geom" type="capsule" fromto="0 0 0 0 0 {-ll:.4f}" size="0.04"/>
          <body name="foot_left" pos="0 0 {-ll:.4f}">
            <joint name="foot_left_joint" type="hinge" axis="0 -1 0" range="-45 45"/>
            <geom name="foot_left_geom" type="capsule" fromto="0 0 0 {fl:.4f} 0 0" size="0.06"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="thigh_joint" gear="100"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="leg_joint" gear="100"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="foot_joint" gear="100"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="thigh_left_joint" gear="100"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="leg_left_joint" gear="100"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="foot_left_joint" gear="100"/>
  </actuator>
</mujoco>'''
        return mjcf


class Walker2DEnv:
    """
    Single-world Walker2D environment (reference implementation).

    Observation (17D):
        - rootz, rooty, 6 joint angles (8)
        - 9 joint velocities (9)

    Action (6D):
        - Joint torques for thigh, leg, foot x2 sides

    Reward:
        - Forward velocity (x direction)
        - Survival bonus
        - Control cost penalty
    """

    def __init__(
        self,
        parametric_model: ParametricWalker2D = None,
        dt: float = 0.05,
        num_substeps: int = 5,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        healthy_z_range: tuple = (0.8, 2.0),
        healthy_angle_range: float = 1.0,
        device: str = "cuda",
    ):
        if not NEWTON_AVAILABLE:
            raise ImportError("Newton/Warp required")

        self.dt = dt
        self.num_substeps = num_substeps
        self.sub_dt = dt / num_substeps
        self.device = device

        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_reward = healthy_reward
        self.healthy_z_range = healthy_z_range
        self.healthy_angle_range = healthy_angle_range

        if parametric_model is None:
            self.parametric_model = ParametricWalker2D()
        else:
            self.parametric_model = parametric_model

        self._build_model()

        self.obs_dim = 17  # 8 qpos + 9 qvel
        self.act_dim = 6   # 6 actuated joints

        self.steps = 0
        self.max_steps = 1000
        self.terminated = False
        self.prev_x = 0.0

    def _build_model(self):
        """Build Newton model from parametric MJCF."""
        wp.init()

        mjcf_str = self.parametric_model.generate_mjcf()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(mjcf_str)
            mjcf_path = f.name

        try:
            builder = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

            builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
                limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
            )
            builder.default_shape_cfg.ke = 5.0e4
            builder.default_shape_cfg.kd = 5.0e2
            builder.default_shape_cfg.kf = 1.0e3
            builder.default_shape_cfg.mu = 0.75

            builder.add_mjcf(
                mjcf_path,
                ignore_names=["floor", "ground"],
            )

            for i in range(len(builder.joint_target_ke)):
                builder.joint_target_ke[i] = 150
                builder.joint_target_kd[i] = 5

            builder.add_ground_plane()

            self.model = builder.finalize(requires_grad=True)

            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                njmax=100,
                nconmax=50,
            )

            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.control = self.model.control()

            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
            self.contacts = self.model.collide(self.state_0)

        finally:
            os.unlink(mjcf_path)

    def reset(self) -> np.ndarray:
        self.steps = 0
        self.terminated = False

        joint_q = self.model.joint_q.numpy()
        joint_qd = self.model.joint_qd.numpy()

        # Reset root: rootx=0, rootz=0 (displacement), rooty=0
        noise = 5e-3
        joint_q[0] = 0.0  # rootx
        joint_q[1] = 0.0  # rootz
        joint_q[2] = np.random.uniform(-noise, noise)  # rooty
        # Actuated joints with small noise
        joint_q[3:] = np.random.uniform(-noise, noise, size=len(joint_q) - 3)
        joint_qd[:] = np.random.uniform(-noise, noise, size=len(joint_qd))

        self.model.joint_q.assign(joint_q)
        self.model.joint_qd.assign(joint_qd)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        body_q = self.state_0.body_q.numpy()
        self.prev_x = body_q[0, 0]

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        joint_q = self.model.joint_q.numpy()
        joint_qd = self.model.joint_qd.numpy()

        # qpos[1:9]: rootz, rooty, 6 joint angles (skip rootx)
        qpos = joint_q[1:9]
        # qvel[0:9]: all velocities
        qvel = joint_qd[0:9]

        obs = np.concatenate([qpos, qvel]).astype(np.float32)
        return obs

    def step(self, action: np.ndarray):
        if self.terminated:
            return self._get_obs(), 0.0, True, False, {}

        action = np.clip(action, -1.0, 1.0)

        joint_act = self.model.joint_act.numpy()
        n_act = min(len(action), len(joint_act))
        joint_act[:n_act] = action[:n_act] * 100.0  # gear ratio
        self.model.joint_act.assign(joint_act)

        # Position before step
        body_q_before = self.state_0.body_q.numpy()
        x_before = body_q_before[0, 0]

        for _ in range(self.num_substeps):
            self.state_0.clear_forces()
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.steps += 1

        body_q_after = self.state_0.body_q.numpy()
        x_after = body_q_after[0, 0]
        z_after = body_q_after[0, 2]

        joint_q = self.model.joint_q.numpy()
        rooty = joint_q[2]

        forward_reward = self.forward_reward_weight * (x_after - x_before) / self.dt
        ctrl_cost = self.ctrl_cost_weight * np.sum(action ** 2)
        healthy = (self.healthy_z_range[0] < z_after < self.healthy_z_range[1]) and (abs(rooty) < self.healthy_angle_range)
        healthy_rew = self.healthy_reward if healthy else 0.0

        reward = forward_reward + healthy_rew - ctrl_cost

        if not healthy:
            self.terminated = True

        truncated = self.steps >= self.max_steps

        info = {
            "x": x_after,
            "z": z_after,
            "forward_reward": forward_reward,
            "ctrl_cost": ctrl_cost,
            "healthy": healthy,
        }

        return self._get_obs(), reward, self.terminated, truncated, info


def test_walker2d_env():
    """Test Walker2D environment."""
    print("Testing Walker2D Environment")
    print("=" * 50)

    env = Walker2DEnv()
    print(f"Thigh: {env.parametric_model.thigh_length:.3f} m")
    print(f"Leg: {env.parametric_model.leg_length:.3f} m")
    print(f"Foot: {env.parametric_model.foot_length:.3f} m")
    print(f"Init z: {env.parametric_model.init_z:.3f} m")
    print(f"Obs dim: {env.obs_dim}, Act dim: {env.act_dim}")

    obs = env.reset()
    print(f"Initial obs shape: {obs.shape}")

    total_reward = 0
    for step in range(100):
        action = np.random.uniform(-1, 1, size=env.act_dim)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 20 == 0:
            print(f"Step {step}: x={info['x']:.3f}, z={info['z']:.3f}, "
                  f"healthy={info['healthy']}, reward={reward:.2f}")

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    print(f"Total reward: {total_reward:.1f}")
    print("\nWalker2D environment test complete!")


if __name__ == "__main__":
    test_walker2d_env()
