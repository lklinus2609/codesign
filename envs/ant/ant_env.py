"""
Newton-based Differentiable Ant Environment

Level 2: Multi-body locomotion with parametric morphology.

Morphology parameters:
- leg_length: Length of upper leg segments
- foot_length: Length of lower leg (foot) segments
- torso_radius: Size of central body

The Ant has 4 legs, each with 2 joints (hip, ankle).
Total: 8 actuated joints + 6 DOF free root = 14 DOF
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


class ParametricAnt:
    """
    Parametric Ant morphology model.

    Design parameters (symmetric for all 4 legs):
    - leg_length: Upper leg segment length [0.15, 0.4] m
    - foot_length: Lower leg (foot) segment length [0.3, 0.7] m
    - torso_radius: Central body radius [0.15, 0.35] m
    """

    def __init__(
        self,
        leg_length: float = 0.28,    # Default from MJCF: 0.2*sqrt(2) ≈ 0.28
        foot_length: float = 0.57,   # Default from MJCF: 0.4*sqrt(2) ≈ 0.57
        torso_radius: float = 0.25,  # Default from MJCF
        leg_thickness: float = 0.08, # Capsule radius
    ):
        # Design parameters
        self.leg_length = leg_length
        self.foot_length = foot_length
        self.torso_radius = torso_radius
        self.leg_thickness = leg_thickness

        # Bounds
        self.leg_length_min, self.leg_length_max = 0.15, 0.4
        self.foot_length_min, self.foot_length_max = 0.3, 0.7
        self.torso_radius_min, self.torso_radius_max = 0.15, 0.35

    def get_params(self) -> dict:
        return {
            "leg_length": self.leg_length,
            "foot_length": self.foot_length,
            "torso_radius": self.torso_radius,
        }

    def set_params(self, leg_length=None, foot_length=None, torso_radius=None):
        if leg_length is not None:
            self.leg_length = np.clip(leg_length, self.leg_length_min, self.leg_length_max)
        if foot_length is not None:
            self.foot_length = np.clip(foot_length, self.foot_length_min, self.foot_length_max)
        if torso_radius is not None:
            self.torso_radius = np.clip(torso_radius, self.torso_radius_min, self.torso_radius_max)

    def generate_mjcf(self) -> str:
        """Generate MJCF XML string with current parameters."""
        L = self.leg_length / np.sqrt(2)  # Diagonal length in MJCF format
        F = self.foot_length / np.sqrt(2)
        R = self.torso_radius
        T = self.leg_thickness

        mjcf = f'''<mujoco model="parametric_ant">
  <compiler inertiafromgeom="true" angle="degree"/>

  <default>
    <joint armature="0.01" damping="0.1" limited="true"/>
    <geom condim="3" density="5.0" friction="1.5 0.1 0.1" margin="0.01" rgba="0.97 0.38 0.06 1"/>
  </default>

  <option timestep="0.016" iterations="50" solver="Newton"/>

  <worldbody>
    <body name="torso" pos="0 0 0.75">
      <geom name="torso_geom" pos="0 0 0" size="{R}" type="sphere"/>

      <!-- Aux geoms connecting torso to legs -->
      <geom fromto="0 0 0 {L} {L} 0" name="aux_1_geom" size="{T}" type="capsule"/>
      <geom fromto="0 0 0 -{L} {L} 0" name="aux_2_geom" size="{T}" type="capsule"/>
      <geom fromto="0 0 0 -{L} -{L} 0" name="aux_3_geom" size="{T}" type="capsule"/>
      <geom fromto="0 0 0 {L} -{L} 0" name="aux_4_geom" size="{T}" type="capsule"/>

      <joint armature="0" damping="0" limited="false" name="root" pos="0 0 0" type="free"/>

      <!-- Front Left Leg -->
      <body name="front_left_leg" pos="{L} {L} 0">
        <joint axis="0 0 1" name="hip_1" pos="0 0 0" range="-40 40" type="hinge"/>
        <geom fromto="0 0 0 {L} {L} 0" name="leg_1_geom" size="{T}" type="capsule"/>
        <body pos="{L} {L} 0" name="front_left_foot">
          <joint axis="-1 1 0" name="ankle_1" pos="0 0 0" range="30 100" type="hinge"/>
          <geom fromto="0 0 0 {F} {F} 0" name="foot_1_geom" size="{T}" type="capsule"/>
        </body>
      </body>

      <!-- Front Right Leg -->
      <body name="front_right_leg" pos="-{L} {L} 0">
        <joint axis="0 0 1" name="hip_2" pos="0 0 0" range="-40 40" type="hinge"/>
        <geom fromto="0 0 0 -{L} {L} 0" name="leg_2_geom" size="{T}" type="capsule"/>
        <body pos="-{L} {L} 0" name="front_right_foot">
          <joint axis="1 1 0" name="ankle_2" pos="0 0 0" range="-100 -30" type="hinge"/>
          <geom fromto="0 0 0 -{F} {F} 0" name="foot_2_geom" size="{T}" type="capsule"/>
        </body>
      </body>

      <!-- Back Left Leg -->
      <body name="back_left_leg" pos="-{L} -{L} 0">
        <joint axis="0 0 1" name="hip_3" pos="0 0 0" range="-40 40" type="hinge"/>
        <geom fromto="0 0 0 -{L} -{L} 0" name="leg_3_geom" size="{T}" type="capsule"/>
        <body pos="-{L} -{L} 0" name="back_left_foot">
          <joint axis="-1 1 0" name="ankle_3" pos="0 0 0" range="-100 -30" type="hinge"/>
          <geom fromto="0 0 0 -{F} -{F} 0" name="foot_3_geom" size="{T}" type="capsule"/>
        </body>
      </body>

      <!-- Back Right Leg -->
      <body name="back_right_leg" pos="{L} -{L} 0">
        <joint axis="0 0 1" name="hip_4" pos="0 0 0" range="-40 40" type="hinge"/>
        <geom fromto="0 0 0 {L} -{L} 0" name="leg_4_geom" size="{T}" type="capsule"/>
        <body pos="{L} -{L} 0" name="back_right_foot">
          <joint axis="1 1 0" name="ankle_4" pos="0 0 0" range="30 100" type="hinge"/>
          <geom fromto="0 0 0 {F} -{F} 0" name="foot_4_geom" size="{T}" type="capsule"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_1" gear="15"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_1" gear="15"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_2" gear="15"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_2" gear="15"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_3" gear="15"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_3" gear="15"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_4" gear="15"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_4" gear="15"/>
  </actuator>
</mujoco>'''
        return mjcf


class AntEnv:
    """
    Ant environment using Newton physics engine.

    Observation (27D):
        - torso z position (1)
        - torso orientation quaternion (4)
        - joint angles (8)
        - torso velocity (6)
        - joint velocities (8)

    Action (8D):
        - Joint torques for 4 hips + 4 ankles

    Reward:
        - Forward velocity (x direction)
        - Survival bonus
        - Control cost penalty
    """

    def __init__(
        self,
        parametric_model: ParametricAnt = None,
        dt: float = 0.05,
        num_substeps: int = 5,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.5,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple = (0.2, 1.0),
        device: str = "cuda",
    ):
        if not NEWTON_AVAILABLE:
            raise ImportError("Newton/Warp required")

        self.dt = dt
        self.num_substeps = num_substeps
        self.sub_dt = dt / num_substeps
        self.device = device

        # Reward weights
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_reward = healthy_reward
        self.terminate_when_unhealthy = terminate_when_unhealthy
        self.healthy_z_range = healthy_z_range

        # Parametric model
        if parametric_model is None:
            self.parametric_model = ParametricAnt()
        else:
            self.parametric_model = parametric_model

        # Build model
        self._build_model()

        # Dimensions
        self.obs_dim = 27  # z(1) + quat(4) + joints(8) + vel(6) + joint_vel(8)
        self.act_dim = 8   # 4 hips + 4 ankles

        # Episode state
        self.steps = 0
        self.max_steps = 1000
        self.terminated = False
        self.prev_x = 0.0

    def _build_model(self):
        """Build Newton model from parametric MJCF."""
        wp.init()

        # Generate MJCF and save to temp file
        mjcf_str = self.parametric_model.generate_mjcf()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(mjcf_str)
            mjcf_path = f.name

        try:
            # Build model
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
                xform=wp.transform(wp.vec3(0, 0, 0.75)),
            )

            # Set joint target stiffness/damping
            for i in range(len(builder.joint_target_ke)):
                builder.joint_target_ke[i] = 150
                builder.joint_target_kd[i] = 5

            # Add ground plane
            builder.add_ground_plane()

            # Finalize with gradients enabled
            self.model = builder.finalize(requires_grad=True)

            # Create solver
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                njmax=100,
                nconmax=50,
            )

            # Allocate states
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.control = self.model.control()

            # Forward kinematics
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

            # Collision pipeline
            self.contacts = self.model.collide(self.state_0)

        finally:
            # Clean up temp file
            os.unlink(mjcf_path)

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.steps = 0
        self.terminated = False

        # Reset joint positions/velocities with small noise
        joint_q = self.model.joint_q.numpy()
        joint_qd = self.model.joint_qd.numpy()

        # Add small noise to joint angles (skip root joint)
        joint_q[7:] += np.random.uniform(-0.1, 0.1, size=len(joint_q) - 7)
        joint_qd[:] = np.random.uniform(-0.1, 0.1, size=len(joint_qd))

        self.model.joint_q.assign(joint_q)
        self.model.joint_qd.assign(joint_qd)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Track initial x position
        body_q = self.state_0.body_q.numpy()
        self.prev_x = body_q[0, 0]  # Torso x position

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Get observation."""
        joint_q = self.model.joint_q.numpy()
        joint_qd = self.model.joint_qd.numpy()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        # Torso state
        torso_z = body_q[0, 2]  # z position
        torso_quat = body_q[0, 3:7]  # quaternion (w, x, y, z)
        torso_vel = body_qd[0, :]  # 6D velocity

        # Joint angles (skip 7 DOF free root joint)
        joint_angles = joint_q[7:15] if len(joint_q) > 15 else joint_q[7:]
        joint_vels = joint_qd[6:14] if len(joint_qd) > 14 else joint_qd[6:]

        obs = np.concatenate([
            [torso_z],
            torso_quat,
            joint_angles,
            torso_vel,
            joint_vels,
        ]).astype(np.float32)

        return obs

    def _get_torso_xy(self) -> tuple:
        """Get torso x, y position."""
        body_q = self.state_0.body_q.numpy()
        return body_q[0, 0], body_q[0, 1]

    def _is_healthy(self) -> bool:
        """Check if ant is in healthy state (not fallen)."""
        body_q = self.state_0.body_q.numpy()
        z = body_q[0, 2]
        return self.healthy_z_range[0] < z < self.healthy_z_range[1]

    def step(self, action: np.ndarray):
        """
        Take a step in the environment.

        Args:
            action: Joint torques (8D), scaled to [-1, 1]

        Returns:
            obs, reward, terminated, truncated, info
        """
        if self.terminated:
            return self._get_obs(), 0.0, True, False, {}

        # Clip and apply action
        action = np.clip(action, -1.0, 1.0)

        joint_act = self.model.joint_act.numpy()
        # Actions control the 8 actuated joints
        n_act = min(len(action), len(joint_act))
        joint_act[:n_act] = action[:n_act] * 15.0  # Scale by gear ratio
        self.model.joint_act.assign(joint_act)

        # Get position before step
        x_before = self._get_torso_xy()[0]

        # Simulate substeps
        for _ in range(self.num_substeps):
            self.state_0.clear_forces()
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.steps += 1

        # Get position after step
        x_after = self._get_torso_xy()[0]

        # Compute reward
        forward_reward = self.forward_reward_weight * (x_after - x_before) / self.dt
        ctrl_cost = self.ctrl_cost_weight * np.sum(action ** 2)
        healthy = self._is_healthy()
        healthy_reward = self.healthy_reward if healthy else 0.0

        reward = forward_reward + healthy_reward - ctrl_cost

        # Check termination
        if self.terminate_when_unhealthy and not healthy:
            self.terminated = True

        truncated = self.steps >= self.max_steps

        obs = self._get_obs()

        info = {
            "x": x_after,
            "forward_reward": forward_reward,
            "ctrl_cost": ctrl_cost,
            "healthy": healthy,
            "leg_length": self.parametric_model.leg_length,
            "foot_length": self.parametric_model.foot_length,
        }

        return obs, reward, self.terminated, truncated, info

    def compute_gradient_wrt_morphology(
        self,
        policy_fn,
        param_name: str = "leg_length",
        horizon: int = 200,
        n_rollouts: int = 3,
        eps: float = 0.02,
    ) -> tuple:
        """
        Compute gradient of return w.r.t. morphology parameter.

        Uses finite difference (model rebuild required for each perturbation).

        Args:
            policy_fn: Frozen policy
            param_name: Which parameter ("leg_length", "foot_length", "torso_radius")
            horizon: Rollout length
            n_rollouts: Number of rollouts for averaging
            eps: Finite difference epsilon

        Returns:
            (mean_return, gradient)
        """
        current_val = getattr(self.parametric_model, param_name)

        # Evaluate at param - eps
        self.parametric_model.set_params(**{param_name: current_val - eps})
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

        # Evaluate at param + eps
        self.parametric_model.set_params(**{param_name: current_val + eps})
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

        # Restore original value
        self.parametric_model.set_params(**{param_name: current_val})
        self._build_model()

        mean_return = (np.mean(returns_plus) + np.mean(returns_minus)) / 2
        gradient = (np.mean(returns_plus) - np.mean(returns_minus)) / (2 * eps)

        return mean_return, gradient


def test_ant_env():
    """Test Ant environment."""
    print("Testing Ant Environment")
    print("=" * 50)

    env = AntEnv()
    print(f"Leg length: {env.parametric_model.leg_length:.3f} m")
    print(f"Foot length: {env.parametric_model.foot_length:.3f} m")
    print(f"Torso radius: {env.parametric_model.torso_radius:.3f} m")
    print(f"Obs dim: {env.obs_dim}, Act dim: {env.act_dim}")

    obs = env.reset()
    print(f"Initial obs shape: {obs.shape}")

    total_reward = 0
    for step in range(100):
        action = np.random.uniform(-1, 1, size=env.act_dim)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 20 == 0:
            print(f"Step {step}: x={info['x']:.3f}, healthy={info['healthy']}, reward={reward:.2f}")

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    print(f"Total reward: {total_reward:.1f}")
    print("\nAnt environment test complete!")


if __name__ == "__main__":
    test_ant_env()
