#!/usr/bin/env python3
"""
G1 Walking Environment Wrapper for Newton Physics Engine.

Wraps MimicKit's initialized Newton engine with IsaacLab-style
locomotion rewards, observations, termination, and domain randomization.

Observation (11 + 3*N_dof):
    [ang_vel(3), projected_gravity(3), commands(3),
     dof_pos_relative(N_dof), dof_vel(N_dof), prev_actions(N_dof),
     sin_phase(1), cos_phase(1)]

Rewards (18 terms, IsaacLab G1 flat terrain):
    tracking_lin_vel, tracking_ang_vel, lin_vel_z, ang_vel_xy,
    orientation, base_height, dof_acc, dof_vel, action_rate,
    dof_pos_limits, alive, hip_pos, contact_no_vel,
    feet_swing_height, contact, dof_torques,
    joint_deviation_arms, joint_deviation_torso

Domain randomization:
    obs noise, random pushes, motor strength scaling

Termination: pelvis ground contact OR episode timeout (20s)
"""

import math
import numpy as np
import torch
import warp as wp


# ---------------------------------------------------------------------------
# Quaternion utility
# ---------------------------------------------------------------------------

def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q.

    Args:
        q: (N, 4) quaternion in (x, y, z, w) format (Warp/IsaacGym convention)
        v: (N, 3) vector in world frame
    Returns:
        (N, 3) vector in local frame
    """
    q_w = q[:, 3:4]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
    return a - b + c


# ---------------------------------------------------------------------------
# Configuration (IsaacLab G1 flat terrain + unitree_rl_gym G1RoughCfg)
# ---------------------------------------------------------------------------

class G1WalkingConfig:
    # --- Observation scales ---
    obs_scale_ang_vel = 0.25
    obs_scale_dof_pos = 1.0
    obs_scale_dof_vel = 0.05
    obs_scale_lin_vel = 2.0  # used for commands_scale

    # --- Commands ---
    cmd_x_vel = [0.3, 1.0]
    cmd_y_vel = [-0.3, 0.3]
    cmd_yaw_vel = [-0.5, 0.5]
    cmd_resample_time = 10.0  # seconds

    # --- Control ---
    action_scale = 0.25

    # --- Reward scales (IsaacLab G1 flat terrain merged with G1RoughCfg) ---
    reward_tracking_lin_vel = 1.0
    reward_tracking_ang_vel = 1.0       # IsaacLab flat: 1.0
    reward_lin_vel_z = -0.2             # IsaacLab flat: -0.2
    reward_ang_vel_xy = -0.05
    reward_orientation = -1.0
    reward_base_height = -10.0
    reward_dof_acc = -1e-7              # IsaacLab flat: -1e-7
    reward_dof_vel = -1e-3
    reward_action_rate = -0.005         # IsaacLab flat: -0.005
    reward_dof_pos_limits = -1.0        # IsaacLab: -1.0
    reward_alive = 0.15
    reward_hip_pos = -0.1              # IsaacLab: -0.1
    reward_contact_no_vel = -0.1       # IsaacLab feet_slide: -0.1
    reward_feet_swing_height = -20.0
    reward_contact = 0.75              # IsaacLab feet_air_time: 0.75
    reward_dof_torques = -2e-6         # IsaacLab flat: -2e-6
    reward_joint_deviation_arms = -0.1  # IsaacLab: -0.1
    reward_joint_deviation_torso = -0.1 # IsaacLab: -0.1

    # --- Reward parameters ---
    tracking_sigma = 0.25
    base_height_target = 0.78
    soft_dof_pos_limit = 0.9
    feet_swing_height_target = 0.08

    # --- Termination ---
    max_episode_length_s = 20.0
    termination_penalty = -10.0  # Known Issue #9: -100 backfires, use -10

    # --- Phase ---
    phase_period = 0.8
    phase_offset = 0.5

    # --- Domain randomization ---
    enable_dr = True
    dr_motor_strength_range = 0.1  # +/-10%
    dr_push_interval = 100         # steps between pushes (~3.3s at 30Hz)
    dr_push_vel_range = 0.5        # m/s
    dr_obs_noise_std = 0.02


# ---------------------------------------------------------------------------
# Domain Randomization
# ---------------------------------------------------------------------------

class DomainRandomization:
    """Per-env randomization of physics parameters.

    Randomized at episode reset per env. Applied during step().
    """

    def __init__(self, num_envs, device, cfg):
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg

        self.motor_strength_scale = torch.ones(num_envs, 1, device=device)
        self.push_interval = cfg.dr_push_interval
        self.push_vel_range = cfg.dr_push_vel_range
        self.obs_noise_std = cfg.dr_obs_noise_std

    def randomize_envs(self, env_ids):
        """Resample randomization for given envs."""
        n = len(env_ids)
        r = self.cfg.dr_motor_strength_range
        self.motor_strength_scale[env_ids] = 1.0 + (
            torch.rand(n, 1, device=self.device) - 0.5) * 2 * r

    def apply_push(self, engine, char_id, episode_length_buf):
        """Apply random velocity pushes at periodic intervals."""
        push_mask = ((episode_length_buf % self.push_interval == 0)
                     & (episode_length_buf > 0))
        if not push_mask.any():
            return
        push_ids = push_mask.nonzero(as_tuple=False).flatten()
        n = len(push_ids)
        vel_noise = torch.empty(n, 3, device=self.device).uniform_(
            -self.push_vel_range, self.push_vel_range)
        vel_noise[:, 2] = 0  # no vertical push
        current_vel = engine.get_root_vel(char_id)[push_ids]
        engine.set_root_vel(push_ids, char_id, current_vel + vel_noise)

    def add_obs_noise(self, obs):
        """Add Gaussian noise to observations."""
        if self.obs_noise_std > 0:
            return obs + torch.randn_like(obs) * self.obs_noise_std
        return obs


# ---------------------------------------------------------------------------
# G1 Walking Environment Wrapper
# ---------------------------------------------------------------------------

class G1WalkingEnvWrapper:
    """Wraps MimicKit's Newton env with locomotion rewards and observations."""

    def __init__(self, base_env, device, cfg=None):
        self._env = base_env
        self._engine = base_env._engine
        self._char_id = base_env._get_char_id()
        self.device = device
        self.cfg = cfg or G1WalkingConfig()

        self.num_envs = self._engine.get_num_envs()
        self.dt = self._engine.get_timestep()

        # DOF / body counts
        self.num_dofs = self._engine.get_obj_num_dofs(self._char_id)
        self.num_bodies = self._engine.get_obj_num_bodies(self._char_id)

        # Dimensions
        self.obs_dim = 3 + 3 + 3 + 3 * self.num_dofs + 2
        self.act_dim = self.num_dofs

        # Body indices (feet, pelvis, penalised contacts)
        self._find_body_indices()

        # DOF limits
        self._setup_dof_limits()

        # DOF group indices for rewards
        self._find_dof_group_indices()

        # Save initial state for resets
        wp.synchronize()
        self.init_root_pos = self._engine.get_root_pos(self._char_id).clone()
        self.init_root_rot = self._engine.get_root_rot(self._char_id).clone()
        self.default_dof_pos = self._engine.get_dof_pos(self._char_id)[0].clone()

        # Commands scale
        s = self.cfg
        self.commands_scale = torch.tensor(
            [s.obs_scale_lin_vel, s.obs_scale_lin_vel, s.obs_scale_ang_vel],
            device=self.device,
        )

        # Gravity direction
        self.gravity_vec = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device,
        ).expand(self.num_envs, -1).contiguous()

        # Max episode length
        self.max_episode_length = int(self.cfg.max_episode_length_s / self.dt)

        # Allocate buffers
        self._alloc_buffers()

        # Domain randomization
        self.dr = None
        if self.cfg.enable_dr:
            self.dr = DomainRandomization(self.num_envs, self.device, self.cfg)

        # Reward function list (name -> method)
        self._reward_fns = {
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_ang_vel": self._reward_tracking_ang_vel,
            "lin_vel_z": self._reward_lin_vel_z,
            "ang_vel_xy": self._reward_ang_vel_xy,
            "orientation": self._reward_orientation,
            "base_height": self._reward_base_height,
            "dof_acc": self._reward_dof_acc,
            "dof_vel": self._reward_dof_vel,
            "action_rate": self._reward_action_rate,
            "dof_pos_limits": self._reward_dof_pos_limits,
            "alive": self._reward_alive,
            "hip_pos": self._reward_hip_pos,
            "contact_no_vel": self._reward_contact_no_vel,
            "feet_swing_height": self._reward_feet_swing_height,
            "contact": self._reward_contact,
            "dof_torques": self._reward_dof_torques,
            "joint_deviation_arms": self._reward_joint_deviation_arms,
            "joint_deviation_torso": self._reward_joint_deviation_torso,
        }
        self._reward_scales = {}
        for name in self._reward_fns:
            attr = f"reward_{name}"
            self._reward_scales[name] = getattr(self.cfg, attr, 0.0)

        print(f"  [G1WalkingEnv] num_envs={self.num_envs}, num_dofs={self.num_dofs}, "
              f"num_bodies={self.num_bodies}")
        print(f"  [G1WalkingEnv] obs_dim={self.obs_dim}, act_dim={self.act_dim}, "
              f"dt={self.dt:.4f}s, max_ep={self.max_episode_length}")
        print(f"  [G1WalkingEnv] DR={'ON' if self.dr else 'OFF'}")

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _find_body_indices(self):
        body_names = self._engine.get_obj_body_names(self._char_id)

        # Feet
        self.feet_names = []
        self.feet_body_ids = []
        for i, name in enumerate(body_names):
            if "ankle_roll" in name:
                self.feet_names.append(name)
                self.feet_body_ids.append(i)
        self.feet_num = len(self.feet_body_ids)

        # Pelvis (termination)
        self.pelvis_body_id = None
        for i, name in enumerate(body_names):
            if "pelvis" in name:
                self.pelvis_body_id = i
                break

        # Penalised contacts (hip, knee)
        self.penalized_body_ids = []
        for i, name in enumerate(body_names):
            if "hip" in name or "knee" in name:
                self.penalized_body_ids.append(i)

    def _setup_dof_limits(self):
        dof_low, dof_high = self._engine.get_obj_dof_limits(0, self._char_id)
        self.dof_pos_lower = torch.tensor(dof_low, dtype=torch.float32, device=self.device)
        self.dof_pos_upper = torch.tensor(dof_high, dtype=torch.float32, device=self.device)

        soft = self.cfg.soft_dof_pos_limit
        mid = (self.dof_pos_upper + self.dof_pos_lower) / 2
        half_range = (self.dof_pos_upper - self.dof_pos_lower) / 2
        self.soft_dof_pos_lower = mid - soft * half_range
        self.soft_dof_pos_upper = mid + soft * half_range

    def _find_dof_group_indices(self):
        """Find DOF indices for hip, arm, and torso joints."""
        model = self._engine._sim_model
        joint_keys = model.joint_key
        joint_qd_start = model.joint_qd_start.numpy()
        articulation_start = model.articulation_start.numpy()
        joints_per_world = model.joint_count // self.num_envs
        body_start = int(articulation_start[0])
        qd_offset = int(joint_qd_start[body_start + 1])  # root DOF offset

        self.hip_dof_indices = []
        self.arm_dof_indices = []
        self.torso_dof_indices = []

        for j in range(1, joints_per_world):
            idx = body_start + j
            name = joint_keys[idx] if idx < len(joint_keys) else ""
            ds = int(joint_qd_start[idx]) - qd_offset
            de = int(joint_qd_start[idx + 1]) - qd_offset if (idx + 1) < len(joint_qd_start) else ds

            for d in range(ds, min(de, self.num_dofs)):
                if "hip" in name and "pitch" not in name:
                    self.hip_dof_indices.append(d)
                elif any(kw in name for kw in ["shoulder", "elbow", "wrist", "hand"]):
                    self.arm_dof_indices.append(d)
                elif "waist" in name or "torso" in name:
                    self.torso_dof_indices.append(d)

        self.hip_dof_indices_t = torch.tensor(
            self.hip_dof_indices, dtype=torch.long, device=self.device) if self.hip_dof_indices else torch.zeros(0, dtype=torch.long, device=self.device)
        self.arm_dof_indices_t = torch.tensor(
            self.arm_dof_indices, dtype=torch.long, device=self.device) if self.arm_dof_indices else torch.zeros(0, dtype=torch.long, device=self.device)
        self.torso_dof_indices_t = torch.tensor(
            self.torso_dof_indices, dtype=torch.long, device=self.device) if self.torso_dof_indices else torch.zeros(0, dtype=torch.long, device=self.device)

        print(f"  [G1WalkingEnv] Hip DOFs: {len(self.hip_dof_indices)}, "
              f"Arm DOFs: {len(self.arm_dof_indices)}, "
              f"Torso DOFs: {len(self.torso_dof_indices)}")

    def _alloc_buffers(self):
        N, D = self.num_envs, self.num_dofs
        dev = self.device

        self.obs_buf = torch.zeros(N, self.obs_dim, device=dev)
        self.rew_buf = torch.zeros(N, device=dev)
        self.reset_buf = torch.zeros(N, dtype=torch.bool, device=dev)
        self.time_out_buf = torch.zeros(N, dtype=torch.bool, device=dev)
        self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=dev)

        self.actions = torch.zeros(N, D, device=dev)
        self.last_actions = torch.zeros(N, D, device=dev)
        self.commands = torch.zeros(N, 3, device=dev)

        self.base_lin_vel = torch.zeros(N, 3, device=dev)
        self.base_ang_vel = torch.zeros(N, 3, device=dev)
        self.projected_gravity = torch.zeros(N, 3, device=dev)
        self.dof_pos = torch.zeros(N, D, device=dev)
        self.dof_vel = torch.zeros(N, D, device=dev)
        self.dof_forces = torch.zeros(N, D, device=dev)
        self.last_dof_vel = torch.zeros(N, D, device=dev)
        self.root_height = torch.zeros(N, device=dev)

        self.phase = torch.zeros(N, device=dev)
        self.phase_left = torch.zeros(N, device=dev)
        self.phase_right = torch.zeros(N, device=dev)
        self.leg_phase = torch.zeros(N, 2, device=dev)

        self.feet_pos = torch.zeros(N, self.feet_num, 3, device=dev)
        self.feet_vel = torch.zeros(N, self.feet_num, 3, device=dev)
        self.feet_contact_forces = torch.zeros(N, self.feet_num, 3, device=dev)
        self.ground_contact_forces = None  # set in _update_state

        # Episode stats for logging
        self.ep_reward_accum = torch.zeros(N, device=dev)
        self.ep_length_accum = torch.zeros(N, dtype=torch.long, device=dev)
        self.completed_rewards = []
        self.completed_lengths = []

        # For continuing rollouts
        self.last_obs = None

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def _update_state(self):
        """Read engine state into local buffers."""
        cid = self._char_id
        root_rot = self._engine.get_root_rot(cid)
        root_vel = self._engine.get_root_vel(cid)
        root_ang = self._engine.get_root_ang_vel(cid)
        root_pos = self._engine.get_root_pos(cid)

        self.base_lin_vel[:] = quat_rotate_inverse(root_rot, root_vel)
        self.base_ang_vel[:] = quat_rotate_inverse(root_rot, root_ang)
        self.projected_gravity[:] = quat_rotate_inverse(root_rot, self.gravity_vec)
        self.root_height[:] = root_pos[:, 2]

        self.dof_pos[:] = self._engine.get_dof_pos(cid)
        self.dof_vel[:] = self._engine.get_dof_vel(cid)

        # DOF forces (for torque penalty)
        raw = self._engine._dof_forces
        if isinstance(raw, torch.Tensor):
            if raw.dim() == 3:
                self.dof_forces[:] = torch.sum(raw, dim=-1)
            else:
                self.dof_forces[:] = raw
        else:
            self.dof_forces[:] = raw[cid]

        # Feet state
        body_pos = self._engine.get_body_pos(cid)
        body_vel = self._engine.get_body_vel(cid)
        gc_forces = self._engine.get_ground_contact_forces(cid)
        self.ground_contact_forces = gc_forces

        for i, bid in enumerate(self.feet_body_ids):
            self.feet_pos[:, i] = body_pos[:, bid]
            self.feet_vel[:, i] = body_vel[:, bid]
            self.feet_contact_forces[:, i] = gc_forces[:, bid]

    def _update_phase(self):
        """Update gait phase (period=0.8s, 50% offset between legs)."""
        period = self.cfg.phase_period
        offset = self.cfg.phase_offset
        self.phase = (self.episode_length_buf.float() * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1.0
        self.leg_phase = torch.stack([self.phase_left, self.phase_right], dim=-1)

    def _resample_commands(self, env_ids):
        """Resample velocity commands for given envs."""
        n = len(env_ids)
        cfg = self.cfg
        self.commands[env_ids, 0] = torch.empty(n, device=self.device).uniform_(
            cfg.cmd_x_vel[0], cfg.cmd_x_vel[1])
        self.commands[env_ids, 1] = torch.empty(n, device=self.device).uniform_(
            cfg.cmd_y_vel[0], cfg.cmd_y_vel[1])
        self.commands[env_ids, 2] = torch.empty(n, device=self.device).uniform_(
            cfg.cmd_yaw_vel[0], cfg.cmd_yaw_vel[1])

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def reset(self, env_ids=None):
        """Reset specified envs (or all if None). Returns (obs, info)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            self._update_state()
            self._update_phase()
            self._compute_observations()
            self.last_obs = self.obs_buf.clone()
            return self.obs_buf.clone(), {}

        self._reset_envs(env_ids)
        self._update_state()
        self._update_phase()
        self._compute_observations()
        self.last_obs = self.obs_buf.clone()
        return self.obs_buf.clone(), {}

    def step(self, action):
        """Step environment. Returns (obs, reward, done, info)."""
        action = torch.clamp(action.to(self.device), -1.0, 1.0)
        self.last_actions[:] = self.actions
        self.actions[:] = action

        # Scale to target positions, apply motor strength DR
        target = action * self.cfg.action_scale + self.default_dof_pos.unsqueeze(0)
        if self.dr is not None:
            target = target * self.dr.motor_strength_scale
        self._engine.set_cmd(self._char_id, target)

        # Physics step
        self._engine.step()
        wp.synchronize()

        # Save pre-update DOF vel for acceleration reward
        self.last_dof_vel[:] = self.dof_vel

        # Update state from engine
        self._update_state()

        # Increment episode length
        self.episode_length_buf += 1

        # Update phase
        self._update_phase()

        # Apply random pushes (DR)
        if self.dr is not None:
            self.dr.apply_push(self._engine, self._char_id, self.episode_length_buf)

        # Check termination
        self._check_termination()

        # Compute rewards
        self._compute_rewards()

        # Track episode stats
        self.ep_reward_accum += self.rew_buf
        self.ep_length_accum += 1

        # Done signal (before reset)
        done = self.reset_buf.clone().float()

        # Reset terminated envs
        reset_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(reset_ids) > 0:
            self._record_completed(reset_ids)
            self._reset_envs(reset_ids)

        # Resample commands periodically
        resample_steps = max(1, int(self.cfg.cmd_resample_time / self.dt))
        resample_ids = (self.episode_length_buf % resample_steps == 0).nonzero(
            as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)

        # Compute observations (post-reset for terminated envs)
        self._compute_observations()

        # Apply observation noise (DR)
        if self.dr is not None:
            self.obs_buf = self.dr.add_obs_noise(self.obs_buf)

        self.last_obs = self.obs_buf.clone()

        return self.obs_buf.clone(), self.rew_buf.clone(), done, {}

    def _reset_envs(self, env_ids):
        """Reset specific environment indices."""
        n = len(env_ids)
        cid = self._char_id

        self._engine.set_root_pos(env_ids, cid, self.init_root_pos[env_ids])
        self._engine.set_root_rot(env_ids, cid, self.init_root_rot[env_ids])
        self._engine.set_root_vel(
            env_ids, cid, torch.zeros(n, 3, device=self.device))
        self._engine.set_root_ang_vel(
            env_ids, cid, torch.zeros(n, 3, device=self.device))

        default_exp = self.default_dof_pos.unsqueeze(0).expand(n, -1)
        self._engine.set_dof_pos(env_ids, cid, default_exp)
        self._engine.set_dof_vel(
            env_ids, cid, torch.zeros(n, self.num_dofs, device=self.device))

        # Sync joint_q from dof_pos (ball-joint exp-map -> quaternion) and FK
        self._engine._sim_state.pre_step_update()
        self._engine._sim_state.eval_fk()
        wp.synchronize()

        # Clear per-env buffers
        self.actions[env_ids] = 0
        self.last_actions[env_ids] = 0
        self.last_dof_vel[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.ep_reward_accum[env_ids] = 0
        self.ep_length_accum[env_ids] = 0

        self._resample_commands(env_ids)

        # Randomize DR params for reset envs
        if self.dr is not None:
            self.dr.randomize_envs(env_ids)

    def _record_completed(self, env_ids):
        for idx in env_ids:
            i = idx.item()
            if self.ep_length_accum[i] > 0:
                self.completed_rewards.append(self.ep_reward_accum[i].item())
                self.completed_lengths.append(self.ep_length_accum[i].item())

    def pop_completed_episodes(self):
        """Drain and return completed episode stats."""
        rewards = self.completed_rewards
        lengths = self.completed_lengths
        self.completed_rewards = []
        self.completed_lengths = []
        return rewards, lengths

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _compute_observations(self):
        s = self.cfg
        self.obs_buf = torch.cat([
            self.base_ang_vel * s.obs_scale_ang_vel,                        # 3
            self.projected_gravity,                                          # 3
            self.commands[:, :3] * self.commands_scale,                      # 3
            (self.dof_pos - self.default_dof_pos.unsqueeze(0)) * s.obs_scale_dof_pos,  # D
            self.dof_vel * s.obs_scale_dof_vel,                              # D
            self.actions,                                                    # D
            torch.sin(2 * math.pi * self.phase).unsqueeze(1),               # 1
            torch.cos(2 * math.pi * self.phase).unsqueeze(1),               # 1
        ], dim=-1)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _check_termination(self):
        # Pelvis ground contact
        if self.pelvis_body_id is not None and self.ground_contact_forces is not None:
            pelvis_force = self.ground_contact_forces[:, self.pelvis_body_id, :]
            pelvis_contact = torch.norm(pelvis_force, dim=-1) > 1.0
        else:
            pelvis_contact = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device)

        # Episode timeout
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length

        self.reset_buf = pelvis_contact | self.time_out_buf

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _compute_rewards(self):
        self.rew_buf[:] = 0.0
        for name, fn in self._reward_fns.items():
            scale = self._reward_scales[name]
            if abs(scale) < 1e-12:
                continue
            rew = fn() * scale
            self.rew_buf += rew

        # Termination penalty (only for failures, not timeouts)
        if self.pelvis_body_id is not None and self.ground_contact_forces is not None:
            pelvis_force = self.ground_contact_forces[:, self.pelvis_body_id, :]
            fell = torch.norm(pelvis_force, dim=-1) > 1.0
            self.rew_buf += self.cfg.termination_penalty * fell.float()

    # --- Individual reward functions ---

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.tracking_sigma)

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        return torch.square(self.root_height - self.cfg.base_height_target)

    def _reward_dof_acc(self):
        return torch.sum(
            torch.square((self.dof_vel - self.last_dof_vel) / self.dt), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.soft_dof_pos_lower).clamp(max=0.0)
        out_of_limits += (self.dof_pos - self.soft_dof_pos_upper).clamp(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_alive(self):
        return torch.ones(self.num_envs, device=self.device)

    def _reward_hip_pos(self):
        if len(self.hip_dof_indices_t) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum(
            torch.square(self.dof_pos[:, self.hip_dof_indices_t]), dim=1)

    def _reward_contact_no_vel(self):
        contact = torch.norm(self.feet_contact_forces, dim=2) > 1.0
        contact_vel = self.feet_vel * contact.unsqueeze(-1)
        return torch.sum(torch.square(contact_vel), dim=(1, 2))

    def _reward_feet_swing_height(self):
        contact = torch.norm(self.feet_contact_forces, dim=2) > 1.0
        target_h = self.cfg.feet_swing_height_target
        pos_error = torch.square(self.feet_pos[:, :, 2] - target_h) * (~contact)
        return torch.sum(pos_error, dim=1)

    def _reward_contact(self):
        """Phase-aware contact reward."""
        res = torch.zeros(self.num_envs, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            has_contact = torch.norm(
                self.feet_contact_forces[:, i, :], dim=-1) > 1.0
            res += (~(has_contact ^ is_stance)).float()
        return res

    def _reward_dof_torques(self):
        """Joint torque penalty (IsaacLab dof_torques_l2)."""
        return torch.sum(torch.square(self.dof_forces), dim=-1)

    def _reward_joint_deviation_arms(self):
        """Arm joint deviation from default (IsaacLab joint_deviation_arms)."""
        if len(self.arm_dof_indices_t) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        default = self.default_dof_pos[self.arm_dof_indices_t].unsqueeze(0)
        return torch.sum(
            torch.abs(self.dof_pos[:, self.arm_dof_indices_t] - default), dim=1)

    def _reward_joint_deviation_torso(self):
        """Torso joint deviation from default (IsaacLab joint_deviation_torso)."""
        if len(self.torso_dof_indices_t) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        default = self.default_dof_pos[self.torso_dof_indices_t].unsqueeze(0)
        return torch.sum(
            torch.abs(self.dof_pos[:, self.torso_dof_indices_t] - default), dim=1)
