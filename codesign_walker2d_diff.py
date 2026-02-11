#!/usr/bin/env python3
"""
Level 2.2: PGHC Co-Design for Walker2D with Backprop Design Gradients

Replaces finite-difference morphology gradients (Level 2.1) with true
backpropagation through physics using Warp's tape mechanism.

Inner loop: PPO with Walker2DVecEnv (SolverMuJoCo) — unchanged from 2.1
Outer loop: BPTT through SolverSemiImplicit to get exact ∂reward/∂morphology

2 design parameters: thigh_length, leg_length
(foot_length dropped — only affects collision geometry, not joint_X_p)

Run:
    python codesign_walker2d_diff.py --wandb --num-worlds 1024
"""

import os
os.environ["PYGLET_HEADLESS"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import tempfile
from collections import deque

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import warp as wp
import newton

from envs.walker2d.walker2d_env import ParametricWalker2D
from envs.walker2d.walker2d_vec_env import Walker2DVecEnv


# ---------------------------------------------------------------------------
# Observation normalization (same as Level 2.1)
# ---------------------------------------------------------------------------

class RunningMeanStdCUDA:
    """Tracks running mean/variance using Welford's algorithm on CUDA tensors."""

    def __init__(self, shape, clip=5.0, device="cuda:0"):
        self.mean = torch.zeros(shape, dtype=torch.float64, device=device)
        self.var = torch.ones(shape, dtype=torch.float64, device=device)
        self.count = 1e-4
        self.clip = clip
        self.device = device

    def update(self, batch):
        """batch: (N, D) CUDA float32 tensor."""
        batch = batch.double()
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total_count) / total_count
        self.count = total_count

    def normalize(self, obs):
        """obs: (N, D) CUDA float32 tensor. Returns (N, D) CUDA float32 tensor."""
        return ((obs - self.mean.float()) / torch.sqrt(self.var.float() + 1e-8)).clamp(-self.clip, self.clip)

    def normalize_np(self, obs_np):
        """Normalize numpy array (for DiffWalker2DEval action pre-collection)."""
        obs_norm = (obs_np - self.mean.cpu().numpy().astype(np.float32)) / np.sqrt(self.var.cpu().numpy().astype(np.float32) + 1e-8)
        return np.clip(obs_norm, -self.clip, self.clip).astype(np.float32)


# ---------------------------------------------------------------------------
# Networks (same as Level 2.1)
# ---------------------------------------------------------------------------

class Walker2DPolicy(nn.Module):
    def __init__(self, obs_dim=17, act_dim=6, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)
        nn.init.uniform_(self.net[-1].weight, -0.1, 0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return torch.tanh(self.net(x))

    def get_action(self, obs, deterministic=False):
        """obs: CUDA tensor or numpy. Returns CUDA tensor."""
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(next(self.parameters()).device)
        with torch.no_grad():
            mean = self.forward(obs)
            if deterministic:
                return mean
            std = torch.exp(self.log_std)
            action = mean + std * torch.randn_like(mean)
            return torch.clamp(action, -1.0, 1.0)

    def get_actions_batch(self, obs_batch, deterministic=False):
        """obs_batch: CUDA tensor or numpy. Returns CUDA tensor."""
        if isinstance(obs_batch, np.ndarray):
            obs_batch = torch.FloatTensor(obs_batch).to(next(self.parameters()).device)
        with torch.no_grad():
            mean = self.forward(obs_batch)
            if deterministic:
                return mean
            std = torch.exp(self.log_std)
            actions = mean + std * torch.randn_like(mean)
            return torch.clamp(actions, -1.0, 1.0)

    def get_action_and_log_prob_batch(self, obs_batch):
        mean = self.forward(obs_batch)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        actions_raw = dist.rsample()
        actions = torch.clamp(actions_raw, -1.0, 1.0)
        log_probs = dist.log_prob(actions).sum(-1)
        return actions, log_probs


class Walker2DValue(nn.Module):
    def __init__(self, obs_dim=17, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# PPO Training (same as Level 2.1)
# ---------------------------------------------------------------------------

def collect_rollout_vec(env, policy, value_net, obs_rms, horizon=64, device="cuda:0"):
    """Collect rollout from vectorized environment. All GPU-resident."""
    num_worlds = env.num_worlds
    obs = env.last_obs if hasattr(env, 'last_obs') and env.last_obs is not None else env.reset()

    if not hasattr(env, 'ep_reward_accum') or env.ep_reward_accum is None:
        env.ep_reward_accum = torch.zeros(num_worlds, device=device)
        env.ep_length_accum = torch.zeros(num_worlds, device=device, dtype=torch.int32)

    # Pre-allocate rollout storage on GPU
    all_obs = torch.zeros(horizon, num_worlds, env.obs_dim, device=device)
    all_actions = torch.zeros(horizon, num_worlds, env.act_dim, device=device)
    all_rewards = torch.zeros(horizon, num_worlds, device=device)
    all_log_probs = torch.zeros(horizon, num_worlds, device=device)
    all_dones = torch.zeros(horizon, num_worlds, device=device)
    all_values = torch.zeros(horizon, num_worlds, device=device)

    completed_rewards, completed_lengths = [], []

    for t in range(horizon):
        obs_rms.update(obs)
        obs_norm = obs_rms.normalize(obs)

        actions, log_probs = policy.get_action_and_log_prob_batch(obs_norm)
        with torch.no_grad():
            values = value_net(obs_norm)

        next_obs, rewards, dones, _ = env.step(actions.detach())

        all_obs[t] = obs_norm
        all_actions[t] = actions.detach()
        all_rewards[t] = rewards
        all_log_probs[t] = log_probs.detach()
        all_dones[t] = dones.float()
        all_values[t] = values

        # Episode tracking on GPU
        env.ep_reward_accum += rewards
        env.ep_length_accum += 1
        done_mask = dones.bool() if dones.dtype != torch.bool else dones
        if done_mask.any():
            completed_rewards.extend(env.ep_reward_accum[done_mask].cpu().tolist())
            completed_lengths.extend(env.ep_length_accum[done_mask].cpu().float().tolist())
            env.ep_reward_accum[done_mask] = 0.0
            env.ep_length_accum[done_mask] = 0

        obs = next_obs

    env.last_obs = obs

    obs_norm = obs_rms.normalize(obs)
    with torch.no_grad():
        last_value = value_net(obs_norm)

    return {
        "observations": all_obs,
        "actions": all_actions,
        "rewards": all_rewards,
        "log_probs": all_log_probs,
        "dones": all_dones,
        "values": all_values,
        "last_value": last_value,
        "completed_rewards": completed_rewards,
        "completed_lengths": completed_lengths,
    }


def ppo_update_vec(policy, value_net, optimizer, rollout, n_epochs=5, clip_ratio=0.2,
                   gamma=0.99, gae_lambda=0.95, value_coeff=0.5, entropy_coeff=0.01,
                   num_mini_batches=8, desired_kl=0.008):
    """PPO update with GAE, mini-batches, and adaptive LR. All tensors on CUDA."""
    device = rollout["rewards"].device
    H, N = rollout["rewards"].shape
    rewards = rollout["rewards"]
    dones = rollout["dones"]
    values = rollout["values"]
    last_value = rollout["last_value"]

    with torch.no_grad():
        advantages = torch.zeros(H, N, device=device)
        last_gae = torch.zeros(N, device=device)
        for t in reversed(range(H)):
            next_value = last_value if t == H - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + values

    total_samples = H * N
    obs_flat = rollout["observations"].reshape(total_samples, -1)
    acts_flat = rollout["actions"].reshape(total_samples, -1)
    old_log_probs_flat = rollout["log_probs"].reshape(total_samples)
    advantages_flat = advantages.reshape(total_samples)
    returns_flat = returns.reshape(total_samples)
    advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

    all_params = list(policy.parameters()) + list(value_net.parameters())
    mini_batch_size = total_samples // num_mini_batches
    mean_kl = 0.0

    for epoch in range(n_epochs):
        perm = torch.randperm(total_samples, device=device)
        for mb in range(num_mini_batches):
            idx = perm[mb * mini_batch_size : (mb + 1) * mini_batch_size]

            mean = policy(obs_flat[idx])
            std = torch.exp(policy.log_std)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(acts_flat[idx]).sum(-1)
            entropy = dist.entropy().sum(-1).mean()
            values_pred = value_net(obs_flat[idx])

            ratio = torch.exp(log_probs - old_log_probs_flat[idx])
            clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            policy_loss = -torch.min(ratio * advantages_flat[idx], clipped_ratio * advantages_flat[idx]).mean()
            value_loss = nn.functional.mse_loss(values_pred, returns_flat[idx])
            loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

        with torch.no_grad():
            mean_all = policy(obs_flat)
            std_all = torch.exp(policy.log_std)
            dist_all = torch.distributions.Normal(mean_all, std_all)
            new_lp = dist_all.log_prob(acts_flat).sum(-1)
            mean_kl = (old_log_probs_flat - new_lp).mean().item()
        if mean_kl > 2.0 * desired_kl:
            break

    current_lr = optimizer.param_groups[0]['lr']
    if mean_kl > 2.0 * desired_kl:
        new_lr = max(current_lr / 1.5, 1e-5)
    elif mean_kl < desired_kl / 2.0:
        new_lr = min(current_lr * 1.5, 3e-4)
    else:
        new_lr = current_lr
    for pg in optimizer.param_groups:
        pg['lr'] = new_lr

    return mean_kl


# ---------------------------------------------------------------------------
# Warp kernels for differentiable evaluation
# ---------------------------------------------------------------------------

@wp.kernel
def diff_extract_obs_kernel(
    joint_q: wp.array2d(dtype=float),
    joint_qd: wp.array2d(dtype=float),
    obs: wp.array2d(dtype=float),
):
    """Extract 17D observation per world (same layout as training env)."""
    tid = wp.tid()
    # qpos: skip rootx (index 0), take indices 1..8
    for i in range(8):
        obs[tid, i] = joint_q[tid, 1 + i]
    # qvel: all 9
    for i in range(9):
        obs[tid, 8 + i] = joint_qd[tid, i]


@wp.kernel
def diff_set_torques_kernel(
    actions: wp.array2d(dtype=float),
    gear_ratio: float,
    joint_f: wp.array2d(dtype=float),
    num_act_dofs: int,
    num_root_dofs: int,
):
    """Set joint torques from pre-collected actions."""
    tid = wp.tid()
    for i in range(num_root_dofs):
        joint_f[tid, i] = 0.0
    for i in range(num_act_dofs):
        joint_f[tid, num_root_dofs + i] = actions[tid, i] * gear_ratio


@wp.kernel
def diff_init_worlds_kernel(
    joint_q: wp.array2d(dtype=float),
    joint_qd: wp.array2d(dtype=float),
    num_joint_q: int,
    num_joint_qd: int,
):
    """Initialize all worlds to default pose (zero noise for eval)."""
    tid = wp.tid()
    for i in range(num_joint_q):
        joint_q[tid, i] = 0.0
    for i in range(num_joint_qd):
        joint_qd[tid, i] = 0.0


@wp.kernel
def compute_forward_loss_kernel(
    joint_q_final: wp.array2d(dtype=float),
    joint_q_initial: wp.array2d(dtype=float),
    loss: wp.array(dtype=float),
    num_worlds: int,
):
    """loss = -mean(x_final - x_initial) over all worlds.

    Uses joint_q (always flat float) instead of body_q.
    Root x position is at joint_q[w, 0].
    Negative because we want gradient ASCENT on forward distance.
    """
    tid = wp.tid()
    if tid == 0:
        total = float(0.0)
        for w in range(num_worlds):
            x_final = joint_q_final[w, 0]
            x_initial = joint_q_initial[w, 0]
            total = total + (x_final - x_initial)
        loss[0] = -total / float(num_worlds)


# ---------------------------------------------------------------------------
# Differentiable Walker2D evaluation environment
# ---------------------------------------------------------------------------

class DiffWalker2DEval:
    """
    Differentiable Walker2D eval for computing design gradients via BPTT.

    Builds a small eval env with SolverSemiImplicit, runs a frozen-policy
    episode on wp.Tape(), then tape.backward() to get ∂reward/∂joint_X_p.

    Two-phase approach:
    1. Pre-collect actions (no tape) — avoids CPU↔GPU sync during taped pass
    2. Replay on tape — all physics operations recorded for BPTT
    """

    def __init__(
        self,
        parametric_model,
        num_worlds=16,
        horizon=100,
        dt=0.05,
        num_substeps=5,
        gear_ratio=100.0,
        device="cuda:0",
    ):
        self.parametric_model = parametric_model
        self.num_worlds = num_worlds
        self.horizon = horizon
        self.dt = dt
        self.num_substeps = num_substeps
        self.sub_dt = dt / num_substeps
        self.gear_ratio = gear_ratio
        self.device = device

        self.obs_dim = 17
        self.act_dim = 6
        self.num_joints_per_world = 9  # 3 root + 6 actuated
        self.num_bodies_per_world = 7  # torso + 3 per leg

        self._build_model()
        self._alloc_buffers()

    def _build_model(self):
        """Build Newton model with SolverSemiImplicit for BPTT."""
        mjcf_str = self.parametric_model.generate_mjcf()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(mjcf_str)
            mjcf_path = f.name

        try:
            single_builder = newton.ModelBuilder()
            # Register MuJoCo attributes even though we use SemiImplicit solver,
            # because add_mjcf may need them for parsing
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

            # Use SolverSemiImplicit for BPTT compatibility
            self.solver = newton.solvers.SolverSemiImplicit(self.model)

            self.control = self.model.control()

        finally:
            os.unlink(mjcf_path)

    def _alloc_buffers(self):
        """Pre-allocate all GPU buffers."""
        total_physics_steps = self.horizon * self.num_substeps
        # Pre-allocate states for the tape: need total_physics_steps + 1
        self.states = [self.model.state(requires_grad=True)
                       for _ in range(total_physics_steps + 1)]

        # Observation buffer (for action pre-collection phase)
        self.obs_wp = wp.zeros((self.num_worlds, self.obs_dim), dtype=float, device=self.device)

        # Pre-collected actions buffer: (horizon, num_worlds, act_dim)
        self.pre_actions = [
            wp.zeros((self.num_worlds, self.act_dim), dtype=float, device=self.device)
            for _ in range(self.horizon)
        ]

        # Temporary CPU action buffer for transfer
        self._actions_cpu = wp.zeros((self.num_worlds, self.act_dim), dtype=float, device="cpu")

        # Loss buffer
        self.loss_buf = wp.zeros(1, dtype=float, requires_grad=True, device=self.device)

        # Store initial joint_q for loss computation
        self.joint_q_initial = wp.zeros(
            (self.num_worlds, self.num_joints_per_world),
            dtype=float, requires_grad=True, device=self.device,
        )

        # joint_q / joint_qd views via ArticulationView
        try:
            from newton.selection import ArticulationView
            self.artic_view = ArticulationView(self.model, "parametric_walker2d", verbose=False)
        except Exception:
            try:
                from newton.selection import ArticulationView
                self.artic_view = ArticulationView(self.model, "*", verbose=False)
            except Exception:
                self.artic_view = None

        # Cache articulation views for all states
        self._joint_q_cache = {}
        self._joint_qd_cache = {}
        self._joint_f = None

        if self.artic_view is not None:
            for s in self.states:
                sid = id(s)
                self._joint_q_cache[sid] = self.artic_view.get_attribute("joint_q", s)
                self._joint_qd_cache[sid] = self.artic_view.get_attribute("joint_qd", s)
            self._joint_f = self.artic_view.get_attribute("joint_f", self.control)

    def _get_joint_q(self, state):
        sid = id(state)
        if sid in self._joint_q_cache:
            return self._joint_q_cache[sid]
        return state.joint_q.reshape((self.num_worlds, self.num_joints_per_world))

    def _get_joint_qd(self, state):
        sid = id(state)
        if sid in self._joint_qd_cache:
            return self._joint_qd_cache[sid]
        return state.joint_qd.reshape((self.num_worlds, self.num_joints_per_world))

    def _get_joint_f(self):
        if self._joint_f is not None:
            return self._joint_f
        return self.control.joint_f.reshape((self.num_worlds, self.num_joints_per_world))

    def _reset_state(self, state):
        """Reset a state to default initial pose."""
        joint_q = self._get_joint_q(state)
        joint_qd = self._get_joint_qd(state)
        wp.launch(diff_init_worlds_kernel, dim=self.num_worlds,
                  inputs=[joint_q, joint_qd,
                          self.num_joints_per_world, self.num_joints_per_world])

    def compute_gradient(self, policy, obs_rms):
        """
        Compute ∂reward/∂(thigh_length, leg_length) via BPTT.

        Returns:
            (grad_thigh, grad_leg, mean_reward)
        """
        wp.synchronize()

        # ---------------------------------------------------------------
        # Phase 1: Pre-collect actions (no tape, deterministic policy)
        # ---------------------------------------------------------------
        state_0 = self.states[0]
        self._reset_state(state_0)

        # Run FK to populate body_q from joint_q
        newton.eval_fk(self.model, state_0.joint_q, state_0.joint_qd, state_0)
        wp.synchronize()

        # We need a temporary forward pass to collect actions
        # Use state swapping with just 2 extra states (not on tape)
        tmp_state_a = self.model.state()
        tmp_state_b = self.model.state()

        # Copy initial state
        wp.copy(tmp_state_a.joint_q, state_0.joint_q)
        wp.copy(tmp_state_a.joint_qd, state_0.joint_qd)
        newton.eval_fk(self.model, tmp_state_a.joint_q, tmp_state_a.joint_qd, tmp_state_a)
        wp.synchronize()

        for step in range(self.horizon):
            # Extract obs
            joint_q_view = self._get_joint_q(tmp_state_a)
            joint_qd_view = self._get_joint_qd(tmp_state_a)
            wp.launch(diff_extract_obs_kernel, dim=self.num_worlds,
                      inputs=[joint_q_view, joint_qd_view, self.obs_wp])
            wp.synchronize()

            obs_np = self.obs_wp.numpy()
            obs_norm = obs_rms.normalize_np(obs_np)
            # Policy is on CUDA — send obs there, get actions back
            obs_t = torch.FloatTensor(obs_norm).to(next(policy.parameters()).device)
            actions_t = policy.get_actions_batch(obs_t, deterministic=True)
            actions_np = actions_t.cpu().numpy().astype(np.float64)

            # Store actions
            self._actions_cpu.numpy()[:] = actions_np
            wp.copy(self.pre_actions[step], self._actions_cpu)

            # Step forward (non-tape, using SemiImplicit solver for consistency)
            joint_f = self._get_joint_f()
            wp.launch(diff_set_torques_kernel, dim=self.num_worlds,
                      inputs=[self.pre_actions[step], self.gear_ratio, joint_f,
                              self.act_dim, 3])

            for _ in range(self.num_substeps):
                tmp_state_a.clear_forces()
                contacts = self.model.collide(tmp_state_a)
                self.solver.step(tmp_state_a, tmp_state_b, self.control, contacts, self.sub_dt)
                tmp_state_a, tmp_state_b = tmp_state_b, tmp_state_a

        # Compute mean reward (forward distance) from phase 1 for diagnostics
        joint_q_final_np = self._get_joint_q(tmp_state_a).numpy()
        # Initial x is 0 (we reset to zero)
        mean_forward_dist = np.mean(joint_q_final_np[:, 0])

        # Clean up tmp states
        del tmp_state_a, tmp_state_b

        # ---------------------------------------------------------------
        # Phase 2: Replay on tape for gradients
        # ---------------------------------------------------------------
        state_0 = self.states[0]
        self._reset_state(state_0)

        tape = wp.Tape()
        self.loss_buf.zero_()

        with tape:
            # FK on tape
            newton.eval_fk(self.model, state_0.joint_q, state_0.joint_qd, state_0)

            # Store initial joint_q for loss
            joint_q_0 = self._get_joint_q(state_0)
            wp.copy(self.joint_q_initial, joint_q_0)

            # Physics loop
            physics_step = 0
            for step in range(self.horizon):
                cur_state = self.states[physics_step]

                # Set pre-collected torques
                joint_f = self._get_joint_f()
                wp.launch(diff_set_torques_kernel, dim=self.num_worlds,
                          inputs=[self.pre_actions[step], self.gear_ratio, joint_f,
                                  self.act_dim, 3])

                # Substeps
                for sub in range(self.num_substeps):
                    src = self.states[physics_step]
                    dst = self.states[physics_step + 1]

                    src.clear_forces()
                    contacts = self.model.collide(src)
                    self.solver.step(src, dst, self.control, contacts, self.sub_dt)

                    physics_step += 1

            # Compute loss: -mean(forward distance)
            final_state = self.states[physics_step]
            joint_q_final = self._get_joint_q(final_state)

            wp.launch(compute_forward_loss_kernel, dim=1,
                      inputs=[joint_q_final, self.joint_q_initial,
                              self.loss_buf, self.num_worlds])

        # Get loss value
        wp.synchronize()
        loss_val = self.loss_buf.numpy()[0]

        if np.isnan(loss_val):
            print("    [WARN] Loss is NaN, skipping backward")
            tape.zero()
            return 0.0, 0.0, 0.0

        # Backward
        tape.backward(self.loss_buf)
        wp.synchronize()

        # ---------------------------------------------------------------
        # Phase 3: Extract gradients
        # ---------------------------------------------------------------
        joint_X_p_grad = self.model.joint_X_p.grad
        if joint_X_p_grad is None:
            print("    [WARN] No joint_X_p gradient available")
            tape.zero()
            return 0.0, 0.0, mean_forward_dist

        grad_np = joint_X_p_grad.numpy().copy()

        # Map joint_X_p gradients → design param gradients
        #
        # Per world (9 joints per world):
        #   thigh_length → joint_X_p for joints 4, 7 (leg, leg_left) z-component (index 2)
        #     joint_X_p[w*9+4, 2] = -thigh_length, so ∂/∂thigh_length = -∂loss/∂joint_X_p[..., 2]
        #   leg_length   → joint_X_p for joints 5, 8 (foot, foot_left) z-component (index 2)
        #     joint_X_p[w*9+5, 2] = -leg_length, so ∂/∂leg_length = -∂loss/∂joint_X_p[..., 2]
        #
        # Since loss = -mean(distance), tape.backward gives us ∂loss/∂joint_X_p.
        # For gradient ASCENT on reward, we need -∂loss/∂param.
        # ∂loss/∂thigh_length = ∂loss/∂joint_X_p[z] × ∂joint_X_p[z]/∂thigh_length
        #                     = grad[z] × (-1)
        # So ∂reward/∂thigh_length = -∂loss/∂thigh_length = grad[z]

        grad_thigh = 0.0
        grad_leg = 0.0

        nj = self.num_joints_per_world  # 9
        for w in range(self.num_worlds):
            # Thigh length: affects joints 4 (leg) and 7 (leg_left), z-component (col 2)
            grad_thigh += grad_np[w * nj + 4, 2]  # right leg
            grad_thigh += grad_np[w * nj + 7, 2]  # left leg

            # Leg length: affects joints 5 (foot) and 8 (foot_left), z-component (col 2)
            grad_leg += grad_np[w * nj + 5, 2]  # right foot
            grad_leg += grad_np[w * nj + 8, 2]  # left foot

        # Average over worlds
        grad_thigh /= self.num_worlds
        grad_leg /= self.num_worlds

        # Clip gradients
        grad_clip = 10.0
        grad_thigh = np.clip(grad_thigh, -grad_clip, grad_clip)
        grad_leg = np.clip(grad_leg, -grad_clip, grad_clip)

        tape.zero()

        return float(grad_thigh), float(grad_leg), float(mean_forward_dist)

    def cleanup(self):
        """Free GPU resources."""
        for attr in ('states', 'artic_view', 'model', 'solver', 'control',
                     'obs_wp', 'pre_actions', '_actions_cpu', 'loss_buf',
                     'joint_q_initial', '_joint_q_cache', '_joint_qd_cache',
                     '_joint_f'):
            if hasattr(self, attr):
                delattr(self, attr)
        import gc
        gc.collect()
        wp.synchronize()


# ---------------------------------------------------------------------------
# FD gradient for comparison / sanity check
# ---------------------------------------------------------------------------

def compute_fd_gradient(parametric_model, policy, obs_rms, eps=0.02, num_eval_worlds=64):
    """Finite-difference gradient for 2 design params (for validation only)."""
    device = next(policy.parameters()).device
    param_names = ["thigh_length", "leg_length"]
    current_vals = {name: getattr(parametric_model, name) for name in param_names}

    def eval_at_params(**overrides):
        for name, val in overrides.items():
            setattr(parametric_model, name, val)
        parametric_model.set_params(
            thigh_length=parametric_model.thigh_length,
            leg_length=parametric_model.leg_length,
        )
        env = Walker2DVecEnv(
            parametric_model=parametric_model,
            num_worlds=num_eval_worlds,
        )
        wp.synchronize()
        obs = env.reset()  # CUDA tensor
        episode_rewards = torch.zeros(num_eval_worlds, device=device)
        completed = torch.zeros(num_eval_worlds, dtype=torch.bool, device=device)
        for step in range(env.max_steps):
            obs_input = obs_rms.normalize(obs) if obs_rms is not None else obs
            actions = policy.get_actions_batch(obs_input, deterministic=True)
            obs, rewards, dones, _ = env.step(actions)
            done_mask = dones.bool() if dones.dtype != torch.bool else dones
            episode_rewards += rewards * (~completed).float()
            completed = completed | done_mask
        env.cleanup()
        wp.synchronize()
        for name, val in current_vals.items():
            setattr(parametric_model, name, val)
        return episode_rewards.mean().item()

    gradients = {}
    for name in param_names:
        val = current_vals[name]
        r_plus = eval_at_params(**{name: val + eps})
        r_minus = eval_at_params(**{name: val - eps})
        gradients[name] = (r_plus - r_minus) / (2 * eps)

    return gradients


# ---------------------------------------------------------------------------
# Stability gate (same as Level 2.1)
# ---------------------------------------------------------------------------

class StabilityGate:
    def __init__(self, rel_threshold=0.02, min_inner_iters=500,
                 stable_iters_required=50, window=5):
        self.rel_threshold = rel_threshold
        self.min_inner_iters = min_inner_iters
        self.stable_iters_required = stable_iters_required
        self.reward_history = deque(maxlen=window)
        self.total_inner_iters = 0
        self.stable_count = 0

    def reset(self):
        self.reward_history.clear()
        self.total_inner_iters = 0
        self.stable_count = 0

    def update(self, mean_reward):
        self.reward_history.append(mean_reward)

    def tick(self, n_iters=1):
        self.total_inner_iters += n_iters
        if self._is_plateau():
            self.stable_count += n_iters
        else:
            self.stable_count = 0

    def _is_plateau(self):
        if len(self.reward_history) < 2:
            return False
        rewards = np.array(self.reward_history)
        mean_val = np.mean(rewards)
        if abs(mean_val) < 1e-6:
            return True
        return (np.max(rewards) - np.min(rewards)) / abs(mean_val) < self.rel_threshold

    def is_converged(self):
        if self.total_inner_iters < self.min_inner_iters:
            return False
        return self.stable_count >= self.stable_iters_required


# ---------------------------------------------------------------------------
# Main PGHC loop
# ---------------------------------------------------------------------------

def pghc_codesign_walker2d_diff(
    n_outer_iterations=50,
    design_lr=0.005,
    initial_thigh_length=0.45,
    initial_leg_length=0.50,
    num_worlds=1024,
    num_eval_worlds=16,
    eval_horizon=100,
    use_wandb=False,
    horizon=64,
    compare_fd=False,
):
    """PGHC Co-Design for Walker2D with backprop design gradients."""
    print("=" * 60)
    print("PGHC Co-Design for Walker2D (Backprop Gradients via BPTT)")
    print("=" * 60)

    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="pghc-codesign",
            name=f"walker2d-diff-{num_worlds}w",
            config={
                "level": "2.2-walker2d-diff",
                "num_worlds": num_worlds,
                "num_eval_worlds": num_eval_worlds,
                "eval_horizon": eval_horizon,
                "n_outer_iterations": n_outer_iterations,
                "design_lr": design_lr,
                "initial_thigh_length": initial_thigh_length,
                "initial_leg_length": initial_leg_length,
                "horizon": horizon,
                "gamma": 0.99,
                "entropy_coeff": 0.01,
                "desired_kl": 0.008,
                "hidden_dim": 128,
                "gear_ratio": 100,
                "ctrl_cost_weight": 1e-3,
                "gradient_method": "backprop",
                "eval_solver": "SolverSemiImplicit",
                "train_solver": "SolverMuJoCo",
                "compare_fd": compare_fd,
            },
        )
        print(f"  [wandb] Logging enabled")
    elif use_wandb:
        print("  [wandb] Not available")
        use_wandb = False

    # Use fixed foot_length since we don't optimize it
    fixed_foot_length = 0.20

    parametric_model = ParametricWalker2D(
        thigh_length=initial_thigh_length,
        leg_length=initial_leg_length,
        foot_length=fixed_foot_length,
    )

    device = torch.device("cuda:0")

    env = Walker2DVecEnv(
        num_worlds=num_worlds,
        parametric_model=parametric_model,
    )

    policy = Walker2DPolicy().to(device)
    value_net = Walker2DValue().to(device)
    optimizer = optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()), lr=3e-4
    )

    obs_rms = RunningMeanStdCUDA(shape=(17,), device=device)
    stability_gate = StabilityGate(rel_threshold=0.02, min_inner_iters=0, stable_iters_required=50)

    param_names = ["thigh_length", "leg_length"]
    design_params = torch.tensor(
        [initial_thigh_length, initial_leg_length],
        dtype=torch.float32, requires_grad=False,
    )
    design_optimizer = torch.optim.Adam([design_params], lr=design_lr)

    param_bounds = {
        "thigh_length": (parametric_model.thigh_length_min, parametric_model.thigh_length_max),
        "leg_length": (parametric_model.leg_length_min, parametric_model.leg_length_max),
    }

    param_history = {name: deque(maxlen=5) for name in param_names}

    history = {
        "thigh_length": [parametric_model.thigh_length],
        "leg_length": [parametric_model.leg_length],
        "rewards": [],
        "gradients": [],
        "inner_iterations": [],
    }

    global_step = 0

    print(f"\nConfiguration:")
    print(f"  Num parallel worlds (training): {num_worlds}")
    print(f"  Num eval worlds (backprop): {num_eval_worlds}")
    print(f"  Eval horizon: {eval_horizon} macro steps")
    print(f"  Training horizon: {horizon}")
    print(f"  Design optimizer: Adam (lr={design_lr})")
    print(f"  Gradient method: BPTT through SolverSemiImplicit")
    if compare_fd:
        print(f"  FD comparison: ENABLED")
    print(f"  Initial morphology:")
    for name in param_names:
        print(f"    {name} = {getattr(parametric_model, name):.3f} m")
    print(f"  foot_length = {fixed_foot_length:.3f} m (fixed)")
    print(f"  Init z: {parametric_model.init_z:.3f} m")

    for outer_iter in range(n_outer_iterations):
        print(f"\n{'='*60}")
        print(f"Outer Iteration {outer_iter + 1}/{n_outer_iterations}")
        print(f"{'='*60}")
        print(f"  thigh={parametric_model.thigh_length:.3f}, "
              f"leg={parametric_model.leg_length:.3f}")

        if use_wandb:
            wandb.log({
                "outer/iteration": outer_iter + 1,
                **{f"outer/{n}": getattr(parametric_model, n) for n in param_names},
            }, step=global_step)

        # =============================================
        # INNER LOOP (same as Level 2.1)
        # =============================================
        print(f"\n  [Inner Loop] Training PPO ({num_worlds} parallel envs)...")
        stability_gate.reset()

        log_every = 10
        inner_iter = 0
        crash_count = 0
        max_crashes = 5
        mean_rew, std_rew, mean_len = 0.0, 0.0, 0.0
        reward_buffer = deque(maxlen=200)
        length_buffer = deque(maxlen=200)

        while True:
            try:
                rollout = collect_rollout_vec(env, policy, value_net, obs_rms, horizon=horizon, device=device)
                crash_count = 0  # reset on success
            except Exception as e:
                crash_count += 1
                print(f"    [WARN] Physics crash at iter {inner_iter+1}: {type(e).__name__}: {e}")
                if crash_count >= max_crashes:
                    print(f"    [ERROR] {max_crashes} consecutive crashes. Rebuilding env...")
                    env.cleanup()
                    wp.synchronize()
                    env = Walker2DVecEnv(
                        num_worlds=num_worlds,
                        parametric_model=parametric_model,
                    )
                    crash_count = 0
                wp.synchronize()
                env.last_obs = None
                if hasattr(env, 'ep_reward_accum'):
                    delattr(env, 'ep_reward_accum')
                if hasattr(env, 'ep_length_accum'):
                    delattr(env, 'ep_length_accum')
                env.reset()
                continue

            mean_kl = ppo_update_vec(policy, value_net, optimizer, rollout)
            global_step += 1

            rollout_mean_reward = rollout["rewards"].mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            reward_buffer.extend(rollout["completed_rewards"])
            length_buffer.extend(rollout["completed_lengths"])

            if len(reward_buffer) > 0:
                mean_rew = np.mean(reward_buffer)
                std_rew = np.std(reward_buffer)
                mean_len = np.mean(length_buffer)

            if (inner_iter + 1) % log_every == 0 and len(reward_buffer) > 0:
                stability_gate.update(mean_rew)
                stability_gate.tick(log_every)

            if use_wandb:
                wandb.log({
                    "inner/reward_per_step": rollout_mean_reward,
                    "inner/iteration": inner_iter + 1,
                    "inner/kl": mean_kl,
                    "inner/lr": current_lr,
                    "inner/mean_reward": mean_rew,
                    "inner/reward_std": std_rew,
                    "inner/episode_length": mean_len,
                    **{f"design/{n}": getattr(parametric_model, n) for n in param_names},
                }, step=global_step)

            if (inner_iter + 1) % log_every == 0:
                print(f"    Iter {inner_iter + 1}: "
                      f"rew/step={rollout_mean_reward:.2f}, "
                      f"kl={mean_kl:.4f}, lr={current_lr:.1e}, "
                      f"ep_rew={mean_rew:.1f} +/-{std_rew:.1f}, len={mean_len:.0f}")

            inner_iter += 1

            if stability_gate.is_converged():
                print(f"    CONVERGED at iter {inner_iter} "
                      f"(stable for {stability_gate.stable_count} iters)")
                break

        history["rewards"].append(mean_rew)
        history["inner_iterations"].append(stability_gate.total_inner_iters)
        print(f"  Policy converged. Reward = {mean_rew:.1f} +/- {std_rew:.1f}, "
              f"Length = {mean_len:.0f}")

        # =============================================
        # OUTER LOOP — Backprop Design Gradient
        # =============================================
        env.cleanup()
        wp.synchronize()

        print(f"\n  [Outer Loop] Computing design gradients via BPTT "
              f"({num_eval_worlds} eval worlds, {eval_horizon} steps)...")

        diff_eval = DiffWalker2DEval(
            parametric_model=parametric_model,
            num_worlds=num_eval_worlds,
            horizon=eval_horizon,
            dt=0.05,
            num_substeps=5,
            gear_ratio=100.0,
        )

        grad_thigh, grad_leg, eval_mean_reward = diff_eval.compute_gradient(policy, obs_rms)

        diff_eval.cleanup()
        wp.synchronize()

        print(f"    Backprop gradients:")
        print(f"      ∂reward/∂thigh_length = {grad_thigh:.6f}")
        print(f"      ∂reward/∂leg_length   = {grad_leg:.6f}")
        print(f"      Eval forward distance  = {eval_mean_reward:.3f} m")

        gradients = {"thigh_length": grad_thigh, "leg_length": grad_leg}
        history["gradients"].append(gradients)

        # Optional: FD comparison for validation
        fd_gradients = None
        if compare_fd:
            print(f"    [FD Comparison] Computing FD gradients...")
            fd_gradients = compute_fd_gradient(
                parametric_model, policy, obs_rms,
                eps=0.02, num_eval_worlds=64,
            )
            print(f"      FD ∂reward/∂thigh_length = {fd_gradients['thigh_length']:.4f}")
            print(f"      FD ∂reward/∂leg_length   = {fd_gradients['leg_length']:.4f}")
            # Check sign agreement
            for name in param_names:
                bp_sign = "+" if gradients[name] > 0 else "-"
                fd_sign = "+" if fd_gradients[name] > 0 else "-"
                agree = "AGREE" if bp_sign == fd_sign else "DISAGREE"
                print(f"      {name}: BP={bp_sign}, FD={fd_sign} → {agree}")

        # =============================================
        # Update design parameters
        # =============================================
        old_params = {name: getattr(parametric_model, name) for name in param_names}

        design_optimizer.zero_grad()
        # For Adam minimizer: grad = -gradient (we want to ASCEND reward)
        grad_tensor = torch.tensor(
            [-gradients[name] for name in param_names],
            dtype=torch.float32,
        )
        design_params.grad = grad_tensor
        design_optimizer.step()

        with torch.no_grad():
            for i, name in enumerate(param_names):
                lo, hi = param_bounds[name]
                design_params[i].clamp_(lo, hi)

        for i, name in enumerate(param_names):
            setattr(parametric_model, name, design_params[i].item())

        # Update foot length consistently (not optimized)
        parametric_model.set_params(
            thigh_length=parametric_model.thigh_length,
            leg_length=parametric_model.leg_length,
            foot_length=fixed_foot_length,
        )

        print(f"\n  Design update:")
        for name in param_names:
            delta = getattr(parametric_model, name) - old_params[name]
            print(f"    {name}: {old_params[name]:.4f} -> "
                  f"{getattr(parametric_model, name):.4f} (delta={delta:+.5f})")

        for name in param_names:
            history[name].append(getattr(parametric_model, name))

        # Rebuild training env with new morphology
        env = Walker2DVecEnv(
            num_worlds=num_worlds,
            parametric_model=parametric_model,
        )
        env.last_obs = None
        wp.synchronize()

        if use_wandb:
            log_dict = {
                "outer/reward_at_convergence": mean_rew,
                "outer/eval_forward_distance": eval_mean_reward,
                "outer/inner_iterations_used": stability_gate.total_inner_iters,
                "outer/bp_grad_thigh": grad_thigh,
                "outer/bp_grad_leg": grad_leg,
                "outer/thigh_length_new": parametric_model.thigh_length,
                "outer/leg_length_new": parametric_model.leg_length,
            }
            if fd_gradients is not None:
                log_dict["outer/fd_grad_thigh"] = fd_gradients["thigh_length"]
                log_dict["outer/fd_grad_leg"] = fd_gradients["leg_length"]
            wandb.log(log_dict, step=global_step)

        # Check outer convergence
        for name in param_names:
            param_history[name].append(getattr(parametric_model, name))

        if all(len(param_history[n]) >= 5 for n in param_names):
            all_stable = True
            for name in param_names:
                vals = list(param_history[name])
                if max(vals) - min(vals) >= 0.005:
                    all_stable = False
                    break
            if all_stable:
                print(f"\n  OUTER CONVERGED: All design params stable over last 5 iters")
                break

    # =============================================
    # Final Results
    # =============================================
    print("\n" + "=" * 60)
    print("PGHC Co-Design Complete (Backprop Gradients)!")
    print("=" * 60)

    print(f"\nMorphology evolution:")
    for name in param_names:
        vals = history[name]
        print(f"  {name}: {vals[0]:.3f} -> {vals[-1]:.3f} (delta={vals[-1]-vals[0]:+.3f})")

    total_samples = sum(history["inner_iterations"]) * num_worlds * horizon
    print(f"\nTotal training samples: {total_samples:,}")

    if use_wandb:
        wandb.log({
            "summary/total_samples": total_samples,
            **{f"summary/{n}_initial": history[n][0] for n in param_names},
            **{f"summary/{n}_final": history[n][-1] for n in param_names},
        })
        wandb.finish()

    return history, policy, parametric_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGHC Co-Design for Walker2D (Backprop Gradients)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--outer-iters", type=int, default=50)
    parser.add_argument("--design-lr", type=float, default=0.005)
    parser.add_argument("--num-worlds", type=int, default=1024)
    parser.add_argument("--num-eval-worlds", type=int, default=16)
    parser.add_argument("--eval-horizon", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--initial-thigh", type=float, default=0.45)
    parser.add_argument("--initial-leg", type=float, default=0.50)
    parser.add_argument("--compare-fd", action="store_true",
                        help="Also compute FD gradient for comparison (slow)")
    args = parser.parse_args()

    history, policy, model = pghc_codesign_walker2d_diff(
        n_outer_iterations=args.outer_iters,
        design_lr=args.design_lr,
        initial_thigh_length=args.initial_thigh,
        initial_leg_length=args.initial_leg,
        num_worlds=args.num_worlds,
        num_eval_worlds=args.num_eval_worlds,
        eval_horizon=args.eval_horizon,
        use_wandb=args.wandb,
        horizon=args.horizon,
        compare_fd=args.compare_fd,
    )
