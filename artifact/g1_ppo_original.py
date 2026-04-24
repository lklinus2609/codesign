#!/usr/bin/env python3
"""
PGHC Co-Design for G1 Humanoid — PPO Inner Loop

Replaces MimicKit AMP with a simple PPO approach using locomotion rewards
(velocity tracking, stability penalties, phase-aware contacts) inspired by
unitree_rl_gym. The outer loop (DiffG1Eval BPTT via SolverSemiImplicit +
wp.Tape) is unchanged from codesign_g1_unified.py.

Architecture:
    Single process, single CUDA context:
    1. Build MimicKit env ONCE (for Newton engine init)
    2. Wrap with G1WalkingEnvWrapper (PPO-style obs/reward/done)
    3. OUTER LOOP:
        a. Update training model joint_X_p in-place from theta
        b. Inner loop: PPO training until convergence
        c. Collect actions (frozen policy, first M worlds)
        d. Build DiffG1Eval lazily (M eval worlds, requires_grad=True)
        e. BPTT gradient via DiffG1Eval.compute_gradient()
        f. Free DiffG1Eval (reclaim VRAM)
        g. Adam update theta, clip to +/-30 deg
        h. Check outer convergence

Run:
    python g1_ppo.py --wandb --num-train-envs 4096 --num-eval-worlds 32
"""

import os
os.environ["PYGLET_HEADLESS"] = "1"

import argparse
import gc
import math
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODESIGN_DIR = Path(__file__).parent.resolve()
MIMICKIT_DIR = (CODESIGN_DIR / ".." / "MimicKit").resolve()
MIMICKIT_SRC_DIR = MIMICKIT_DIR / "mimickit"

BASE_MJCF_PATH = MIMICKIT_DIR / "data" / "assets" / "g1" / "g1.xml"
BASE_ENV_CONFIG = MIMICKIT_DIR / "data" / "envs" / "amp_g1_env.yaml"
BASE_ENGINE_CONFIG = MIMICKIT_DIR / "data" / "engines" / "newton_engine.yaml"

# ---------------------------------------------------------------------------
# MimicKit + GPU imports
# ---------------------------------------------------------------------------
if str(MIMICKIT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(MIMICKIT_SRC_DIR))

import warp as wp
import newton  # noqa: F401

import util.mp_util as mp_util
import envs.env_builder as env_builder

# Co-design modules
from g1_mjcf_modifier import (
    G1MJCFModifier, SYMMETRIC_PAIRS, NUM_DESIGN_PARAMS,
    quat_from_x_rotation, quat_multiply, quat_normalize,
)
from g1_eval_worker import DiffG1Eval
from g1_ppo_env import G1WalkingEnvWrapper, G1WalkingConfig

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Adam optimizer for design parameters (UNCHANGED from codesign_g1_unified)
# ---------------------------------------------------------------------------

class AdamOptimizer:
    """Adam optimizer for numpy arrays (gradient ascent)."""

    def __init__(self, n_params, lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(n_params)
        self.v = np.zeros(n_params)
        self.t = 0

    def step(self, params, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return params + self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Joint info extraction + in-place update (UNCHANGED from codesign_g1_unified)
# ---------------------------------------------------------------------------

def extract_joint_info(model, num_worlds):
    body_keys = model.body_key
    joint_keys = model.joint_key
    bodies_per_world = model.body_count // num_worlds
    joints_per_world = model.joint_count // num_worlds
    body_name_to_idx = {body_keys[i]: i for i in range(bodies_per_world)}
    joint_name_to_idx = {joint_keys[i]: i for i in range(joints_per_world)}
    joint_X_p_np = model.joint_X_p.numpy()
    base_quats_dict = {}
    joint_idx_map = {}
    for left_body, right_body in SYMMETRIC_PAIRS:
        for body_name in (left_body, right_body):
            if body_name in body_name_to_idx:
                idx = body_name_to_idx[body_name]
            else:
                joint_name = body_name.replace("_link", "_joint")
                if joint_name in joint_name_to_idx:
                    idx = joint_name_to_idx[joint_name]
                else:
                    print(f"  [WARN] Body/joint not found: {body_name}")
                    continue
            base_quats_dict[body_name] = tuple(joint_X_p_np[idx, 3:7].tolist())
            joint_idx_map[body_name] = idx
    print(f"  [JointInfo] Found {len(joint_idx_map)} parameterized joints "
          f"(expected {NUM_DESIGN_PARAMS * 2})")
    return base_quats_dict, joint_idx_map


def update_training_joint_X_p(model, theta_np, base_quats_dict, joint_idx_map,
                               num_worlds):
    joints_per_world = model.joint_count // num_worlds
    joint_X_p_np = model.joint_X_p.numpy()
    for i, (left_body, right_body) in enumerate(SYMMETRIC_PAIRS):
        angle = float(theta_np[i])
        delta_q = quat_from_x_rotation(angle)
        for body_name in (left_body, right_body):
            if body_name not in joint_idx_map:
                continue
            base_q = base_quats_dict[body_name]
            new_q = quat_normalize(quat_multiply(delta_q, base_q))
            local_idx = joint_idx_map[body_name]
            for w in range(num_worlds):
                global_idx = w * joints_per_world + local_idx
                joint_X_p_np[global_idx, 3:7] = new_q
    model.joint_X_p.assign(joint_X_p_np)
    wp.synchronize()


# ---------------------------------------------------------------------------
# Running Mean/Std for observation normalization
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Welford's online algorithm for running mean/variance."""

    def __init__(self, shape, clip=5.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, batch):
        batch = np.asarray(batch, dtype=np.float64)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count *
                    batch_count / total_count) / total_count
        self.count = total_count

    def normalize(self, obs):
        obs_norm = (obs - self.mean.astype(np.float32)) / np.sqrt(
            self.var.astype(np.float32) + 1e-8)
        return np.clip(obs_norm, -self.clip, self.clip).astype(np.float32)


# ---------------------------------------------------------------------------
# LSTM Policy and Value networks
# ---------------------------------------------------------------------------

class G1Policy(nn.Module):
    """LSTM actor for G1 locomotion (matches G1RoughCfgPPO config)."""

    def __init__(self, obs_dim, act_dim, lstm_hidden=64, mlp_hidden=32):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lstm_hidden = lstm_hidden

        self.lstm = nn.LSTM(obs_dim, lstm_hidden, num_layers=1, batch_first=False)
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, act_dim),
        )
        # init_noise_std = 0.8 → log(0.8) ≈ -0.223
        self.log_std = nn.Parameter(torch.ones(act_dim) * math.log(0.8))

        # Small init for final layer
        nn.init.uniform_(self.mlp[-1].weight, -0.1, 0.1)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, obs_seq, hidden, masks=None):
        """Forward pass through LSTM + MLP.

        Args:
            obs_seq: (seq_len, batch, obs_dim)
            hidden:  tuple (h, c) each (1, batch, lstm_hidden)
            masks:   (seq_len, batch) — 1.0=continue, 0.0=reset hidden
        Returns:
            means: (seq_len, batch, act_dim)
            hidden: updated (h, c)
        """
        if masks is not None:
            outputs = []
            for t in range(obs_seq.size(0)):
                m = masks[t].unsqueeze(0).unsqueeze(-1)  # (1, B, 1)
                hidden = (hidden[0] * m, hidden[1] * m)
                out, hidden = self.lstm(obs_seq[t:t+1], hidden)
                outputs.append(out)
            lstm_out = torch.cat(outputs, dim=0)
        else:
            lstm_out, hidden = self.lstm(obs_seq, hidden)

        means = self.mlp(lstm_out)
        return means, hidden

    def init_hidden(self, batch_size, device):
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return (h, c)

    def get_deterministic_action(self, obs, hidden):
        """Single-step deterministic action (for action collection)."""
        obs_in = obs.unsqueeze(0)  # (1, N, obs_dim)
        means, hidden = self.forward(obs_in, hidden)
        return means.squeeze(0), hidden  # (N, act_dim)


class G1Value(nn.Module):
    """LSTM critic for G1 locomotion."""

    def __init__(self, obs_dim, lstm_hidden=64, mlp_hidden=32):
        super().__init__()
        self.lstm_hidden = lstm_hidden

        self.lstm = nn.LSTM(obs_dim, lstm_hidden, num_layers=1, batch_first=False)
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, obs_seq, hidden, masks=None):
        """Forward pass.

        Args:
            obs_seq: (seq_len, batch, obs_dim)
            hidden:  tuple (h, c)
            masks:   (seq_len, batch)
        Returns:
            values: (seq_len, batch, 1)
            hidden: updated (h, c)
        """
        if masks is not None:
            outputs = []
            for t in range(obs_seq.size(0)):
                m = masks[t].unsqueeze(0).unsqueeze(-1)
                hidden = (hidden[0] * m, hidden[1] * m)
                out, hidden = self.lstm(obs_seq[t:t+1], hidden)
                outputs.append(out)
            lstm_out = torch.cat(outputs, dim=0)
        else:
            lstm_out, hidden = self.lstm(obs_seq, hidden)

        return self.mlp(lstm_out), hidden

    def init_hidden(self, batch_size, device):
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return (h, c)


# ---------------------------------------------------------------------------
# PPO rollout collection (LSTM-aware)
# ---------------------------------------------------------------------------

def collect_rollout(walking_env, policy, value_net, obs_rms,
                    hidden_p, hidden_v, horizon=24, device="cuda:0"):
    """Collect H-step rollout from vectorized env with LSTM hidden state tracking.

    Args:
        walking_env: G1WalkingEnvWrapper instance
        policy, value_net: LSTM-based networks
        obs_rms: RunningMeanStd for observation normalization
        hidden_p, hidden_v: LSTM hidden states from previous rollout
        horizon: steps per rollout
        device: torch device

    Returns:
        rollout dict, updated (hidden_p, hidden_v)
    """
    N = walking_env.num_envs

    # Use last obs from env (continuing rollout) or reset
    if walking_env.last_obs is not None:
        obs = walking_env.last_obs.clone()
    else:
        obs, _ = walking_env.reset()

    # Save start hidden states for PPO update
    start_hp = (hidden_p[0].detach().clone(), hidden_p[1].detach().clone())
    start_hv = (hidden_v[0].detach().clone(), hidden_v[1].detach().clone())

    all_obs, all_actions, all_rewards = [], [], []
    all_log_probs, all_dones, all_values, all_masks = [], [], [], []

    for t in range(horizon):
        obs_np = obs.cpu().numpy()
        obs_rms.update(obs_np)
        obs_norm = torch.FloatTensor(obs_rms.normalize(obs_np)).to(device)

        with torch.no_grad():
            # Policy forward (single step)
            obs_in = obs_norm.unsqueeze(0)  # (1, N, obs_dim)
            means, hidden_p = policy(obs_in, hidden_p)
            means = means.squeeze(0)  # (N, act_dim)

            std = torch.exp(policy.log_std)
            dist = torch.distributions.Normal(means, std)
            actions = dist.sample()
            actions = torch.clamp(actions, -1.0, 1.0)
            log_probs = dist.log_prob(actions).sum(-1)

            # Value forward
            val_out, hidden_v = value_net(obs_in, hidden_v)
            values = val_out.squeeze(0).squeeze(-1)  # (N,)

        # Step env
        next_obs, rewards, dones, info = walking_env.step(actions)

        # Store data
        all_obs.append(obs_norm)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_log_probs.append(log_probs)
        all_dones.append(dones)
        all_values.append(values)

        # Masks for PPO LSTM replay: 0 where previous step was done
        if t == 0:
            all_masks.append(torch.ones(N, device=device))
        else:
            all_masks.append(1.0 - all_dones[t - 1])

        # Reset hidden states for done envs
        done_mask = (1.0 - dones).unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
        hidden_p = (hidden_p[0] * done_mask, hidden_p[1] * done_mask)
        hidden_v = (hidden_v[0] * done_mask, hidden_v[1] * done_mask)

        obs = next_obs

    # Bootstrap value
    with torch.no_grad():
        obs_np = obs.cpu().numpy()
        obs_norm = torch.FloatTensor(obs_rms.normalize(obs_np)).to(device)
        val_out, hidden_v = value_net(obs_norm.unsqueeze(0), hidden_v)
        last_value = val_out.squeeze(0).squeeze(-1)

    walking_env.last_obs = obs

    return {
        "observations": torch.stack(all_obs),      # (H, N, obs_dim)
        "actions": torch.stack(all_actions),         # (H, N, act_dim)
        "rewards": torch.stack(all_rewards),         # (H, N)
        "log_probs": torch.stack(all_log_probs),     # (H, N)
        "dones": torch.stack(all_dones),             # (H, N)
        "values": torch.stack(all_values),           # (H, N)
        "masks": torch.stack(all_masks),             # (H, N)
        "last_value": last_value,                    # (N,)
        "start_hidden_p": start_hp,
        "start_hidden_v": start_hv,
    }, hidden_p, hidden_v


# ---------------------------------------------------------------------------
# PPO update (LSTM sequence-aware, mini-batches over environments)
# ---------------------------------------------------------------------------

def ppo_update(policy, value_net, optimizer, rollout,
               n_epochs=5, clip_ratio=0.2, gamma=0.99, gae_lambda=0.95,
               value_coeff=0.5, entropy_coeff=0.01,
               num_mini_batches=4, desired_kl=0.01, max_grad_norm=1.0,
               device="cuda:0"):
    """PPO update with GAE, LSTM sequence handling, adaptive LR."""

    H, N = rollout["rewards"].shape
    rewards = rollout["rewards"]
    dones = rollout["dones"]
    values = rollout["values"]
    last_value = rollout["last_value"]

    # --- GAE advantages ---
    with torch.no_grad():
        advantages = torch.zeros(H, N, device=device)
        last_gae = torch.zeros(N, device=device)
        for t in reversed(range(H)):
            next_val = last_value if t == H - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + values

    # Normalize advantages
    adv_flat = advantages.reshape(-1)
    advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    all_params = list(policy.parameters()) + list(value_net.parameters())
    mini_batch_size = max(1, N // num_mini_batches)
    mean_kl = 0.0

    for epoch in range(n_epochs):
        env_perm = torch.randperm(N, device=device)

        for mb in range(num_mini_batches):
            start = mb * mini_batch_size
            end = min(start + mini_batch_size, N)
            mb_inds = env_perm[start:end]
            mb_size = len(mb_inds)

            obs_mb = rollout["observations"][:, mb_inds]       # (H, mb, obs_dim)
            acts_mb = rollout["actions"][:, mb_inds]            # (H, mb, act_dim)
            old_lp_mb = rollout["log_probs"][:, mb_inds]        # (H, mb)
            adv_mb = advantages[:, mb_inds]                     # (H, mb)
            ret_mb = returns[:, mb_inds]                        # (H, mb)
            masks_mb = rollout["masks"][:, mb_inds]             # (H, mb)

            # Policy forward (sequential LSTM)
            hp = (rollout["start_hidden_p"][0][:, mb_inds].contiguous(),
                  rollout["start_hidden_p"][1][:, mb_inds].contiguous())
            means, _ = policy(obs_mb, hp, masks=masks_mb)       # (H, mb, act)
            std = torch.exp(policy.log_std)
            dist = torch.distributions.Normal(means, std)
            log_probs = dist.log_prob(acts_mb).sum(-1)           # (H, mb)
            entropy = dist.entropy().sum(-1).mean()

            # Value forward (sequential LSTM)
            hv = (rollout["start_hidden_v"][0][:, mb_inds].contiguous(),
                  rollout["start_hidden_v"][1][:, mb_inds].contiguous())
            val_pred, _ = value_net(obs_mb, hv, masks=masks_mb)  # (H, mb, 1)
            val_pred = val_pred.squeeze(-1)                      # (H, mb)

            # PPO losses
            ratio = torch.exp(log_probs - old_lp_mb)
            clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            policy_loss = -torch.min(ratio * adv_mb, clipped * adv_mb).mean()
            value_loss = F.mse_loss(val_pred, ret_mb)
            loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
            optimizer.step()

        # Approximate KL after each epoch
        with torch.no_grad():
            hp_full = (rollout["start_hidden_p"][0].contiguous(),
                       rollout["start_hidden_p"][1].contiguous())
            masks_full = rollout["masks"]
            means_all, _ = policy(rollout["observations"], hp_full, masks=masks_full)
            std_all = torch.exp(policy.log_std)
            dist_all = torch.distributions.Normal(means_all, std_all)
            new_lp = dist_all.log_prob(rollout["actions"]).sum(-1)
            mean_kl = (rollout["log_probs"] - new_lp).mean().item()

        if mean_kl > 2.0 * desired_kl:
            break

    # Adaptive LR (RSL-RL style)
    current_lr = optimizer.param_groups[0]['lr']
    if mean_kl > 2.0 * desired_kl:
        new_lr = max(current_lr / 1.5, 1e-5)
    elif mean_kl < desired_kl / 2.0:
        new_lr = min(current_lr * 1.5, 1e-2)
    else:
        new_lr = current_lr
    for pg in optimizer.param_groups:
        pg['lr'] = new_lr

    return mean_kl


# ---------------------------------------------------------------------------
# StabilityGate (convergence detection for inner loop)
# ---------------------------------------------------------------------------

class StabilityGate:
    """Convergence detection: reward plateau over a window."""

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
        relative_change = (np.max(rewards) - np.min(rewards)) / abs(mean_val)
        return relative_change < self.rel_threshold

    def is_converged(self):
        if self.total_inner_iters < self.min_inner_iters:
            return False
        return self.stable_count >= self.stable_iters_required


# ---------------------------------------------------------------------------
# PPO Inner Loop Controller
# ---------------------------------------------------------------------------

class PPOInnerLoop:
    """Drives PPO training with convergence detection."""

    def __init__(self, walking_env, policy, value_net, obs_rms,
                 device="cuda:0",
                 horizon=24, n_epochs=5, num_mini_batches=4,
                 lr=1e-3, gamma=0.99, gae_lambda=0.95,
                 clip_ratio=0.2, entropy_coeff=0.01,
                 desired_kl=0.01, max_grad_norm=1.0,
                 rel_threshold=0.02, min_inner_iters=500,
                 stable_iters_required=50, max_samples=200_000_000,
                 use_wandb=False, log_every=10):
        self.walking_env = walking_env
        self.policy = policy
        self.value_net = value_net
        self.obs_rms = obs_rms
        self.device = device

        self.horizon = horizon
        self.n_epochs = n_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.desired_kl = desired_kl
        self.max_grad_norm = max_grad_norm
        self.max_samples = max_samples
        self.use_wandb = use_wandb
        self.log_every = log_every

        self.optimizer = optim.Adam(
            list(policy.parameters()) + list(value_net.parameters()), lr=lr,
        )
        self.gate = StabilityGate(
            rel_threshold=rel_threshold,
            min_inner_iters=min_inner_iters,
            stable_iters_required=stable_iters_required,
        )

    def train_until_converged(self, out_dir):
        """Run PPO inner loop until convergence or sample cap.

        Returns (converged: bool, reward_history: list[float]).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        env = self.walking_env
        policy = self.policy
        value_net = self.value_net
        N = env.num_envs

        # Reset env and LSTM hidden states
        env.last_obs = None
        obs, _ = env.reset()
        env.last_obs = obs
        hidden_p = policy.init_hidden(N, self.device)
        hidden_v = value_net.init_hidden(N, self.device)

        policy.train()
        value_net.train()
        self.gate.reset()

        reward_buffer = deque(maxlen=200)
        length_buffer = deque(maxlen=200)
        reward_history = []

        sample_count = 0
        inner_iter = 0
        mean_rew, std_rew, mean_len = 0.0, 0.0, 0.0
        start_time = time.time()

        while sample_count < self.max_samples:
            # Collect rollout
            rollout, hidden_p, hidden_v = collect_rollout(
                env, policy, value_net, self.obs_rms,
                hidden_p, hidden_v,
                horizon=self.horizon, device=self.device,
            )

            # PPO update
            mean_kl = ppo_update(
                policy, value_net, self.optimizer, rollout,
                n_epochs=self.n_epochs, clip_ratio=self.clip_ratio,
                gamma=self.gamma, gae_lambda=self.gae_lambda,
                entropy_coeff=self.entropy_coeff,
                num_mini_batches=self.num_mini_batches,
                desired_kl=self.desired_kl, max_grad_norm=self.max_grad_norm,
                device=self.device,
            )

            sample_count += N * self.horizon
            inner_iter += 1
            current_lr = self.optimizer.param_groups[0]['lr']

            # Track completed episodes
            ep_rews, ep_lens = env.pop_completed_episodes()
            reward_buffer.extend(ep_rews)
            length_buffer.extend(ep_lens)

            if len(reward_buffer) > 0:
                mean_rew = np.mean(reward_buffer)
                std_rew = np.std(reward_buffer)
                mean_len = np.mean(length_buffer)

            # Stability gate
            if inner_iter % self.log_every == 0 and len(reward_buffer) > 0:
                self.gate.update(mean_rew)
                self.gate.tick(self.log_every)
                reward_history.append(mean_rew)

            # Logging
            if inner_iter % self.log_every == 0:
                rollout_mean = rollout["rewards"].mean().item()
                print(f"    PPO iter {inner_iter}: "
                      f"rew/step={rollout_mean:.3f}, kl={mean_kl:.4f}, "
                      f"lr={current_lr:.1e}, ep_rew={mean_rew:.1f}±{std_rew:.1f}, "
                      f"ep_len={mean_len:.0f}, samples={sample_count:,}")

                if self.use_wandb:
                    wandb.log({
                        "inner/reward_per_step": rollout_mean,
                        "inner/kl": mean_kl,
                        "inner/lr": current_lr,
                        "inner/mean_reward": mean_rew,
                        "inner/reward_std": std_rew,
                        "inner/episode_length": mean_len,
                        "inner/iteration": inner_iter,
                        "inner/samples": sample_count,
                    })

            if self.gate.is_converged():
                print(f"    CONVERGED at iter {inner_iter} "
                      f"(stable for {self.gate.stable_count} iters)")
                break

        elapsed = time.time() - start_time

        # Save checkpoint
        ckpt = {
            "policy": policy.state_dict(),
            "value_net": value_net.state_dict(),
            "obs_rms_mean": self.obs_rms.mean,
            "obs_rms_var": self.obs_rms.var,
            "obs_rms_count": self.obs_rms.count,
        }
        torch.save(ckpt, str(out_dir / "ppo_checkpoint.pt"))

        converged = self.gate.is_converged()
        print(f"    Inner loop done: {inner_iter} iters, {sample_count:,} samples, "
              f"{elapsed / 60:.1f} min, converged={converged}")

        return converged, reward_history


# ---------------------------------------------------------------------------
# Action collection from frozen PPO policy (for BPTT replay)
# ---------------------------------------------------------------------------

def collect_actions_from_ppo(walking_env, policy, obs_rms,
                             num_eval_worlds, horizon, device="cuda:0"):
    """Collect deterministic actions from frozen PPO policy.

    Runs the full training env but only keeps first num_eval_worlds actions.
    Returns list of (num_eval_worlds, act_dim) numpy arrays — one per step,
    in the same format DiffG1Eval expects (target joint positions).
    """
    policy.eval()
    env = walking_env
    N = env.num_envs

    # Reset and get fresh obs
    obs, _ = env.reset()
    hidden = policy.init_hidden(N, device)

    action_scale = env.cfg.action_scale
    default_dof = env.default_dof_pos.unsqueeze(0)  # (1, D)

    actions_list = []

    with torch.no_grad():
        for _ in range(horizon):
            obs_np = obs.cpu().numpy()
            obs_norm = torch.FloatTensor(obs_rms.normalize(obs_np)).to(device)

            # Deterministic action
            action_mean, hidden = policy.get_deterministic_action(obs_norm, hidden)
            action = torch.clamp(action_mean, -1.0, 1.0)

            # Scale to target positions (what engine.set_cmd receives)
            target = action * action_scale + default_dof
            actions_list.append(
                target[:num_eval_worlds].cpu().numpy().copy()
            )

            obs, reward, done, info = env.step(action)

            # Reset hidden for done envs
            done_mask = (1.0 - done).unsqueeze(0).unsqueeze(-1)
            hidden = (hidden[0] * done_mask, hidden[1] * done_mask)

    policy.train()
    return actions_list


# ---------------------------------------------------------------------------
# Main PGHC loop
# ---------------------------------------------------------------------------

def pghc_codesign_g1_ppo(args):
    """PGHC co-design for G1 humanoid with PPO inner loop."""

    print("=" * 70)
    print("PGHC Co-Design for G1 Humanoid (PPO Inner Loop)")
    print("=" * 70)

    device = "cuda:0"
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- wandb ---
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project="pghc-codesign",
            name=f"g1-ppo-{args.num_train_envs}env",
            config=vars(args),
        )
    elif args.wandb:
        print("  [wandb] Not available, continuing without")

    # =========================================================
    # One-time initialization
    # =========================================================
    print("\n[1/4] Initializing MimicKit (for Newton engine)...")
    try:
        mp_util.init(0, 1, device, np.random.randint(6000, 7000))
    except (AssertionError, Exception):
        pass

    # Generate initial MJCF (theta=0 = unmodified G1)
    theta = np.zeros(NUM_DESIGN_PARAMS, dtype=np.float64)
    mjcf_modifier = G1MJCFModifier(str(BASE_MJCF_PATH))
    modified_mjcf = out_dir / "g1_modified.xml"
    mjcf_modifier.generate(theta, str(modified_mjcf))

    # Mesh directory symlink
    mesh_src = BASE_MJCF_PATH.parent / "meshes"
    mesh_dst = out_dir / "meshes"
    if not mesh_dst.exists():
        try:
            mesh_dst.symlink_to(mesh_src)
        except (OSError, NotImplementedError):
            shutil.copytree(str(mesh_src), str(mesh_dst))

    # Env config pointing to modified MJCF
    modified_env_config = out_dir / "env_config.yaml"
    mjcf_modifier.generate_env_config(
        str(modified_mjcf), str(BASE_ENV_CONFIG), str(modified_env_config),
    )

    print("[2/4] Building training env...")
    base_env = env_builder.build_env(
        str(modified_env_config), str(BASE_ENGINE_CONFIG),
        args.num_train_envs, device, visualize=False,
    )

    # Wrap with walking env
    walking_env = G1WalkingEnvWrapper(base_env, device)

    obs_dim = walking_env.obs_dim
    act_dim = walking_env.act_dim
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}")

    # Build policy / value / obs normalizer
    policy = G1Policy(obs_dim, act_dim, lstm_hidden=64, mlp_hidden=32).to(device)
    value_net = G1Value(obs_dim, lstm_hidden=64, mlp_hidden=32).to(device)
    obs_rms = RunningMeanStd(shape=(obs_dim,))

    # Extract joint info from training Newton model
    engine = base_env._engine
    sim_model = engine._sim_model
    base_quats_dict, joint_idx_map = extract_joint_info(
        sim_model, args.num_train_envs,
    )

    diff_eval = None
    print("[3/4] DiffG1Eval will be built lazily before each BPTT phase")

    # =========================================================
    # Outer loop setup
    # =========================================================
    print("[4/4] Starting PGHC outer loop...\n")

    design_optimizer = AdamOptimizer(NUM_DESIGN_PARAMS, lr=args.design_lr)
    theta_bounds = (-0.5236, 0.5236)  # +/-30 deg

    param_names = [
        f"theta_{i}_{SYMMETRIC_PAIRS[i][0].replace('_link', '')}"
        for i in range(NUM_DESIGN_PARAMS)
    ]

    ppo_inner = PPOInnerLoop(
        walking_env, policy, value_net, obs_rms,
        device=device,
        horizon=args.ppo_horizon,
        n_epochs=args.ppo_epochs,
        num_mini_batches=args.ppo_mini_batches,
        lr=args.ppo_lr,
        gamma=0.99, gae_lambda=0.95,
        clip_ratio=0.2, entropy_coeff=0.01,
        desired_kl=0.01, max_grad_norm=1.0,
        rel_threshold=args.plateau_threshold,
        min_inner_iters=args.min_plateau_iters,
        stable_iters_required=args.plateau_window,
        max_samples=args.max_inner_samples,
        use_wandb=use_wandb,
        log_every=10,
    )

    history = {
        "theta": [theta.copy()],
        "forward_dist": [],
        "cot": [],
        "gradients": [],
        "inner_times": [],
    }
    theta_history = deque(maxlen=5)

    print(f"Configuration:")
    print(f"  Training envs:     {args.num_train_envs}")
    print(f"  Eval worlds:       {args.num_eval_worlds}")
    print(f"  Eval horizon:      {args.eval_horizon} steps "
          f"({args.eval_horizon / 30:.1f}s)")
    print(f"  PPO horizon:       {args.ppo_horizon} steps")
    print(f"  PPO LR:            {args.ppo_lr}")
    print(f"  Max inner samples: {args.max_inner_samples:,}")
    print(f"  Design optimizer:  Adam (lr={args.design_lr})")
    print(f"  Design params:     {NUM_DESIGN_PARAMS} (symmetric lower-body)")

    # =========================================================
    # Outer loop (structure IDENTICAL to codesign_g1_unified.py)
    # =========================================================
    for outer_iter in range(args.outer_iters):
        print(f"\n{'=' * 70}")
        print(f"Outer Iteration {outer_iter + 1}/{args.outer_iters}")
        print(f"{'=' * 70}")

        theta_deg = np.degrees(theta)
        for i, name in enumerate(param_names):
            print(f"  {name}: {theta[i]:+.4f} rad ({theta_deg[i]:+.2f} deg)")

        iter_dir = out_dir / f"outer_{outer_iter:03d}"

        # ----- Inner Loop (PPO) -----
        print(f"\n  [Inner Loop] PPO training ({args.num_train_envs} envs)...")
        t0 = time.time()

        try:
            converged, rew_hist = ppo_inner.train_until_converged(iter_dir)
            inner_time = time.time() - t0
            print(f"  [Inner Loop] Done in {inner_time / 60:.1f} min")
        except Exception as e:
            print(f"  [Inner Loop] FAILED: {e}")
            import traceback
            traceback.print_exc()
            history["inner_times"].append(time.time() - t0)
            continue

        history["inner_times"].append(inner_time)

        # ----- Collect Actions -----
        print(f"\n  [Actions] Collecting ({args.num_eval_worlds} worlds x "
              f"{args.eval_horizon} steps)...")

        actions_list = collect_actions_from_ppo(
            walking_env, policy, obs_rms,
            args.num_eval_worlds, args.eval_horizon, device,
        )
        print(f"    Got {len(actions_list)} steps x {actions_list[0].shape}")

        # ----- BPTT Gradient -----
        print(f"\n  [BPTT] Building DiffG1Eval lazily...")

        wp.config.enable_backward = True

        diff_eval = DiffG1Eval(
            mjcf_modifier=mjcf_modifier,
            theta_np=theta.copy(),
            num_worlds=args.num_eval_worlds,
            horizon=args.eval_horizon,
            dt=1.0 / 30.0,
            num_substeps=8,
            device=device,
        )

        print(f"  [BPTT] Computing gradient...")
        grad_theta, fwd_dist, cot = diff_eval.compute_gradient(actions_list)

        # Free DiffG1Eval
        diff_eval.cleanup()
        diff_eval = None
        gc.collect()
        torch.cuda.empty_cache()
        wp.config.enable_backward = False
        print(f"  [BPTT] DiffG1Eval freed, VRAM reclaimed")

        print(f"    BPTT gradients:")
        for i, name in enumerate(param_names):
            print(f"      d_reward/d_{name} = {grad_theta[i]:+.6f}")
        print(f"    Forward distance = {fwd_dist:.3f} m")
        print(f"    Cost of Transport = {cot:.4f}")

        history["forward_dist"].append(fwd_dist)
        history["cot"].append(cot)
        history["gradients"].append(grad_theta.copy())

        # ----- Design Update -----
        old_theta = theta.copy()
        theta = design_optimizer.step(theta, grad_theta)
        theta = np.clip(theta, theta_bounds[0], theta_bounds[1])

        print(f"\n  Design update:")
        for i, name in enumerate(param_names):
            delta = theta[i] - old_theta[i]
            print(f"    {name}: {old_theta[i]:+.4f} -> {theta[i]:+.4f} "
                  f"(delta={delta:+.5f}, {np.degrees(delta):+.3f} deg)")

        history["theta"].append(theta.copy())

        # Update training model for next outer iteration
        update_training_joint_X_p(
            sim_model, theta, base_quats_dict, joint_idx_map,
            args.num_train_envs,
        )

        # Also regenerate MJCF and env config for DiffG1Eval
        mjcf_modifier.generate(theta, str(modified_mjcf))
        mjcf_modifier.generate_env_config(
            str(modified_mjcf), str(BASE_ENV_CONFIG), str(modified_env_config),
        )

        # Save checkpoints
        np.save(str(out_dir / "theta_latest.npy"), theta)
        np.save(str(iter_dir / "theta.npy"), theta)
        np.save(str(iter_dir / "grad.npy"), grad_theta)

        # wandb
        if use_wandb:
            log_dict = {
                "outer/iteration": outer_iter + 1,
                "outer/eval_forward_distance": fwd_dist,
                "outer/cot": cot,
                "outer/inner_time_min": inner_time / 60.0,
                "outer/grad_norm": np.linalg.norm(grad_theta),
            }
            for i, name in enumerate(param_names):
                log_dict[f"outer/{name}_rad"] = theta[i]
                log_dict[f"outer/{name}_deg"] = np.degrees(theta[i])
                log_dict[f"outer/grad_{name}"] = grad_theta[i]
            if rew_hist:
                log_dict["outer/final_ppo_reward"] = rew_hist[-1]
            wandb.log(log_dict)

        # Outer convergence check
        theta_history.append(theta.copy())
        if len(theta_history) >= 5:
            theta_stack = np.array(list(theta_history))
            ranges = theta_stack.max(axis=0) - theta_stack.min(axis=0)
            max_range = ranges.max()
            if max_range < np.radians(0.5):
                print(f"\n  OUTER CONVERGED: theta stable "
                      f"(max range = {np.degrees(max_range):.3f} deg)")
                break

    # =========================================================
    # Final results
    # =========================================================
    print("\n" + "=" * 70)
    print("PGHC Co-Design Complete!")
    print("=" * 70)

    initial = history["theta"][0]
    final = history["theta"][-1]
    for i, name in enumerate(param_names):
        print(f"  {name}: {initial[i]:+.4f} -> {final[i]:+.4f} "
              f"({np.degrees(initial[i]):+.2f} -> "
              f"{np.degrees(final[i]):+.2f} deg)")

    if history["forward_dist"]:
        print(f"\nForward distance: {history['forward_dist'][0]:.3f} -> "
              f"{history['forward_dist'][-1]:.3f} m")
    if history["cot"]:
        print(f"Cost of Transport: {history['cot'][0]:.4f} -> "
              f"{history['cot'][-1]:.4f}")

    total_time = sum(history["inner_times"])
    print(f"Total inner loop time: {total_time / 3600:.1f} hours")

    if diff_eval is not None:
        diff_eval.cleanup()
    if use_wandb:
        wandb.finish()

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PGHC Co-Design for G1 Humanoid (PPO Inner Loop)"
    )
    # Outer loop
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--outer-iters", type=int, default=20)
    parser.add_argument("--design-lr", type=float, default=0.005)
    parser.add_argument("--num-train-envs", type=int, default=4096)
    parser.add_argument("--num-eval-worlds", type=int, default=32)
    parser.add_argument("--eval-horizon", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="output_g1_ppo")

    # PPO inner loop
    parser.add_argument("--ppo-lr", type=float, default=1e-3)
    parser.add_argument("--ppo-horizon", type=int, default=24)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--ppo-mini-batches", type=int, default=4)
    parser.add_argument("--max-inner-samples", type=int, default=200_000_000)

    # Convergence
    parser.add_argument("--plateau-threshold", type=float, default=0.02)
    parser.add_argument("--plateau-window", type=int, default=50)
    parser.add_argument("--min-plateau-iters", type=int, default=500)

    args = parser.parse_args()

    pghc_codesign_g1_ppo(args)
