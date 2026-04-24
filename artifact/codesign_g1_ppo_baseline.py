#!/usr/bin/env python3
"""
GBC (Gradient-Based Co-Design) for G1 Humanoid — PPO Baseline

Pure PPO inner loop with IsaacLab-style shaped reward and domain randomization.
Outer loop (orthogonal SPSA + Cautious BFGS) is identical to codesign_g1_unified.py.

This serves as a controlled baseline:
  - Same morphology parameterization (6 symmetric lower-body joint angles)
  - Same outer loop gradient estimation and optimizer
  - Different inner loop: pure PPO + shaped reward (no AMP discriminator)
  - Domain randomization for policy robustness across morphology changes

Run:
    # Single GPU
    python codesign_g1_ppo_baseline.py --num-train-envs 4096

    # Multi-GPU (4x)
    python codesign_g1_ppo_baseline.py --num-train-envs 16380 \
        --devices cuda:0 cuda:1 cuda:2 cuda:3
"""

import os
os.environ["PYGLET_HEADLESS"] = "1"

# ---------------------------------------------------------------------------
# Multi-GPU device isolation (identical to codesign_g1_unified.py)
# ---------------------------------------------------------------------------
_GBC_GPU_INDEX = os.environ.pop("_GBC_GPU_INDEX", None)
if _GBC_GPU_INDEX is not None:
    _orig_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if _orig_cvd:
        _gpu_list = [g.strip() for g in _orig_cvd.split(",")]
        _idx = int(_GBC_GPU_INDEX)
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            _gpu_list[_idx] if _idx < len(_gpu_list) else _GBC_GPU_INDEX
        )
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _GBC_GPU_INDEX

import argparse
import math
import platform
import random
import shutil
import sys
import time
import types
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
# MimicKit + GPU imports (env_builder only — no agent_builder)
# ---------------------------------------------------------------------------
if str(MIMICKIT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(MIMICKIT_SRC_DIR))

import warp as wp

if _GBC_GPU_INDEX is not None:
    _warp_cache = os.path.join(
        os.path.expanduser("~"), ".cache", "warp_per_rank", f"rank_{_GBC_GPU_INDEX}"
    )
    os.makedirs(_warp_cache, exist_ok=True)
    wp.config.kernel_cache_dir = _warp_cache

import newton  # noqa: F401
import util.mp_util as mp_util
import envs.env_builder as env_builder

from g1_mjcf_modifier import (
    G1MJCFModifier, SYMMETRIC_PAIRS, NUM_DESIGN_PARAMS,
    quat_from_x_rotation, quat_multiply, quat_normalize,
)
from g1_ppo_env import G1WalkingEnvWrapper, G1WalkingConfig
from convergence_gate import CompositeConvergenceGate

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------
GRAVITY = 9.81
VELOCITY_EPSILON = 0.01
THETA_BOUNDS = (-np.radians(30), np.radians(30))


def compute_cot(mean_power, mean_fwd_vel, total_mass):
    """Cost of Transport: power / (m * g * |v|), with velocity smoothing."""
    safe_v = torch.sqrt(mean_fwd_vel ** 2 + VELOCITY_EPSILON)
    return mean_power / (total_mass * GRAVITY * safe_v)


# ---------------------------------------------------------------------------
# Episode-averaged CoT tracker (identical to codesign_g1_unified.py)
# ---------------------------------------------------------------------------

class EpisodeCoTTracker:
    """Tracks per-episode Cost of Transport (all joints)."""

    @staticmethod
    def get_dof_forces_safe(engine, char_id):
        raw = engine._dof_forces
        if isinstance(raw, torch.Tensor):
            if raw.dim() == 3:
                return torch.sum(raw, dim=-1)
            return raw
        return raw[char_id]

    def __init__(self, num_envs, total_mass, device):
        self.total_mass = total_mass
        self.power_sum = torch.zeros(num_envs, device=device)
        self.vel_sum = torch.zeros(num_envs, device=device)
        self.step_count = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.episode_cots = []
        self.episode_powers = []
        self.episode_vels = []

    def reset_accumulators(self):
        self.power_sum.zero_()
        self.vel_sum.zero_()
        self.step_count.zero_()

    def accumulate(self, engine, char_id):
        dof_forces = self.get_dof_forces_safe(engine, char_id)
        dof_vel = engine.get_dof_vel(char_id)
        root_vel = engine.get_root_vel(char_id)
        power = torch.sum(torch.abs(dof_forces * dof_vel), dim=-1)
        self.power_sum += power
        self.vel_sum += root_vel[:, 0]
        self.step_count += 1

    def finalize_episodes(self, done_buf):
        done_mask = done_buf > 0
        if not done_mask.any():
            return
        done_idx = done_mask.nonzero(as_tuple=True)[0]
        steps = self.step_count[done_idx].float()
        valid = steps > 0
        if valid.any():
            vi = done_idx[valid]
            vs = steps[valid]
            mp = self.power_sum[vi] / vs
            mv = self.vel_sum[vi] / vs
            cots = compute_cot(mp, mv, self.total_mass)
            self.episode_cots.extend(cots.tolist())
            self.episode_powers.extend(mp.tolist())
            self.episode_vels.extend(mv.tolist())
        self.power_sum[done_idx] = 0
        self.vel_sum[done_idx] = 0
        self.step_count[done_idx] = 0

    def get_stats(self):
        if not self.episode_cots:
            return None, None, None, 0
        n = len(self.episode_cots)
        cot = sum(self.episode_cots) / n
        pwr = sum(self.episode_powers) / n
        vel = sum(self.episode_vels) / n
        self.episode_cots.clear()
        self.episode_powers.clear()
        self.episode_vels.clear()
        return cot, pwr, vel, n


# ---------------------------------------------------------------------------
# Cautious BFGS (identical to codesign_g1_unified.py)
# ---------------------------------------------------------------------------

class CautiousBFGS:
    """Cautious BFGS for morphology optimization (gradient ascent)."""

    COND_RESET_THRESHOLD = 1e6

    def __init__(self, n_params):
        self.n = n_params
        self.H = np.eye(n_params, dtype=np.float64)
        self.prev_grad = None
        self.initialized = False
        self.num_updates = 0
        self.num_skipped = 0

    def compute_direction(self, grad):
        return self.H @ grad

    def update(self, actual_s, grad_new):
        if self.prev_grad is not None:
            y = self.prev_grad - grad_new
            sTy = actual_s @ y
            if sTy > 1e-10:
                if not self.initialized:
                    yTy = y @ y
                    if yTy > 1e-20:
                        gamma = sTy / yTy
                        self.H = gamma * np.eye(self.n, dtype=np.float64)
                    self.initialized = True
                rho = 1.0 / sTy
                I_n = np.eye(self.n, dtype=np.float64)
                V = I_n - rho * np.outer(actual_s, y)
                W = I_n - rho * np.outer(y, actual_s)
                self.H = V @ self.H @ W + rho * np.outer(actual_s, actual_s)
                self.num_updates += 1
                cond = np.linalg.cond(self.H)
                if cond > self.COND_RESET_THRESHOLD:
                    print(f"    [BFGS] Condition number {cond:.1e} — resetting H")
                    self.H = np.eye(self.n, dtype=np.float64)
                    self.initialized = False
            else:
                self.num_skipped += 1
        self.prev_grad = grad_new.copy()


# ---------------------------------------------------------------------------
# Joint info extraction + in-place update (identical to codesign_g1_unified.py)
# ---------------------------------------------------------------------------

def extract_joint_info(model, num_worlds):
    joint_keys = model.joint_key
    joints_per_world = model.joint_count // num_worlds
    joint_name_to_idx = {joint_keys[i]: i for i in range(joints_per_world)}
    joint_X_p_np = model.joint_X_p.numpy()
    base_quats_dict = {}
    joint_idx_map = {}
    for left_body, right_body in SYMMETRIC_PAIRS:
        for body_name in (left_body, right_body):
            joint_name = body_name.replace("_link", "_joint")
            if joint_name in joint_name_to_idx:
                idx = joint_name_to_idx[joint_name]
            else:
                print(f"  [WARN] Joint not found: {joint_name}")
                continue
            raw = joint_X_p_np[idx, 3:7].tolist()
            base_quats_dict[body_name] = (raw[3], raw[0], raw[1], raw[2])
            joint_idx_map[body_name] = idx
    print(f"  [JointInfo] Found {len(joint_idx_map)} parameterized joints "
          f"(expected {NUM_DESIGN_PARAMS * 2})")
    return base_quats_dict, joint_idx_map


def update_training_joint_X_p(model, theta_np, base_quats_dict, joint_idx_map,
                               num_worlds):
    joints_per_world = model.joint_count // num_worlds
    joint_X_p_np = model.joint_X_p.numpy()
    world_indices = np.arange(num_worlds)
    for i, (left_body, right_body) in enumerate(SYMMETRIC_PAIRS):
        angle = float(theta_np[i])
        delta_q = quat_from_x_rotation(angle)
        for body_name in (left_body, right_body):
            if body_name not in joint_idx_map:
                continue
            base_q = base_quats_dict[body_name]
            new_q = quat_normalize(quat_multiply(delta_q, base_q))
            local_idx = joint_idx_map[body_name]
            global_indices = world_indices * joints_per_world + local_idx
            joint_X_p_np[global_indices, 3:7] = [new_q[1], new_q[2], new_q[3], new_q[0]]
    model.joint_X_p.assign(joint_X_p_np)
    wp.synchronize()


# ---------------------------------------------------------------------------
# MLP Actor-Critic (replaces LSTM for fair comparison with MimicKit MLP)
# ---------------------------------------------------------------------------

class G1ActorCritic(nn.Module):
    """MLP Actor-Critic for G1 locomotion.

    Architecture matches MimicKit's fc_2layers_1024units for fair comparison.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes=(1024, 1024),
                 log_std_init=-1.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Actor
        actor_layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            actor_layers.extend([nn.Linear(in_dim, h), nn.ELU()])
            in_dim = h
        actor_layers.append(nn.Linear(in_dim, act_dim))
        self.actor = nn.Sequential(*actor_layers)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

        # Critic
        critic_layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            critic_layers.extend([nn.Linear(in_dim, h), nn.ELU()])
            in_dim = h
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Small init for output layers
        nn.init.uniform_(self.actor[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.uniform_(self.critic[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.critic[-1].bias)

    def get_action(self, obs, deterministic=False):
        """Returns (action, log_prob, value)."""
        mean = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        if deterministic:
            return torch.clamp(mean, -1.0, 1.0), torch.zeros(obs.shape[0], device=obs.device), value
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        """Returns (log_prob, value, entropy) for PPO update."""
        mean = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1).mean()
        return log_prob, value, entropy


# ---------------------------------------------------------------------------
# GPU-resident running mean/std for observation normalization
# ---------------------------------------------------------------------------

class RunningMeanStdCUDA:
    """Welford's online algorithm on CUDA tensors."""

    def __init__(self, shape, device, clip=5.0):
        self.mean = torch.zeros(shape, device=device, dtype=torch.float64)
        self.var = torch.ones(shape, device=device, dtype=torch.float64)
        self.count = 1e-4
        self.clip = clip
        self.device = device

    def update(self, batch):
        batch = batch.to(dtype=torch.float64)
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count *
                    batch_count / total_count) / total_count
        self.count = total_count

    def normalize(self, obs):
        return torch.clamp(
            (obs.float() - self.mean.float()) / torch.sqrt(self.var.float() + 1e-8),
            -self.clip, self.clip,
        )

    def sync_across_ranks(self, num_procs):
        """All-reduce normalizer stats across GPU ranks."""
        if num_procs <= 1:
            return
        n = self.mean.shape[0]
        # Flatten into a single 1-D buffer: [mean (n), var (n), count (1)]
        buf = torch.cat([
            self.mean,
            self.var,
            torch.tensor([self.count], device=self.device, dtype=torch.float64),
        ])
        torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
        buf /= num_procs
        self.mean = buf[:n]
        self.var = buf[n:2*n]
        self.count = buf[2*n].item()


# ---------------------------------------------------------------------------
# PPO rollout collection (MLP, no LSTM hidden states)
# ---------------------------------------------------------------------------

def collect_rollout(walking_env, model, obs_rms, horizon, device):
    """Collect H-step rollout. GPU-resident, no CPU roundtrips."""
    N = walking_env.num_envs

    if walking_env.last_obs is not None:
        obs = walking_env.last_obs.clone()
    else:
        obs, _ = walking_env.reset()

    all_obs, all_actions, all_rewards = [], [], []
    all_log_probs, all_dones, all_values = [], [], []

    for t in range(horizon):
        obs_rms.update(obs)
        obs_norm = obs_rms.normalize(obs)

        with torch.no_grad():
            action, log_prob, value = model.get_action(obs_norm)

        next_obs, rewards, dones, info = walking_env.step(action)

        all_obs.append(obs_norm)
        all_actions.append(action)
        all_rewards.append(rewards)
        all_log_probs.append(log_prob)
        all_dones.append(dones)
        all_values.append(value)

        obs = next_obs

    # Bootstrap value
    with torch.no_grad():
        obs_norm = obs_rms.normalize(obs)
        _, _, last_value = model.get_action(obs_norm)

    walking_env.last_obs = obs

    return {
        "observations": torch.stack(all_obs),
        "actions": torch.stack(all_actions),
        "rewards": torch.stack(all_rewards),
        "log_probs": torch.stack(all_log_probs),
        "dones": torch.stack(all_dones),
        "values": torch.stack(all_values),
        "last_value": last_value,
    }


# ---------------------------------------------------------------------------
# PPO update (MLP, multi-GPU gradient sync)
# ---------------------------------------------------------------------------

def ppo_update(model, optimizer, rollout, n_epochs=5, clip_ratio=0.2,
               gamma=0.99, gae_lambda=0.95, value_coeff=0.5,
               entropy_coeff=0.01, num_mini_batches=4, desired_kl=0.01,
               max_grad_norm=1.0, num_procs=1, device="cuda:0"):
    """PPO update with GAE, adaptive LR, multi-GPU gradient averaging."""

    H, N = rollout["rewards"].shape
    rewards = rollout["rewards"]
    dones = rollout["dones"]
    values = rollout["values"]
    last_value = rollout["last_value"]

    # GAE advantages
    with torch.no_grad():
        advantages = torch.zeros(H, N, device=device)
        last_gae = torch.zeros(N, device=device)
        for t in reversed(range(H)):
            next_val = last_value if t == H - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + values

    adv_flat = advantages.reshape(-1)
    adv_std_raw = adv_flat.std().item()
    advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    # Pre-update explained variance: how well the critic predicted returns
    with torch.no_grad():
        ret_var = returns.var()
        explained_variance = (1.0 - (returns - values).var() / (ret_var + 1e-8)).item()

    # Flatten for mini-batch sampling
    flat_obs = rollout["observations"].reshape(H * N, -1)
    flat_acts = rollout["actions"].reshape(H * N, -1)
    flat_old_lp = rollout["log_probs"].reshape(-1)
    flat_adv = advantages.reshape(-1)
    flat_ret = returns.reshape(-1)

    total_samples = H * N
    mini_batch_size = max(1, total_samples // num_mini_batches)
    mean_kl = 0.0
    last_epoch_critic_loss = 0.0
    last_epoch_entropy = 0.0

    for epoch in range(n_epochs):
        epoch_critic_loss = 0.0
        epoch_entropy = 0.0
        epoch_batches = 0

        perm = torch.randperm(total_samples, device=device)
        # Sync permutation across ranks to keep mini-batch count identical
        if num_procs > 1:
            torch.distributed.broadcast(perm, src=0)

        for start in range(0, total_samples, mini_batch_size):
            end = min(start + mini_batch_size, total_samples)
            mb_idx = perm[start:end]

            log_probs, val_pred, entropy = model.evaluate_actions(
                flat_obs[mb_idx], flat_acts[mb_idx])

            # Clamp log-ratio to prevent exp() overflow → NaN
            log_ratio = log_probs - flat_old_lp[mb_idx]
            log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
            ratio = torch.exp(log_ratio)
            clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            policy_loss = -torch.min(ratio * flat_adv[mb_idx],
                                     clipped * flat_adv[mb_idx]).mean()
            value_loss = F.mse_loss(val_pred, flat_ret[mb_idx])
            loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

            optimizer.zero_grad()
            loss.backward()

            # NaN guard: zero out gradients if loss was invalid
            # (all ranks see same data via synced permutation, so this
            # fires consistently — no NCCL desync risk)
            if torch.isnan(loss) or torch.isinf(loss):
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.zero_()

            # Multi-GPU: average gradients
            if num_procs > 1:
                for p in model.parameters():
                    if p.grad is not None:
                        torch.distributed.all_reduce(
                            p.grad, op=torch.distributed.ReduceOp.SUM)
                        p.grad /= num_procs

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            epoch_critic_loss += value_loss.item()
            epoch_entropy += entropy.item()
            epoch_batches += 1

        last_epoch_critic_loss = epoch_critic_loss / max(epoch_batches, 1)
        last_epoch_entropy = epoch_entropy / max(epoch_batches, 1)

        # Approximate KL
        with torch.no_grad():
            new_lp, _, _ = model.evaluate_actions(flat_obs, flat_acts)
            mean_kl = (flat_old_lp - new_lp).mean().item()

        # Sync early-stop decision across ranks to prevent NCCL deadlock
        # (different ranks may compute different KL values)
        if num_procs > 1:
            should_break = torch.tensor(
                [1 if mean_kl > 2.0 * desired_kl else 0],
                device=device, dtype=torch.int32)
            torch.distributed.broadcast(should_break, src=0)
            if should_break.item():
                break
        elif mean_kl > 2.0 * desired_kl:
            break

    # Adaptive LR
    current_lr = optimizer.param_groups[0]['lr']
    if mean_kl > 2.0 * desired_kl:
        new_lr = max(current_lr / 1.5, 1e-5)
    elif mean_kl < desired_kl / 2.0:
        new_lr = min(current_lr * 1.5, 1e-2)
    else:
        new_lr = current_lr
    for pg in optimizer.param_groups:
        pg['lr'] = new_lr

    return {
        "mean_kl": mean_kl,
        "value_loss": last_epoch_critic_loss,
        "entropy": last_epoch_entropy,
        "adv_std": adv_std_raw,
        "explained_variance": explained_variance,
        "lr": optimizer.param_groups[0]['lr'],
    }


# ---------------------------------------------------------------------------
# PPO Inner Loop with convergence detection
# ---------------------------------------------------------------------------

class PPOInnerLoop:
    """Drives PPO training with convergence detection. Multi-GPU aware."""

    def __init__(self, walking_env, model, obs_rms, device,
                 rank, num_procs,
                 horizon=24, n_epochs=5, num_mini_batches=4,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_ratio=0.2, entropy_coeff=0.01,
                 desired_kl=0.01, max_grad_norm=1.0,
                 kl_conv_threshold=1e-4, kl_window=10,
                 max_inner_iters=10000,
                 warmup_iters=5000,
                 convergence_mode='composite',
                 convergence_window=10,
                 critic_loss_plateau_threshold=0.05,
                 adv_std_threshold=0.1,
                 entropy_plateau_threshold=0.05,
                 explained_variance_threshold=0.95,
                 use_wandb=False, log_every=10):
        self.walking_env = walking_env
        self.model = model
        self.obs_rms = obs_rms
        self.device = device
        self.rank = rank
        self.num_procs = num_procs
        self.is_root = (rank == 0)

        self.horizon = horizon
        self.n_epochs = n_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.desired_kl = desired_kl
        self.max_grad_norm = max_grad_norm
        self.max_inner_iters = max_inner_iters
        self.warmup_iters = warmup_iters
        self.convergence_mode = convergence_mode
        self.use_wandb = use_wandb
        self.log_every = log_every

        # Legacy KL convergence (fallback)
        self.kl_conv_threshold = kl_conv_threshold
        self.kl_window = kl_window

        # Composite convergence gate
        self.convergence_gate = CompositeConvergenceGate(window=convergence_window)
        self.convergence_gate.add_signal(
            'critic_loss', 'plateau', critic_loss_plateau_threshold)
        self.convergence_gate.add_signal(
            'adv_std', 'below', adv_std_threshold)
        self.convergence_gate.add_signal(
            'entropy', 'plateau', entropy_plateau_threshold)
        self.convergence_gate.add_signal(
            'explained_variance', 'above', explained_variance_threshold)

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self._global_inner_iter = 0  # monotonic for wandb

    def _check_kl_converged(self, kl_values):
        if len(kl_values) < self.kl_window:
            return False
        recent = list(kl_values)[-self.kl_window:]
        return np.mean(recent) < self.kl_conv_threshold

    def train_until_converged(self, out_dir, min_iters=None):
        """Run PPO until convergence or iteration cap.

        Returns (converged, reward_history, final_mean_reward).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        env = self.walking_env
        model = self.model
        N = env.num_envs

        # Reset env (preserves network weights = warm-start)
        env.last_obs = None
        obs, _ = env.reset()
        env.last_obs = obs

        model.train()
        self.convergence_gate.reset()
        kl_values = []
        reward_buffer = deque(maxlen=200)
        length_buffer = deque(maxlen=200)
        reward_history = []
        inter_kl = 0.0

        inner_iters = 0
        mean_rew, std_rew, mean_len = 0.0, 0.0, 0.0
        start_time = time.time()
        effective_min = min_iters if min_iters is not None else 0

        while inner_iters < self.max_inner_iters:
            rollout = collect_rollout(
                env, model, self.obs_rms, self.horizon, self.device)

            # Inter-iteration KL: measure policy shift on current rollout.
            # Subsample to keep the post-update forward pass cheap.
            _KL_SUBSAMPLE = 1024
            H, N_envs = rollout["rewards"].shape
            total = H * N_envs
            flat_obs = rollout["observations"].reshape(total, -1)
            flat_acts = rollout["actions"].reshape(total, -1)
            flat_old_lp = rollout["log_probs"].reshape(-1)
            if total > _KL_SUBSAMPLE:
                kl_idx = torch.randperm(total, device=self.device)[:_KL_SUBSAMPLE]
                kl_obs = flat_obs[kl_idx]
                kl_acts = flat_acts[kl_idx]
                kl_old_lp = flat_old_lp[kl_idx]
            else:
                kl_obs, kl_acts, kl_old_lp = flat_obs, flat_acts, flat_old_lp

            update_info = ppo_update(
                model, self.optimizer, rollout,
                n_epochs=self.n_epochs, clip_ratio=self.clip_ratio,
                gamma=self.gamma, gae_lambda=self.gae_lambda,
                entropy_coeff=self.entropy_coeff,
                num_mini_batches=self.num_mini_batches,
                desired_kl=self.desired_kl, max_grad_norm=self.max_grad_norm,
                num_procs=self.num_procs, device=self.device,
            )
            mean_kl = update_info["mean_kl"]

            with torch.no_grad():
                new_lp, _, _ = model.evaluate_actions(kl_obs, kl_acts)
                inter_kl = (kl_old_lp - new_lp).mean().item()

            # Sync obs normalizer periodically
            if inner_iters % 50 == 0:
                self.obs_rms.sync_across_ranks(self.num_procs)

            inner_iters += 1
            self._global_inner_iter += 1
            current_lr = self.optimizer.param_groups[0]['lr']

            # Track completed episodes
            ep_rews, ep_lens = env.pop_completed_episodes()
            reward_buffer.extend(ep_rews)
            length_buffer.extend(ep_lens)

            if len(reward_buffer) > 0:
                mean_rew = np.mean(reward_buffer)
                std_rew = np.std(reward_buffer)
                mean_len = np.mean(length_buffer)

            # Feed convergence gate every iteration
            self.convergence_gate.update({
                'critic_loss': update_info["value_loss"],
                'adv_std': update_info["adv_std"],
                'entropy': update_info["entropy"],
                'explained_variance': update_info["explained_variance"],
            })

            # KL convergence tracking at log intervals (legacy fallback)
            if inner_iters % self.log_every == 0:
                kl_values.append(abs(inter_kl))
                if len(reward_buffer) > 0:
                    reward_history.append(mean_rew)

            # Logging (root only)
            if self.is_root and inner_iters % self.log_every == 0:
                rollout_mean = rollout["rewards"].mean().item()
                gate_diag = self.convergence_gate.get_diagnostics()
                print(f"    PPO iter {inner_iters}: "
                      f"rew/step={rollout_mean:.3f}, kl={mean_kl:.4f}, "
                      f"inter_kl={abs(inter_kl):.6f}, "
                      f"lr={current_lr:.1e}, ep_rew={mean_rew:.1f}+/-{std_rew:.1f}, "
                      f"ep_len={mean_len:.0f}, "
                      f"v_loss={update_info['value_loss']:.4f}, "
                      f"adv_std={update_info['adv_std']:.3f}, "
                      f"ev={update_info['explained_variance']:.3f}, "
                      f"gate={'ALL' if gate_diag.get('gate/all_converged') else 'no'}")

                log_dict = {
                    "inner/reward_per_step": rollout_mean,
                    "inner/kl": mean_kl,
                    "inner/inter_iter_kl": abs(inter_kl),
                    "inner/lr": current_lr,
                    "inner/mean_reward": mean_rew,
                    "inner/reward_std": std_rew,
                    "inner/episode_length": mean_len,
                    "inner/value_loss": update_info["value_loss"],
                    "inner/adv_std": update_info["adv_std"],
                    "inner/entropy": update_info["entropy"],
                    "inner/explained_variance": update_info["explained_variance"],
                    "inner/iteration": self._global_inner_iter,
                }
                log_dict.update(gate_diag)
                if self.use_wandb:
                    wandb.log(log_dict, step=self._global_inner_iter)

            # Convergence check
            should_stop = False
            if self.is_root and inner_iters >= effective_min:
                if self.convergence_mode == 'composite':
                    should_stop = self.convergence_gate.check_converged()
                else:
                    should_stop = self._check_kl_converged(kl_values)

            if mp_util.enable_mp():
                flag = torch.tensor([int(should_stop)], dtype=torch.int32,
                                    device=self.device)
                flag = mp_util.broadcast(flag)
                should_stop = flag.item() == 1

            if should_stop:
                if self.is_root:
                    diag = self.convergence_gate.get_diagnostics()
                    print(f"    [Inner] CONVERGED at iter {inner_iters} "
                          f"(mode={self.convergence_mode}, {diag})")
                break

        elapsed = time.time() - start_time

        # Save checkpoint
        if self.is_root:
            ckpt = {
                "model": model.state_dict(),
                "obs_rms_mean": self.obs_rms.mean.cpu(),
                "obs_rms_var": self.obs_rms.var.cpu(),
                "obs_rms_count": self.obs_rms.count,
            }
            torch.save(ckpt, str(out_dir / "ppo_checkpoint.pt"))

        converged = should_stop
        if self.is_root and not converged:
            print(f"    [Inner] Iteration cap ({inner_iters} iters)")
        if self.is_root:
            print(f"    [Inner] Done: {inner_iters} iters, {elapsed / 60:.1f} min")

        return converged, reward_history, mean_rew


# ---------------------------------------------------------------------------
# SPSA gradient (adapted for walking_env + PPO model)
# ---------------------------------------------------------------------------

def generate_orthogonal_directions(n_params, num_sets, seed):
    rng = np.random.RandomState(seed)
    blocks = []
    for _ in range(num_sets):
        A = rng.standard_normal((n_params, n_params))
        Q, R = np.linalg.qr(A)
        Q *= np.sign(np.diag(R))
        blocks.append(Q)
    return np.vstack(blocks)


def apply_partitioned_morphologies(model, theta_np, eps, n_params,
                                   base_quats_dict, joint_idx_map,
                                   num_worlds, worlds_per_pert,
                                   directions=None):
    if directions is None:
        directions = np.eye(n_params)
    M = directions.shape[0]
    n_pert = 1 + 2 * M
    joints_per_world = model.joint_count // num_worlds
    joint_X_p_np = model.joint_X_p.numpy()
    for p in range(n_pert):
        if p == 0:
            theta_pert = theta_np
        else:
            m = (p - 1) // 2
            is_plus = (p % 2 == 1)
            sign = 1.0 if is_plus else -1.0
            theta_pert = theta_np + sign * eps * directions[m]
        w_start = p * worlds_per_pert
        w_end = (p + 1) * worlds_per_pert
        world_indices = np.arange(w_start, w_end)
        for i, (left_body, right_body) in enumerate(SYMMETRIC_PAIRS):
            angle = float(theta_pert[i])
            delta_q = quat_from_x_rotation(angle)
            for body_name in (left_body, right_body):
                if body_name not in joint_idx_map:
                    continue
                base_q = base_quats_dict[body_name]
                new_q = quat_normalize(quat_multiply(delta_q, base_q))
                local_idx = joint_idx_map[body_name]
                global_indices = world_indices * joints_per_world + local_idx
                joint_X_p_np[global_indices, 3:7] = [new_q[1], new_q[2], new_q[3], new_q[0]]
    model.joint_X_p.assign(joint_X_p_np)
    wp.synchronize()


def compute_spsa_gradient_parallel(walking_env, model, obs_rms,
                                   engine, char_id, total_mass,
                                   theta_np, base_quats_dict, joint_idx_map,
                                   num_worlds, horizon,
                                   rank, num_procs, device, param_names,
                                   num_seeds=10, base_seed=42,
                                   eps=0.05,
                                   num_spsa_sets=3):
    """SPSA gradient using walking_env's 18-term IsaacLab reward as objective.

    Returns (grad, grad_stderr, snr, center_reward, cot_center).
    center_reward is the mean episode reward from the env.
    cot_center is a diagnostic-only CoT for the center morphology.
    """
    n_params = len(theta_np)
    directions = generate_orthogonal_directions(n_params, num_spsa_sets,
                                                 seed=base_seed + 9999)
    M = directions.shape[0]
    n_pert = 1 + 2 * M
    wpp = num_worlds // n_pert
    assert num_worlds >= n_pert

    if rank == 0:
        print(f"    [SPSA] {num_spsa_sets} sets x {n_params} params = {M} dirs, "
              f"n_pert={n_pert}, {wpp} worlds/pert, {num_seeds} seeds")

    # Env reward for all perturbations
    reward_all = torch.zeros(n_pert, num_seeds, dtype=torch.float64, device=device)
    # CoT diagnostic for center perturbation only
    cot_diag_all = torch.zeros(num_seeds, dtype=torch.float64, device=device)

    sim_model = engine._sim_model

    apply_partitioned_morphologies(sim_model, theta_np, eps, n_params,
                                   base_quats_dict, joint_idx_map,
                                   num_worlds, wpp, directions=directions)

    model.eval()

    for seed_idx in range(num_seeds):
        seed = base_seed + seed_idx
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        obs, _ = walking_env.reset()

        # Per-world env reward accumulators
        reward_sum = torch.zeros(num_worlds, device=device)
        step_count = torch.zeros(num_worlds, device=device, dtype=torch.long)
        ep_reward_sum = torch.zeros(num_worlds, device=device)
        ep_count = torch.zeros(num_worlds, device=device, dtype=torch.long)

        # CoT diagnostic accumulators (center worlds only, indices 0..wpp-1)
        power_sum_c = torch.zeros(wpp, device=device)
        vel_sum_c = torch.zeros(wpp, device=device)
        step_count_c = torch.zeros(wpp, device=device, dtype=torch.long)
        ep_cot_sum_c = torch.zeros(wpp, device=device)
        ep_count_c = torch.zeros(wpp, device=device, dtype=torch.long)

        with torch.no_grad():
            for _ in range(horizon):
                obs_norm = obs_rms.normalize(obs)
                action, _, _ = model.get_action(obs_norm, deterministic=True)
                obs, reward, done, info = walking_env.step(action)

                # Accumulate env reward (18-term IsaacLab) for all worlds
                reward_sum += walking_env.rew_buf[:num_worlds]
                step_count += 1

                # CoT diagnostic: center worlds only (perturbation 0)
                dof_forces = EpisodeCoTTracker.get_dof_forces_safe(engine, char_id)
                dof_vel_eng = engine.get_dof_vel(char_id)
                root_vel = engine.get_root_vel(char_id)
                power = torch.sum(torch.abs(
                    dof_forces[:wpp] *
                    dof_vel_eng[:wpp]), dim=-1)
                power_sum_c += power
                vel_sum_c += root_vel[:wpp, 0]
                step_count_c += 1

                done_mask = (done[:num_worlds] > 0)
                if done_mask.any():
                    di = done_mask.nonzero(as_tuple=True)[0]
                    valid = step_count[di] > 0
                    if valid.any():
                        vi = di[valid]
                        ep_reward_sum[vi] += reward_sum[vi]
                        ep_count[vi] += 1
                    reward_sum[di] = 0
                    step_count[di] = 0

                    # CoT diagnostic reset for center worlds
                    center_done = done_mask[:wpp]
                    if center_done.any():
                        cdi = center_done.nonzero(as_tuple=True)[0]
                        cv = step_count_c[cdi] > 0
                        if cv.any():
                            cvi = cdi[cv]
                            mp = power_sum_c[cvi] / step_count_c[cvi].float()
                            mv = vel_sum_c[cvi] / step_count_c[cvi].float()
                            ep_cot_sum_c[cvi] += compute_cot(mp, mv, total_mass)
                            ep_count_c[cvi] += 1
                        power_sum_c[cdi] = 0
                        vel_sum_c[cdi] = 0
                        step_count_c[cdi] = 0

        # Finalize still-alive episodes
        alive = step_count > 0
        if alive.any():
            ai = alive.nonzero(as_tuple=True)[0]
            ep_reward_sum[ai] += reward_sum[ai]
            ep_count[ai] += 1

        alive_c = step_count_c > 0
        if alive_c.any():
            aci = alive_c.nonzero(as_tuple=True)[0]
            mp = power_sum_c[aci] / step_count_c[aci].float()
            mv = vel_sum_c[aci] / step_count_c[aci].float()
            ep_cot_sum_c[aci] += compute_cot(mp, mv, total_mass)
            ep_count_c[aci] += 1

        # Aggregate per perturbation
        for p in range(n_pert):
            s, e = p * wpp, (p + 1) * wpp
            counts = ep_count[s:e].float().clamp(min=1)
            reward_all[p, seed_idx] = (ep_reward_sum[s:e] / counts).mean().item()

        # CoT diagnostic (center only)
        cot_counts = ep_count_c.float().clamp(min=1)
        cot_diag_all[seed_idx] = (ep_cot_sum_c / cot_counts).mean().item()

    model.train()

    # Drain SPSA-phase episode stats so they don't leak into inner loop
    walking_env.pop_completed_episodes()

    if num_procs > 1:
        torch.distributed.all_reduce(reward_all, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(cot_diag_all, op=torch.distributed.ReduceOp.SUM)
        reward_all /= num_procs
        cot_diag_all /= num_procs

    # Restore original morphology
    update_training_joint_X_p(sim_model, theta_np, base_quats_dict,
                               joint_idx_map, num_worlds)

    # Gradient via least-squares on env reward
    reward_np = reward_all.cpu().numpy()
    center_reward = float(np.mean(reward_np[0]))
    cot_center = float(cot_diag_all.mean().item())

    grad_per_seed = []
    for k in range(num_seeds):
        dJ = np.array([
            (reward_np[2 * m + 1, k] - reward_np[2 * m + 2, k]) / (2 * eps)
            for m in range(M)
        ])
        if np.any(np.isnan(dJ)):
            continue
        grad_k, _, _, _ = np.linalg.lstsq(directions, dJ, rcond=None)
        grad_per_seed.append(grad_k)

    if len(grad_per_seed) == 0:
        grad = np.zeros(n_params, dtype=np.float64)
        grad_stderr = np.full(n_params, float('inf'), dtype=np.float64)
    else:
        grad_samples = np.stack(grad_per_seed)
        grad = grad_samples.mean(axis=0)
        if len(grad_per_seed) > 1:
            grad_stderr = grad_samples.std(axis=0, ddof=1) / np.sqrt(len(grad_per_seed))
        else:
            grad_stderr = np.full(n_params, float('inf'), dtype=np.float64)

    snr = np.zeros(n_params, dtype=np.float64)
    for i in range(n_params):
        if grad_stderr[i] > 0 and np.isfinite(grad_stderr[i]):
            snr[i] = abs(grad[i]) / grad_stderr[i]
        else:
            snr[i] = float('inf') if grad[i] != 0 else 0.0

    grad = np.clip(grad, -10.0, 10.0)

    if rank == 0:
        print(f"    [SPSA] Center: env_reward={center_reward:.4f}, "
              f"CoT(diag)={cot_center:.4f}")
        print(f"    [SPSA] Gradient ({len(grad_per_seed)}/{num_seeds} valid seeds):")
        for i, name in enumerate(param_names):
            print(f"      d_reward/d_{name} = {grad[i]:+.6f} "
                  f"+/- {grad_stderr[i]:.6f} (SNR={snr[i]:.2f})")
        finite_snr = snr[np.isfinite(snr)]
        if len(finite_snr) > 0:
            print(f"    [SPSA] SNR: min={np.min(finite_snr):.2f}, "
                  f"mean={np.mean(finite_snr):.2f}, max={np.max(finite_snr):.2f}")

    return grad, grad_stderr, snr, center_reward, cot_center


# ---------------------------------------------------------------------------
# Worker initialization
# ---------------------------------------------------------------------------

def _init_gbc_worker(rank, num_procs, device, master_port, args):
    is_root = (rank == 0)
    per_rank_envs = args.num_train_envs // num_procs

    if "cuda" in device:
        torch.cuda.set_device(device)

    seed = int(np.uint64(42) + np.uint64(41 * rank))
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if "cuda" in device:
        torch.cuda.manual_seed(seed)

    mp_util.global_mp_device = device
    mp_util.global_num_procs = num_procs

    if num_procs > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        backend = getattr(args, "dist_backend", "nccl")
        if backend == "auto":
            backend = "nccl" if ("cuda" in device and platform.system() != "Windows") else "gloo"
        torch.distributed.init_process_group(backend, rank=rank, world_size=num_procs)

    if is_root:
        print("=" * 70)
        print("GBC for G1 Humanoid (PPO Baseline)")
        if num_procs > 1:
            print(f"  Multi-GPU: {num_procs} GPUs, {per_rank_envs} envs/GPU")
        print("=" * 70)

    out_dir = Path(args.out_dir).resolve()
    if is_root:
        out_dir.mkdir(parents=True, exist_ok=True)
    if mp_util.enable_mp():
        torch.distributed.barrier()

    use_wandb = is_root and not args.no_wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project="gbc-codesign",
            name=f"g1-ppo-baseline-{args.num_train_envs}env-{num_procs}gpu",
            config=vars(args),
        )

    # Build MimicKit env (for Newton engine only)
    if is_root:
        print("\n[1/3] Initializing Newton engine...")

    mjcf_modifier = G1MJCFModifier(str(BASE_MJCF_PATH))
    modified_mjcf = out_dir / "g1_modified.xml"
    modified_env_config = out_dir / "env_config.yaml"

    theta_init = np.zeros(NUM_DESIGN_PARAMS, dtype=np.float64)
    if is_root:
        mjcf_modifier.generate(theta_init, str(modified_mjcf))
        mesh_src = BASE_MJCF_PATH.parent / "meshes"
        mesh_dst = out_dir / "meshes"
        if not mesh_dst.exists():
            try:
                mesh_dst.symlink_to(mesh_src)
            except (OSError, NotImplementedError):
                shutil.copytree(str(mesh_src), str(mesh_dst))
        mjcf_modifier.generate_env_config(
            str(modified_mjcf), str(BASE_ENV_CONFIG), str(modified_env_config))

    if mp_util.enable_mp():
        torch.distributed.barrier()

    if is_root:
        print("[2/3] Building training env + PPO model...")

    # Staggered build for Warp cache
    if mp_util.enable_mp():
        if is_root:
            base_env = env_builder.build_env(
                str(modified_env_config), str(BASE_ENGINE_CONFIG),
                per_rank_envs, device, visualize=False)
            wp.synchronize()
        torch.distributed.barrier()
        if not is_root:
            base_env = env_builder.build_env(
                str(modified_env_config), str(BASE_ENGINE_CONFIG),
                per_rank_envs, device, visualize=False)
    else:
        base_env = env_builder.build_env(
            str(modified_env_config), str(BASE_ENGINE_CONFIG),
            per_rank_envs, device, visualize=False)

    # Configure DR
    cfg = G1WalkingConfig()
    cfg.enable_dr = args.enable_dr

    # Wrap with walking env
    walking_env = G1WalkingEnvWrapper(base_env, device, cfg=cfg)

    obs_dim = walking_env.obs_dim
    act_dim = walking_env.act_dim

    # Build PPO model
    hidden_sizes = tuple(int(x) for x in args.ppo_hidden_sizes.split(","))
    model = G1ActorCritic(obs_dim, act_dim, hidden_sizes=hidden_sizes).to(device)
    obs_rms = RunningMeanStdCUDA(shape=(obs_dim,), device=device)

    if is_root:
        n_params_model = sum(p.numel() for p in model.parameters())
        print(f"  obs_dim={obs_dim}, act_dim={act_dim}, "
              f"model_params={n_params_model:,}, hidden={hidden_sizes}")

    # Extract joint info
    engine = base_env._engine
    sim_model = engine._sim_model
    char_id = base_env._get_char_id()
    total_mass = engine.calc_obj_mass(0, char_id)
    base_quats_dict, joint_idx_map = extract_joint_info(sim_model, per_rank_envs)
    if is_root:
        print(f"  Robot mass: {total_mass:.2f} kg")

    return types.SimpleNamespace(
        is_root=is_root, per_rank_envs=per_rank_envs,
        walking_env=walking_env, base_env=base_env,
        model=model, obs_rms=obs_rms,
        engine=engine, sim_model=sim_model,
        char_id=char_id, total_mass=total_mass,
        base_quats_dict=base_quats_dict, joint_idx_map=joint_idx_map,
        use_wandb=use_wandb, out_dir=out_dir, mjcf_modifier=mjcf_modifier,
    )


# ---------------------------------------------------------------------------
# Main GBC loop (one per GPU rank)
# ---------------------------------------------------------------------------

def gbc_worker(rank, num_procs, device, master_port, args):
    ws = _init_gbc_worker(rank, num_procs, device, master_port, args)
    is_root = ws.is_root
    per_rank_envs = ws.per_rank_envs
    walking_env, model, obs_rms = ws.walking_env, ws.model, ws.obs_rms
    engine, sim_model = ws.engine, ws.sim_model
    char_id, total_mass = ws.char_id, ws.total_mass
    base_quats_dict, joint_idx_map = ws.base_quats_dict, ws.joint_idx_map
    use_wandb, out_dir = ws.use_wandb, ws.out_dir

    if is_root:
        print("[3/3] Starting GBC outer loop...\n")

    theta = np.zeros(NUM_DESIGN_PARAMS, dtype=np.float64)
    bfgs = CautiousBFGS(NUM_DESIGN_PARAMS)
    theta_bounds = THETA_BOUNDS

    param_names = [
        f"theta_{i}_{SYMMETRIC_PAIRS[i][0].replace('_link', '')}"
        for i in range(NUM_DESIGN_PARAMS)
    ]

    ppo_inner = PPOInnerLoop(
        walking_env, model, obs_rms, device,
        rank=rank, num_procs=num_procs,
        horizon=args.ppo_horizon,
        n_epochs=args.ppo_epochs,
        num_mini_batches=args.ppo_mini_batches,
        lr=args.ppo_lr,
        clip_ratio=args.ppo_clip_ratio,
        entropy_coeff=args.ppo_entropy_coeff,
        kl_conv_threshold=args.kl_conv_threshold,
        kl_window=args.kl_window,
        max_inner_iters=args.max_inner_iters,
        warmup_iters=args.warmup_iters,
        convergence_mode=args.convergence_mode,
        convergence_window=args.convergence_window,
        critic_loss_plateau_threshold=args.critic_loss_plateau_threshold,
        adv_std_threshold=args.adv_std_threshold,
        entropy_plateau_threshold=args.entropy_plateau_threshold,
        explained_variance_threshold=args.explained_variance_threshold,
        use_wandb=use_wandb,
    )

    # Adaptive trust region
    trust_radius = np.radians(args.max_step_deg)
    TRUST_RADIUS_MIN = np.radians(0.05)
    TRUST_RADIUS_MAX = np.radians(5.0)
    prev_center_reward = None
    predicted_improvement = None

    history = {
        "theta": [theta.copy()],
        "center_reward": [],
        "cot": [],
        "gradients": [],
        "grad_snr": [],
        "inner_times": [],
    }
    theta_history = deque(maxlen=5)

    if is_root:
        _M = args.spsa_sets * NUM_DESIGN_PARAMS
        _n_pert = 1 + 2 * _M
        _wpp = per_rank_envs // _n_pert
        print(f"Configuration:")
        print(f"  Training envs:     {args.num_train_envs} total "
              f"({per_rank_envs}/GPU x {num_procs} GPUs)")
        print(f"  Inner loop:        PPO (lr={args.ppo_lr}, horizon={args.ppo_horizon}, "
              f"hidden={args.ppo_hidden_sizes})")
        print(f"  Domain rand:       {'ON' if args.enable_dr else 'OFF'}")
        print(f"  Eval horizon:      {args.eval_horizon} steps")
        print(f"  Design optimizer:  Cautious BFGS (lr={args.design_lr})")
        print(f"  Max step size:     {args.max_step_deg} deg/iter (adaptive trust region)")
        print(f"  SPSA:              {args.spsa_sets} sets, {args.num_spsa_seeds} seeds, "
              f"eps={args.spsa_epsilon} rad")
        print(f"  Warmup iters:      {args.warmup_iters} (first outer iter)")

    # =========================================================
    # Outer loop
    # =========================================================
    for outer_iter in range(args.outer_iters):
        if is_root:
            print(f"\n{'=' * 70}")
            print(f"Outer Iteration {outer_iter + 1}/{args.outer_iters}")
            print(f"{'=' * 70}")
            for i, name in enumerate(param_names):
                print(f"  {name}: {theta[i]:+.4f} rad ({np.degrees(theta[i]):+.2f} deg)")

        iter_dir = out_dir / f"outer_{outer_iter:03d}"

        # ----- Inner Loop (ALL ranks) -----
        if is_root:
            print(f"\n  [PPO] Training ({per_rank_envs} envs/GPU x {num_procs} GPUs)...")
        t0 = time.time()

        min_iters = args.warmup_iters if outer_iter == 0 else args.post_warmup_min_iters

        converged, rew_hist, final_mean_rew = ppo_inner.train_until_converged(
            iter_dir, min_iters=min_iters)
        inner_time = time.time() - t0
        if is_root:
            print(f"  [PPO] Done in {inner_time / 60:.1f} min, "
                  f"mean_rew={final_mean_rew:.1f}")
        history["inner_times"].append(inner_time)

        # ----- SPSA Gradient Phase (ALL ranks) -----
        if is_root:
            print(f"\n  [SPSA] Computing gradient...")

        spsa_base_seed = 42 + outer_iter * 1000

        grad_theta, grad_stderr, grad_snr, center_reward, cot_diag = compute_spsa_gradient_parallel(
            walking_env, model, obs_rms,
            engine, char_id, total_mass,
            theta.copy(), base_quats_dict, joint_idx_map,
            per_rank_envs, args.eval_horizon,
            rank, num_procs, device, param_names,
            num_seeds=args.num_spsa_seeds, base_seed=spsa_base_seed,
            eps=args.spsa_epsilon,
            num_spsa_sets=args.spsa_sets,
        )

        if is_root:
            print(f"    Center env reward = {center_reward:.4f}")
            print(f"    Cost of Transport (diag) = {cot_diag:.4f}")
            history["center_reward"].append(center_reward)
            history["cot"].append(cot_diag)
            history["gradients"].append(grad_theta.copy())
            history["grad_snr"].append(grad_snr.copy())
            rho = None
            if prev_center_reward is not None and predicted_improvement is not None:
                actual_improvement = center_reward - prev_center_reward
                if abs(predicted_improvement) > 1e-12:
                    rho = actual_improvement / predicted_improvement
                else:
                    rho = 0.0
                if rho > 0.75:
                    trust_radius = min(trust_radius * 1.5, TRUST_RADIUS_MAX)
                elif rho < 0.25:
                    trust_radius = max(trust_radius * 0.5, TRUST_RADIUS_MIN)
                print(f"    [Trust Region] rho={rho:.3f}, radius={np.degrees(trust_radius):.3f} deg")
            prev_center_reward = center_reward

            # Design update
            n_clamped = 0
            raw_step = np.zeros(NUM_DESIGN_PARAMS, dtype=np.float64)

            if np.any(np.isnan(grad_theta)) or np.all(grad_theta == 0):
                print(f"\n  [FATAL] Degenerate gradient")
                theta[:] = np.nan
                predicted_improvement = None
            else:
                grad_theta_raw = grad_theta.copy()

                skip_update = False
                if args.snr_threshold > 0:
                    snr_mask = (grad_snr > args.snr_threshold) | ~np.isfinite(grad_snr)
                    n_masked = NUM_DESIGN_PARAMS - snr_mask.sum()
                    if n_masked == NUM_DESIGN_PARAMS:
                        print(f"    [SNR gate] All params masked, skipping update")
                        skip_update = True
                        predicted_improvement = 0.0
                    elif n_masked > 0:
                        print(f"    [SNR gate] Masking {n_masked}/{NUM_DESIGN_PARAMS} params")
                        grad_theta = grad_theta * snr_mask.astype(float)

                if not skip_update:
                    old_theta = theta.copy()
                    direction = bfgs.compute_direction(grad_theta)
                    step = args.design_lr * direction
                    max_step_rad = trust_radius
                    raw_step = step.copy()
                    step = np.clip(step, -max_step_rad, max_step_rad)
                    n_clamped = np.sum(np.abs(raw_step) > max_step_rad)
                    if n_clamped > 0:
                        print(f"    [Trust region] Clamped {n_clamped}/{NUM_DESIGN_PARAMS} "
                              f"params to +/-{np.degrees(max_step_rad):.3f} deg")

                    theta = old_theta + step
                    theta = np.clip(theta, theta_bounds[0], theta_bounds[1])
                    actual_s = theta - old_theta
                    bfgs.update(actual_s, grad_theta_raw)
                    predicted_improvement = grad_theta_raw @ actual_s

                    print(f"\n  Design update (BFGS, {bfgs.num_updates} H updates, "
                          f"{bfgs.num_skipped} skipped):")
                    for i, name in enumerate(param_names):
                        delta = theta[i] - old_theta[i]
                        print(f"    {name}: {old_theta[i]:+.4f} -> {theta[i]:+.4f} "
                              f"({np.degrees(delta):+.3f} deg)")

        # Broadcast theta
        theta_tensor = torch.from_numpy(theta).double().to(device)
        if mp_util.enable_mp():
            theta_tensor = mp_util.broadcast(theta_tensor)
        theta = theta_tensor.cpu().double().numpy()

        if np.any(np.isnan(theta)):
            break

        history["theta"].append(theta.copy())

        # Update morphology
        update_training_joint_X_p(sim_model, theta, base_quats_dict,
                                   joint_idx_map, per_rank_envs)

        # Outer convergence check
        outer_converged = False
        if is_root:
            theta_history.append(theta.copy())
            if len(theta_history) >= 5:
                theta_stack = np.array(list(theta_history))
                ranges = theta_stack.max(axis=0) - theta_stack.min(axis=0)
                if ranges.max() < np.radians(0.5):
                    print(f"\n  OUTER CONVERGED")
                    outer_converged = True

        if mp_util.enable_mp():
            flag = torch.tensor([int(outer_converged)], dtype=torch.int32, device=device)
            flag = mp_util.broadcast(flag)
            outer_converged = flag.item() == 1

        # Save + wandb (root only)
        if is_root:
            np.save(str(out_dir / "theta_latest.npy"), theta)
            np.save(str(iter_dir / "theta.npy"), theta)
            np.save(str(iter_dir / "grad.npy"), grad_theta)

            if use_wandb:
                finite_snr = grad_snr[np.isfinite(grad_snr)]
                log_dict = {
                    "outer/iteration": outer_iter + 1,
                    "outer/center_reward": center_reward,
                    "outer/cot_diag": cot_diag,
                    "outer/inner_time_min": inner_time / 60.0,
                    "outer/grad_norm": np.linalg.norm(grad_theta),
                    "outer/grad_snr_min": float(np.min(finite_snr)) if len(finite_snr) > 0 else 0.0,
                    "outer/grad_snr_mean": float(np.mean(finite_snr)) if len(finite_snr) > 0 else 0.0,
                    "outer/bfgs_updates": bfgs.num_updates,
                    "outer/bfgs_skipped": bfgs.num_skipped,
                    "outer/bfgs_cond": float(np.linalg.cond(bfgs.H)),
                    "outer/step_clamped": int(n_clamped),
                    "outer/max_raw_step_deg": float(np.degrees(np.max(np.abs(raw_step)))),
                    "outer/trust_radius_deg": np.degrees(trust_radius),
                    "outer/final_ppo_reward": final_mean_rew,
                }
                for i, name in enumerate(param_names):
                    log_dict[f"outer/{name}_rad"] = theta[i]
                    log_dict[f"outer/{name}_deg"] = np.degrees(theta[i])
                    log_dict[f"outer/grad_{name}"] = grad_theta[i]
                    log_dict[f"outer/snr_{name}"] = grad_snr[i] if np.isfinite(grad_snr[i]) else 0.0
                if rho is not None:
                    log_dict["outer/trust_ratio_rho"] = rho
                wandb.log(log_dict, step=ppo_inner._global_inner_iter)

        if outer_converged:
            break

    # Final results
    if is_root:
        print("\n" + "=" * 70)
        print("GBC Co-Design Complete (PPO Baseline)!")
        print("=" * 70)
        initial = history["theta"][0]
        final = history["theta"][-1]
        for i, name in enumerate(param_names):
            print(f"  {name}: {np.degrees(initial[i]):+.2f} -> {np.degrees(final[i]):+.2f} deg")
        if history["center_reward"]:
            print(f"\nEnv reward: {history['center_reward'][0]:.4f} -> "
                  f"{history['center_reward'][-1]:.4f}")
        if history["cot"]:
            print(f"CoT (diag): {history['cot'][0]:.4f} -> {history['cot'][-1]:.4f}")
        total_time = sum(history["inner_times"])
        print(f"Total inner loop time: {total_time / 3600:.1f} hours")

    if use_wandb:
        wandb.finish()

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GBC for G1 Humanoid (PPO Baseline)")

    # Outer loop
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--outer-iters", type=int, default=100)
    parser.add_argument("--design-lr", type=float, default=1.0)
    parser.add_argument("--num-train-envs", type=int, default=4096)
    parser.add_argument("--eval-horizon", type=int, default=300)
    parser.add_argument("--num-spsa-seeds", type=int, default=30)
    parser.add_argument("--spsa-epsilon", type=float, default=0.05)
    parser.add_argument("--spsa-sets", type=int, default=3)
    parser.add_argument("--snr-threshold", type=float, default=0.0)
    parser.add_argument("--max-step-deg", type=float, default=0.5)
    parser.add_argument("--max-inner-iters", type=int, default=10000)
    parser.add_argument("--out-dir", type=str, default="output_g1_ppo_baseline")

    # PPO inner loop
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-horizon", type=int, default=24)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--ppo-mini-batches", type=int, default=4)
    parser.add_argument("--ppo-clip-ratio", type=float, default=0.2)
    parser.add_argument("--ppo-entropy-coeff", type=float, default=0.01)
    parser.add_argument("--ppo-hidden-sizes", type=str, default="1024,1024")

    # Domain randomization
    parser.add_argument("--enable-dr", action="store_true", default=True)
    parser.add_argument("--no-dr", dest="enable_dr", action="store_false")

    # Convergence
    parser.add_argument("--convergence-mode", type=str, default="composite",
                        choices=["composite", "kl"],
                        help="Inner convergence: 'composite' (multi-signal gate) or 'kl' (legacy)")
    parser.add_argument("--convergence-window", type=int, default=10,
                        help="Window size for convergence gate signals")
    parser.add_argument("--critic-loss-plateau-threshold", type=float, default=0.05)
    parser.add_argument("--adv-std-threshold", type=float, default=0.1)
    parser.add_argument("--entropy-plateau-threshold", type=float, default=0.05)
    parser.add_argument("--explained-variance-threshold", type=float, default=0.95)
    parser.add_argument("--kl-conv-threshold", type=float, default=1e-4,
                        help="Legacy KL convergence threshold (only used with --convergence-mode=kl)")
    parser.add_argument("--kl-window", type=int, default=10,
                        help="Legacy KL convergence window (only used with --convergence-mode=kl)")
    parser.add_argument("--warmup-iters", type=int, default=5000)
    parser.add_argument("--post-warmup-min-iters", type=int, default=100)

    # Multi-GPU
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument("--dist-backend", type=str, default="nccl",
                        choices=["nccl", "gloo", "auto"])

    args = parser.parse_args()

    if args.devices is None:
        n_gpus = torch.cuda.device_count()
        args.devices = [f"cuda:{i}" for i in range(n_gpus)] if n_gpus > 0 else ["cuda:0"]

    num_workers = len(args.devices)
    assert args.num_train_envs % num_workers == 0

    gpu_indices = []
    for d in args.devices:
        assert "cuda" in d
        gpu_indices.append(int(d.split(":")[1]))

    master_port = args.master_port if args.master_port is not None else random.randint(6000, 7000)

    if num_workers > 1:
        torch.multiprocessing.set_start_method("spawn")
        processes = []
        for r in range(num_workers):
            os.environ["_GBC_GPU_INDEX"] = str(gpu_indices[r])
            p = torch.multiprocessing.Process(
                target=gbc_worker,
                args=[r, num_workers, "cuda:0", master_port, args])
            p.start()
            processes.append(p)
        os.environ.pop("_GBC_GPU_INDEX", None)

        failed = []
        for r, p in enumerate(processes):
            p.join()
            if p.exitcode != 0:
                failed.append((r, p.exitcode))
        if failed:
            print(f"\n[ERROR] {len(failed)} worker(s) failed:")
            for r, code in failed:
                print(f"  Rank {r}: exit code {code}")
            sys.exit(1)
    else:
        gbc_worker(0, 1, args.devices[0], master_port, args)
