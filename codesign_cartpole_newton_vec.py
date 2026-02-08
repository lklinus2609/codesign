#!/usr/bin/env python3
"""
Level 1.5: PGHC Co-Design for Cart-Pole using Vectorized Newton Physics

Uses N parallel environments on GPU for dramatically faster training.

Run with wandb logging:
    python codesign_cartpole_newton_vec.py --wandb --num-worlds 64
"""

# CRITICAL: Must set BEFORE importing pyglet/newton for headless video recording
import os
os.environ["PYGLET_HEADLESS"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from collections import deque

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import warp as wp
import newton

from envs.cartpole_newton import CartPoleNewtonVecEnv, ParametricCartPoleNewton


def _record_video_subprocess(L_value, policy_state_dict, max_steps, width, height):
    """
    Record video in a subprocess to get a fresh OpenGL context.

    Newton's ViewerGL doesn't properly clean up OpenGL resources on close(),
    causing black screens when creating multiple viewers. Running in a subprocess
    guarantees a fresh OpenGL context each time.
    """
    import multiprocessing as mp
    import pickle
    import tempfile

    # Save policy weights to temp file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        pickle.dump({
            'state_dict': policy_state_dict,
            'L_value': L_value,
            'max_steps': max_steps,
            'width': width,
            'height': height,
        }, f)
        config_path = f.name

    # Create output file for frames
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        output_path = f.name

    # Run recording in subprocess
    import subprocess
    import sys

    script = f'''
import os
os.environ["PYGLET_HEADLESS"] = "1"

import pickle
import numpy as np
import torch

import warp as wp
import newton

# Initialize warp
wp.init()

# Load config
with open("{config_path}", "rb") as f:
    config = pickle.load(f)

# Reconstruct policy
from codesign_cartpole_newton_vec import CartPolePolicy
policy = CartPolePolicy()
policy.load_state_dict(config["state_dict"])
policy.eval()

# Import environment
from envs.cartpole_newton import CartPoleNewtonVecEnv, ParametricCartPoleNewton

# Create environment (single world for video)
parametric = ParametricCartPoleNewton(L_init=config["L_value"])
env = CartPoleNewtonVecEnv(parametric_model=parametric, num_worlds=1, force_max=30.0, x_limit=3.0, start_near_upright=True)
wp.synchronize()

# Create viewer
viewer = newton.viewer.ViewerGL(headless=True, width=config["width"], height=config["height"])

frames = []
obs = env.reset()[0]  # Get single world obs
sim_time = 0.0

# Forward kinematics
newton.eval_fk(env.model, env.model.joint_q, env.model.joint_qd, env.state_0)
wp.synchronize()

# Setup viewer
viewer.set_model(env.model)
viewer.set_camera(pos=wp.vec3(0.0, -3.0, 1.0), pitch=-10.0, yaw=90.0)

# Warm-up
viewer.begin_frame(0.0)
viewer.log_state(env.state_0)
viewer.end_frame()
_ = viewer.get_frame()
wp.synchronize()

# Record frames
for step in range(config["max_steps"]):
    action = policy.get_action(obs, deterministic=True)
    force = float(action[0]) * env.force_max

    viewer.begin_frame(sim_time)
    viewer.log_state(env.state_0)
    viewer.end_frame()

    frame_wp = viewer.get_frame()
    if frame_wp is not None:
        frames.append(frame_wp.numpy())

    obs_all, rewards, dones, _ = env.step(np.array([force]))
    obs = obs_all[0]
    sim_time += env.dt

wp.synchronize()
viewer.close()

# Save frames
if frames:
    video = np.stack(frames)
    with open("{output_path}", "wb") as f:
        pickle.dump(video, f)
'''

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    if result.returncode != 0:
        print(f"    [video] Subprocess error: {result.stderr[:500]}")
        # Cleanup temp files
        try:
            os.unlink(config_path)
            os.unlink(output_path)
        except:
            pass
        return None

    # Load frames
    try:
        with open(output_path, 'rb') as f:
            video = pickle.load(f)
    except:
        video = None

    # Cleanup temp files
    try:
        os.unlink(config_path)
        os.unlink(output_path)
    except:
        pass

    return video


def record_episode_video(L_value, policy, max_steps=200, width=640, height=480):
    """
    Record a video of one episode.

    Uses subprocess to get a fresh OpenGL context, avoiding Newton's ViewerGL
    resource cleanup issues that cause black screens on subsequent recordings.
    """
    # Get policy state dict for subprocess
    policy_state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}

    return _record_video_subprocess(
        L_value, policy_state_dict, max_steps, width, height
    )


class CartPolePolicy(nn.Module):
    """Policy network for cart-pole (supports batched inference)."""

    def __init__(self, obs_dim=4, act_dim=1, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, act_dim),
        )
        # init_noise_std=0.5 → log(0.5)≈-0.7 (enough exploration without drowning signal)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.7)

        # Small random init for final layer
        nn.init.uniform_(self.net[-1].weight, -0.1, 0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        # Use tanh to bound output to [-1, 1] - gives proper gradients
        return torch.tanh(self.net(x))

    def get_action(self, obs, deterministic=False):
        """Get action for single observation."""
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        with torch.no_grad():
            mean = self.forward(obs)
            if deterministic:
                return mean.numpy()
            std = torch.exp(self.log_std)
            action = mean + std * torch.randn_like(mean)
            # Clip to [-1, 1] after adding noise
            action = torch.clamp(action, -1.0, 1.0)
            return action.numpy()

    def get_actions_batch(self, obs_batch, deterministic=False):
        """Get actions for batch of observations."""
        if isinstance(obs_batch, np.ndarray):
            obs_batch = torch.FloatTensor(obs_batch)
        with torch.no_grad():
            mean = self.forward(obs_batch)
            if deterministic:
                return mean.numpy()
            std = torch.exp(self.log_std)
            actions = mean + std * torch.randn_like(mean)
            # Clip to [-1, 1] after adding noise
            actions = torch.clamp(actions, -1.0, 1.0)
            return actions.numpy()

    def get_action_and_log_prob_batch(self, obs_batch):
        """Get actions and log probs for batch."""
        mean = self.forward(obs_batch)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        actions_raw = dist.rsample()
        actions = torch.clamp(actions_raw, -1.0, 1.0)
        # Evaluate log_prob at CLAMPED actions so it matches PPO update
        log_probs = dist.log_prob(actions).sum(-1)
        return actions, log_probs


class CartPoleValue(nn.Module):
    """Value network for cart-pole (baseline for PPO advantage estimation)."""

    def __init__(self, obs_dim=4, hidden_dim=32):
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


def collect_rollout_vec(env, policy, value_net, horizon=32):
    """Collect rollout from vectorized environment (no reset between rollouts).

    Tracks per-world episode accumulators so completed episode stats
    are returned alongside rollout data (RSL-RL style, no separate eval).
    """
    wp.synchronize()  # Ensure clean state before rollout

    num_worlds = env.num_worlds
    # NOTE: no env.reset() here — continue from where we left off
    # The env auto-resets individual worlds when they terminate
    obs = env.last_obs if hasattr(env, 'last_obs') and env.last_obs is not None else env.reset()

    # Initialize episode accumulators if not present
    if not hasattr(env, 'ep_reward_accum'):
        env.ep_reward_accum = np.zeros(num_worlds)
        env.ep_length_accum = np.zeros(num_worlds)

    all_obs = []
    all_actions = []
    all_rewards = []
    all_log_probs = []
    all_dones = []
    all_values = []

    completed_rewards = []
    completed_lengths = []

    for _ in range(horizon):
        obs_t = torch.FloatTensor(obs)
        actions, log_probs = policy.get_action_and_log_prob_batch(obs_t)
        with torch.no_grad():
            values = value_net(obs_t)

        actions_np = actions.detach().numpy()
        next_obs, rewards, dones, _ = env.step(actions_np)

        all_obs.append(obs)
        all_actions.append(actions.detach())
        all_rewards.append(rewards)
        all_log_probs.append(log_probs.detach())
        all_dones.append(dones)
        all_values.append(values)

        # Track episode stats
        env.ep_reward_accum += rewards
        env.ep_length_accum += 1

        # Record completed episodes and reset their accumulators
        done_indices = np.where(dones)[0]
        for idx in done_indices:
            completed_rewards.append(env.ep_reward_accum[idx])
            completed_lengths.append(env.ep_length_accum[idx])
            env.ep_reward_accum[idx] = 0.0
            env.ep_length_accum[idx] = 0.0

        obs = next_obs

    # Store last obs for next rollout (no unnecessary resets)
    env.last_obs = obs

    wp.synchronize()  # Ensure all GPU operations complete

    # Bootstrap value at last observation (critical for short rollouts)
    with torch.no_grad():
        last_value = value_net(torch.FloatTensor(obs))

    # Stack into tensors: (horizon, num_worlds, ...)
    return {
        "observations": torch.FloatTensor(np.array(all_obs)),  # (H, N, obs_dim)
        "actions": torch.stack(all_actions),                    # (H, N, act_dim)
        "rewards": torch.FloatTensor(np.array(all_rewards)),    # (H, N)
        "log_probs": torch.stack(all_log_probs),                # (H, N)
        "dones": torch.FloatTensor(np.array(all_dones)),        # (H, N)
        "values": torch.stack(all_values),                      # (H, N)
        "last_value": last_value,                               # (N,) bootstrap
        "completed_rewards": completed_rewards,
        "completed_lengths": completed_lengths,
    }


def ppo_update_vec(policy, value_net, optimizer, rollout, n_epochs=5, clip_ratio=0.2,
                   gamma=0.99, gae_lambda=0.95, value_coeff=1.0, entropy_coeff=0.005,
                   num_mini_batches=8, desired_kl=0.005):
    """PPO update with GAE, mini-batches, and adaptive LR (RSL-RL style).

    Returns mean KL divergence for logging and LR adaptation.
    """
    H, N = rollout["rewards"].shape

    rewards = rollout["rewards"]       # (H, N)
    dones = rollout["dones"]           # (H, N)
    values = rollout["values"]         # (H, N)
    last_value = rollout["last_value"] # (N,) bootstrap value

    # --- Compute GAE advantages with bootstrap ---
    with torch.no_grad():
        advantages = torch.zeros(H, N)
        last_gae = torch.zeros(N)

        for t in reversed(range(H)):
            if t == H - 1:
                next_value = last_value  # Bootstrap from value estimate
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values

    # Reshape for updates
    total_samples = H * N
    obs_flat = rollout["observations"].reshape(total_samples, -1)
    acts_flat = rollout["actions"].reshape(total_samples, -1)
    old_log_probs_flat = rollout["log_probs"].reshape(total_samples)
    advantages_flat = advantages.reshape(total_samples)
    returns_flat = returns.reshape(total_samples)

    # Normalize advantages
    advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

    all_params = list(policy.parameters()) + list(value_net.parameters())
    mini_batch_size = total_samples // num_mini_batches
    mean_kl = 0.0

    for epoch in range(n_epochs):
        # Shuffle indices for mini-batches
        perm = torch.randperm(total_samples)

        for mb in range(num_mini_batches):
            idx = perm[mb * mini_batch_size : (mb + 1) * mini_batch_size]

            obs_mb = obs_flat[idx]
            acts_mb = acts_flat[idx]
            old_lp_mb = old_log_probs_flat[idx]
            adv_mb = advantages_flat[idx]
            ret_mb = returns_flat[idx]

            # Policy: recompute log probs and entropy
            mean = policy(obs_mb)
            std = torch.exp(policy.log_std)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(acts_mb).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            # Value prediction
            values_pred = value_net(obs_mb)

            # PPO clipped policy loss
            ratio = torch.exp(log_probs - old_lp_mb)
            clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            policy_loss = -torch.min(ratio * adv_mb, clipped_ratio * adv_mb).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(values_pred, ret_mb)

            # Total loss
            loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

        # Track KL after each epoch (approximate KL)
        with torch.no_grad():
            mean_all = policy(obs_flat)
            std_all = torch.exp(policy.log_std)
            dist_all = torch.distributions.Normal(mean_all, std_all)
            new_lp = dist_all.log_prob(acts_flat).sum(-1)
            mean_kl = (old_log_probs_flat - new_lp).mean().item()

        # Early stop if KL too large (RSL-RL style)
        if mean_kl > 2.0 * desired_kl:
            break

    # Adaptive LR (RSL-RL style)
    current_lr = optimizer.param_groups[0]['lr']
    if mean_kl > 2.0 * desired_kl:
        new_lr = max(current_lr / 1.5, 1e-5)
    elif mean_kl < desired_kl / 2.0:
        new_lr = min(current_lr * 1.5, 2e-4)
    else:
        new_lr = current_lr
    for pg in optimizer.param_groups:
        pg['lr'] = new_lr

    return mean_kl


def evaluate_policy_vec(env, policy):
    """Evaluate policy across all worlds for a full episode (no early break)."""
    obs = env.reset()
    episode_rewards = np.zeros(env.num_worlds)
    episode_lengths = np.zeros(env.num_worlds)
    completed = np.zeros(env.num_worlds, dtype=bool)

    for step in range(env.max_steps):
        actions = policy.get_actions_batch(obs, deterministic=True)
        obs, rewards, dones, _ = env.step(actions)

        # Accumulate rewards only for still-running episodes
        episode_rewards += rewards * (~completed)
        episode_lengths += (~completed).astype(float)

        # Mark completed episodes
        completed = completed | dones

    # Restore last_obs so training rollouts continue normally
    env.last_obs = None

    # Average ALL worlds — no selection bias
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    return mean_reward, std_reward, mean_length


def compute_design_gradient(parametric_model, policy, eps=0.02, horizon=200, n_rollouts=3):
    """Compute dReward/dL using finite differences with vec env (num_worlds=1)."""

    def eval_at_L(L_val):
        """Evaluate mean reward at a specific L value."""
        parametric_model.set_L(L_val)
        # Create vec env with single world for gradient evaluation
        env = CartPoleNewtonVecEnv(
            parametric_model=parametric_model,
            num_worlds=1,
            force_max=30.0,
            x_limit=3.0,
            start_near_upright=True,
        )

        rollout_rewards = []
        for _ in range(n_rollouts):
            obs = env.reset()[0]  # Get single world obs
            total_reward = 0
            for _ in range(horizon):
                action = policy.get_action(obs, deterministic=True)
                force = float(action[0]) * env.force_max
                obs_all, rewards, dones, _ = env.step(np.array([force]))
                obs = obs_all[0]
                total_reward += rewards[0]
                if dones[0]:
                    break
            rollout_rewards.append(total_reward)
        return np.mean(rollout_rewards)

    L_current = parametric_model.L
    wp.synchronize()

    # Evaluate at L - eps
    reward_minus = eval_at_L(L_current - eps)
    wp.synchronize()

    # Evaluate at L + eps
    reward_plus = eval_at_L(L_current + eps)
    wp.synchronize()

    # Restore L
    parametric_model.set_L(L_current)

    gradient = (reward_plus - reward_minus) / (2 * eps)
    mean_reward = (reward_plus + reward_minus) / 2

    return mean_reward, gradient


class StabilityGate:
    """Stability gating for PGHC inner loop convergence detection.

    Converges when mean reward changes less than `rel_threshold` (1%) over
    a window of recent evaluations, for `stable_iters_required` consecutive
    iterations, after at least `min_inner_iters` total iterations.
    """

    def __init__(self, rel_threshold=0.01, min_inner_iters=500, stable_iters_required=50, window=5):
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
        """Called periodically with the current mean reward."""
        self.reward_history.append(mean_reward)

    def tick(self, n_iters=1):
        """Called every inner iteration to count total and stable iters."""
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


def pghc_codesign_vec(
    n_outer_iterations=15,
    design_lr=0.02,
    max_step=0.01,
    initial_L=0.6,
    num_worlds=1024,
    use_wandb=False,
    video_every_n_iters=100,
):
    """
    PGHC Co-Design using vectorized environments for fast training.
    """
    print("=" * 60)
    print("PGHC Co-Design (Vectorized Newton Physics)")
    print("=" * 60)

    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="pghc-codesign",
            name=f"cartpole-vec-{num_worlds}w",
            config={
                "level": "1.5-vec",
                "num_worlds": num_worlds,
                "n_outer_iterations": n_outer_iterations,
                "convergence": "reward plateau (1% rel change, window=5)",
                "min_inner_iters": 500,
                "stable_iters_required": 50,
                "design_lr": design_lr,
                "initial_L": initial_L,
                "force_max": 30.0,
                "x_limit": 3.0,
            },
        )
        print(f"  [wandb] Logging enabled")
    elif use_wandb and not WANDB_AVAILABLE:
        print("  [wandb] Not available")
        use_wandb = False

    # Initialize parametric model
    parametric_model = ParametricCartPoleNewton(L_init=initial_L)

    # Create vectorized environment
    env = CartPoleNewtonVecEnv(
        num_worlds=num_worlds,
        parametric_model=parametric_model,
        force_max=30.0,
        x_limit=3.0,  # IsaacLab uses (-3, 3)
        start_near_upright=True,  # Balance task first (like IsaacLab)
    )

    policy = CartPolePolicy()
    value_net = CartPoleValue()
    optimizer = optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()), lr=2e-4
    )

    stability_gate = StabilityGate(
        rel_threshold=0.02,
        min_inner_iters=500,
        stable_iters_required=50,
    )

    history = {
        "L": [parametric_model.L],
        "rewards": [],
        "gradients": [],
        "inner_iterations": [],
    }

    global_step = 0

    print(f"\nConfiguration:")
    print(f"  Num parallel worlds: {num_worlds}")
    print(f"  Initial L: {parametric_model.L:.3f} m")
    print(f"  Force max: {env.force_max} N")
    print(f"  Cart bounds: (-{env.x_limit}, {env.x_limit})")
    print(f"  Start near upright: {env.start_near_upright}")
    print(f"  Convergence: reward plateau (<1% change, window=5, 50 stable iters after 500 min)")

    # Record initial video (untrained policy)
    if use_wandb and video_every_n_iters > 0:
        print("\n  [wandb] Recording initial policy video...")
        video = record_episode_video(parametric_model.L, policy, max_steps=200)
        if video is not None:
            wandb.log({"video/episode": wandb.Video(video.transpose(0, 3, 1, 2), fps=50, format="mp4")}, step=0)

    for outer_iter in range(n_outer_iterations):
        print(f"\n{'='*60}")
        print(f"Outer Iteration {outer_iter + 1}/{n_outer_iterations}")
        print(f"{'='*60}")
        print(f"  Current L = {parametric_model.L:.3f} m")

        if use_wandb:
            wandb.log({
                "outer/iteration": outer_iter + 1,
                "outer/L_current": parametric_model.L,
            }, step=global_step)

        # =============================================
        # INNER LOOP: Train until convergence
        # =============================================
        print(f"\n  [Inner Loop] Training PPO ({num_worlds} parallel envs)...")
        stability_gate.reset()

        log_every = 10  # Print/log stats every N iters
        inner_iter = 0
        mean_rew, std_rew, mean_len = 0.0, 0.0, 0.0

        # Rolling buffer of completed episodes (RSL-RL style)
        reward_buffer = deque(maxlen=200)
        length_buffer = deque(maxlen=200)

        while True:
            # Collect rollout from all worlds
            try:
                rollout = collect_rollout_vec(env, policy, value_net, horizon=32)
            except Exception as e:
                print(f"    [WARN] Physics crash at iter {inner_iter+1}: {type(e).__name__}. Resetting env...")
                wp.synchronize()
                env.last_obs = None
                env.ep_reward_accum = None
                env.ep_length_accum = None
                delattr(env, 'ep_reward_accum')
                delattr(env, 'ep_length_accum')
                env.reset()
                continue
            mean_kl = ppo_update_vec(policy, value_net, optimizer, rollout)

            global_step += 1

            # Per-step mean reward from rollout (all worlds, all timesteps)
            rollout_mean_reward = rollout["rewards"].mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            # Track completed episodes from training rollouts
            reward_buffer.extend(rollout["completed_rewards"])
            length_buffer.extend(rollout["completed_lengths"])

            # Update stats from buffer
            if len(reward_buffer) > 0:
                mean_rew = np.mean(reward_buffer)
                std_rew = np.std(reward_buffer)
                mean_len = np.mean(length_buffer)

            # Update stability gate every log_every iters
            if (inner_iter + 1) % log_every == 0 and len(reward_buffer) > 0:
                stability_gate.update(mean_rew)

            if use_wandb:
                log_dict = {
                    "inner/reward_per_step": rollout_mean_reward,
                    "inner/iteration": inner_iter + 1,
                    "inner/kl": mean_kl,
                    "inner/lr": current_lr,
                    "inner/mean_reward": mean_rew,
                    "inner/reward_std": std_rew,
                    "inner/episode_length": mean_len,
                    "design/L": parametric_model.L,
                }
                wandb.log(log_dict, step=global_step)

            if (inner_iter + 1) % log_every == 0:
                print(f"    Iter {inner_iter + 1}: "
                      f"reward/step={rollout_mean_reward:.2f}, "
                      f"kl={mean_kl:.4f}, lr={current_lr:.1e}, "
                      f"mean_reward={mean_rew:.1f} ±{std_rew:.1f}, mean_len={mean_len:.0f}")

            # Record video every N inner iterations
            if use_wandb and video_every_n_iters > 0 and (inner_iter + 1) % video_every_n_iters == 0:
                try:
                    print(f"    [wandb] Recording video (iter {inner_iter + 1}, L={parametric_model.L:.2f}m)...")
                    video = record_episode_video(parametric_model.L, policy, max_steps=500)
                    if video is not None:
                        wandb.log({
                            "video/episode": wandb.Video(video.transpose(0, 3, 1, 2), fps=50, format="mp4"),
                            "video/inner_iter": inner_iter + 1,
                            "video/outer_iter": outer_iter + 1,
                            "video/L": parametric_model.L,
                            "video/reward": mean_rew,
                        }, step=global_step)
                except Exception as e:
                    print(f"    [wandb] Video recording failed: {type(e).__name__}. Skipping.")

            stability_gate.tick()
            inner_iter += 1

            if stability_gate.is_converged():
                print(f"    CONVERGED at iter {inner_iter} (stable for {stability_gate.stable_count} iters)")
                break

        history["rewards"].append(mean_rew)
        history["inner_iterations"].append(stability_gate.total_inner_iters)
        print(f"  Policy converged. Reward = {mean_rew:.1f} +/- {std_rew:.1f}, Length = {mean_len:.0f}")

        # =============================================
        # OUTER LOOP: Compute design gradient
        # =============================================
        print(f"\n  [Outer Loop] Computing dReward/dL (frozen policy)...")
        wp.synchronize()

        _, gradient = compute_design_gradient(
            parametric_model, policy,
            eps=0.02, horizon=200, n_rollouts=3
        )
        history["gradients"].append(gradient)

        print(f"  dReward/dL = {gradient:.4f}")

        # =============================================
        # Update L (gradient ascent)
        # =============================================
        step = np.clip(design_lr * gradient, -max_step, max_step)
        old_L = parametric_model.L
        parametric_model.set_L(old_L + step)

        # Rebuild vectorized environment with new L
        wp.synchronize()
        env = CartPoleNewtonVecEnv(
            num_worlds=num_worlds,
            parametric_model=parametric_model,
            force_max=30.0,
            x_limit=3.0,
            start_near_upright=True,
        )
        env.last_obs = None  # Force fresh reset on next rollout
        wp.synchronize()

        print(f"\n  L update: {old_L:.3f} -> {parametric_model.L:.3f} m (step = {step:+.4f})")
        history["L"].append(parametric_model.L)

        if use_wandb:
            wandb.log({
                "outer/reward_at_convergence": mean_rew,
                "outer/gradient": gradient,
                "outer/L_step": step,
                "outer/L_new": parametric_model.L,
                "outer/inner_iterations_used": stability_gate.total_inner_iters,
            }, step=global_step)

    # =============================================
    # Final Results
    # =============================================
    print("\n" + "=" * 60)
    print("PGHC Co-Design Complete!")
    print("=" * 60)

    print(f"\nPole length evolution:")
    print(f"  Initial: {history['L'][0]:.3f} m")
    print(f"  Final:   {history['L'][-1]:.3f} m")
    print(f"  Change:  {history['L'][-1] - history['L'][0]:+.3f} m")

    total_samples = sum(history["inner_iterations"]) * num_worlds * 16
    print(f"\nTotal training samples: {total_samples:,}")
    print(f"  ({num_worlds} worlds x 16 steps x {sum(history['inner_iterations'])} iters)")

    if use_wandb:
        # Record final video
        if video_every_n_iters > 0:
            print("\n  [wandb] Recording final policy video...")
            video = record_episode_video(parametric_model.L, policy, max_steps=300)
            if video is not None:
                wandb.log({
                    "video/final": wandb.Video(video.transpose(0, 3, 1, 2), fps=50, format="mp4"),
                    "video/L_final": parametric_model.L,
                })

        wandb.log({
            "summary/L_initial": history["L"][0],
            "summary/L_final": history["L"][-1],
            "summary/total_samples": total_samples,
        })
        wandb.finish()

    return history, policy, parametric_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGHC Co-Design (Vectorized)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--outer-iters", type=int, default=10, help="Number of outer iterations")
    parser.add_argument("--design-lr", type=float, default=0.02, help="Design learning rate")
    parser.add_argument("--initial-L", type=float, default=0.6, help="Initial pole length")
    parser.add_argument("--num-worlds", type=int, default=2048, help="Number of parallel worlds")
    parser.add_argument("--video-every", type=int, default=100, help="Record video every N inner iterations (0 to disable)")
    args = parser.parse_args()

    history, policy, model = pghc_codesign_vec(
        n_outer_iterations=args.outer_iters,
        design_lr=args.design_lr,
        initial_L=args.initial_L,
        num_worlds=args.num_worlds,
        use_wandb=args.wandb,
        video_every_n_iters=args.video_every,
    )
