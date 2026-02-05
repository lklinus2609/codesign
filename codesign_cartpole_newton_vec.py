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
import gc

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
env = CartPoleNewtonVecEnv(parametric_model=parametric, num_worlds=1, force_max=100.0, x_limit=3.0, start_near_upright=True)
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
    terminated = dones[0]
    sim_time += env.dt

    if terminated or env.steps[0] >= config["max_steps"]:
        for _ in range(10):
            viewer.begin_frame(sim_time)
            viewer.log_state(env.state_0)
            viewer.end_frame()
            frame_wp = viewer.get_frame()
            if frame_wp is not None:
                frames.append(frame_wp.numpy())
        break

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

    def __init__(self, obs_dim=4, act_dim=1, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
        # Moderate exploration (exp(-0.5) ≈ 0.6)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

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
        log_probs = dist.log_prob(actions_raw).sum(-1)
        # Clip actions to [-1, 1]
        actions = torch.clamp(actions_raw, -1.0, 1.0)
        return actions, log_probs


def collect_rollout_vec(env, policy, horizon=200):
    """Collect rollout from vectorized environment."""
    num_worlds = env.num_worlds
    obs = env.reset()

    all_obs = []
    all_actions = []
    all_rewards = []
    all_log_probs = []
    all_dones = []

    for _ in range(horizon):
        obs_t = torch.FloatTensor(obs)
        actions, log_probs = policy.get_action_and_log_prob_batch(obs_t)

        actions_np = actions.detach().numpy()
        next_obs, rewards, dones, _ = env.step(actions_np)

        all_obs.append(obs)
        all_actions.append(actions.detach())
        all_rewards.append(rewards)
        all_log_probs.append(log_probs.detach())
        all_dones.append(dones)

        obs = next_obs

    # Stack into tensors: (horizon, num_worlds, ...)
    return {
        "observations": torch.FloatTensor(np.array(all_obs)),  # (H, N, obs_dim)
        "actions": torch.stack(all_actions),                    # (H, N, act_dim)
        "rewards": torch.FloatTensor(np.array(all_rewards)),    # (H, N)
        "log_probs": torch.stack(all_log_probs),                # (H, N)
        "dones": torch.FloatTensor(np.array(all_dones)),        # (H, N)
    }


def ppo_update_vec(policy, optimizer, rollout, n_epochs=4, clip_ratio=0.2, gamma=0.99):
    """PPO update with vectorized rollout data."""
    # Reshape from (H, N, ...) to (H*N, ...)
    H, N = rollout["rewards"].shape

    obs = rollout["observations"].reshape(H * N, -1)
    acts = rollout["actions"].reshape(H * N, -1)
    old_log_probs = rollout["log_probs"].reshape(H * N)
    rewards = rollout["rewards"]  # Keep (H, N) for return computation
    dones = rollout["dones"]

    # Compute returns per trajectory (handle resets)
    returns = torch.zeros(H, N)
    running_return = torch.zeros(N)

    for t in reversed(range(H)):
        # Reset return where episode ended
        running_return = running_return * (1 - dones[t])
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    returns = returns.reshape(H * N)

    # Normalize returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    for _ in range(n_epochs):
        # Recompute log probs
        mean = policy(obs)
        std = torch.exp(policy.log_std)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(acts).sum(-1)

        # PPO loss
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        loss = -torch.min(ratio * returns, clipped_ratio * returns).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()


def evaluate_policy_vec(env, policy, n_episodes=5):
    """Evaluate policy using vectorized environment."""
    # Use first n_episodes worlds for evaluation
    obs = env.reset()
    episode_returns = np.zeros(env.num_worlds)
    episode_lengths = np.zeros(env.num_worlds)
    completed = np.zeros(env.num_worlds, dtype=bool)

    for step in range(env.max_steps):
        actions = policy.get_actions_batch(obs, deterministic=True)
        obs, rewards, dones, _ = env.step(actions)

        # Accumulate rewards for incomplete episodes
        episode_returns += rewards * (~completed)
        episode_lengths += (~completed).astype(float)

        # Mark completed episodes
        completed = completed | dones

        # If enough episodes completed, stop
        if np.sum(completed) >= n_episodes:
            break

    # Return stats from first n completed episodes
    mean_return = np.mean(episode_returns[:n_episodes])
    std_return = np.std(episode_returns[:n_episodes])
    mean_length = np.mean(episode_lengths[:n_episodes])
    return mean_return, std_return, mean_length


def compute_design_gradient(parametric_model, policy, eps=0.02, horizon=200, n_rollouts=3):
    """Compute dReturn/dL using finite differences with vec env (num_worlds=1)."""

    def eval_at_L(L_val):
        """Evaluate mean return at a specific L value."""
        parametric_model.set_L(L_val)
        # Create vec env with single world for gradient evaluation
        env = CartPoleNewtonVecEnv(
            parametric_model=parametric_model,
            num_worlds=1,
            force_max=100.0,
            x_limit=3.0,
            start_near_upright=True,
        )

        returns = []
        for _ in range(n_rollouts):
            obs = env.reset()[0]  # Get single world obs
            total_return = 0
            for _ in range(horizon):
                action = policy.get_action(obs, deterministic=True)
                force = float(action[0]) * env.force_max
                obs_all, rewards, dones, _ = env.step(np.array([force]))
                obs = obs_all[0]
                total_return += rewards[0]
                if dones[0]:
                    break
            returns.append(total_return)
        return np.mean(returns)

    L_current = parametric_model.L
    wp.synchronize()

    # Evaluate at L - eps
    return_minus = eval_at_L(L_current - eps)
    wp.synchronize()

    # Evaluate at L + eps
    return_plus = eval_at_L(L_current + eps)
    wp.synchronize()

    # Restore L
    parametric_model.set_L(L_current)

    gradient = (return_plus - return_minus) / (2 * eps)
    mean_return = (return_plus + return_minus) / 2

    return mean_return, gradient


class StabilityGate:
    """Stability gating for PGHC inner loop convergence detection."""

    def __init__(self, window_size=10, threshold=0.01, min_iterations=20):
        self.window_size = window_size
        self.threshold = threshold
        self.min_iterations = min_iterations
        self.returns = deque(maxlen=window_size)
        self.iteration = 0

    def reset(self):
        self.returns.clear()
        self.iteration = 0

    def update(self, mean_return):
        self.returns.append(mean_return)
        self.iteration += 1

    def is_converged(self):
        if self.iteration < self.min_iterations:
            return False
        if len(self.returns) < self.window_size:
            return False

        returns_arr = np.array(self.returns)
        mean_val = np.mean(returns_arr)
        if abs(mean_val) < 1e-6:
            return True

        relative_change = (np.max(returns_arr) - np.min(returns_arr)) / abs(mean_val)
        return relative_change < self.threshold

    def get_stats(self):
        if len(self.returns) < 2:
            return {"mean": 0, "std": 0, "relative_change": 1.0}

        returns_arr = np.array(self.returns)
        mean_val = np.mean(returns_arr)
        std_val = np.std(returns_arr)
        relative_change = (np.max(returns_arr) - np.min(returns_arr)) / max(abs(mean_val), 1e-6)

        return {
            "mean": mean_val,
            "std": std_val,
            "relative_change": relative_change,
        }


def pghc_codesign_vec(
    n_outer_iterations=15,
    stability_window=10,
    stability_threshold=0.01,
    min_inner_iterations=20,
    outer_loop_start_iter=1000,  # Skip outer loop until this many total inner iters
    design_lr=0.02,
    max_step=0.1,
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
                "stability_threshold": stability_threshold,
                "design_lr": design_lr,
                "initial_L": initial_L,
                "force_max": 100.0,
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
        force_max=30.0,  # 30N on 1kg = 30 m/s², reasonable for swing-up
        x_limit=3.0,  # IsaacLab uses (-3, 3)
        start_near_upright=True,  # Balance task first (like IsaacLab)
    )

    policy = CartPolePolicy()
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    stability_gate = StabilityGate(
        window_size=stability_window,
        threshold=stability_threshold,
        min_iterations=min_inner_iterations,
    )

    history = {
        "L": [parametric_model.L],
        "returns": [],
        "gradients": [],
        "inner_iterations": [],
    }

    global_step = 0
    total_inner_iterations = 0  # Track total inner iters across all outer loops

    print(f"\nConfiguration:")
    print(f"  Num parallel worlds: {num_worlds}")
    print(f"  Initial L: {parametric_model.L:.3f} m")
    print(f"  Force max: {env.force_max} N")
    print(f"  Cart bounds: (-{env.x_limit}, {env.x_limit})")
    print(f"  Start near upright: {env.start_near_upright}")
    print(f"  Stability threshold: {stability_threshold:.1%}")
    print(f"  Outer loop starts after: {outer_loop_start_iter} inner iterations")

    # Record initial video (untrained policy)
    if use_wandb and video_every_n_iters > 0:
        print("\n  [wandb] Recording initial policy video...")
        video = record_episode_video(parametric_model.L, policy, max_steps=200)
        if video is not None:
            wandb.log({"video/episode": wandb.Video(video.transpose(0, 3, 1, 2), fps=30, format="mp4")}, step=0)

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

        inner_iter = 0
        while True:
            # Collect rollout from all worlds
            rollout = collect_rollout_vec(env, policy, horizon=200)
            ppo_update_vec(policy, optimizer, rollout)

            # Evaluate
            mean_ret, std_ret, mean_len = evaluate_policy_vec(env, policy, n_episodes=min(10, num_worlds))
            stability_gate.update(mean_ret)
            stats = stability_gate.get_stats()

            global_step += 1

            if use_wandb:
                wandb.log({
                    "inner/return_mean": mean_ret,
                    "inner/return_std": std_ret,
                    "inner/episode_length": mean_len,
                    "inner/relative_change": stats["relative_change"],
                    "inner/iteration": inner_iter + 1,
                    "design/L": parametric_model.L,
                }, step=global_step)

            if (inner_iter + 1) % 5 == 0:
                samples_per_iter = num_worlds * 200
                # Debug: check action distribution
                test_obs = torch.FloatTensor(env.reset())
                with torch.no_grad():
                    test_actions = policy.get_actions_batch(test_obs)
                print(f"    Iter {inner_iter + 1}: return={mean_ret:.1f}, len={mean_len:.0f}, "
                      f"actions=[{test_actions.min():.2f}, {test_actions.max():.2f}]")

            # Record video every N inner iterations
            if use_wandb and video_every_n_iters > 0 and (inner_iter + 1) % video_every_n_iters == 0:
                print(f"    [wandb] Recording video (iter {inner_iter + 1}, L={parametric_model.L:.2f}m)...")
                video = record_episode_video(parametric_model.L, policy, max_steps=200)
                if video is not None:
                    wandb.log({
                        "video/episode": wandb.Video(video.transpose(0, 3, 1, 2), fps=30, format="mp4"),
                        "video/inner_iter": inner_iter + 1,
                        "video/outer_iter": outer_iter + 1,
                        "video/L": parametric_model.L,
                        "video/return": mean_ret,
                    }, step=global_step)

            total_inner_iterations += 1
            inner_iter += 1

            # Only allow convergence break AFTER we've trained enough
            if total_inner_iterations >= outer_loop_start_iter and stability_gate.is_converged():
                print(f"    CONVERGED at iter {inner_iter} (total: {total_inner_iterations})")
                break

        # Final evaluation
        mean_return, std_return, mean_length = evaluate_policy_vec(env, policy, n_episodes=10)
        history["returns"].append(mean_return)
        history["inner_iterations"].append(stability_gate.iteration)
        print(f"  Policy converged. Return = {mean_return:.1f} +/- {std_return:.1f}, Length = {mean_length:.0f}")

        # =============================================
        # OUTER LOOP: Compute design gradient
        # =============================================
        print(f"\n  [Outer Loop] Computing dReturn/dL (frozen policy)...")
        wp.synchronize()

        _, gradient = compute_design_gradient(
            parametric_model, policy,
            eps=0.02, horizon=200, n_rollouts=3
        )
        history["gradients"].append(gradient)

        print(f"  dReturn/dL = {gradient:.4f}")

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
            force_max=100.0,
            x_limit=3.0,
            start_near_upright=True,
        )
        wp.synchronize()

        print(f"\n  L update: {old_L:.3f} -> {parametric_model.L:.3f} m (step = {step:+.4f})")
        history["L"].append(parametric_model.L)

        if use_wandb:
            wandb.log({
                "outer/return_at_convergence": mean_return,
                "outer/gradient": gradient,
                "outer/L_step": step,
                "outer/L_new": parametric_model.L,
                "outer/inner_iterations_used": stability_gate.iteration,
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

    total_samples = sum(history["inner_iterations"]) * num_worlds * 200
    print(f"\nTotal training samples: {total_samples:,}")
    print(f"  ({num_worlds} worlds x 200 steps x {sum(history['inner_iterations'])} iters)")

    if use_wandb:
        # Record final video
        if video_every_n_iters > 0:
            print("\n  [wandb] Recording final policy video...")
            video = record_episode_video(parametric_model.L, policy, max_steps=300)
            if video is not None:
                wandb.log({
                    "video/final": wandb.Video(video.transpose(0, 3, 1, 2), fps=30, format="mp4"),
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
    parser.add_argument("--num-worlds", type=int, default=1024, help="Number of parallel worlds")
    parser.add_argument("--video-every", type=int, default=100, help="Record video every N inner iterations (0 to disable)")
    args = parser.parse_args()

    history, policy, model = pghc_codesign_vec(
        n_outer_iterations=args.outer_iters,
        stability_window=10,
        stability_threshold=0.01,
        min_inner_iterations=15,
        design_lr=args.design_lr,
        initial_L=args.initial_L,
        num_worlds=args.num_worlds,
        use_wandb=args.wandb,
        video_every_n_iters=args.video_every,
    )
