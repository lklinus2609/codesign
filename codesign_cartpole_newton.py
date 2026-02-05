#!/usr/bin/env python3
"""
Level 1.5: PGHC Co-Design for Cart-Pole using Newton Physics

This validates the full PGHC pipeline with Newton before moving to Level 2 (Ant).

Uses stability gating: inner loop runs until policy converges (return plateau),
then outer loop updates morphology.

Run with wandb logging:
    python codesign_cartpole_newton.py --wandb
"""

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

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import tempfile
    import os
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from envs.cartpole_newton import CartPoleNewtonEnv, ParametricCartPoleNewton


def render_cartpole_frame(x, theta, L, force=0, step=0, return_so_far=0):
    """Render a single frame of the cart-pole as a numpy array."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=100)

    # Cart dimensions
    cart_width = 0.4
    cart_height = 0.2

    # Draw track
    ax.hlines(0, -2.5, 2.5, colors='gray', linewidth=2)

    # Draw cart
    cart = patches.Rectangle(
        (x - cart_width/2, 0),
        cart_width, cart_height,
        linewidth=2, edgecolor='blue', facecolor='lightblue'
    )
    ax.add_patch(cart)

    # Draw pole
    pole_x = x
    pole_y = cart_height
    pole_end_x = pole_x + L * np.sin(theta)
    pole_end_y = pole_y + L * np.cos(theta)

    ax.plot([pole_x, pole_end_x], [pole_y, pole_end_y],
            color='brown', linewidth=6, solid_capstyle='round')

    # Draw pole tip
    ax.plot(pole_end_x, pole_end_y, 'o', color='red', markersize=10)

    # Draw force arrow
    if abs(force) > 0.5:
        arrow_scale = 0.3
        ax.arrow(x, cart_height/2, force * arrow_scale, 0,
                head_width=0.05, head_length=0.02, fc='green', ec='green')

    # Labels
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 2.0)
    ax.set_aspect('equal')
    ax.set_title(f'Step: {step} | L: {L:.2f}m | θ: {np.degrees(theta):.1f}° | Return: {return_so_far:.1f}')
    ax.axis('off')

    # Convert to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    frame = np.asarray(buf)[:, :, :3]  # Remove alpha channel

    plt.close(fig)
    return frame


def record_episode_video(env, policy, max_steps=200):
    """Record a video of one episode."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    frames = []
    obs = env.reset()
    total_reward = 0

    for step in range(max_steps):
        # Get action
        action = policy.get_action(obs, deterministic=True)
        force = float(action[0]) * env.force_max

        # Render frame
        x, theta = obs[0], obs[1]
        L = env.parametric_model.L
        frame = render_cartpole_frame(x, theta, L, force, step, total_reward)
        frames.append(frame)

        # Step
        obs, reward, terminated, truncated, _ = env.step(force)
        total_reward += reward

        if terminated or truncated:
            # Add a few more frames showing final state
            for _ in range(10):
                x, theta = obs[0], obs[1]
                frame = render_cartpole_frame(x, theta, L, 0, step, total_reward)
                frames.append(frame)
            break

    return np.stack(frames)


class CartPolePolicy(nn.Module):
    """Simple policy network for cart-pole."""

    def __init__(self, obs_dim=4, act_dim=1, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        return self.net(x)

    def get_action(self, obs, deterministic=False):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        with torch.no_grad():
            mean = self.forward(obs)
            if deterministic:
                return mean.numpy()
            std = torch.exp(self.log_std)
            action = mean + std * torch.randn_like(mean)
            return action.numpy()

    def get_action_and_log_prob(self, obs):
        mean = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


def collect_rollout(env, policy, horizon=200):
    """Collect a rollout for PPO training."""
    obs = env.reset()
    observations = []
    actions = []
    rewards = []
    log_probs = []

    for _ in range(horizon):
        obs_t = torch.FloatTensor(obs)
        action, log_prob = policy.get_action_and_log_prob(obs_t)

        action_np = action.detach().numpy()
        # Scale action to force range
        force = float(action_np[0]) * env.force_max

        next_obs, reward, terminated, truncated, _ = env.step(force)

        observations.append(obs)
        actions.append(action.detach())
        rewards.append(reward)
        log_probs.append(log_prob.detach())

        obs = next_obs
        if terminated or truncated:
            obs = env.reset()

    return {
        "observations": torch.FloatTensor(np.array(observations)),
        "actions": torch.stack(actions),
        "rewards": torch.FloatTensor(rewards),
        "log_probs": torch.stack(log_probs),
    }


def ppo_update(policy, optimizer, rollout, n_epochs=4, clip_ratio=0.2):
    """Simple PPO update."""
    obs = rollout["observations"]
    acts = rollout["actions"]
    old_log_probs = rollout["log_probs"]
    rewards = rollout["rewards"]

    # Compute returns (simple sum, no discounting for short horizon)
    returns = torch.zeros_like(rewards)
    running_return = 0
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + 0.99 * running_return
        returns[t] = running_return

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


def evaluate_policy(env, policy, n_episodes=5):
    """Evaluate policy performance."""
    returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        for _ in range(env.max_steps):
            action = policy.get_action(obs, deterministic=True)
            force = float(action[0]) * env.force_max
            obs, reward, terminated, truncated, _ = env.step(force)
            total_reward += reward
            if terminated or truncated:
                break
        returns.append(total_reward)
    return np.mean(returns), np.std(returns)


def compute_design_gradient(env, policy, eps=0.02, horizon=200, n_rollouts=3):
    """Compute dReturn/dL with frozen policy via finite difference."""
    L_current = env.parametric_model.L

    def policy_fn(obs):
        action = policy.get_action(obs, deterministic=True)
        return float(action[0]) * env.force_max

    # Evaluate at L - eps
    env.parametric_model.set_L(L_current - eps)
    env._build_model()
    returns_minus = []
    for _ in range(n_rollouts):
        obs = env.reset()
        total_return = 0
        for _ in range(horizon):
            force = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(force)
            total_return += reward
            if terminated or truncated:
                break
        returns_minus.append(total_return)

    # Evaluate at L + eps
    env.parametric_model.set_L(L_current + eps)
    env._build_model()
    returns_plus = []
    for _ in range(n_rollouts):
        obs = env.reset()
        total_return = 0
        for _ in range(horizon):
            force = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(force)
            total_return += reward
            if terminated or truncated:
                break
        returns_plus.append(total_return)

    # Restore L
    env.parametric_model.set_L(L_current)
    env._build_model()

    gradient = (np.mean(returns_plus) - np.mean(returns_minus)) / (2 * eps)
    mean_return = (np.mean(returns_plus) + np.mean(returns_minus)) / 2

    return mean_return, gradient


class StabilityGate:
    """
    Stability gating for PGHC.

    Monitors policy performance and signals when policy has converged
    (return has plateaued), satisfying the Envelope Theorem assumption.
    """

    def __init__(self, window_size=10, threshold=0.01, min_iterations=20):
        """
        Args:
            window_size: Number of recent returns to track
            threshold: Relative change threshold (5% = 0.05)
            min_iterations: Minimum iterations before allowing convergence
        """
        self.window_size = window_size
        self.threshold = threshold
        self.min_iterations = min_iterations
        self.returns = deque(maxlen=window_size)
        self.iteration = 0

    def reset(self):
        """Reset for new morphology."""
        self.returns.clear()
        self.iteration = 0

    def update(self, mean_return):
        """Update with new return value."""
        self.returns.append(mean_return)
        self.iteration += 1

    def is_converged(self):
        """Check if policy has converged (return plateau)."""
        if self.iteration < self.min_iterations:
            return False

        if len(self.returns) < self.window_size:
            return False

        # Compute relative change: (max - min) / |mean|
        returns_arr = np.array(self.returns)
        mean_val = np.mean(returns_arr)
        if abs(mean_val) < 1e-6:
            return True  # Near zero, consider converged

        relative_change = (np.max(returns_arr) - np.min(returns_arr)) / abs(mean_val)
        return relative_change < self.threshold

    def get_stats(self):
        """Get current stability statistics."""
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


def pghc_codesign(
    n_outer_iterations=15,
    max_inner_iterations=100,
    stability_window=10,
    stability_threshold=0.01,
    min_inner_iterations=20,
    design_lr=0.02,
    max_step=0.1,
    initial_L=0.6,
    ctrl_cost_weight=0.5,
    use_wandb=False,
):
    """
    PGHC Co-Design for Newton Cart-Pole with Stability Gating.

    Inner loop runs until policy converges (return plateau), then outer loop
    updates morphology. This enforces the Envelope Theorem assumption.
    """
    print("=" * 60)
    print("PGHC Co-Design for Cart-Pole (Newton Physics)")
    print("Stability Gating Mode")
    print("=" * 60)

    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="pghc-codesign",
            name="cartpole-newton-L1.5",
            config={
                "level": "1.5",
                "environment": "cartpole-newton",
                "n_outer_iterations": n_outer_iterations,
                "max_inner_iterations": max_inner_iterations,
                "stability_window": stability_window,
                "stability_threshold": stability_threshold,
                "min_inner_iterations": min_inner_iterations,
                "design_lr": design_lr,
                "max_step": max_step,
                "initial_L": initial_L,
                "ctrl_cost_weight": ctrl_cost_weight,
            },
        )
        print("  [wandb] Logging enabled")
    elif use_wandb and not WANDB_AVAILABLE:
        print("  [wandb] Not available, install with: pip install wandb")
        use_wandb = False

    # Initialize
    parametric_model = ParametricCartPoleNewton(L_init=initial_L)
    env = CartPoleNewtonEnv(parametric_model=parametric_model, ctrl_cost_weight=ctrl_cost_weight)

    policy = CartPolePolicy()
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    # Stability gate
    stability_gate = StabilityGate(
        window_size=stability_window,
        threshold=stability_threshold,
        min_iterations=min_inner_iterations,
    )

    # Track history
    history = {
        "L": [parametric_model.L],
        "returns": [],
        "gradients": [],
        "inner_iterations": [],
    }

    # Global step counter for wandb x-axis
    global_step = 0

    print(f"\nInitial pole length L = {parametric_model.L:.3f} m")
    print(f"L range: [{parametric_model.L_min}, {parametric_model.L_max}] m")
    print(f"Control cost weight: {ctrl_cost_weight}")
    print(f"\nStability gating:")
    print(f"  Window size: {stability_window}")
    print(f"  Threshold: {stability_threshold:.1%}")
    print(f"  Min iterations: {min_inner_iterations}")

    # Record initial video (untrained policy)
    if use_wandb and MATPLOTLIB_AVAILABLE:
        print("\n  [wandb] Recording initial policy video...")
        video = record_episode_video(env, policy, max_steps=200)
        if video is not None:
            wandb.log({"video/episode": wandb.Video(video.transpose(0, 3, 1, 2), fps=30, format="mp4")}, step=0)

    for outer_iter in range(n_outer_iterations):
        print(f"\n{'='*60}")
        print(f"Outer Iteration {outer_iter + 1}/{n_outer_iterations}")
        print(f"{'='*60}")
        print(f"  Current L = {parametric_model.L:.3f} m")

        # Log outer loop start
        if use_wandb:
            wandb.log({
                "outer/iteration": outer_iter + 1,
                "outer/L_current": parametric_model.L,
                "outer/event": 1,  # Marker for outer loop start
            }, step=global_step)

        # =============================================
        # INNER LOOP: Train until convergence
        # =============================================
        print(f"\n  [Inner Loop] Training PPO until convergence...")
        stability_gate.reset()

        for inner_iter in range(max_inner_iterations):
            rollout = collect_rollout(env, policy, horizon=200)
            ppo_update(policy, optimizer, rollout)

            # Evaluate and update stability gate
            mean_ret, std_ret = evaluate_policy(env, policy, n_episodes=3)
            stability_gate.update(mean_ret)
            stats = stability_gate.get_stats()

            global_step += 1

            # Log every inner iteration to wandb
            if use_wandb:
                wandb.log({
                    "inner/return_mean": mean_ret,
                    "inner/return_std": std_ret,
                    "inner/relative_change": stats["relative_change"],
                    "inner/iteration": inner_iter + 1,
                    "design/L": parametric_model.L,
                    "design/pole_mass": parametric_model.pole_mass,
                    "outer/iteration": outer_iter + 1,
                }, step=global_step)

            if (inner_iter + 1) % 10 == 0:
                print(f"    Iter {inner_iter + 1}: return = {mean_ret:.1f}, "
                      f"rel_change = {stats['relative_change']:.3f}")

            # Check convergence
            if stability_gate.is_converged():
                print(f"    CONVERGED at iter {inner_iter + 1} "
                      f"(rel_change = {stats['relative_change']:.3f} < {stability_threshold})")
                if use_wandb:
                    wandb.log({
                        "inner/converged": 1,
                        "inner/convergence_iter": inner_iter + 1,
                    }, step=global_step)
                break
        else:
            print(f"    MAX ITERATIONS reached ({max_inner_iterations})")
            if use_wandb:
                wandb.log({"inner/converged": 0}, step=global_step)

        # Final evaluation
        mean_return, std_return = evaluate_policy(env, policy, n_episodes=5)
        history["returns"].append(mean_return)
        history["inner_iterations"].append(stability_gate.iteration)
        print(f"  Policy converged. Return = {mean_return:.1f} +/- {std_return:.1f}")

        # Record video of converged policy (before morphology change)
        if use_wandb and MATPLOTLIB_AVAILABLE:
            print(f"  [wandb] Recording video (L={parametric_model.L:.2f}m)...")
            video = record_episode_video(env, policy, max_steps=200)
            if video is not None:
                wandb.log({
                    "video/episode": wandb.Video(video.transpose(0, 3, 1, 2), fps=30, format="mp4"),
                    "video/outer_iter": outer_iter + 1,
                    "video/L": parametric_model.L,
                    "video/return": mean_return,
                }, step=global_step)

        # =============================================
        # OUTER LOOP: Compute design gradient
        # =============================================
        print(f"\n  [Outer Loop] Computing dReturn/dL (frozen policy)...")

        _, gradient = compute_design_gradient(env, policy, eps=0.02, horizon=200, n_rollouts=3)
        history["gradients"].append(gradient)

        print(f"  dReturn/dL = {gradient:.4f}")

        # =============================================
        # Update L (gradient ascent)
        # =============================================
        step = np.clip(design_lr * gradient, -max_step, max_step)
        old_L = parametric_model.L
        parametric_model.set_L(old_L + step)

        # Rebuild environment with new L
        env = CartPoleNewtonEnv(parametric_model=parametric_model, ctrl_cost_weight=ctrl_cost_weight)

        print(f"\n  L update: {old_L:.3f} -> {parametric_model.L:.3f} m (step = {step:+.4f})")
        history["L"].append(parametric_model.L)

        # Log outer loop results
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

    print(f"\nInner loop iterations per outer loop:")
    for i, iters in enumerate(history["inner_iterations"]):
        print(f"  Outer {i+1}: {iters} inner iterations")

    print(f"\nReturn progression:")
    for i, ret in enumerate(history["returns"]):
        print(f"  Iter {i+1}: {ret:.1f}")

    print(f"\nGradient history:")
    for i, grad in enumerate(history["gradients"]):
        print(f"  Iter {i+1}: {grad:+.4f}")

    # Log final summary to wandb
    if use_wandb:
        # Record final video
        if MATPLOTLIB_AVAILABLE:
            print("\n  [wandb] Recording final policy video...")
            video = record_episode_video(env, policy, max_steps=300)
            if video is not None:
                wandb.log({
                    "video/final": wandb.Video(video.transpose(0, 3, 1, 2), fps=30, format="mp4"),
                    "video/L_final": parametric_model.L,
                })

        wandb.log({
            "summary/L_initial": history["L"][0],
            "summary/L_final": history["L"][-1],
            "summary/L_change": history["L"][-1] - history["L"][0],
            "summary/final_return": history["returns"][-1],
            "summary/total_inner_iterations": sum(history["inner_iterations"]),
        })
        wandb.finish()

    return history, policy, parametric_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGHC Co-Design for Cart-Pole (Newton)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--outer-iters", type=int, default=10, help="Number of outer iterations")
    parser.add_argument("--max-inner-iters", type=int, default=100, help="Max inner iterations")
    parser.add_argument("--design-lr", type=float, default=0.02, help="Design learning rate")
    parser.add_argument("--initial-L", type=float, default=0.6, help="Initial pole length")
    parser.add_argument("--ctrl-cost", type=float, default=0.5, help="Control cost weight")
    args = parser.parse_args()

    history, policy, model = pghc_codesign(
        n_outer_iterations=args.outer_iters,
        max_inner_iterations=args.max_inner_iters,
        stability_window=10,
        stability_threshold=0.01,
        min_inner_iterations=20,
        design_lr=args.design_lr,
        initial_L=args.initial_L,
        ctrl_cost_weight=args.ctrl_cost,
        use_wandb=args.wandb,
    )
