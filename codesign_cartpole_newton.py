#!/usr/bin/env python3
"""
Level 1.5: PGHC Co-Design for Cart-Pole using Newton Physics

This validates the full PGHC pipeline with Newton before moving to Level 2 (Ant).

Uses simplified PGHC (trusts gradient with small steps).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from envs.cartpole_newton import CartPoleNewtonEnv, ParametricCartPoleNewton


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


def pghc_codesign(
    n_outer_iterations=15,
    n_inner_iterations=20,
    design_lr=0.02,
    max_step=0.1,
    initial_L=0.6,
):
    """
    PGHC Co-Design for Newton Cart-Pole.

    Simplified version: trusts gradient direction with small steps.
    """
    print("=" * 60)
    print("PGHC Co-Design for Cart-Pole (Newton Physics)")
    print("=" * 60)

    # Initialize
    parametric_model = ParametricCartPoleNewton(L_init=initial_L)
    env = CartPoleNewtonEnv(parametric_model=parametric_model)

    policy = CartPolePolicy()
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    # Track history
    history = {
        "L": [parametric_model.L],
        "returns": [],
        "gradients": [],
    }

    print(f"\nInitial pole length L = {parametric_model.L:.3f} m")
    print(f"L range: [{parametric_model.L_min}, {parametric_model.L_max}] m")

    for outer_iter in range(n_outer_iterations):
        print(f"\n{'='*60}")
        print(f"Outer Iteration {outer_iter + 1}/{n_outer_iterations}")
        print(f"{'='*60}")
        print(f"  Current L = {parametric_model.L:.3f} m")

        # =============================================
        # INNER LOOP: Train policy at current L
        # =============================================
        print(f"\n  [Inner Loop] Training PPO for {n_inner_iterations} iterations...")

        for inner_iter in range(n_inner_iterations):
            rollout = collect_rollout(env, policy, horizon=200)
            ppo_update(policy, optimizer, rollout)

            if (inner_iter + 1) % 10 == 0:
                mean_ret, std_ret = evaluate_policy(env, policy, n_episodes=3)
                print(f"    Inner iter {inner_iter + 1}: return = {mean_ret:.1f} +/- {std_ret:.1f}")

        # Final evaluation
        mean_return, std_return = evaluate_policy(env, policy, n_episodes=5)
        history["returns"].append(mean_return)
        print(f"  Policy trained. Return = {mean_return:.1f} +/- {std_return:.1f}")

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
        env = CartPoleNewtonEnv(parametric_model=parametric_model)

        print(f"\n  L update: {old_L:.3f} -> {parametric_model.L:.3f} m (step = {step:+.4f})")
        history["L"].append(parametric_model.L)

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

    print(f"\nReturn progression:")
    for i, ret in enumerate(history["returns"]):
        print(f"  Iter {i+1}: {ret:.1f}")

    print(f"\nGradient history:")
    for i, grad in enumerate(history["gradients"]):
        print(f"  Iter {i+1}: {grad:+.4f}")

    return history, policy, parametric_model


if __name__ == "__main__":
    history, policy, model = pghc_codesign(
        n_outer_iterations=10,
        n_inner_iterations=15,
        design_lr=0.02,
        initial_L=0.6,
    )
