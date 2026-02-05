#!/usr/bin/env python3
"""
Level 2: PGHC Co-Design for Ant Locomotion

Optimizes Ant morphology (leg_length, foot_length) for forward locomotion
using the Performance-Gated Hybrid Co-Design algorithm.

Uses simplified PGHC (trust gradient with small steps) due to compute cost.

Run on machine with Newton/Warp installed:
    python codesign_ant.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import deque

try:
    from envs.ant import AntEnv, ParametricAnt
    NEWTON_AVAILABLE = True
except ImportError as e:
    print(f"Newton not available: {e}")
    NEWTON_AVAILABLE = False


class AntPolicy(nn.Module):
    """Actor-Critic policy for Ant."""

    def __init__(self, obs_dim=27, act_dim=8, hidden_dim=256):
        super().__init__()

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor (policy) head
        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic (value) head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.features(x)
        return self.actor_mean(features), self.critic(features)

    def get_action(self, obs, deterministic=False):
        """Get action from policy."""
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            mean, _ = self.forward(obs)
            if deterministic:
                action = mean
            else:
                std = torch.exp(self.actor_log_std)
                action = mean + std * torch.randn_like(mean)

        return torch.tanh(action).squeeze(0).numpy()


def collect_rollout(env, policy, horizon=1000):
    """Collect a single rollout."""
    obs = env.reset()
    observations = []
    actions = []
    rewards = []
    values = []
    log_probs = []
    dones = []

    for _ in range(horizon):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)

        with torch.no_grad():
            mean, value = policy(obs_t)
            std = torch.exp(policy.actor_log_std)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)

        action_np = torch.tanh(action).squeeze(0).numpy()

        next_obs, reward, terminated, truncated, info = env.step(action_np)

        observations.append(obs)
        actions.append(action.squeeze(0).numpy())
        rewards.append(reward)
        values.append(value.item())
        log_probs.append(log_prob.item())
        dones.append(terminated or truncated)

        obs = next_obs
        if terminated or truncated:
            obs = env.reset()

    return {
        "observations": np.array(observations),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "values": np.array(values),
        "log_probs": np.array(log_probs),
        "dones": np.array(dones),
    }


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = np.zeros_like(rewards)
    last_gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(policy, optimizer, rollout, n_epochs=4, clip_ratio=0.2, batch_size=64):
    """PPO policy update."""
    obs = torch.FloatTensor(rollout["observations"])
    acts = torch.FloatTensor(rollout["actions"])
    old_log_probs = torch.FloatTensor(rollout["log_probs"])

    advantages, returns = compute_gae(
        rollout["rewards"], rollout["values"], rollout["dones"]
    )
    advantages = torch.FloatTensor(advantages)
    returns = torch.FloatTensor(returns)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n_samples = len(obs)
    indices = np.arange(n_samples)

    for _ in range(n_epochs):
        np.random.shuffle(indices)

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            batch_obs = obs[batch_idx]
            batch_acts = acts[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]

            # Forward pass
            mean, values = policy(batch_obs)
            std = torch.exp(policy.actor_log_std)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(batch_acts).sum(-1)

            # PPO clipped loss
            ratio = torch.exp(log_probs - batch_old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            policy_loss = -torch.min(
                ratio * batch_advantages,
                clipped_ratio * batch_advantages
            ).mean()

            # Value loss
            value_loss = 0.5 * (batch_returns - values.squeeze()).pow(2).mean()

            # Entropy bonus
            entropy = dist.entropy().sum(-1).mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

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
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        returns.append(total_reward)

    return np.mean(returns), np.std(returns)


def train_ppo(env, policy, optimizer, n_iterations=50, horizon=2048, verbose=True):
    """Train policy with PPO."""
    best_return = -float("inf")
    returns_history = []

    for iteration in range(n_iterations):
        # Collect rollout
        rollout = collect_rollout(env, policy, horizon=horizon)

        # PPO update
        ppo_update(policy, optimizer, rollout)

        # Evaluate
        mean_return, std_return = evaluate_policy(env, policy, n_episodes=3)
        returns_history.append(mean_return)

        if mean_return > best_return:
            best_return = mean_return

        if verbose and iteration % 10 == 0:
            print(f"    PPO iter {iteration}: return = {mean_return:.1f} +/- {std_return:.1f}")

    return best_return, returns_history


def compute_design_gradient(env, policy, param_name, eps=0.02, horizon=500, n_rollouts=3):
    """
    Compute gradient of expected return w.r.t. morphology parameter.

    Uses finite difference with FROZEN policy (envelope theorem).
    """
    current_val = getattr(env.parametric_model, param_name)

    # Evaluate at param - eps
    env.parametric_model.set_params(**{param_name: current_val - eps})
    env._build_model()
    returns_minus = []
    for _ in range(n_rollouts):
        obs = env.reset()
        total_return = 0
        for _ in range(horizon):
            action = policy.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_return += reward
            if terminated or truncated:
                break
        returns_minus.append(total_return)

    # Evaluate at param + eps
    env.parametric_model.set_params(**{param_name: current_val + eps})
    env._build_model()
    returns_plus = []
    for _ in range(n_rollouts):
        obs = env.reset()
        total_return = 0
        for _ in range(horizon):
            action = policy.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_return += reward
            if terminated or truncated:
                break
        returns_plus.append(total_return)

    # Restore original
    env.parametric_model.set_params(**{param_name: current_val})
    env._build_model()

    gradient = (np.mean(returns_plus) - np.mean(returns_minus)) / (2 * eps)
    return gradient


def pghc_codesign(
    n_outer_iterations=20,
    n_inner_iterations=30,
    design_lr=0.01,
    max_step=0.05,
    verbose=True,
):
    """
    PGHC Co-Design for Ant.

    Simplified version: trusts gradient direction with small steps.
    """
    if not NEWTON_AVAILABLE:
        print("ERROR: Newton/Warp required for Ant environment")
        return

    print("=" * 70)
    print("PGHC Co-Design for Ant Locomotion")
    print("=" * 70)

    # Initialize parametric model
    parametric_model = ParametricAnt(
        leg_length=0.28,
        foot_length=0.57,
    )

    env = AntEnv(parametric_model=parametric_model)

    # Initialize policy
    policy = AntPolicy(obs_dim=env.obs_dim, act_dim=env.act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    # Track history
    history = {
        "leg_length": [parametric_model.leg_length],
        "foot_length": [parametric_model.foot_length],
        "returns": [],
        "gradients_leg": [],
        "gradients_foot": [],
    }

    print(f"\nInitial morphology:")
    print(f"  leg_length = {parametric_model.leg_length:.3f} m")
    print(f"  foot_length = {parametric_model.foot_length:.3f} m")

    for outer_iter in range(n_outer_iterations):
        print(f"\n{'='*70}")
        print(f"Outer Iteration {outer_iter + 1}/{n_outer_iterations}")
        print(f"{'='*70}")
        print(f"  Current: leg={parametric_model.leg_length:.3f}, foot={parametric_model.foot_length:.3f}")

        # ============================================
        # INNER LOOP: Train policy at current morphology
        # ============================================
        print(f"\n  [Inner Loop] Training PPO...")
        env._build_model()  # Rebuild with current params
        best_return, _ = train_ppo(
            env, policy, optimizer,
            n_iterations=n_inner_iterations,
            horizon=2048,
            verbose=verbose,
        )
        history["returns"].append(best_return)
        print(f"  Best return after training: {best_return:.1f}")

        # ============================================
        # OUTER LOOP: Compute design gradients
        # ============================================
        print(f"\n  [Outer Loop] Computing design gradients (frozen policy)...")

        # Gradient w.r.t. leg_length
        grad_leg = compute_design_gradient(
            env, policy, "leg_length",
            eps=0.02, horizon=500, n_rollouts=3
        )
        history["gradients_leg"].append(grad_leg)

        # Gradient w.r.t. foot_length
        grad_foot = compute_design_gradient(
            env, policy, "foot_length",
            eps=0.02, horizon=500, n_rollouts=3
        )
        history["gradients_foot"].append(grad_foot)

        print(f"  dReturn/d(leg_length) = {grad_leg:.4f}")
        print(f"  dReturn/d(foot_length) = {grad_foot:.4f}")

        # ============================================
        # Update morphology (gradient ascent)
        # ============================================
        step_leg = np.clip(design_lr * grad_leg, -max_step, max_step)
        step_foot = np.clip(design_lr * grad_foot, -max_step, max_step)

        new_leg = parametric_model.leg_length + step_leg
        new_foot = parametric_model.foot_length + step_foot

        # Apply with bounds
        parametric_model.set_params(leg_length=new_leg, foot_length=new_foot)

        print(f"\n  Morphology update:")
        print(f"    leg_length: {history['leg_length'][-1]:.3f} -> {parametric_model.leg_length:.3f}")
        print(f"    foot_length: {history['foot_length'][-1]:.3f} -> {parametric_model.foot_length:.3f}")

        history["leg_length"].append(parametric_model.leg_length)
        history["foot_length"].append(parametric_model.foot_length)

    # ============================================
    # Final Results
    # ============================================
    print("\n" + "=" * 70)
    print("PGHC Co-Design Complete!")
    print("=" * 70)
    print(f"\nFinal morphology:")
    print(f"  leg_length = {parametric_model.leg_length:.3f} m")
    print(f"  foot_length = {parametric_model.foot_length:.3f} m")

    print(f"\nMorphology evolution:")
    print(f"  leg_length: {history['leg_length'][0]:.3f} -> {history['leg_length'][-1]:.3f}")
    print(f"  foot_length: {history['foot_length'][0]:.3f} -> {history['foot_length'][-1]:.3f}")

    print(f"\nReturn progression:")
    for i, ret in enumerate(history["returns"]):
        print(f"  Iter {i+1}: {ret:.1f}")

    return history, policy, parametric_model


if __name__ == "__main__":
    if NEWTON_AVAILABLE:
        history, policy, model = pghc_codesign(
            n_outer_iterations=10,
            n_inner_iterations=20,
            design_lr=0.01,
        )
    else:
        print("Newton/Warp not available. Please install Newton to run this script.")
        print("  pip install -e path/to/newton")
