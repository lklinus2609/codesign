#!/usr/bin/env python3
"""
Level 2.1: PGHC Co-Design for Walker2D Locomotion (Vectorized)

Optimizes Walker2D morphology (thigh_length, leg_length, foot_length) for
forward locomotion using PGHC with vectorized Newton physics environments.

Walker2D is a 2D biped: simpler dynamics than Ant, more relevant to humanoid.
- 17D observations, 6D actions
- 3 design parameters (symmetric for both legs)

Run with wandb logging:
    python codesign_walker2d.py --wandb --num-worlds 1024
"""

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

from envs.walker2d.walker2d_env import ParametricWalker2D
from envs.walker2d.walker2d_vec_env import Walker2DVecEnv


# ---------------------------------------------------------------------------
# Observation normalization
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Tracks running mean/variance using Welford's algorithm."""

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
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total_count) / total_count
        self.count = total_count

    def normalize(self, obs):
        obs_norm = (obs - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + 1e-8)
        return np.clip(obs_norm, -self.clip, self.clip).astype(np.float32)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class Walker2DPolicy(nn.Module):
    """Policy network for Walker2D."""

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
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        with torch.no_grad():
            mean = self.forward(obs)
            if deterministic:
                return mean.numpy()
            std = torch.exp(self.log_std)
            action = mean + std * torch.randn_like(mean)
            return torch.clamp(action, -1.0, 1.0).numpy()

    def get_actions_batch(self, obs_batch, deterministic=False):
        if isinstance(obs_batch, np.ndarray):
            obs_batch = torch.FloatTensor(obs_batch)
        with torch.no_grad():
            mean = self.forward(obs_batch)
            if deterministic:
                return mean.numpy()
            std = torch.exp(self.log_std)
            actions = mean + std * torch.randn_like(mean)
            return torch.clamp(actions, -1.0, 1.0).numpy()

    def get_action_and_log_prob_batch(self, obs_batch):
        mean = self.forward(obs_batch)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        actions_raw = dist.rsample()
        actions = torch.clamp(actions_raw, -1.0, 1.0)
        log_probs = dist.log_prob(actions).sum(-1)
        return actions, log_probs


class Walker2DValue(nn.Module):
    """Value network for Walker2D."""

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
# PPO Training
# ---------------------------------------------------------------------------

def collect_rollout_vec(env, policy, value_net, obs_rms, horizon=64):
    """Collect rollout from vectorized environment (RSL-RL style)."""
    wp.synchronize()

    num_worlds = env.num_worlds
    obs = env.last_obs if hasattr(env, 'last_obs') and env.last_obs is not None else env.reset()

    if not hasattr(env, 'ep_reward_accum') or env.ep_reward_accum is None:
        env.ep_reward_accum = np.zeros(num_worlds)
        env.ep_length_accum = np.zeros(num_worlds)

    all_obs, all_actions, all_rewards = [], [], []
    all_log_probs, all_dones, all_values = [], [], []
    completed_rewards, completed_lengths = [], []

    for _ in range(horizon):
        obs_rms.update(obs)
        obs_norm = obs_rms.normalize(obs)

        obs_t = torch.FloatTensor(obs_norm)
        actions, log_probs = policy.get_action_and_log_prob_batch(obs_t)
        with torch.no_grad():
            values = value_net(obs_t)

        actions_np = actions.detach().numpy()
        next_obs, rewards, dones, _ = env.step(actions_np)

        all_obs.append(obs_norm)
        all_actions.append(actions.detach())
        all_rewards.append(rewards)
        all_log_probs.append(log_probs.detach())
        all_dones.append(dones)
        all_values.append(values)

        env.ep_reward_accum += rewards
        env.ep_length_accum += 1

        for idx in np.where(dones)[0]:
            completed_rewards.append(env.ep_reward_accum[idx])
            completed_lengths.append(env.ep_length_accum[idx])
            env.ep_reward_accum[idx] = 0.0
            env.ep_length_accum[idx] = 0.0

        obs = next_obs

    env.last_obs = obs
    wp.synchronize()

    obs_norm = obs_rms.normalize(obs)
    with torch.no_grad():
        last_value = value_net(torch.FloatTensor(obs_norm))

    return {
        "observations": torch.FloatTensor(np.array(all_obs)),
        "actions": torch.stack(all_actions),
        "rewards": torch.FloatTensor(np.array(all_rewards)),
        "log_probs": torch.stack(all_log_probs),
        "dones": torch.FloatTensor(np.array(all_dones)),
        "values": torch.stack(all_values),
        "last_value": last_value,
        "completed_rewards": completed_rewards,
        "completed_lengths": completed_lengths,
    }


def ppo_update_vec(policy, value_net, optimizer, rollout, n_epochs=5, clip_ratio=0.2,
                   gamma=0.99, gae_lambda=0.95, value_coeff=0.5, entropy_coeff=0.01,
                   num_mini_batches=8, desired_kl=0.008):
    """PPO update with GAE, mini-batches, and adaptive LR."""
    H, N = rollout["rewards"].shape
    rewards = rollout["rewards"]
    dones = rollout["dones"]
    values = rollout["values"]
    last_value = rollout["last_value"]

    with torch.no_grad():
        advantages = torch.zeros(H, N)
        last_gae = torch.zeros(N)
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
        perm = torch.randperm(total_samples)
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


def evaluate_policy_vec(env, policy, obs_rms=None):
    """Evaluate policy across all worlds for a full episode."""
    obs = env.reset()
    episode_rewards = np.zeros(env.num_worlds)
    episode_lengths = np.zeros(env.num_worlds)
    completed = np.zeros(env.num_worlds, dtype=bool)

    for step in range(env.max_steps):
        obs_input = obs_rms.normalize(obs) if obs_rms is not None else obs
        actions = policy.get_actions_batch(obs_input, deterministic=True)
        obs, rewards, dones, _ = env.step(actions)

        episode_rewards += rewards * (~completed)
        episode_lengths += (~completed).astype(float)
        completed = completed | dones

    env.last_obs = None
    return np.mean(episode_rewards), np.std(episode_rewards), np.mean(episode_lengths)


# ---------------------------------------------------------------------------
# Design gradient (finite differences)
# ---------------------------------------------------------------------------

def compute_design_gradient(parametric_model, policy, obs_rms=None, eps=0.02,
                            num_eval_worlds=512):
    """Compute dReward/d(param) for all 3 design params via finite differences."""
    param_names = ["thigh_length", "leg_length", "foot_length"]
    current_vals = {name: getattr(parametric_model, name) for name in param_names}

    def eval_at_params(**overrides):
        for name, val in overrides.items():
            setattr(parametric_model, name, val)
        parametric_model.set_params(
            thigh_length=parametric_model.thigh_length,
            leg_length=parametric_model.leg_length,
            foot_length=parametric_model.foot_length,
        )
        env = Walker2DVecEnv(
            parametric_model=parametric_model,
            num_worlds=num_eval_worlds,
        )
        wp.synchronize()
        mean_reward, std_reward, _ = evaluate_policy_vec(env, policy, obs_rms=obs_rms)
        env.cleanup()
        wp.synchronize()
        for name, val in current_vals.items():
            setattr(parametric_model, name, val)
        return mean_reward, std_reward

    gradients = {}
    diagnostics = {}

    for name in param_names:
        val = current_vals[name]
        r_plus, std_plus = eval_at_params(**{name: val + eps})
        r_minus, std_minus = eval_at_params(**{name: val - eps})

        grad = (r_plus - r_minus) / (2 * eps)
        gradients[name] = grad
        diagnostics[name] = {
            "r_plus": r_plus, "r_minus": r_minus,
            "std_plus": std_plus, "std_minus": std_minus,
            "gradient": grad,
        }

    mean_reward = np.mean([
        (diagnostics[n]["r_plus"] + diagnostics[n]["r_minus"]) / 2.0
        for n in param_names
    ])

    return gradients, diagnostics, mean_reward


# ---------------------------------------------------------------------------
# Stability gate
# ---------------------------------------------------------------------------

class StabilityGate:
    """Inner loop convergence detection via reward plateau."""

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

def pghc_codesign_walker2d(
    n_outer_iterations=50,
    design_lr=0.005,
    initial_thigh_length=0.45,
    initial_leg_length=0.50,
    initial_foot_length=0.20,
    num_worlds=1024,
    num_eval_worlds=512,
    use_wandb=False,
    horizon=64,
):
    """PGHC Co-Design for Walker2D using vectorized environments."""
    print("=" * 60)
    print("PGHC Co-Design for Walker2D (Vectorized Newton Physics)")
    print("=" * 60)

    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="pghc-codesign",
            name=f"walker2d-vec-{num_worlds}w",
            config={
                "level": "2.1-walker2d-vec",
                "num_worlds": num_worlds,
                "num_eval_worlds": num_eval_worlds,
                "n_outer_iterations": n_outer_iterations,
                "design_lr": design_lr,
                "initial_thigh_length": initial_thigh_length,
                "initial_leg_length": initial_leg_length,
                "initial_foot_length": initial_foot_length,
                "horizon": horizon,
                "gamma": 0.99,
                "entropy_coeff": 0.01,
                "desired_kl": 0.008,
                "hidden_dim": 128,
                "gear_ratio": 100,
                "ctrl_cost_weight": 1e-3,
            },
        )
        print(f"  [wandb] Logging enabled")
    elif use_wandb:
        print("  [wandb] Not available")
        use_wandb = False

    parametric_model = ParametricWalker2D(
        thigh_length=initial_thigh_length,
        leg_length=initial_leg_length,
        foot_length=initial_foot_length,
    )

    env = Walker2DVecEnv(
        num_worlds=num_worlds,
        parametric_model=parametric_model,
    )

    policy = Walker2DPolicy()
    value_net = Walker2DValue()
    optimizer = optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()), lr=3e-4
    )

    obs_rms = RunningMeanStd(shape=(17,))
    stability_gate = StabilityGate(rel_threshold=0.02, min_inner_iters=0, stable_iters_required=50)

    param_names = ["thigh_length", "leg_length", "foot_length"]
    design_params = torch.tensor(
        [initial_thigh_length, initial_leg_length, initial_foot_length],
        dtype=torch.float32, requires_grad=False
    )
    design_optimizer = torch.optim.Adam([design_params], lr=design_lr)

    param_bounds = {
        "thigh_length": (parametric_model.thigh_length_min, parametric_model.thigh_length_max),
        "leg_length": (parametric_model.leg_length_min, parametric_model.leg_length_max),
        "foot_length": (parametric_model.foot_length_min, parametric_model.foot_length_max),
    }

    param_history = {name: deque(maxlen=5) for name in param_names}

    history = {
        "thigh_length": [parametric_model.thigh_length],
        "leg_length": [parametric_model.leg_length],
        "foot_length": [parametric_model.foot_length],
        "rewards": [],
        "gradients": [],
        "inner_iterations": [],
    }

    global_step = 0

    print(f"\nConfiguration:")
    print(f"  Num parallel worlds: {num_worlds}")
    print(f"  Num eval worlds (FD): {num_eval_worlds}")
    print(f"  Horizon: {horizon}")
    print(f"  Design optimizer: Adam (lr={design_lr})")
    print(f"  Initial morphology:")
    for name in param_names:
        print(f"    {name} = {getattr(parametric_model, name):.3f} m")
    print(f"  Init z: {parametric_model.init_z:.3f} m")

    for outer_iter in range(n_outer_iterations):
        print(f"\n{'='*60}")
        print(f"Outer Iteration {outer_iter + 1}/{n_outer_iterations}")
        print(f"{'='*60}")
        print(f"  thigh={parametric_model.thigh_length:.3f}, "
              f"leg={parametric_model.leg_length:.3f}, "
              f"foot={parametric_model.foot_length:.3f}")

        if use_wandb:
            wandb.log({
                "outer/iteration": outer_iter + 1,
                **{f"outer/{n}": getattr(parametric_model, n) for n in param_names},
            }, step=global_step)

        # =============================================
        # INNER LOOP
        # =============================================
        print(f"\n  [Inner Loop] Training PPO ({num_worlds} parallel envs)...")
        stability_gate.reset()

        log_every = 10
        inner_iter = 0
        mean_rew, std_rew, mean_len = 0.0, 0.0, 0.0
        reward_buffer = deque(maxlen=200)
        length_buffer = deque(maxlen=200)

        while True:
            try:
                rollout = collect_rollout_vec(env, policy, value_net, obs_rms, horizon=horizon)
            except Exception as e:
                print(f"    [WARN] Physics crash at iter {inner_iter+1}: {type(e).__name__}. Resetting...")
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
        # OUTER LOOP
        # =============================================
        env.cleanup()
        wp.synchronize()

        print(f"\n  [Outer Loop] Computing design gradients "
              f"(frozen policy, {num_eval_worlds} eval worlds, 6 evals)...")

        gradients, diagnostics, fd_mean_reward = compute_design_gradient(
            parametric_model, policy, obs_rms=obs_rms,
            eps=0.02, num_eval_worlds=num_eval_worlds,
        )
        history["gradients"].append(gradients)

        for name in param_names:
            d = diagnostics[name]
            print(f"  d/d({name}): R+={d['r_plus']:.1f}, R-={d['r_minus']:.1f}, "
                  f"grad={d['gradient']:.4f}")

        # =============================================
        # Update design parameters
        # =============================================
        old_params = {name: getattr(parametric_model, name) for name in param_names}

        design_optimizer.zero_grad()
        grad_tensor = torch.tensor(
            [-gradients[name] for name in param_names],
            dtype=torch.float32
        )
        design_params.grad = grad_tensor
        design_optimizer.step()

        with torch.no_grad():
            for i, name in enumerate(param_names):
                lo, hi = param_bounds[name]
                design_params[i].clamp_(lo, hi)

        for i, name in enumerate(param_names):
            setattr(parametric_model, name, design_params[i].item())

        print(f"\n  Design update:")
        for name in param_names:
            delta = getattr(parametric_model, name) - old_params[name]
            print(f"    {name}: {old_params[name]:.4f} -> "
                  f"{getattr(parametric_model, name):.4f} (delta={delta:+.5f})")

        for name in param_names:
            history[name].append(getattr(parametric_model, name))

        # Rebuild env
        env = Walker2DVecEnv(
            num_worlds=num_worlds,
            parametric_model=parametric_model,
        )
        env.last_obs = None
        wp.synchronize()

        if use_wandb:
            log_dict = {
                "outer/reward_at_convergence": mean_rew,
                "outer/fd_mean_reward": fd_mean_reward,
                "outer/inner_iterations_used": stability_gate.total_inner_iters,
            }
            for name in param_names:
                d = diagnostics[name]
                log_dict[f"outer/grad_{name}"] = d["gradient"]
                log_dict[f"outer/r_plus_{name}"] = d["r_plus"]
                log_dict[f"outer/r_minus_{name}"] = d["r_minus"]
                log_dict[f"outer/{name}_new"] = getattr(parametric_model, name)
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
    print("PGHC Co-Design Complete!")
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
    parser = argparse.ArgumentParser(description="PGHC Co-Design for Walker2D (Vectorized)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--outer-iters", type=int, default=50)
    parser.add_argument("--design-lr", type=float, default=0.005)
    parser.add_argument("--num-worlds", type=int, default=1024)
    parser.add_argument("--num-eval-worlds", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--initial-thigh", type=float, default=0.45)
    parser.add_argument("--initial-leg", type=float, default=0.50)
    parser.add_argument("--initial-foot", type=float, default=0.20)
    args = parser.parse_args()

    history, policy, model = pghc_codesign_walker2d(
        n_outer_iterations=args.outer_iters,
        design_lr=args.design_lr,
        initial_thigh_length=args.initial_thigh,
        initial_leg_length=args.initial_leg,
        initial_foot_length=args.initial_foot,
        num_worlds=args.num_worlds,
        num_eval_worlds=args.num_eval_worlds,
        use_wandb=args.wandb,
        horizon=args.horizon,
    )
