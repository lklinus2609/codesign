"""
PPO (Proximal Policy Optimization) for Cart-Pole

Inner loop policy optimizer for PGHC co-design.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Tuple, List, Optional


class ActorCritic(nn.Module):
    """
    Actor-Critic network for continuous control.

    Actor: Outputs mean of Gaussian policy
    Critic: Outputs state value V(s)
    """

    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        log_std_init: float = -0.5,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Actor network (policy mean)
        actor_layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            actor_layers.extend([
                nn.Linear(in_dim, h),
                nn.Tanh(),
            ])
            in_dim = h
        actor_layers.append(nn.Linear(in_dim, act_dim))
        self.actor_mean = nn.Sequential(*actor_layers)

        # Learnable log std
        self.actor_log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

        # Critic network (value function)
        critic_layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            critic_layers.extend([
                nn.Linear(in_dim, h),
                nn.Tanh(),
            ])
            in_dim = h
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean and value."""
        mean = self.actor_mean(obs)
        value = self.critic(obs)
        return mean, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: State value estimate
        """
        mean, value = self.forward(obs)
        std = torch.exp(self.actor_log_std)

        if deterministic:
            action = mean
            log_prob = torch.zeros(1)
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Returns:
            log_prob: Log probability of actions
            value: State value estimates
            entropy: Policy entropy
        """
        mean, value = self.forward(obs)
        std = torch.exp(self.actor_log_std)

        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, value.squeeze(-1), entropy


class RolloutBuffer:
    """Buffer for storing rollout data."""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, obs, action, reward, value, log_prob, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def compute_returns_and_advantages(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute returns and GAE advantages."""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [False])

        # GAE computation
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t + 1]
            delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        returns = advantages + values[:-1]

        return returns, advantages

    def get_tensors(self, returns, advantages, device='cpu'):
        """Convert buffer to tensors for training."""
        obs = torch.stack(self.obs).float().to(device)  # Ensure float32
        actions = torch.stack(self.actions).float().to(device)  # Ensure float32
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return obs, actions, old_log_probs, returns, advantages


class PPO:
    """
    Proximal Policy Optimization algorithm.

    For use as inner loop optimizer in PGHC.
    """

    def __init__(
        self,
        env,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        """
        Initialize PPO.

        Args:
            env: Environment
            hidden_sizes: MLP hidden layer sizes
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            vf_coef: Value function loss coefficient
            ent_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm for clipping
            n_epochs: Number of optimization epochs per update
            batch_size: Minibatch size
            device: Computation device
        """
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

        # Create actor-critic network
        self.policy = ActorCritic(
            obs_dim=env.obs_dim,
            act_dim=env.act_dim,
            hidden_sizes=hidden_sizes,
        ).to(device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Training stats
        self.total_timesteps = 0

    def collect_rollouts(self, n_steps: int) -> dict:
        """
        Collect rollouts from environment.

        Args:
            n_steps: Number of steps to collect

        Returns:
            Dictionary with rollout statistics
        """
        self.buffer.clear()

        obs = self.env.reset()
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        else:
            obs = obs.float()  # Convert to float32
        obs = obs.to(self.device)

        episode_rewards = []
        episode_lengths = []
        current_reward = 0
        current_length = 0

        for step in range(n_steps):
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(obs)

            # Clip action to environment limits
            action_clipped = torch.clamp(action, -self.env.force_max, self.env.force_max)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_clipped)
            done = terminated or truncated

            if not isinstance(next_obs, torch.Tensor):
                next_obs = torch.tensor(next_obs, dtype=torch.float32)
            else:
                next_obs = next_obs.float()  # Convert to float32
            next_obs = next_obs.to(self.device)

            # Store transition (detach to avoid graph issues)
            self.buffer.add(
                obs.cpu().detach(),
                action.cpu().detach(),
                reward.item() if isinstance(reward, torch.Tensor) else reward,
                value.item(),
                log_prob.item(),
                done,
            )

            current_reward += reward.item() if isinstance(reward, torch.Tensor) else reward
            current_length += 1
            self.total_timesteps += 1

            if done:
                episode_rewards.append(current_reward)
                episode_lengths.append(current_length)
                current_reward = 0
                current_length = 0
                obs = self.env.reset()
                if not isinstance(obs, torch.Tensor):
                    obs = torch.tensor(obs, dtype=torch.float32)
                else:
                    obs = obs.float()  # Convert to float32
                obs = obs.to(self.device)
            else:
                obs = next_obs

        # Get last value for GAE
        with torch.no_grad():
            _, _, last_value = self.policy.get_action(obs)

        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'mean_length': np.mean(episode_lengths) if episode_lengths else n_steps,
            'last_value': last_value.item(),
        }

    def update(self, last_value: float) -> dict:
        """
        Perform PPO update.

        Args:
            last_value: Value estimate for last state

        Returns:
            Dictionary with training statistics
        """
        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )

        # Get tensors
        obs, actions, old_log_probs, returns_t, advantages_t = self.buffer.get_tensors(
            returns, advantages, self.device
        )

        # Training loop
        n_samples = len(obs)
        indices = np.arange(n_samples)

        pg_losses = []
        vf_losses = []
        entropy_losses = []
        clip_fractions = []

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Get batch
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]

                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(batch_obs, batch_actions)

                # Policy loss (clipped surrogate)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                vf_loss = nn.functional.mse_loss(values, batch_returns)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = pg_loss + self.vf_coef * vf_loss + self.ent_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track stats
                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(((ratio - 1).abs() > self.clip_range).float().mean().item())

        return {
            'pg_loss': np.mean(pg_losses),
            'vf_loss': np.mean(vf_losses),
            'entropy': -np.mean(entropy_losses),
            'clip_fraction': np.mean(clip_fractions),
        }

    def train(self, total_timesteps: int, rollout_steps: int = 2048, log_interval: int = 1, verbose: bool = True) -> List[dict]:
        """
        Train PPO for specified timesteps.

        Args:
            total_timesteps: Total training timesteps
            rollout_steps: Steps per rollout
            log_interval: How often to log (in iterations)
            verbose: Print progress

        Returns:
            List of training statistics per iteration
        """
        history = []
        iteration = 0

        while self.total_timesteps < total_timesteps:
            # Collect rollouts
            rollout_stats = self.collect_rollouts(rollout_steps)

            # Update policy
            update_stats = self.update(rollout_stats['last_value'])

            # Combine stats
            stats = {**rollout_stats, **update_stats, 'timesteps': self.total_timesteps}
            history.append(stats)

            if verbose and iteration % log_interval == 0:
                print(f"Iter {iteration}: timesteps={self.total_timesteps}, "
                      f"mean_reward={rollout_stats['mean_reward']:.1f}, "
                      f"mean_length={rollout_stats['mean_length']:.0f}")

            iteration += 1

        return history

    def get_action(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Get action for evaluation (deterministic by default)."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        else:
            obs = obs.float()  # Convert to float32
        obs = obs.to(self.device)

        with torch.no_grad():
            action, _, _ = self.policy.get_action(obs, deterministic=deterministic)

        return torch.clamp(action, -self.env.force_max, self.env.force_max)

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """Callable interface for use with rollout_differentiable."""
        return self.get_action(obs, deterministic=True)

    def save(self, path: str):
        """Save policy weights."""
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        """Load policy weights."""
        self.policy.load_state_dict(torch.load(path))


if __name__ == "__main__":
    from envs.cartpole import CartPoleEnv

    print("Testing PPO on CartPole...")

    env = CartPoleEnv()
    ppo = PPO(env, hidden_sizes=(64, 64), lr=3e-4)

    # Train
    print("\nTraining PPO...")
    history = ppo.train(total_timesteps=20000, rollout_steps=2048, log_interval=1)

    # Evaluate
    print("\nEvaluating trained policy...")
    obs = env.reset()
    total_reward = 0

    for step in range(500):
        action = ppo.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward.item()

        if terminated or truncated:
            break

    print(f"Evaluation reward: {total_reward:.0f} / {step+1} steps")
