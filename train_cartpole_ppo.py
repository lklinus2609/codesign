#!/usr/bin/env python3
"""
Train PPO on Cart-Pole environment.

Usage:
    python train_cartpole_ppo.py [--timesteps 100000] [--eval-episodes 5]
"""

import argparse
from envs.cartpole import CartPoleEnv, PPO


def main():
    parser = argparse.ArgumentParser(description="Train PPO on Cart-Pole")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--rollout-steps", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Evaluation episodes")
    parser.add_argument("--log-interval", type=int, default=5, help="Log every N iterations")
    args = parser.parse_args()

    print("Cart-Pole PPO Training")
    print("=" * 50)
    print(f"Timesteps: {args.timesteps}")
    print(f"Hidden size: {args.hidden}")
    print(f"Learning rate: {args.lr}")
    print()

    # Create environment and PPO
    env = CartPoleEnv()
    print(f"Environment: obs_dim={env.obs_dim}, act_dim={env.act_dim}")
    print(f"Pole length: {env.parametric_model.get_L():.2f} m")
    print(f"Force limit: {env.force_max:.1f} N")
    print(f"Theta threshold: {env.theta_threshold:.2f} rad")
    print()

    ppo = PPO(
        env,
        hidden_sizes=(args.hidden, args.hidden),
        lr=args.lr,
    )

    # Train
    print("Training...")
    print("-" * 50)
    history = ppo.train(
        total_timesteps=args.timesteps,
        rollout_steps=args.rollout_steps,
        log_interval=args.log_interval,
    )

    # Evaluate
    print()
    print("Evaluation")
    print("-" * 50)
    rewards = []
    for ep in range(args.eval_episodes):
        obs = env.reset()
        total_reward = 0
        for step in range(500):
            action = ppo.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward.item()
            if terminated or truncated:
                break
        rewards.append(total_reward)
        print(f"Episode {ep+1}: {total_reward:.0f} steps")

    mean_reward = sum(rewards) / len(rewards)
    print()
    print(f"Mean reward: {mean_reward:.0f}")
    if mean_reward >= 450:
        print("PASS - Policy learned to balance!")
    elif mean_reward >= 200:
        print("OK - Partial learning")
    else:
        print("NEEDS MORE TRAINING - Try increasing --timesteps")


if __name__ == "__main__":
    main()
