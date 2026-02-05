#!/usr/bin/env python3
"""
Find Optimal Pole Length for Cart-Pole

Grid search over L values to find the optimal pole length L*.
This provides a baseline to verify PGHC converges correctly.

Usage:
    python find_optimal_L.py [--timesteps 100000] [--n-points 10]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from envs.cartpole import CartPoleEnv, ParametricCartPole, PPO


def evaluate_policy(env: CartPoleEnv, ppo: PPO, n_episodes: int = 20) -> Tuple[float, float]:
    """Evaluate policy and return mean and std of rewards."""
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        for _ in range(500):
            action = ppo.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward.item()
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)


def find_optimal_L(
    L_min: float = 0.3,
    L_max: float = 1.2,
    n_points: int = 10,
    timesteps: int = 100000,
    eval_episodes: int = 20,
    n_seeds: int = 3,
    verbose: bool = True,
) -> dict:
    """
    Grid search to find optimal pole length.

    Args:
        L_min, L_max: Range of L values to search
        n_points: Number of grid points
        timesteps: PPO training timesteps per L
        eval_episodes: Episodes for evaluation
        n_seeds: Number of random seeds per L (for robustness)
        verbose: Print progress

    Returns:
        Dictionary with results
    """
    L_values = np.linspace(L_min, L_max, n_points)

    results = {
        'L_values': L_values.tolist(),
        'returns_mean': [],
        'returns_std': [],
        'mass_values': [],
    }

    if verbose:
        print("Finding Optimal Pole Length for Cart-Pole")
        print("=" * 60)
        print(f"L range: [{L_min:.2f}, {L_max:.2f}] m")
        print(f"Grid points: {n_points}")
        print(f"Training timesteps: {timesteps}")
        print(f"Seeds per L: {n_seeds}")
        print()

    best_L = None
    best_return = -np.inf

    for i, L in enumerate(L_values):
        if verbose:
            print(f"[{i+1}/{n_points}] L = {L:.3f} m", end="", flush=True)

        seed_returns = []

        for seed in range(n_seeds):
            # Create environment with this L
            np.random.seed(seed * 100 + i)
            parametric_model = ParametricCartPole(L_init=L, L_min=L_min, L_max=L_max)
            env = CartPoleEnv(parametric_model=parametric_model, shaped_reward=True)

            # Train PPO
            ppo = PPO(env, hidden_sizes=(64, 64), lr=3e-4)
            ppo.train(
                total_timesteps=timesteps,
                rollout_steps=2048,
                log_interval=1000,
                verbose=False,
            )

            # Evaluate
            mean_return, std_return = evaluate_policy(env, ppo, n_episodes=eval_episodes)
            seed_returns.append(mean_return)

        # Average over seeds
        avg_return = np.mean(seed_returns)
        std_return = np.std(seed_returns)

        # Get pole mass at this L
        pole_mass = parametric_model.pole_mass

        results['returns_mean'].append(avg_return)
        results['returns_std'].append(std_return)
        results['mass_values'].append(pole_mass)

        if verbose:
            print(f" | m = {pole_mass:.3f} kg | Return = {avg_return:.1f} +/- {std_return:.1f}")

        if avg_return > best_return:
            best_return = avg_return
            best_L = L

    results['optimal_L'] = best_L
    results['optimal_return'] = best_return
    results['optimal_mass'] = ParametricCartPole(L_init=best_L).pole_mass

    if verbose:
        print()
        print("=" * 60)
        print(f"OPTIMAL L* = {best_L:.3f} m")
        print(f"Optimal pole mass = {results['optimal_mass']:.3f} kg")
        print(f"Optimal return = {best_return:.1f}")
        print()
        print("Use this to verify PGHC:")
        print(f"  - Start PGHC from L=0.6 (or other suboptimal)")
        print(f"  - Check if it converges to L* ~ {best_L:.2f}")
        print(f"  - Check if final return ~ {best_return:.0f}")

    return results


def plot_results(results: dict, save_path: str = None):
    """Plot the L vs Return curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    L_values = results['L_values']
    returns_mean = results['returns_mean']
    returns_std = results['returns_std']
    mass_values = results['mass_values']
    optimal_L = results['optimal_L']
    optimal_return = results['optimal_return']

    # Left plot: L vs Return
    ax = axes[0]
    ax.errorbar(L_values, returns_mean, yerr=returns_std, fmt='b-o', capsize=3, label='Mean +/- Std')
    ax.axvline(x=optimal_L, color='r', linestyle='--', label=f'Optimal L* = {optimal_L:.3f} m')
    ax.axhline(y=optimal_return, color='g', linestyle=':', alpha=0.5)
    ax.set_xlabel("Pole Half-Length L (m)")
    ax.set_ylabel("Return (after PPO training)")
    ax.set_title("Cart-Pole: Optimal Pole Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right plot: L vs Mass and Return (dual axis)
    ax1 = axes[1]
    ax2 = ax1.twinx()

    line1, = ax1.plot(L_values, returns_mean, 'b-o', label='Return')
    line2, = ax2.plot(L_values, mass_values, 'r-s', label='Pole Mass')

    ax1.set_xlabel("Pole Half-Length L (m)")
    ax1.set_ylabel("Return", color='b')
    ax2.set_ylabel("Pole Mass (kg)", color='r')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.set_title("Return vs Mass Trade-off")

    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Find optimal pole length for cart-pole")
    parser.add_argument("--L-min", type=float, default=0.3, help="Minimum L")
    parser.add_argument("--L-max", type=float, default=1.2, help="Maximum L")
    parser.add_argument("--n-points", type=int, default=10, help="Grid points")
    parser.add_argument("--timesteps", type=int, default=100000, help="PPO timesteps per L")
    parser.add_argument("--n-seeds", type=int, default=3, help="Seeds per L")
    parser.add_argument("--save-plot", type=str, default=None, help="Save plot path")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    results = find_optimal_L(
        L_min=args.L_min,
        L_max=args.L_max,
        n_points=args.n_points,
        timesteps=args.timesteps,
        n_seeds=args.n_seeds,
    )

    if not args.no_plot:
        plot_results(results, save_path=args.save_plot)

    return results


if __name__ == "__main__":
    main()
