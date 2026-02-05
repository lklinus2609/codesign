#!/usr/bin/env python3
"""
PGHC Co-Design on Cart-Pole (Simplified)

Practical version that trusts the gradient with small steps.
No candidate evaluation - suitable for expensive environments like humanoids.

Algorithm:
    1. Train policy at current L until converged
    2. Compute dReturn/dL with frozen policy (envelope theorem)
    3. Update: L = L + lr * gradient (small step)
    4. Repeat

Usage:
    python codesign_cartpole_simple.py [--outer-iters 20] [--inner-timesteps 50000]
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict

from envs.cartpole import CartPoleEnv, ParametricCartPole, PPO


def evaluate_policy(env: CartPoleEnv, ppo: PPO, n_episodes: int = 10) -> float:
    """Evaluate policy and return mean reward."""
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
    return np.mean(rewards)


def compute_design_gradient(
    env: CartPoleEnv,
    ppo: PPO,
    n_rollouts: int = 10,
    horizon: int = 200,
) -> tuple:
    """
    Compute gradient of return w.r.t. pole length L.
    Uses frozen policy (envelope theorem).
    """
    returns = []
    gradients = []

    env.parametric_model.L.requires_grad_(True)

    for _ in range(n_rollouts):
        obs = env.reset()
        total_return = torch.tensor(0.0, dtype=torch.float64)

        for t in range(horizon):
            with torch.no_grad():
                action = ppo.get_action(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            total_return = total_return + reward

            if terminated or truncated:
                break

        if total_return.requires_grad:
            total_return.backward()
            grad = env.parametric_model.L.grad.item() if env.parametric_model.L.grad is not None else 0.0
            env.parametric_model.L.grad = None
        else:
            grad = 0.0

        returns.append(total_return.item())
        gradients.append(grad)

    return np.mean(returns), np.mean(gradients)


def run_codesign_simple(
    L_init: float = 0.6,
    L_min: float = 0.3,
    L_max: float = 1.2,
    outer_iters: int = 20,
    inner_timesteps: int = 50000,
    design_lr: float = 0.005,  # Small LR = implicit trust region
    max_step: float = 0.03,    # Max 3cm per iteration
    eval_episodes: int = 10,
    grad_rollouts: int = 10,
    verbose: bool = True,
) -> Dict:
    """
    Run simplified PGHC co-design.

    No candidate evaluation - just follow gradient with small steps.
    Suitable for expensive environments (humanoids).
    """
    # Initialize
    parametric_model = ParametricCartPole(L_init=L_init, L_min=L_min, L_max=L_max)
    env = CartPoleEnv(parametric_model=parametric_model, shaped_reward=True)

    history = {
        'L': [L_init],
        'return': [],
        'gradient': [],
    }

    if verbose:
        print("PGHC Co-Design (Simplified - Trust Gradient)")
        print("=" * 60)
        print(f"Initial L: {L_init:.3f} m")
        print(f"L bounds: [{L_min:.2f}, {L_max:.2f}] m")
        print(f"Design LR: {design_lr} (small = implicit trust region)")
        print(f"Max step: {max_step} m")
        print(f"Outer iterations: {outer_iters}")
        print(f"Inner timesteps: {inner_timesteps}")
        print()

    best_return = -np.inf
    best_L = L_init

    for outer_iter in range(outer_iters):
        current_L = env.parametric_model.get_L()
        current_m = env.parametric_model.pole_mass

        if verbose:
            print(f"Outer Iter {outer_iter + 1}/{outer_iters}")
            print(f"  L = {current_L:.4f} m, m = {current_m:.4f} kg")

        # === INNER LOOP: Train PPO at current L ===
        ppo = PPO(env, hidden_sizes=(64, 64), lr=3e-4)
        ppo.train(
            total_timesteps=inner_timesteps,
            rollout_steps=2048,
            log_interval=100,
            verbose=False,
        )

        # Evaluate
        current_return = evaluate_policy(env, ppo, n_episodes=eval_episodes)

        if verbose:
            print(f"  Return = {current_return:.1f}")

        if current_return > best_return:
            best_return = current_return
            best_L = current_L

        # === COMPUTE GRADIENT (envelope theorem) ===
        mean_return, gradient = compute_design_gradient(env, ppo, n_rollouts=grad_rollouts)

        if verbose:
            print(f"  dReturn/dL = {gradient:.4f}")

        # === UPDATE L (gradient ascent with small step) ===
        step = design_lr * gradient
        step = np.clip(step, -max_step, max_step)  # Limit step size
        new_L = current_L + step
        new_L = np.clip(new_L, L_min, L_max)

        env.parametric_model.set_L(new_L)

        if verbose:
            print(f"  L: {current_L:.4f} -> {new_L:.4f} (step={step:+.4f})")
            print()

        # Record history
        history['L'].append(new_L)
        history['return'].append(current_return)
        history['gradient'].append(gradient)

    if verbose:
        print("=" * 60)
        print(f"Final L: {env.parametric_model.get_L():.4f} m")
        print(f"Best L: {best_L:.4f} m (return={best_return:.1f})")

    history['best_L'] = best_L
    history['best_return'] = best_return
    history['final_L'] = env.parametric_model.get_L()

    return history


def plot_history(history: Dict, save_path: str = None):
    """Plot co-design history."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("PGHC Co-Design (Simplified)", fontsize=14)

    iters = range(len(history['return']))

    # L over iterations
    ax = axes[0]
    ax.plot(history['L'], 'b-o', markersize=4)
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("Pole Length L (m)")
    ax.set_title("Design Parameter")
    ax.grid(True, alpha=0.3)

    # Return over iterations
    ax = axes[1]
    ax.plot(iters, history['return'], 'g-o', markersize=4)
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("Mean Return")
    ax.set_title("Policy Performance")
    ax.grid(True, alpha=0.3)

    # Gradient over iterations
    ax = axes[2]
    ax.plot(iters, history['gradient'], 'r-o', markersize=4)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("dReturn/dL")
    ax.set_title("Design Gradient")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="PGHC Co-Design (Simplified)")
    parser.add_argument("--L-init", type=float, default=0.6, help="Initial pole length")
    parser.add_argument("--L-min", type=float, default=0.3, help="Minimum pole length")
    parser.add_argument("--L-max", type=float, default=1.2, help="Maximum pole length")
    parser.add_argument("--outer-iters", type=int, default=20, help="Outer loop iterations")
    parser.add_argument("--inner-timesteps", type=int, default=50000, help="PPO timesteps")
    parser.add_argument("--design-lr", type=float, default=0.005, help="Design learning rate")
    parser.add_argument("--save-plot", type=str, default=None, help="Save plot path")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    history = run_codesign_simple(
        L_init=args.L_init,
        L_min=args.L_min,
        L_max=args.L_max,
        outer_iters=args.outer_iters,
        inner_timesteps=args.inner_timesteps,
        design_lr=args.design_lr,
    )

    if not args.no_plot:
        plot_history(history, save_path=args.save_plot)


if __name__ == "__main__":
    main()
