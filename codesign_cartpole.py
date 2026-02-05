#!/usr/bin/env python3
"""
PGHC Co-Design on Cart-Pole

Demonstrates the full PGHC algorithm:
- Inner loop: PPO learns to balance the pole
- Outer loop: Gradient descent on pole length L

Usage:
    python codesign_cartpole.py [--outer-iters 20] [--inner-timesteps 50000]
"""

import argparse
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict

from envs.cartpole import CartPoleEnv, ParametricCartPole, PPO


class TrustRegion:
    """Trust region for outer loop updates."""

    def __init__(
        self,
        xi: float = 0.1,
        lr_init: float = 0.01,
        lr_min: float = 1e-4,
        lr_max: float = 0.1,
        decay_factor: float = 0.5,
        growth_factor: float = 1.5,
    ):
        self.xi = xi
        self.lr = lr_init
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.decay_factor = decay_factor
        self.growth_factor = growth_factor

    def check_acceptance(self, current_obj: float, candidate_obj: float):
        """Check if update should be accepted."""
        D = candidate_obj - current_obj  # Improvement (positive = better)
        threshold = -self.xi * abs(current_obj)
        accepted = D >= threshold
        return accepted, D

    def update_lr(self, accepted: bool, D: float, current_obj: float):
        """Adapt learning rate based on acceptance."""
        if not accepted:
            self.lr = max(self.lr * self.decay_factor, self.lr_min)
        elif D < 0.01 * abs(current_obj):  # Small improvement
            self.lr = min(self.lr * self.growth_factor, self.lr_max)


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

    Returns:
        (mean_return, gradient)
    """
    returns = []
    gradients = []

    # Ensure L requires grad
    env.parametric_model.L.requires_grad_(True)

    for _ in range(n_rollouts):
        obs = env.reset()
        total_return = torch.tensor(0.0, dtype=torch.float64)

        for t in range(horizon):
            # Frozen policy - no grad through policy
            with torch.no_grad():
                action = ppo.get_action(obs)

            # Step with gradient through L
            obs, reward, terminated, truncated, info = env.step(action)
            total_return = total_return + reward

            if terminated or truncated:
                break

        # Compute gradient
        if total_return.requires_grad:
            total_return.backward()
            grad = env.parametric_model.L.grad.item() if env.parametric_model.L.grad is not None else 0.0
            env.parametric_model.L.grad = None
        else:
            grad = 0.0

        returns.append(total_return.item())
        gradients.append(grad)

    return np.mean(returns), np.mean(gradients)


def run_codesign(
    L_init: float = 0.6,
    L_min: float = 0.3,
    L_max: float = 1.2,
    outer_iters: int = 20,
    inner_timesteps: int = 50000,
    eval_episodes: int = 10,
    grad_rollouts: int = 10,
    verbose: bool = True,
) -> Dict:
    """
    Run PGHC co-design on cart-pole.

    Args:
        L_init: Initial pole length
        L_min, L_max: Bounds on pole length
        outer_iters: Number of outer loop iterations
        inner_timesteps: PPO training timesteps per outer iteration
        eval_episodes: Episodes for evaluation
        grad_rollouts: Rollouts for gradient estimation
        verbose: Print progress

    Returns:
        Dictionary with history
    """
    # Initialize
    parametric_model = ParametricCartPole(L_init=L_init, L_min=L_min, L_max=L_max)
    # Use shaped_reward=True for gradient flow through L
    env = CartPoleEnv(parametric_model=parametric_model, shaped_reward=True)

    trust_region = TrustRegion(xi=0.1, lr_init=0.01, lr_min=1e-4, lr_max=0.05)

    history = {
        'L': [L_init],
        'return': [],
        'gradient': [],
        'lr': [],
        'accepted': [],
    }

    if verbose:
        print("PGHC Co-Design on Cart-Pole (Full Physics)")
        print("=" * 60)
        print(f"Initial L: {L_init:.3f} m")
        print(f"Initial pole mass: {parametric_model.pole_mass:.3f} kg")
        print(f"L bounds: [{L_min:.2f}, {L_max:.2f}] m")
        print(f"Cart mass: {parametric_model.cart_mass:.1f} kg")
        print(f"Pole linear density: {parametric_model.pole_linear_density:.2f} kg/m")
        print(f"Outer iterations: {outer_iters}")
        print(f"Inner timesteps: {inner_timesteps}")
        print(f"Reward: shaped (cos(theta) - provides gradient)")
        print()

    # Current best
    best_return = -np.inf
    best_L = L_init

    for outer_iter in range(outer_iters):
        current_L = env.parametric_model.get_L()
        current_m = env.parametric_model.pole_mass

        if verbose:
            print(f"Outer Iter {outer_iter + 1}/{outer_iters}")
            print(f"  Current: L = {current_L:.4f} m, m = {current_m:.4f} kg")

        # === INNER LOOP: Train PPO at current L ===
        ppo_current = PPO(env, hidden_sizes=(64, 64), lr=3e-4)
        ppo_current.train(
            total_timesteps=inner_timesteps,
            rollout_steps=2048,
            log_interval=100,
            verbose=False,
        )

        # Evaluate policy trained at current L
        current_return = evaluate_policy(env, ppo_current, n_episodes=eval_episodes)

        if verbose:
            print(f"  Return(current) = {current_return:.1f}")

        # Track best
        if current_return > best_return:
            best_return = current_return
            best_L = current_L

        # === COMPUTE GRADIENT with frozen policy (envelope theorem) ===
        mean_return, gradient = compute_design_gradient(env, ppo_current, n_rollouts=grad_rollouts)

        if verbose:
            print(f"  dReturn/dL = {gradient:.4f}")

        # Propose new L (gradient ascent - maximize return)
        step = trust_region.lr * gradient
        step = np.clip(step, -0.05, 0.05)  # Max 5cm per step
        candidate_L = current_L + step
        candidate_L = np.clip(candidate_L, L_min, L_max)

        if verbose:
            candidate_m = parametric_model.pole_linear_density * 2.0 * candidate_L
            print(f"  Candidate: L = {candidate_L:.4f} m, m = {candidate_m:.4f} kg")

        # === ADAPT POLICY to candidate L (warm-start from current policy) ===
        # Save policy weights in case we need to reject
        saved_policy_state = copy.deepcopy(ppo_current.policy.state_dict())

        env.parametric_model.set_L(candidate_L)

        # Continue training the SAME policy at new L (not fresh)
        # This lets it adapt to the new morphology
        ppo_current.train(
            total_timesteps=inner_timesteps,
            rollout_steps=2048,
            log_interval=100,
            verbose=False,
        )

        # Evaluate adapted policy at candidate L
        candidate_return = evaluate_policy(env, ppo_current, n_episodes=eval_episodes)

        if verbose:
            print(f"  Return(candidate) = {candidate_return:.1f}")

        # === TRUST REGION: Compare policies trained at each L ===
        accepted, D = trust_region.check_acceptance(current_return, candidate_return)
        trust_region.update_lr(accepted, D, current_return)

        if accepted:
            # Keep candidate L and adapted policy
            if verbose:
                print(f"  ACCEPTED: L {current_L:.4f} -> {candidate_L:.4f} (D={D:+.1f})")
        else:
            # Reject - revert L AND restore policy weights
            env.parametric_model.set_L(current_L)
            ppo_current.policy.load_state_dict(saved_policy_state)
            if verbose:
                print(f"  REJECTED: L stays at {current_L:.4f} (D={D:+.1f})")

        # Record history
        history['L'].append(env.parametric_model.get_L())
        history['return'].append(current_return)
        history['gradient'].append(gradient)
        history['lr'].append(trust_region.lr)
        history['accepted'].append(accepted)

        if verbose:
            print(f"  LR = {trust_region.lr:.6f}")
            print()

    if verbose:
        print("=" * 60)
        print(f"Final L: {env.parametric_model.get_L():.4f} m")
        print(f"Best L: {best_L:.4f} m (return={best_return:.1f})")

    history['best_L'] = best_L
    history['best_return'] = best_return

    return history


def plot_history(history: Dict, save_path: str = None):
    """Plot co-design history."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("PGHC Co-Design on Cart-Pole", fontsize=14)

    iters = range(len(history['return']))

    # L over iterations
    ax = axes[0, 0]
    ax.plot(history['L'], 'b-o', markersize=4)
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("Pole Length L (m)")
    ax.set_title("Design Parameter Evolution")
    ax.axhline(y=history['best_L'], color='g', linestyle='--', alpha=0.5, label=f"Best L={history['best_L']:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Return over iterations
    ax = axes[0, 1]
    ax.plot(iters, history['return'], 'g-o', markersize=4)
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("Mean Return")
    ax.set_title("Policy Performance")
    ax.axhline(y=history['best_return'], color='r', linestyle='--', alpha=0.5, label=f"Best={history['best_return']:.0f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gradient over iterations
    ax = axes[1, 0]
    ax.plot(iters, history['gradient'], 'r-o', markersize=4)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("dReturn/dL")
    ax.set_title("Design Gradient")
    ax.grid(True, alpha=0.3)

    # Learning rate over iterations
    ax = axes[1, 1]
    ax.semilogy(iters, history['lr'], 'purple', marker='o', markersize=4)
    # Mark accepted/rejected
    for i, (acc, lr) in enumerate(zip(history['accepted'], history['lr'])):
        color = 'green' if acc else 'red'
        ax.scatter(i, lr, c=color, s=50, zorder=5)
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Adaptive LR (green=accept, red=reject)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="PGHC Co-Design on Cart-Pole")
    parser.add_argument("--L-init", type=float, default=0.6, help="Initial pole length")
    parser.add_argument("--L-min", type=float, default=0.3, help="Minimum pole length")
    parser.add_argument("--L-max", type=float, default=1.2, help="Maximum pole length")
    parser.add_argument("--outer-iters", type=int, default=20, help="Outer loop iterations")
    parser.add_argument("--inner-timesteps", type=int, default=50000, help="PPO timesteps per outer iter")
    parser.add_argument("--save-plot", type=str, default=None, help="Save plot to file")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    history = run_codesign(
        L_init=args.L_init,
        L_min=args.L_min,
        L_max=args.L_max,
        outer_iters=args.outer_iters,
        inner_timesteps=args.inner_timesteps,
    )

    if not args.no_plot:
        plot_history(history, save_path=args.save_plot)


if __name__ == "__main__":
    main()
