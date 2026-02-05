#!/usr/bin/env python3
"""
Co-Design Visualization

Visualizes the PGHC co-design algorithm in action:
1. Start with initial L (e.g., 0.5m)
2. Run policy to evaluate performance
3. Compute gradient dReturn/dL
4. Update L using trust region
5. Repeat - watch L evolve toward better values

This shows the actual algorithm, not just fixed pendulums.

Usage:
    python visualize_codesign.py
"""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

from envs.pendulum import PendulumEnv
from envs.pendulum.pendulum_env import SimplePendulumPolicy
from test_level0_verification import MockTrustRegion


def run_codesign_loop(num_outer_loops=100, horizon=100, L_init=0.8, L_min=0.4, L_max=1.2, verbose=True):
    """
    Run the full co-design loop and collect data for visualization.

    Returns all the data needed to animate the co-design process.
    """

    # Trust region - use smaller lr since gradients can be large
    tr = MockTrustRegion(xi=0.1, lr_init=0.001, lr_min=1e-5, lr_max=0.01)

    # Storage for visualization
    all_data = {
        'outer_iterations': [],
        'L_history': [L_init],
        'return_history': [],
        'grad_history': [],
        'accepted_history': [],
        'lr_history': [],
        'trajectories': [],  # Store full trajectories for animation
    }

    current_L = L_init

    for outer_iter in range(num_outer_loops):
        print_freq = 1 if num_outer_loops <= 30 else 5
        if verbose and outer_iter % print_freq == 0:
            print(f"Outer iteration {outer_iter}: L={current_L:.3f}m")

        # Create environment and policy for current L
        env = PendulumEnv()
        env.parametric_model.set_L(current_L)
        policy = SimplePendulumPolicy(env, L_reference=current_L)

        # Run episode and collect trajectory
        obs = env.reset(theta_init=math.pi)
        trajectory = {'theta': [], 'theta_dot': [], 'reward': [], 'time': []}

        for step in range(horizon):
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            trajectory['theta'].append(info['theta'])
            trajectory['theta_dot'].append(info['theta_dot'])
            trajectory['reward'].append(reward.item())
            trajectory['time'].append(step * env.dt)

        # Compute gradient using differentiable rollout
        env2 = PendulumEnv()
        env2.parametric_model.set_L(current_L)
        policy2 = SimplePendulumPolicy(env2, L_reference=current_L)
        result = env2.rollout_differentiable(policy2, horizon=horizon)

        current_return = result['total_return']
        L_grad = result['L_grad']

        # Store raw gradient for logging
        L_grad_raw = L_grad

        # Compute step with max step size limit
        step = tr.lr * L_grad
        max_step = 0.05  # Maximum 5cm per iteration
        step = max(-max_step, min(max_step, step))

        # Propose new L
        L_candidate = current_L + step
        L_candidate = max(L_min, min(L_max, L_candidate))

        # Evaluate candidate
        env_cand = PendulumEnv()
        env_cand.parametric_model.set_L(L_candidate)
        policy_cand = SimplePendulumPolicy(env_cand, L_reference=L_candidate)
        result_cand = env_cand.rollout_differentiable(policy_cand, horizon=horizon)
        candidate_return = result_cand['total_return']

        # Trust region decision
        accepted, D = tr.check_acceptance(-current_return, -candidate_return)
        tr.update_lr(accepted, D, -current_return)

        if accepted:
            new_L = L_candidate
        else:
            new_L = current_L

        # Store data
        all_data['outer_iterations'].append(outer_iter)
        all_data['return_history'].append(current_return)
        all_data['grad_history'].append(L_grad_raw)
        all_data['accepted_history'].append(accepted)
        all_data['lr_history'].append(tr.lr)
        all_data['trajectories'].append({
            'L': current_L,
            'trajectory': trajectory,
            'return': current_return,
            'grad': L_grad_raw,
            'step': step,
            'accepted': accepted,
        })
        all_data['L_history'].append(new_L)

        if verbose:
            # Print every iteration for small runs, every 5 for large runs
            print_freq = 1 if num_outer_loops <= 30 else 5
            if outer_iter % print_freq == 0 or accepted:
                status = "ACCEPTED" if accepted else "rejected"
                print(f"  Return={current_return:.1f}, grad={L_grad_raw:.1f}, step={step:+.3f}, {status}, new_L={new_L:.3f}")

        current_L = new_L

    return all_data


def visualize_codesign():
    """
    Visualize the co-design process with animation.
    """
    # Settings
    L_init = 0.8
    L_min = 0.4
    L_max = 1.2
    num_outer_loops = 100

    print("="*60)
    print("PGHC CO-DESIGN VISUALIZATION")
    print("="*60)
    print("\nHyperparameters:")
    print(f"  - Initial L: {L_init}m")
    print(f"  - L bounds: [{L_min}m, {L_max}m]")
    print("  - Learning rate (beta): 0.001 (initial)")
    print("  - Max step size: 0.05m per iteration")
    print("  - Trust region xi: 0.1 (10% degradation allowed)")
    print(f"  - Outer loop iterations: {num_outer_loops}")
    print("\nRunning co-design loop...")
    print("Watch how pole length L evolves based on gradients!\n")

    # Run the co-design loop
    data = run_codesign_loop(num_outer_loops=num_outer_loops, horizon=80,
                              L_init=L_init, L_min=L_min, L_max=L_max, verbose=True)

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("PGHC Co-Design: Optimizing Pole Length", fontsize=14, fontweight='bold')

    # Layout: 2x3 grid
    # Top left: Pendulum animation
    ax_pend = fig.add_subplot(2, 3, 1)
    # Top middle: L over iterations
    ax_L = fig.add_subplot(2, 3, 2)
    # Top right: Return over iterations
    ax_return = fig.add_subplot(2, 3, 3)
    # Bottom left: Current trajectory (angle)
    ax_theta = fig.add_subplot(2, 3, 4)
    # Bottom middle: Gradient visualization
    ax_grad = fig.add_subplot(2, 3, 5)
    # Bottom right: Algorithm status
    ax_status = fig.add_subplot(2, 3, 6)
    ax_status.axis('off')

    L_max = 1.5

    # Setup pendulum axis
    ax_pend.set_xlim(-L_max * 1.3, L_max * 1.3)
    ax_pend.set_ylim(-L_max * 1.3, L_max * 1.3)
    ax_pend.set_aspect('equal')
    ax_pend.axhline(y=0, color='brown', linewidth=3)
    ax_pend.set_title("Pendulum (current L)")
    ax_pend.set_xticks([])
    ax_pend.set_yticks([])

    # Pendulum elements
    pivot = plt.Circle((0, 0), 0.04, color='black', zorder=5)
    ax_pend.add_patch(pivot)
    pend_line, = ax_pend.plot([], [], 'b-', linewidth=4, zorder=3)
    pend_bob = plt.Circle((0, 0), 0.08, color='red', zorder=4)
    ax_pend.add_patch(pend_bob)
    pend_text = ax_pend.text(0.5, 0.95, '', transform=ax_pend.transAxes,
                             ha='center', va='top', fontsize=11, fontweight='bold')

    # Setup L plot
    ax_L.set_xlabel("Outer Iteration")
    ax_L.set_ylabel("Pole Length L (m)")
    ax_L.set_title("Design Parameter Evolution")
    ax_L.set_xlim(-0.5, len(data['outer_iterations']) + 0.5)
    ax_L.set_ylim(L_min - 0.1, L_max + 0.1)
    ax_L.axhline(y=L_min, color='gray', linestyle='--', alpha=0.5, label='L_min')
    ax_L.axhline(y=L_max, color='gray', linestyle='--', alpha=0.5, label='L_max')
    L_line, = ax_L.plot([], [], 'bo-', linewidth=2, markersize=8)
    L_current_marker, = ax_L.plot([], [], 'ro', markersize=15, zorder=5)
    ax_L.legend(loc='upper right')

    # Setup return plot
    ax_return.set_xlabel("Outer Iteration")
    ax_return.set_ylabel("Episode Return")
    ax_return.set_title("Performance Over Iterations")
    ax_return.set_xlim(-0.5, len(data['outer_iterations']) + 0.5)
    all_returns = data['return_history']
    ax_return.set_ylim(min(all_returns) - 10, max(all_returns) + 10)
    return_line, = ax_return.plot([], [], 'g^-', linewidth=2, markersize=8)
    return_current_marker, = ax_return.plot([], [], 'r^', markersize=15, zorder=5)

    # Setup theta plot
    ax_theta.set_xlabel("Time (s)")
    ax_theta.set_ylabel("Angle (rad)")
    ax_theta.set_title("Current Episode Trajectory")
    ax_theta.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Upright')
    ax_theta.set_xlim(0, 4)
    ax_theta.set_ylim(-math.pi - 0.5, math.pi + 0.5)
    theta_line, = ax_theta.plot([], [], 'b-', linewidth=1.5)
    ax_theta.legend(loc='upper right')

    # Setup gradient visualization
    ax_grad.set_xlim(L_min - 0.1, L_max + 0.1)
    ax_grad.set_ylim(-1, 1)
    ax_grad.set_xlabel("Pole Length L (m)")
    ax_grad.set_title("Proposed Step Direction")
    ax_grad.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax_grad.axvline(x=L_min, color='gray', linestyle='--', alpha=0.5)
    ax_grad.axvline(x=L_max, color='gray', linestyle='--', alpha=0.5)
    grad_arrow = None
    L_marker_grad, = ax_grad.plot([], [], 'bo', markersize=15)
    ax_grad.set_yticks([])

    # Status text
    status_text = ax_status.text(0.1, 0.9, '', transform=ax_status.transAxes,
                                  fontsize=12, verticalalignment='top',
                                  fontfamily='monospace')

    # Current outer iteration
    current_outer = [0]
    current_step = [0]
    # Fewer frames per outer iteration when running many iterations
    steps_per_outer = 20 if num_outer_loops > 30 else 80

    def animate(frame):
        nonlocal grad_arrow

        outer_idx = frame // steps_per_outer
        step_idx = frame % steps_per_outer

        if outer_idx >= len(data['trajectories']):
            outer_idx = len(data['trajectories']) - 1
            step_idx = steps_per_outer - 1

        traj_data = data['trajectories'][outer_idx]
        traj = traj_data['trajectory']
        L = traj_data['L']

        # Map step_idx to trajectory index
        traj_idx = min(step_idx, len(traj['theta']) - 1)

        # Update pendulum
        theta = traj['theta'][traj_idx]
        x_bob = L * math.sin(theta)
        y_bob = L * math.cos(theta)
        pend_line.set_data([0, x_bob], [0, y_bob])
        pend_bob.center = (x_bob, y_bob)
        pend_bob.set_radius(0.05 + 0.02 * L)  # Size varies with L

        upright = "UPRIGHT!" if abs(theta) < 0.3 else ""
        pend_text.set_text(f"L = {L:.2f}m  {upright}")
        if upright:
            pend_text.set_color('green')
        else:
            pend_text.set_color('black')

        # Update L history plot
        L_line.set_data(range(outer_idx + 1), data['L_history'][:outer_idx + 1])
        L_current_marker.set_data([outer_idx], [L])

        # Update return plot
        return_line.set_data(range(outer_idx + 1), data['return_history'][:outer_idx + 1])
        return_current_marker.set_data([outer_idx], [data['return_history'][outer_idx]])

        # Update theta trajectory
        theta_line.set_data(traj['time'][:traj_idx + 1], traj['theta'][:traj_idx + 1])

        # Update gradient arrow (shows proposed step direction)
        if grad_arrow is not None:
            grad_arrow.remove()

        step = traj_data.get('step', 0)
        # Scale arrow: 0.05m step = 0.3 units on plot
        arrow_scale = step * 6
        arrow_color = 'green' if traj_data['accepted'] else 'red'
        grad_arrow = ax_grad.annotate('',
            xy=(L + arrow_scale, 0),
            xytext=(L, 0),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=3))
        L_marker_grad.set_data([L], [0])

        # Update status text
        status = "ACCEPTED" if traj_data['accepted'] else "REJECTED"
        status_color = "green" if traj_data['accepted'] else "red"

        step = traj_data.get('step', 0)
        status_str = f"""
OUTER ITERATION: {outer_idx + 1}/{len(data['trajectories'])}

Current L:      {L:.3f} m
Episode Return: {traj_data['return']:.1f}
Gradient dR/dL: {traj_data['grad']:.1f}
Proposed step:  {step:+.4f} m

Trust Region:   {status}
Learning Rate:  {data['lr_history'][outer_idx]:.5f}

Next L:         {data['L_history'][outer_idx + 1]:.3f} m
Actual change:  {data['L_history'][outer_idx + 1] - L:+.4f} m
"""
        status_text.set_text(status_str)

    total_frames = len(data['trajectories']) * steps_per_outer

    print(f"\nStarting animation ({len(data['trajectories'])} outer iterations)...")
    print("Watch the pole length L adapt based on gradients!")
    print("Green = update accepted, Red = update rejected\n")

    anim = FuncAnimation(fig, animate, frames=total_frames,
                         interval=30, blit=False, repeat=True)

    plt.tight_layout()
    plt.show()

    return data


def visualize_codesign_summary(data=None):
    """
    Show a static summary of the co-design results.
    """
    if data is None:
        print("Running co-design loop...")
        data = run_codesign_loop(num_outer_loops=15, horizon=100)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("PGHC Co-Design Summary", fontsize=14, fontweight='bold')

    iterations = range(len(data['return_history']))

    # L evolution
    ax = axes[0, 0]
    ax.plot(data['L_history'][:-1], 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=0.4, color='gray', linestyle='--', alpha=0.5, label='L_min')
    ax.axhline(y=1.5, color='gray', linestyle='--', alpha=0.5, label='L_max')
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("Pole Length L (m)")
    ax.set_title("Design Parameter Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Return evolution
    ax = axes[0, 1]
    colors = ['green' if acc else 'red' for acc in data['accepted_history']]
    ax.scatter(iterations, data['return_history'], c=colors, s=100, zorder=3)
    ax.plot(iterations, data['return_history'], 'b-', alpha=0.5, linewidth=1)
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("Episode Return")
    ax.set_title("Performance (green=accepted, red=rejected)")
    ax.grid(True, alpha=0.3)

    # Gradient evolution
    ax = axes[1, 0]
    ax.bar(iterations, data['grad_history'], color='purple', alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("Gradient dR/dL")
    ax.set_title("Gradient Direction")
    ax.grid(True, alpha=0.3)

    # L vs Return scatter
    ax = axes[1, 1]
    L_vals = data['L_history'][:-1]
    returns = data['return_history']
    scatter = ax.scatter(L_vals, returns, c=iterations, cmap='viridis', s=100)
    ax.set_xlabel("Pole Length L (m)")
    ax.set_ylabel("Episode Return")
    ax.set_title("Return vs L (color = iteration)")
    plt.colorbar(scatter, ax=ax, label='Iteration')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize PGHC co-design")
    parser.add_argument("--summary", action="store_true", help="Show static summary only")
    args = parser.parse_args()

    if args.summary:
        visualize_codesign_summary()
    else:
        data = visualize_codesign()
        print("\n" + "="*60)
        print("CO-DESIGN COMPLETE")
        print("="*60)
        print(f"Initial L: {data['L_history'][0]:.3f} m")
        print(f"Final L:   {data['L_history'][-1]:.3f} m")
        print(f"L change:  {data['L_history'][-1] - data['L_history'][0]:+.3f} m")
        print(f"\nInitial Return: {data['return_history'][0]:.1f}")
        print(f"Final Return:   {data['return_history'][-1]:.1f}")

        # Show summary after animation
        visualize_codesign_summary(data)
