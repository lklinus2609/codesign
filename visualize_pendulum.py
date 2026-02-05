#!/usr/bin/env python3
"""
Pendulum Visualization

Visualizes the inverted pendulum environment with the swing-up policy.
Shows the pendulum animation and real-time plots of angle, velocity, and reward.

Usage:
    python visualize_pendulum.py [--L 0.6] [--steps 500]
"""

import math
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

from envs.pendulum import PendulumEnv
from envs.pendulum.pendulum_env import SimplePendulumPolicy


def run_episode(env, policy, max_steps=500):
    """Run an episode and collect trajectory data."""
    obs = env.reset(theta_init=math.pi)  # Start hanging down

    trajectory = {
        "theta": [],
        "theta_dot": [],
        "torque": [],
        "reward": [],
        "time": [],
    }

    total_reward = 0
    for step in range(max_steps):
        action = policy(obs)
        obs, reward, done, info = env.step(action)

        trajectory["theta"].append(info["theta"])
        trajectory["theta_dot"].append(info["theta_dot"])
        trajectory["torque"].append(info["torque"])
        trajectory["reward"].append(reward.item())
        trajectory["time"].append(step * env.dt)

        total_reward += reward.item()

        if done:
            break

    return trajectory, total_reward


def visualize_pendulum(L=0.6, max_steps=500, save_path=None):
    """
    Visualize the pendulum with animation.

    Args:
        L: Pole length
        max_steps: Number of simulation steps
        save_path: If provided, save animation to this path
    """
    # Create environment and policy
    env = PendulumEnv()
    env.parametric_model.set_L(L)
    policy = SimplePendulumPolicy(env, L_reference=L)

    # Run episode to collect data
    print(f"Running simulation with L={L}m...")
    trajectory, total_reward = run_episode(env, policy, max_steps)
    print(f"Total reward: {total_reward:.1f}")

    # Setup figure
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f"Inverted Pendulum Swing-Up (L={L}m)", fontsize=14)

    # Pendulum animation subplot
    ax_pend = fig.add_subplot(2, 2, 1)
    ax_pend.set_xlim(-L*1.5, L*1.5)
    ax_pend.set_ylim(-L*1.5, L*1.5)
    ax_pend.set_aspect('equal')
    ax_pend.set_title("Pendulum")
    ax_pend.axhline(y=0, color='brown', linewidth=2)  # Ground/pivot line

    # Pendulum elements
    pivot = plt.Circle((0, 0), 0.02*L, color='black', zorder=5)
    ax_pend.add_patch(pivot)

    line, = ax_pend.plot([], [], 'b-', linewidth=3, zorder=3)
    bob = plt.Circle((0, 0), 0.08*L, color='red', zorder=4)
    ax_pend.add_patch(bob)

    # Torque indicator
    torque_arrow = ax_pend.annotate('', xy=(0, 0), xytext=(0, 0),
                                     arrowprops=dict(arrowstyle='->', color='green', lw=2))

    time_text = ax_pend.text(0.02, 0.98, '', transform=ax_pend.transAxes,
                              verticalalignment='top', fontsize=10)

    # Angle plot
    ax_theta = fig.add_subplot(2, 2, 2)
    ax_theta.set_xlim(0, max_steps * env.dt)
    ax_theta.set_ylim(-math.pi - 0.5, math.pi + 0.5)
    ax_theta.set_xlabel("Time (s)")
    ax_theta.set_ylabel("Angle (rad)")
    ax_theta.set_title("Angle from Upright")
    ax_theta.axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Upright')
    ax_theta.axhline(y=math.pi, color='r', linestyle='--', alpha=0.5, label='Hanging')
    ax_theta.axhline(y=-math.pi, color='r', linestyle='--', alpha=0.5)
    ax_theta.legend(loc='upper right')
    line_theta, = ax_theta.plot([], [], 'b-', linewidth=1)

    # Velocity plot
    ax_vel = fig.add_subplot(2, 2, 3)
    ax_vel.set_xlim(0, max_steps * env.dt)
    max_vel = max(abs(v) for v in trajectory["theta_dot"]) * 1.2
    ax_vel.set_ylim(-max_vel, max_vel)
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("Angular Velocity (rad/s)")
    ax_vel.set_title("Angular Velocity")
    ax_vel.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    line_vel, = ax_vel.plot([], [], 'orange', linewidth=1)

    # Reward plot
    ax_reward = fig.add_subplot(2, 2, 4)
    ax_reward.set_xlim(0, max_steps * env.dt)
    ax_reward.set_ylim(min(trajectory["reward"]) - 0.5, max(trajectory["reward"]) + 0.5)
    ax_reward.set_xlabel("Time (s)")
    ax_reward.set_ylabel("Reward")
    ax_reward.set_title("Instantaneous Reward")
    ax_reward.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Max (upright)')
    ax_reward.legend(loc='lower right')
    line_reward, = ax_reward.plot([], [], 'purple', linewidth=1)

    plt.tight_layout()

    def init():
        line.set_data([], [])
        bob.center = (0, -L)
        line_theta.set_data([], [])
        line_vel.set_data([], [])
        line_reward.set_data([], [])
        time_text.set_text('')
        return line, bob, line_theta, line_vel, line_reward, time_text

    def animate(frame):
        if frame >= len(trajectory["theta"]):
            return line, bob, line_theta, line_vel, line_reward, time_text

        theta = trajectory["theta"][frame]
        torque = trajectory["torque"][frame]
        t = trajectory["time"][frame]

        # Pendulum position (theta=0 is up, positive is clockwise)
        # Bob position relative to pivot at origin
        x_bob = L * math.sin(theta)
        y_bob = L * math.cos(theta)  # Positive y is up

        line.set_data([0, x_bob], [0, y_bob])
        bob.center = (x_bob, y_bob)

        # Update time text
        reward_so_far = sum(trajectory["reward"][:frame+1])
        upright = "YES" if abs(theta) < 0.3 else "no"
        time_text.set_text(f"t={t:.2f}s\nReward: {reward_so_far:.1f}\nUpright: {upright}")

        # Update plots
        line_theta.set_data(trajectory["time"][:frame+1], trajectory["theta"][:frame+1])
        line_vel.set_data(trajectory["time"][:frame+1], trajectory["theta_dot"][:frame+1])
        line_reward.set_data(trajectory["time"][:frame+1], trajectory["reward"][:frame+1])

        return line, bob, line_theta, line_vel, line_reward, time_text

    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(trajectory["theta"]),
                         interval=env.dt * 1000, blit=True, repeat=True)

    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='pillow', fps=int(1/env.dt))
        print("Saved!")

    plt.show()

    return trajectory


def compare_L_values(L_values=[0.4, 0.6, 0.8, 1.0, 1.2], max_steps=300):
    """
    Compare performance across different pole lengths with side-by-side animation.
    """
    n_pendulums = len(L_values)
    colors = plt.cm.viridis(np.linspace(0, 1, n_pendulums))

    # First, collect all trajectories
    print("Running simulations...")
    results = {}
    for L in L_values:
        env = PendulumEnv()
        env.parametric_model.set_L(L)
        policy = SimplePendulumPolicy(env, L_reference=L)
        trajectory, total_reward = run_episode(env, policy, max_steps)
        results[L] = {"trajectory": trajectory, "total_reward": total_reward}
        upright_count = sum(1 for theta in trajectory["theta"] if abs(theta) < 0.3)
        print(f"  L={L}m: Total reward={total_reward:.1f}, Upright={upright_count}/{len(trajectory['theta'])} steps")

    # Create figure with pendulum animations on top, plots on bottom
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Pendulum Comparison - Different Pole Lengths", fontsize=14)

    # Create grid: top row for pendulum animations, bottom rows for plots
    gs = fig.add_gridspec(3, n_pendulums, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)

    # Setup pendulum animation axes
    L_max = max(L_values)
    pend_axes = []
    pend_elements = []

    for i, (L, color) in enumerate(zip(L_values, colors)):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(-L_max * 1.3, L_max * 1.3)
        ax.set_ylim(-L_max * 1.3, L_max * 1.3)
        ax.set_aspect('equal')
        ax.set_title(f"L = {L}m", fontsize=12, fontweight='bold', color=color)
        ax.axhline(y=0, color='brown', linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])

        # Pivot
        pivot = plt.Circle((0, 0), 0.03 * L_max, color='black', zorder=5)
        ax.add_patch(pivot)

        # Rod
        line, = ax.plot([], [], color=color, linewidth=3, zorder=3)

        # Bob (size proportional to L for visual distinction)
        bob = plt.Circle((0, 0), 0.06 * L_max + 0.02 * L, color=color, zorder=4, alpha=0.8)
        ax.add_patch(bob)

        # Status text
        status_text = ax.text(0.5, 0.02, '', transform=ax.transAxes,
                               ha='center', fontsize=9)

        pend_axes.append(ax)
        pend_elements.append({
            'line': line,
            'bob': bob,
            'status': status_text,
            'L': L,
            'color': color,
            'trajectory': results[L]['trajectory']
        })

    # Shared angle plot (bottom left spanning half)
    ax_theta = fig.add_subplot(gs[1, :n_pendulums//2 + 1])
    ax_theta.set_xlabel("Time (s)")
    ax_theta.set_ylabel("Angle (rad)")
    ax_theta.set_title("Angle from Upright (0 = upright)")
    ax_theta.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Upright')
    theta_lines = {}
    for L, color in zip(L_values, colors):
        line, = ax_theta.plot([], [], color=color, label=f"L={L}m", alpha=0.8)
        theta_lines[L] = line
    ax_theta.legend(loc='upper right', fontsize=8)
    ax_theta.set_xlim(0, max_steps * 0.05)
    ax_theta.set_ylim(-math.pi - 0.5, math.pi + 0.5)

    # Shared reward plot (bottom right spanning half)
    ax_reward = fig.add_subplot(gs[1, n_pendulums//2 + 1:])
    ax_reward.set_xlabel("Time (s)")
    ax_reward.set_ylabel("Cumulative Reward")
    ax_reward.set_title("Cumulative Reward Over Time")
    reward_lines = {}
    for L, color in zip(L_values, colors):
        line, = ax_reward.plot([], [], color=color, label=f"L={L}m", alpha=0.8)
        reward_lines[L] = line
    ax_reward.legend(loc='lower right', fontsize=8)
    ax_reward.set_xlim(0, max_steps * 0.05)
    max_cumulative = max(sum(results[L]['trajectory']['reward']) for L in L_values)
    min_cumulative = min(
        min(np.cumsum(results[L]['trajectory']['reward'])) for L in L_values
    )
    ax_reward.set_ylim(min_cumulative - 10, max_cumulative + 10)

    # Bar chart of final rewards
    ax_bar = fig.add_subplot(gs[2, :])
    rewards = [results[L]["total_reward"] for L in L_values]
    bars = ax_bar.bar([f"L={L}m" for L in L_values], rewards, color=colors)
    ax_bar.set_ylabel("Total Episode Reward")
    ax_bar.set_title("Final Performance Comparison")
    ax_bar.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    best_idx = np.argmax(rewards)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    ax_bar.annotate('BEST', xy=(best_idx, rewards[best_idx]),
                    ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')

    # Time display on first pendulum axis
    time_text = pend_axes[0].text(0.02, 0.98, '', transform=pend_axes[0].transAxes,
                                   va='top', fontsize=10, fontweight='bold')

    def animate(frame):
        for elem in pend_elements:
            traj = elem['trajectory']
            L = elem['L']

            if frame < len(traj['theta']):
                theta = traj['theta'][frame]
                x_bob = L * math.sin(theta)
                y_bob = L * math.cos(theta)

                elem['line'].set_data([0, x_bob], [0, y_bob])
                elem['bob'].center = (x_bob, y_bob)

                # Status
                upright = "UPRIGHT!" if abs(theta) < 0.3 else ""
                cum_reward = sum(traj['reward'][:frame+1])
                elem['status'].set_text(f"R={cum_reward:.0f} {upright}")
                if upright:
                    elem['status'].set_color('green')
                    elem['status'].set_fontweight('bold')
                else:
                    elem['status'].set_color('black')
                    elem['status'].set_fontweight('normal')

        # Update plots
        for L in L_values:
            traj = results[L]['trajectory']
            if frame < len(traj['theta']):
                theta_lines[L].set_data(traj['time'][:frame+1], traj['theta'][:frame+1])
                cum_rewards = np.cumsum(traj['reward'][:frame+1])
                reward_lines[L].set_data(traj['time'][:frame+1], cum_rewards)

        # Update time
        if frame < len(results[L_values[0]]['trajectory']['time']):
            t = results[L_values[0]]['trajectory']['time'][frame]
            time_text.set_text(f"t={t:.2f}s")

    # Find max trajectory length
    max_len = max(len(results[L]['trajectory']['theta']) for L in L_values)

    print("\nStarting animation... (close window to exit)")
    anim = FuncAnimation(fig, animate, frames=max_len,
                         interval=50, blit=False, repeat=True)

    plt.show()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize inverted pendulum")
    parser.add_argument("--L", type=float, default=0.6, help="Pole length (meters)")
    parser.add_argument("--steps", type=int, default=400, help="Number of simulation steps")
    parser.add_argument("--compare", action="store_true", help="Compare multiple L values")
    parser.add_argument("--save", type=str, default=None, help="Save animation to file (e.g., pendulum.gif)")

    args = parser.parse_args()

    if args.compare:
        compare_L_values()
    else:
        visualize_pendulum(L=args.L, max_steps=args.steps, save_path=args.save)
