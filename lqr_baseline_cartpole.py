#!/usr/bin/env python3
"""
LQR Baseline for Level 1.5 Cart-Pole Co-Design

Sweeps pole length L and evaluates a near-optimal LQR controller using the
exact same reward function as the RL environment. Provides a ground-truth
answer to: "What pole length does an optimal controller prefer?"

Physics parameters match cartpole_newton_vec_env.py exactly.
Reward function matches the Warp kernel step_rewards_done_kernel exactly.

Usage:
    python lqr_baseline_cartpole.py
"""

import os
import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Physical parameters (matching cartpole_newton_vec_env.py)
# ---------------------------------------------------------------------------
CART_MASS = 1.0          # kg
POLE_DENSITY = 0.2       # kg/m (linear density)
GRAVITY = 9.81           # m/s²
DT = 0.02                # s (50 Hz)
FORCE_MAX = 30.0         # N
X_LIMIT = 3.0            # m (termination boundary)
MAX_STEPS = 500          # 10s episodes
ARMATURE = 0.1           # Joint damping (both joints)


def pole_mass(L):
    """Pole mass = linear_density * total_length = 0.2 * 2L."""
    return POLE_DENSITY * 2.0 * L


def nonlinear_dynamics(state, F, L):
    """
    Full nonlinear cart-pole dynamics (Euler-Lagrange).

    State: [x, theta, x_dot, theta_dot]
    theta = 0 is upright.

    Uniform rod of total length 2L, mass m = 0.4L.
    COM distance from pivot: l = L
    Moment of inertia about pivot: I = (1/3)*m*(2L)^2 = (4/3)*m*L^2

    Equations of motion:
        (M + m) * x_dd + m*l*theta_dd*cos(theta) - m*l*theta_d^2*sin(theta) = F - b_x*x_d
        I * theta_dd + m*l*x_dd*cos(theta) - m*g*l*sin(theta) = -b_theta*theta_d

    Solved for [x_dd, theta_dd] as a 2x2 linear system.
    """
    x, theta, x_dot, theta_dot = state
    M = CART_MASS
    m = pole_mass(L)
    l = L                           # COM distance from pivot
    I = (4.0 / 3.0) * m * L**2     # Moment of inertia about pivot
    g = GRAVITY
    b_x = ARMATURE
    b_theta = ARMATURE

    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    # 2x2 system: A_mat @ [x_dd, theta_dd] = rhs
    #   (M + m) * x_dd   + m*l*cos(th) * theta_dd = F - b_x*x_d + m*l*theta_d^2*sin(th)
    #   m*l*cos(th)*x_dd  + I * theta_dd           = m*g*l*sin(th) - b_theta*theta_d
    A_mat = np.array([
        [M + m,         m * l * cos_th],
        [m * l * cos_th, I],
    ])
    rhs = np.array([
        F - b_x * x_dot + m * l * theta_dot**2 * sin_th,
        m * g * l * sin_th - b_theta * theta_dot,
    ])

    acc = np.linalg.solve(A_mat, rhs)
    x_dd, theta_dd = acc

    return np.array([x_dot, theta_dot, x_dd, theta_dd])


def linearize(L):
    """
    Linearized state-space around upright equilibrium (theta=0, all else 0).

    State: [x, theta, x_dot, theta_dot]
    Input: F

    Returns A (4x4), B (4x1).
    """
    M = CART_MASS
    m = pole_mass(L)
    l = L
    I = (4.0 / 3.0) * m * L**2
    g = GRAVITY
    b_x = ARMATURE
    b_theta = ARMATURE

    # Mass matrix at theta=0: cos(0)=1, sin(0)=0
    # [M+m,  m*l] [x_dd    ]   [F - b_x*x_d    ]
    # [m*l,  I  ] [theta_dd] = [m*g*l*theta - b_theta*theta_d]
    det = (M + m) * I - (m * l)**2

    # Invert mass matrix
    # inv(A_mat) = (1/det) * [[I, -m*l], [-m*l, M+m]]
    inv_00 = I / det
    inv_01 = -m * l / det
    inv_10 = -m * l / det
    inv_11 = (M + m) / det

    # x_dd = inv_00 * (F - b_x*x_d) + inv_01 * (m*g*l*theta - b_theta*theta_d)
    # theta_dd = inv_10 * (F - b_x*x_d) + inv_11 * (m*g*l*theta - b_theta*theta_d)

    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, inv_00 * 0 + inv_01 * m * g * l, -inv_00 * b_x, -inv_01 * b_theta],
        [0, inv_10 * 0 + inv_11 * m * g * l, -inv_10 * b_x, -inv_11 * b_theta],
    ])

    B = np.array([
        [0],
        [0],
        [inv_00],
        [inv_10],
    ])

    return A, B


def solve_lqr(L, Q, R):
    """
    Solve continuous-time LQR for given pole length.

    Returns gain K such that u = -K @ x is optimal.
    Returns None if Riccati equation fails (unstabilizable).
    """
    A, B = linearize(L)
    try:
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.solve(R, B.T @ P)
        return K
    except Exception:
        return None


def simulate_episode(L, K, initial_state, max_steps=MAX_STEPS):
    """
    Simulate one episode with LQR controller on nonlinear dynamics.

    Uses RK4 integration at dt=0.02 with 4 substeps (matching Newton).
    Returns total reward using the exact RL reward function.
    """
    state = np.array(initial_state, dtype=np.float64)
    sub_dt = DT / 4  # 4 substeps, matching Newton
    total_reward = 0.0
    prev_action = 0.0  # Normalized action in [-1, 1]

    for step in range(max_steps):
        # LQR control: u = -K @ state, clipped to force limits
        F = float(-K @ state)
        F = np.clip(F, -FORCE_MAX, FORCE_MAX)

        # Normalized action (matching RL: action in [-1,1], force = action * FORCE_MAX)
        action = F / FORCE_MAX

        # Check termination BEFORE computing reward (matching env logic:
        # env computes reward at the NEW state after physics step, but
        # termination is also checked at the new state)
        # Actually, in the env, step() applies force, runs physics, THEN
        # computes reward+termination on the resulting state.

        # --- RK4 integration with 4 substeps ---
        for _ in range(4):
            k1 = nonlinear_dynamics(state, F, L)
            k2 = nonlinear_dynamics(state + 0.5 * sub_dt * k1, F, L)
            k3 = nonlinear_dynamics(state + 0.5 * sub_dt * k2, F, L)
            k4 = nonlinear_dynamics(state + sub_dt * k3, F, L)
            state = state + (sub_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # --- Reward (exact match to step_rewards_done_kernel) ---
        x, theta, x_dot, theta_dot = state
        action_diff = action - prev_action

        terminated = 1.0 if abs(x) > X_LIMIT else 0.0

        r = (1.0 * (1.0 - terminated)
             - 2.0 * terminated
             - 1.0 * theta**2
             - 0.1 * x**2
             - 1.0 * action**2
             - 0.1 * action_diff**2
             - 0.01 * abs(x_dot)
             - 0.005 * abs(theta_dot))

        total_reward += r
        prev_action = action

        if terminated > 0.5 or step + 1 >= max_steps:
            break

    return total_reward


def sweep_pole_length(L_values, n_episodes=50):
    """
    Sweep pole length, solving LQR and simulating nonlinear episodes at each L.

    Returns arrays of mean rewards, std rewards, and LQR gains.
    """
    # LQR cost matrices (matching RL reward quadratic terms)
    # Reward has: -1.0*theta^2, -0.1*x^2, -1.0*action^2
    # action = F/FORCE_MAX, so action^2 = F^2/FORCE_MAX^2
    # LQR minimizes integral x'Qx + u'Ru where u = F
    # To match: Q_theta = 1.0, Q_x = 0.1 (from reward penalties)
    # R = 1.0/FORCE_MAX^2 (since reward penalizes (F/30)^2 with weight 1.0)
    Q = np.diag([0.1, 1.0, 0.0, 0.0])
    R = np.array([[1.0 / FORCE_MAX**2]])

    mean_rewards = np.zeros(len(L_values))
    std_rewards = np.zeros(len(L_values))
    gain_norms = np.zeros(len(L_values))

    for i, L in enumerate(L_values):
        K = solve_lqr(L, Q, R)

        if K is None:
            mean_rewards[i] = np.nan
            std_rewards[i] = np.nan
            gain_norms[i] = np.nan
            continue

        gain_norms[i] = np.linalg.norm(K)

        # Simulate episodes with random initial conditions
        # Matching env: theta ~ Uniform(-pi/9, pi/9), cart at 0, zero velocities
        episode_rewards = []
        for ep in range(n_episodes):
            rng = np.random.RandomState(ep * 1000 + i)
            theta0 = rng.uniform(-np.pi / 9, np.pi / 9)
            initial_state = [0.0, theta0, 0.0, 0.0]
            reward = simulate_episode(L, K, initial_state)
            episode_rewards.append(reward)

        mean_rewards[i] = np.mean(episode_rewards)
        std_rewards[i] = np.std(episode_rewards)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  L={L:.3f}m: reward={mean_rewards[i]:.1f} ±{std_rewards[i]:.1f}, "
                  f"||K||={gain_norms[i]:.2f}")

    return mean_rewards, std_rewards, gain_norms


def main():
    print("=" * 60)
    print("LQR Baseline for Cart-Pole Co-Design")
    print("=" * 60)
    print(f"\nPhysics parameters (matching Newton env):")
    print(f"  Cart mass:       {CART_MASS} kg")
    print(f"  Pole density:    {POLE_DENSITY} kg/m")
    print(f"  Gravity:         {GRAVITY} m/s^2")
    print(f"  dt:              {DT} s (50 Hz, 4 substeps)")
    print(f"  Force max:       {FORCE_MAX} N")
    print(f"  Cart limit:      +/-{X_LIMIT} m")
    print(f"  Episode length:  {MAX_STEPS} steps ({MAX_STEPS * DT:.0f}s)")
    print(f"  Armature:        {ARMATURE}")

    # Dense sweep
    L_values = np.linspace(0.3, 1.2, 50)
    n_episodes = 50

    print(f"\nSweeping L in [{L_values[0]:.2f}, {L_values[-1]:.2f}] m "
          f"({len(L_values)} points, {n_episodes} episodes each)...")

    mean_rewards, std_rewards, gain_norms = sweep_pole_length(L_values, n_episodes)

    # Find optimal
    valid = ~np.isnan(mean_rewards)
    best_idx = np.argmax(mean_rewards[valid])
    best_L = L_values[valid][best_idx]
    best_reward = mean_rewards[valid][best_idx]

    print(f"\n{'=' * 60}")
    print(f"Results")
    print(f"{'=' * 60}")
    print(f"  LQR optimal L:     {best_L:.3f} m")
    print(f"  LQR best reward:   {best_reward:.1f}")
    print(f"  Pole mass at opt:  {pole_mass(best_L):.3f} kg")
    print(f"  ||K|| at optimal:  {gain_norms[valid][best_idx]:.2f}")

    # Show a few neighbors for context
    print(f"\n  Reward landscape near optimum:")
    for j in range(max(0, best_idx - 3), min(len(L_values[valid]), best_idx + 4)):
        marker = " <-- optimal" if j == best_idx else ""
        print(f"    L={L_values[valid][j]:.3f}m: {mean_rewards[valid][j]:.1f} "
              f"±{std_rewards[valid][j]:.1f}{marker}")

    print(f"\n  Comparison point:")
    print(f"    RL initial L:    0.600 m (PGHC starting point)")
    idx_06 = np.argmin(np.abs(L_values - 0.6))
    print(f"    LQR at L=0.6m:   {mean_rewards[idx_06]:.1f} ±{std_rewards[idx_06]:.1f}")

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Reward vs pole length
    ax1 = axes[0]
    ax1.plot(L_values[valid], mean_rewards[valid], 'b-', linewidth=2, label='Mean reward')
    ax1.fill_between(L_values[valid],
                     mean_rewards[valid] - std_rewards[valid],
                     mean_rewards[valid] + std_rewards[valid],
                     alpha=0.2, color='blue', label='+/-1 std')
    ax1.axvline(best_L, color='red', linestyle='--', alpha=0.7,
                label=f'Optimal L = {best_L:.3f} m')
    ax1.axvline(0.6, color='green', linestyle=':', alpha=0.7,
                label='RL initial L = 0.600 m')
    ax1.set_ylabel('Total Episode Reward')
    ax1.set_title('LQR Baseline: Cart-Pole Reward vs Pole Length')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gain norm vs pole length
    ax2 = axes[1]
    ax2.plot(L_values[valid], gain_norms[valid], 'r-', linewidth=2)
    ax2.axvline(best_L, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Pole Length L (m)')
    ax2.set_ylabel('LQR Gain ||K||')
    ax2.set_title('Controller Effort vs Pole Length')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lqr_baseline_cartpole.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to {plot_path}")
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()


if __name__ == "__main__":
    main()
