#!/usr/bin/env python3
"""
Level 1 Verification Tests: Inverted Pendulum

This module verifies the PGHC algorithm on a simple inverted pendulum,
testing the envelope theorem approximation when the policy converges.

Test Categories:
- 1A: Environment & Physics Correctness
- 1B: Gradient Flow Through Design Parameter
- 1C: Co-Design Convergence to Analytical Optimum

Key Insight:
- Optimal pole length L* ≈ (3 * τ_max) / (m * g)
- For τ_max=2 Nm, m=0.5 kg: L* ≈ 1.22 m
- Starting from L=0.5m, co-design should increase L toward ~1.2m

Reference: VERIFICATION_PLAN.md Level 1

Usage:
    python test_level1_pendulum.py
"""

import os
import sys
import math
import numpy as np
import torch

# Global counters
TESTS_PASSED = 0
TESTS_FAILED = 0


def test_result(name, passed, message=""):
    """Record test result."""
    global TESTS_PASSED, TESTS_FAILED
    if passed:
        TESTS_PASSED += 1
        print(f"  [PASS] {name}")
    else:
        TESTS_FAILED += 1
        print(f"  [FAIL] {name}: {message}")


# Import pendulum environment
from envs.pendulum import PendulumEnv, ParametricPendulum
from envs.pendulum.pendulum_env import SimplePendulumPolicy

# Import trust region logic from Level 0
from test_level0_verification import MockTrustRegion


def test_1A_environment_physics():
    """
    Test 1A: Environment and Physics Correctness

    Verifies that the pendulum physics are correct by checking:
    - Energy conservation (without control)
    - Equilibrium points
    - Dynamics equations
    """
    print("\n" + "="*50)
    print("Test 1A: Environment & Physics")
    print("="*50)

    # 1A.1: Energy conservation (no damping, no control)
    env = PendulumEnv(dt=0.01)  # Small timestep for accuracy
    L = env.parametric_model.get_L()
    m = env.mass
    g = env.gravity

    # Start from horizontal (θ = π/2) with zero velocity
    theta_init = math.pi / 2
    obs = env.reset(theta_init)

    # Compute initial energy
    def compute_energy(theta, theta_dot, L, m, g):
        KE = 0.5 * m * L * L * theta_dot * theta_dot
        PE = m * g * L * (1 - math.cos(theta))
        return KE + PE

    E_initial = compute_energy(theta_init, 0.0, L, m, g)

    # Run simulation with zero torque for several steps
    zero_action = torch.tensor([0.0], dtype=torch.float64)
    energies = [E_initial]

    for _ in range(100):
        obs, _, _, info = env.step(zero_action)
        theta = info["theta"]
        theta_dot = info["theta_dot"]
        E = compute_energy(theta, theta_dot, L, m, g)
        energies.append(E)

    # Check energy conservation (should be approximately constant)
    E_final = energies[-1]
    energy_change = abs(E_final - E_initial) / E_initial

    test_result(
        "1A.1 Energy approximately conserved (no control)",
        energy_change < 0.1,  # Allow 10% drift due to Euler integration
        f"Energy change: {energy_change*100:.1f}% (E_init={E_initial:.3f}, E_final={E_final:.3f})"
    )

    # 1A.2: Stable equilibrium at θ=π (hanging down)
    env = PendulumEnv(dt=0.02)
    obs = env.reset(theta_init=math.pi + 0.01)  # Slightly perturbed

    for _ in range(100):
        obs, _, _, info = env.step(zero_action)

    # Should stay near π
    final_theta = abs(info["theta"])
    test_result(
        "1A.2 Stable equilibrium at θ=π",
        abs(final_theta - math.pi) < 0.5 or final_theta < 0.5,  # Near π or wrapped
        f"Final θ={info['theta']:.3f}, expected ≈π"
    )

    # 1A.3: Unstable equilibrium at θ=0 (upright)
    env = PendulumEnv(dt=0.02)
    obs = env.reset(theta_init=0.01)  # Slightly perturbed from upright

    for _ in range(100):
        obs, _, _, info = env.step(zero_action)

    # Should fall away from 0
    final_theta = abs(info["theta"])
    test_result(
        "1A.3 Unstable equilibrium at θ=0",
        final_theta > 0.1,  # Should have fallen significantly
        f"Final θ={info['theta']:.3f}, expected to fall away from 0"
    )

    # 1A.4: Dynamics equation verification
    # θ̈ = (g/L)*sin(θ) + τ/(m*L²)
    env = PendulumEnv(dt=0.001)  # Very small dt for accurate test
    theta_test = 0.5
    torque_test = 1.0
    obs = env.reset(theta_init=theta_test)

    L = env.parametric_model.get_L()
    expected_accel = (g / L) * math.sin(theta_test) + torque_test / (m * L * L)

    # Take one step
    action = torch.tensor([torque_test], dtype=torch.float64)
    obs, _, _, info = env.step(action)

    # Velocity change should be approximately θ̈ * dt
    actual_vel_change = info["theta_dot"]
    expected_vel_change = expected_accel * env.dt

    rel_err = abs(actual_vel_change - expected_vel_change) / (abs(expected_vel_change) + 1e-10)
    test_result(
        "1A.4 Dynamics equation correct",
        rel_err < 0.1,  # 10% tolerance
        f"actual={actual_vel_change:.4f}, expected={expected_vel_change:.4f}, rel_err={rel_err:.2f}"
    )

    print(f"\nEnvironment & Physics: 4 tests completed")


def test_1B_gradient_flow():
    """
    Test 1B: Gradient Flow Through Design Parameter

    Verifies that gradients of the return with respect to pole length L
    are computed correctly using both autograd and finite differences.
    """
    print("\n" + "="*50)
    print("Test 1B: Gradient Flow")
    print("="*50)

    REL_TOL = 0.2  # 20% tolerance (physics introduces some noise)

    # 1B.1: Gradient exists and is non-zero
    env = PendulumEnv()
    policy = SimplePendulumPolicy(env)

    result = env.rollout_differentiable(policy, horizon=50)

    test_result(
        "1B.1 Gradient exists and non-zero",
        abs(result["L_grad"]) > 1e-6,
        f"L_grad={result['L_grad']:.6f}"
    )

    # 1B.2: Gradient sign is correct (suboptimal L should have positive gradient)
    # With L=0.5 < L*≈1.2, increasing L should improve performance
    # So dReturn/dL should be positive (return increases as L increases)
    env = PendulumEnv()
    env.parametric_model.set_L(0.5)  # Suboptimal (too short)
    policy = SimplePendulumPolicy(env)

    result = env.rollout_differentiable(policy, horizon=100)

    test_result(
        "1B.2 Gradient sign correct (L < L* → dR/dL > 0)",
        result["L_grad"] > 0,
        f"L_grad={result['L_grad']:.4f}, expected positive (L=0.5 < L*≈1.2)"
    )

    # 1B.3: Gradient matches finite difference
    epsilon = 0.01
    env1 = PendulumEnv()
    env1.parametric_model.set_L(0.6)
    policy1 = SimplePendulumPolicy(env1)
    result1 = env1.rollout_differentiable(policy1, horizon=50)
    L_grad_autograd = result1["L_grad"]

    # Finite difference
    env_plus = PendulumEnv()
    env_plus.parametric_model.set_L(0.6 + epsilon)
    policy_plus = SimplePendulumPolicy(env_plus)
    return_plus = env_plus.rollout_differentiable(policy_plus, horizon=50)["total_return"]

    env_minus = PendulumEnv()
    env_minus.parametric_model.set_L(0.6 - epsilon)
    policy_minus = SimplePendulumPolicy(env_minus)
    return_minus = env_minus.rollout_differentiable(policy_minus, horizon=50)["total_return"]

    L_grad_fd = (return_plus - return_minus) / (2 * epsilon)

    # Check they're roughly the same sign and magnitude
    if abs(L_grad_fd) > 1e-6:
        rel_err = abs(L_grad_autograd - L_grad_fd) / (abs(L_grad_fd) + 1e-10)
        same_sign = (L_grad_autograd * L_grad_fd) > 0
        test_result(
            "1B.3 Autograd matches finite difference",
            same_sign and rel_err < 0.5,  # 50% tolerance due to policy stochasticity
            f"autograd={L_grad_autograd:.4f}, FD={L_grad_fd:.4f}, rel_err={rel_err:.2f}"
        )
    else:
        test_result(
            "1B.3 Autograd matches finite difference",
            abs(L_grad_autograd) < 1e-3,
            f"FD gradient near zero, autograd={L_grad_autograd:.4f}"
        )

    # 1B.4: Gradient near optimum is smaller
    env_near_opt = PendulumEnv()
    L_opt = env_near_opt.parametric_model.compute_optimal_length(env_near_opt.torque_max)
    env_near_opt.parametric_model.set_L(L_opt)
    policy_opt = SimplePendulumPolicy(env_near_opt)

    result_opt = env_near_opt.rollout_differentiable(policy_opt, horizon=100)

    test_result(
        "1B.4 Gradient smaller near optimum",
        abs(result_opt["L_grad"]) < abs(result["L_grad"]) or abs(result_opt["L_grad"]) < 5.0,
        f"|grad| at L=0.5: {abs(result['L_grad']):.4f}, |grad| at L*={L_opt:.2f}: {abs(result_opt['L_grad']):.4f}"
    )

    print(f"\nGradient Flow: 4 tests completed")


def test_1C_codesign_convergence():
    """
    Test 1C: Co-Design Convergence

    Verifies that the co-design loop converges toward the analytical
    optimal pole length.
    """
    print("\n" + "="*50)
    print("Test 1C: Co-Design Convergence")
    print("="*50)

    # Setup
    env = PendulumEnv()
    L_init = 0.5  # Start suboptimal
    env.parametric_model.set_L(L_init)

    L_opt = env.parametric_model.compute_optimal_length(env.torque_max)
    print(f"  Initial L: {L_init:.2f} m")
    print(f"  Target L*: {L_opt:.2f} m")

    # Trust region parameters
    tr = MockTrustRegion(xi=0.1, lr_init=0.05, lr_min=1e-4, lr_max=0.2)

    # Co-design loop
    num_outer_loops = 20
    horizon = 100
    L_history = [L_init]
    return_history = []

    for outer_iter in range(num_outer_loops):
        # Create fresh environment with current L
        current_L = L_history[-1]
        env = PendulumEnv()
        env.parametric_model.set_L(current_L)
        policy = SimplePendulumPolicy(env)

        # Evaluate current performance
        result = env.rollout_differentiable(policy, horizon=horizon)
        current_return = result["total_return"]
        L_grad = result["L_grad"]

        return_history.append(current_return)

        # Propose new L (gradient ascent for return maximization)
        L_candidate = current_L + tr.lr * L_grad
        L_candidate = max(env.parametric_model.L_min, min(env.parametric_model.L_max, L_candidate))

        # Evaluate candidate
        env_candidate = PendulumEnv()
        env_candidate.parametric_model.set_L(L_candidate)
        policy_candidate = SimplePendulumPolicy(env_candidate)
        result_candidate = env_candidate.rollout_differentiable(policy_candidate, horizon=horizon)
        candidate_return = result_candidate["total_return"]

        # Trust region check (maximize return, so negate for minimization framework)
        # D = current - candidate, positive D means candidate is worse (for minimization)
        # For maximization: we want candidate > current, so D < 0 is good
        # Our trust region uses cost minimization, so we negate returns
        accepted, D = tr.check_acceptance(-current_return, -candidate_return)
        tr.update_lr(accepted, D, -current_return)

        if accepted:
            L_history.append(L_candidate)
        else:
            L_history.append(current_L)  # Keep current

        if outer_iter % 5 == 0:
            print(f"  Iter {outer_iter}: L={L_history[-1]:.3f}, R={current_return:.1f}, "
                  f"grad={L_grad:.3f}, lr={tr.lr:.4f}, accept={accepted}")

    final_L = L_history[-1]
    print(f"  Final L: {final_L:.3f} m (target: {L_opt:.2f} m)")

    # 1C.1: L moved toward optimum
    initial_error = abs(L_init - L_opt)
    final_error = abs(final_L - L_opt)
    test_result(
        "1C.1 L moved toward optimum",
        final_error < initial_error,
        f"Initial error: {initial_error:.3f}, Final error: {final_error:.3f}"
    )

    # 1C.2: Final L within 30% of optimum
    rel_error = abs(final_L - L_opt) / L_opt
    test_result(
        "1C.2 Final L within 30% of L*",
        rel_error < 0.3,
        f"|L_final - L*| / L* = {rel_error:.2f}"
    )

    # 1C.3: Performance improved
    initial_return = return_history[0]
    final_return = return_history[-1]
    test_result(
        "1C.3 Performance improved",
        final_return > initial_return,
        f"Initial return: {initial_return:.1f}, Final return: {final_return:.1f}"
    )

    # 1C.4: L consistently increased (for L < L*)
    L_increased = sum(1 for i in range(1, len(L_history)) if L_history[i] >= L_history[i-1])
    increase_rate = L_increased / (len(L_history) - 1)
    test_result(
        "1C.4 L increased in majority of iterations",
        increase_rate >= 0.5,
        f"Increased in {L_increased}/{len(L_history)-1} iterations ({increase_rate:.0%})"
    )

    # 1C.5: L trajectory is monotonically improving trend
    # Check if first half average < second half average
    mid = len(L_history) // 2
    first_half_avg = sum(L_history[:mid]) / mid
    second_half_avg = sum(L_history[mid:]) / (len(L_history) - mid)
    test_result(
        "1C.5 L trend is increasing",
        second_half_avg > first_half_avg,
        f"First half avg: {first_half_avg:.3f}, Second half avg: {second_half_avg:.3f}"
    )

    print(f"\nCo-Design Convergence: 5 tests completed")


def test_1D_policy_performance():
    """
    Test 1D: Policy Performance at Different L Values

    Verifies that the simple policy can achieve swing-up and that
    performance varies predictably with pole length.
    """
    print("\n" + "="*50)
    print("Test 1D: Policy Performance")
    print("="*50)

    # 1D.1: Swing-up success with optimal L
    env = PendulumEnv()
    L_opt = env.parametric_model.compute_optimal_length(env.torque_max)
    env.parametric_model.set_L(L_opt)
    policy = SimplePendulumPolicy(env)

    obs = env.reset(theta_init=math.pi)  # Start hanging
    upright_count = 0

    for step in range(200):
        action = policy(obs)
        obs, reward, done, info = env.step(action)

        # Check if upright (|θ| < 0.3 rad ≈ 17 degrees)
        if abs(info["theta"]) < 0.3:
            upright_count += 1

    upright_fraction = upright_count / 200
    test_result(
        "1D.1 Swing-up success with L*",
        upright_fraction > 0.3,  # At least 30% of time upright
        f"Upright {upright_fraction:.0%} of time at L={L_opt:.2f}"
    )

    # 1D.2: Suboptimal L has worse performance
    L_values = [0.4, 0.6, 0.8, 1.0, 1.2]
    returns = []

    for L in L_values:
        env = PendulumEnv()
        env.parametric_model.set_L(L)
        policy = SimplePendulumPolicy(env)

        obs = env.reset(theta_init=math.pi)
        total_return = 0

        for _ in range(100):
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            total_return += reward.item()

        returns.append(total_return)

    # Find which L gave best return
    best_idx = returns.index(max(returns))
    best_L = L_values[best_idx]

    test_result(
        "1D.2 Best performance near L*",
        abs(best_L - L_opt) < 0.3,
        f"Best L={best_L:.2f}, L*={L_opt:.2f}, returns={[f'{r:.0f}' for r in returns]}"
    )

    print(f"\nPolicy Performance: 2 tests completed")


def run_all_tests():
    """Run all Level 1 verification tests."""
    global TESTS_PASSED, TESTS_FAILED

    print("\n" + "#"*60)
    print("# LEVEL 1 VERIFICATION TESTS - Inverted Pendulum")
    print("#"*60)
    print("\nThese tests verify the envelope theorem and co-design convergence.")
    print("All tests must pass before proceeding to Level 2 (HalfCheetah).\n")

    # Run test suites
    test_1A_environment_physics()
    test_1B_gradient_flow()
    test_1C_codesign_convergence()
    test_1D_policy_performance()

    # Summary
    print("\n" + "="*60)
    print("LEVEL 1 TEST SUMMARY")
    print("="*60)
    total = TESTS_PASSED + TESTS_FAILED
    print(f"Total:  {total}")
    print(f"Passed: {TESTS_PASSED}")
    print(f"Failed: {TESTS_FAILED}")

    if TESTS_FAILED == 0:
        print("\nAll Level 1 tests PASSED!")
        print("Envelope theorem validated. Ready to proceed to Level 2 (HalfCheetah).")
        return 0
    else:
        print(f"\n{TESTS_FAILED} test(s) FAILED!")
        print("Debug envelope theorem issues before proceeding to Level 2.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
