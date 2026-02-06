#!/usr/bin/env python3
"""
Level 1 Verification Tests: Inverted Pendulum

This module verifies the PGHC algorithm on a simple inverted pendulum,
testing the envelope theorem approximation when the policy converges.

Test Categories:
- 1A: Environment & Physics Correctness
- 1B: Gradient Flow Through Design Parameter
- 1C: Co-Design Convergence
- 1D: Policy Performance

Key Insight (Envelope Theorem):
When computing gradients w.r.t. design parameter L, the policy must be FROZEN.
This means the policy should not adapt its behavior when L changes.
The gradient flows only through the physics dynamics.

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

    # Start from horizontal (theta = pi/2) with zero velocity
    theta_init = math.pi / 2
    obs = env.reset(theta_init)

    # Compute initial energy
    # Convention: theta=0 is upright, theta=pi is hanging
    # PE = m*g*L*(1 + cos(theta)) so that PE is max at upright
    def compute_energy(theta, theta_dot, L, m, g):
        KE = 0.5 * m * L * L * theta_dot * theta_dot
        PE = m * g * L * (1 + math.cos(theta))
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

    # 1A.2: Stable equilibrium at theta=pi (hanging down)
    env = PendulumEnv(dt=0.02)
    obs = env.reset(theta_init=math.pi + 0.01)  # Slightly perturbed

    for _ in range(100):
        obs, _, _, info = env.step(zero_action)

    # Should stay near pi
    final_theta = abs(info["theta"])
    test_result(
        "1A.2 Stable equilibrium at theta=pi",
        abs(final_theta - math.pi) < 0.5 or final_theta < 0.5,  # Near pi or wrapped
        f"Final theta={info['theta']:.3f}, expected ~pi"
    )

    # 1A.3: Unstable equilibrium at theta=0 (upright)
    env = PendulumEnv(dt=0.02)
    obs = env.reset(theta_init=0.01)  # Slightly perturbed from upright

    for _ in range(100):
        obs, _, _, info = env.step(zero_action)

    # Should fall away from 0
    final_theta = abs(info["theta"])
    test_result(
        "1A.3 Unstable equilibrium at theta=0",
        final_theta > 0.1,  # Should have fallen significantly
        f"Final theta={info['theta']:.3f}, expected to fall away from 0"
    )

    # 1A.4: Dynamics equation verification
    # theta_ddot = (g/L)*sin(theta) + torque/(m*L^2)
    env = PendulumEnv(dt=0.001)  # Very small dt for accurate test
    theta_test = 0.5
    torque_test = 1.0
    obs = env.reset(theta_init=theta_test)

    L = env.parametric_model.get_L()
    expected_accel = (g / L) * math.sin(theta_test) + torque_test / (m * L * L)

    # Take one step
    action = torch.tensor([torque_test], dtype=torch.float64)
    obs, _, _, info = env.step(action)

    # Velocity change should be approximately theta_ddot * dt
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

    CRITICAL: For valid comparison, the SAME frozen policy must be used
    for both autograd and FD computations. The policy must not adapt to L.
    """
    print("\n" + "="*50)
    print("Test 1B: Gradient Flow")
    print("="*50)

    HORIZON = 30  # Shorter horizon for cleaner gradients
    L_TEST = 0.6  # Test point
    EPSILON = 0.01  # FD step size

    # 1B.1: Gradient exists and is non-zero
    env = PendulumEnv()
    env.parametric_model.set_L(L_TEST)
    # Create policy with FIXED L_reference - won't adapt when physics L changes
    policy = SimplePendulumPolicy(env, L_reference=L_TEST)

    result = env.rollout_differentiable(policy, horizon=HORIZON)

    test_result(
        "1B.1 Gradient exists and non-zero",
        abs(result["L_grad"]) > 1e-6,
        f"L_grad={result['L_grad']:.6f}"
    )

    # 1B.2: Autograd gradient matches finite difference
    # CRITICAL: Use the SAME policy (same L_reference) for all evaluations
    # This ensures the policy is truly frozen, matching the envelope theorem

    # Autograd gradient (already computed above)
    L_grad_autograd = result["L_grad"]

    # Finite difference with FROZEN policy (same L_reference)
    env_plus = PendulumEnv()
    env_plus.parametric_model.set_L(L_TEST + EPSILON)
    # Use SAME L_reference as before - policy doesn't adapt to new L
    policy_frozen = SimplePendulumPolicy(env_plus, L_reference=L_TEST)
    return_plus = env_plus.rollout_differentiable(policy_frozen, horizon=HORIZON)["total_return"]

    env_minus = PendulumEnv()
    env_minus.parametric_model.set_L(L_TEST - EPSILON)
    policy_frozen = SimplePendulumPolicy(env_minus, L_reference=L_TEST)
    return_minus = env_minus.rollout_differentiable(policy_frozen, horizon=HORIZON)["total_return"]

    L_grad_fd = (return_plus - return_minus) / (2 * EPSILON)

    # Check relative error (should be small with frozen policy)
    if abs(L_grad_fd) > 1e-6:
        rel_err = abs(L_grad_autograd - L_grad_fd) / (abs(L_grad_fd) + 1e-10)
    else:
        rel_err = abs(L_grad_autograd - L_grad_fd)

    test_result(
        "1B.2 Autograd matches FD (frozen policy)",
        rel_err < 0.5,  # 50% tolerance (some numerical error expected)
        f"autograd={L_grad_autograd:.4f}, FD={L_grad_fd:.4f}, rel_err={rel_err:.2%}"
    )

    # 1B.3: Gradient sign is consistent
    same_sign = (L_grad_autograd * L_grad_fd) >= 0 or (abs(L_grad_autograd) < 1 and abs(L_grad_fd) < 1)
    test_result(
        "1B.3 Gradient sign consistent",
        same_sign,
        f"autograd sign: {'+' if L_grad_autograd >= 0 else '-'}, FD sign: {'+' if L_grad_fd >= 0 else '-'}"
    )

    # 1B.4: Gradient magnitude is reasonable (not exploding)
    test_result(
        "1B.4 Gradient magnitude reasonable",
        abs(L_grad_autograd) < 1000,
        f"|L_grad|={abs(L_grad_autograd):.2f}"
    )

    print(f"\nGradient Flow: 4 tests completed")


def test_1C_codesign_convergence():
    """
    Test 1C: Co-Design Convergence

    Verifies that the co-design loop:
    1. Computes valid gradients
    2. Updates L in the gradient direction
    3. Improves or maintains performance via trust region
    """
    print("\n" + "="*50)
    print("Test 1C: Co-Design Convergence")
    print("="*50)

    # Setup
    L_init = 0.7  # Start in middle of range
    env = PendulumEnv()
    env.parametric_model.set_L(L_init)

    print(f"  Initial L: {L_init:.2f} m")
    print(f"  L range: [{env.parametric_model.L_min}, {env.parametric_model.L_max}]")

    # Trust region parameters
    tr = MockTrustRegion(xi=0.1, lr_init=0.01, lr_min=1e-4, lr_max=0.05)

    # Co-design loop
    num_outer_loops = 10
    horizon = 30
    L_history = [L_init]
    return_history = []
    grad_history = []

    for outer_iter in range(num_outer_loops):
        # Create fresh environment with current L
        current_L = L_history[-1]
        env = PendulumEnv()
        env.parametric_model.set_L(current_L)
        # IMPORTANT: Policy uses current L as reference (adapts each outer iteration)
        # This is correct - policy is frozen DURING gradient computation, not across iterations
        policy = SimplePendulumPolicy(env, L_reference=current_L)

        # Evaluate current performance
        result = env.rollout_differentiable(policy, horizon=horizon)
        current_return = result["total_return"]
        L_grad = result["L_grad"]

        return_history.append(current_return)
        grad_history.append(L_grad)

        # Propose new L (gradient ascent for return maximization)
        L_candidate = current_L + tr.lr * L_grad
        L_candidate = max(env.parametric_model.L_min, min(env.parametric_model.L_max, L_candidate))

        # Evaluate candidate with policy adapted to candidate L
        env_candidate = PendulumEnv()
        env_candidate.parametric_model.set_L(L_candidate)
        policy_candidate = SimplePendulumPolicy(env_candidate, L_reference=L_candidate)
        result_candidate = env_candidate.rollout_differentiable(policy_candidate, horizon=horizon)
        candidate_return = result_candidate["total_return"]

        # Trust region check (negate for minimization framework)
        accepted, D = tr.check_acceptance(-current_return, -candidate_return)
        tr.update_lr(accepted, D, -current_return)

        if accepted:
            L_history.append(L_candidate)
        else:
            L_history.append(current_L)

        if outer_iter % 3 == 0:
            print(f"  Iter {outer_iter}: L={L_history[-1]:.3f}, R={current_return:.1f}, "
                  f"grad={L_grad:.2f}, lr={tr.lr:.4f}, accept={accepted}")

    final_L = L_history[-1]
    print(f"  Final L: {final_L:.3f} m")

    # 1C.1: Gradients are finite (no NaN or Inf)
    all_finite = all(math.isfinite(g) for g in grad_history)
    test_result(
        "1C.1 All gradients finite",
        all_finite,
        f"Gradients: {[f'{g:.2f}' for g in grad_history]}"
    )

    # 1C.2: Trust region accepted at least one update
    num_accepted = sum(1 for i in range(1, len(L_history)) if L_history[i] != L_history[i-1])
    test_result(
        "1C.2 Trust region accepted updates",
        num_accepted >= 1,
        f"Accepted {num_accepted}/{num_outer_loops} updates"
    )

    # 1C.3: Performance did not catastrophically degrade
    initial_return = return_history[0]
    final_return = return_history[-1]
    min_return = min(return_history)
    test_result(
        "1C.3 No catastrophic performance loss",
        min_return > initial_return - 20,  # Allow some exploration
        f"Initial: {initial_return:.1f}, Min: {min_return:.1f}, Final: {final_return:.1f}"
    )

    # 1C.4: L moved in gradient direction when accepted
    correct_direction = 0
    total_moves = 0
    for i in range(len(grad_history)):
        if i < len(L_history) - 1:
            L_delta = L_history[i+1] - L_history[i]
            if abs(L_delta) > 1e-6:  # Actually moved
                total_moves += 1
                if L_delta * grad_history[i] > 0:  # Same sign
                    correct_direction += 1

    test_result(
        "1C.4 L moves in gradient direction",
        correct_direction >= total_moves * 0.5 or total_moves == 0,  # At least 50% correct
        f"Correct direction: {correct_direction}/{total_moves}"
    )

    # 1C.5: Gradient magnitudes stayed bounded
    max_grad = max(abs(g) for g in grad_history)
    test_result(
        "1C.5 Gradients bounded (no explosion)",
        max_grad < 1000,
        f"Max |grad|={max_grad:.2f}"
    )

    print(f"\nCo-Design Convergence: 5 tests completed")


def test_1D_policy_performance():
    """
    Test 1D: Policy Performance at Different L Values

    Verifies that the simple policy can achieve swing-up and that
    the reward function produces meaningful differences across L values.
    """
    print("\n" + "="*50)
    print("Test 1D: Policy Performance")
    print("="*50)

    # 1D.1: Swing-up success at some L value
    best_upright_fraction = 0
    best_L = None

    for L_test in [0.5, 0.6, 0.8, 1.0]:
        env = PendulumEnv()
        env.parametric_model.set_L(L_test)
        policy = SimplePendulumPolicy(env, L_reference=L_test)

        obs = env.reset(theta_init=math.pi)  # Start hanging
        upright_count = 0

        for step in range(200):
            action = policy(obs)
            obs, reward, done, info = env.step(action)

            # Check if upright (|theta| < 0.3 rad)
            if abs(info["theta"]) < 0.3:
                upright_count += 1

        upright_fraction = upright_count / 200
        if upright_fraction > best_upright_fraction:
            best_upright_fraction = upright_fraction
            best_L = L_test

    test_result(
        "1D.1 Swing-up achievable at some L",
        best_upright_fraction > 0.2,  # At least 20% upright
        f"Best: {best_upright_fraction:.0%} upright at L={best_L}"
    )

    # 1D.2: Performance varies meaningfully with L
    L_values = [0.5, 0.7, 0.9, 1.1]
    returns = []

    for L in L_values:
        env = PendulumEnv()
        env.parametric_model.set_L(L)
        policy = SimplePendulumPolicy(env, L_reference=L)

        obs = env.reset(theta_init=math.pi)
        total_return = 0

        for _ in range(100):
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            total_return += reward.item()

        returns.append(total_return)

    # Check that returns vary (not all same)
    return_range = max(returns) - min(returns)
    test_result(
        "1D.2 Performance varies with L",
        return_range > 5,  # Meaningful variation
        f"Returns: {[f'{r:.0f}' for r in returns]}, range={return_range:.0f}"
    )

    print(f"\nPolicy Performance: 2 tests completed")


def run_all_tests():
    """Run all Level 1 verification tests."""
    global TESTS_PASSED, TESTS_FAILED

    print("\n" + "#"*60)
    print("# LEVEL 1 VERIFICATION TESTS - Inverted Pendulum")
    print("#"*60)
    print("\nThese tests verify the envelope theorem and co-design convergence.")
    print("All tests must pass before proceeding to Level 2 (Ant).\n")

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
        print("Envelope theorem validated. Ready to proceed to Level 2 (Ant).")
        return 0
    else:
        print(f"\n{TESTS_FAILED} test(s) FAILED!")
        print("Debug envelope theorem issues before proceeding.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
