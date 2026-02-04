#!/usr/bin/env python3
"""
Level 0 Verification Tests for PGHC Algorithm Components

This module tests the mathematical correctness of PGHC algorithm components
WITHOUT physics simulation. This is the gate before proceeding to Level 1 (Pendulum).

Test Categories:
- 0A: Gradient Pipeline (autograd vs finite-difference)
- 0B: Trust Region (accept/reject/LR adaptation)
- 0C: Quaternion Chain Rule (analytical vs autograd derivatives)

Usage:
    python test_level0_verification.py

Pass criteria: All tests pass (exit code 0)
"""

import os
import sys
import math
import numpy as np
import torch

# Global counters (from test_implementation.py pattern)
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


def _finite_difference_gradient(func, theta_val, epsilon=1e-5):
    """
    Compute gradient using central finite differences.

    Args:
        func: Callable that takes scalar and returns scalar
        theta_val: Point at which to compute gradient
        epsilon: Step size for finite difference

    Returns:
        Approximate gradient (scalar)
    """
    f_plus = func(theta_val + epsilon)
    f_minus = func(theta_val - epsilon)
    return (f_plus - f_minus) / (2 * epsilon)


class MockTrustRegion:
    """
    Isolated trust region logic matching hybrid_agent.py defaults.

    This class extracts the trust region decision logic for unit testing
    without requiring the full HybridAMPAgent infrastructure.
    """

    def __init__(self, xi=0.1, lr_init=0.01, lr_min=1e-5, lr_max=0.1,
                 decay_factor=0.5, growth_factor=1.5):
        """
        Initialize trust region parameters.

        Args:
            xi: Performance degradation threshold (default: 0.1 = 10%)
            lr_init: Initial learning rate
            lr_min: Minimum learning rate (prevents collapse)
            lr_max: Maximum learning rate (prevents instability)
            decay_factor: LR multiplier on reject
            growth_factor: LR multiplier on small improvement
        """
        self.xi = xi
        self.lr = lr_init
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.decay_factor = decay_factor
        self.growth_factor = growth_factor

    def check_acceptance(self, current_obj, candidate_obj):
        """
        Check if a candidate solution should be accepted.

        The trust region accepts if:
            D = current_obj - candidate_obj >= -xi * |current_obj|

        For maximization problems (like reward), D > 0 means improvement.

        Args:
            current_obj: Current objective value
            candidate_obj: Candidate objective value after update

        Returns:
            Tuple of (accepted: bool, D: float)
        """
        D = current_obj - candidate_obj
        # Reject if D < -xi * |current_obj| (i.e., got worse by more than xi%)
        accepted = D >= -self.xi * abs(current_obj)
        return accepted, D

    def update_lr(self, accepted, D, current_obj):
        """
        Update learning rate based on acceptance and improvement.

        Rules (matching hybrid_agent.py:518-540):
        - On reject: lr *= decay_factor (shrink step)
        - On accept with small improvement: lr *= growth_factor (expand step)
        - Small improvement defined as D < 0.01 * |current_obj| (< 1% gain)

        Args:
            accepted: Whether the step was accepted
            D: Improvement D = current - candidate
            current_obj: Current objective value
        """
        if not accepted:
            # Shrink step on reject
            self.lr *= self.decay_factor
            self.lr = max(self.lr, self.lr_min)
        else:
            # Check for small improvement to potentially grow LR
            small_improvement_threshold = 0.01 * abs(current_obj) if current_obj != 0 else 0.01
            if D < small_improvement_threshold and D >= 0:
                # Small positive improvement -> grow LR for faster convergence
                self.lr *= self.growth_factor
                self.lr = min(self.lr, self.lr_max)


def test_0A_gradient_pipeline():
    """
    Test 0A: Gradient Pipeline Verification

    Verifies that PyTorch autograd produces correct gradients by comparing
    against finite difference approximations for various functions.

    Tolerance: Relative error < 1% for FD comparison
    """
    print("\n" + "="*50)
    print("Test 0A: Gradient Pipeline")
    print("="*50)

    REL_TOL = 0.01  # 1% relative tolerance

    # 0A.1: Basic quadratic f(theta) = (theta - 0.5)^2
    def f1_numpy(theta):
        return (theta - 0.5) ** 2

    def f1_torch(theta):
        return (theta - 0.5) ** 2

    theta_val = 0.3
    theta_t = torch.tensor([theta_val], requires_grad=True, dtype=torch.float64)
    loss = f1_torch(theta_t)
    loss.backward()
    autograd_grad = theta_t.grad.item()
    fd_grad = _finite_difference_gradient(f1_numpy, theta_val)

    rel_err = abs(autograd_grad - fd_grad) / (abs(fd_grad) + 1e-10)
    test_result(
        "0A.1 Basic quadratic autograd",
        rel_err < REL_TOL,
        f"autograd={autograd_grad:.6f}, FD={fd_grad:.6f}, rel_err={rel_err:.4f}"
    )

    # 0A.2: Non-trivial f(theta) = (theta - 0.5)^2 + 0.1*sin(10*theta)
    def f2_numpy(theta):
        return (theta - 0.5) ** 2 + 0.1 * np.sin(10 * theta)

    def f2_torch(theta):
        return (theta - 0.5) ** 2 + 0.1 * torch.sin(10 * theta)

    theta_val = 0.7
    theta_t = torch.tensor([theta_val], requires_grad=True, dtype=torch.float64)
    loss = f2_torch(theta_t)
    loss.backward()
    autograd_grad = theta_t.grad.item()
    fd_grad = _finite_difference_gradient(f2_numpy, theta_val)

    rel_err = abs(autograd_grad - fd_grad) / (abs(fd_grad) + 1e-10)
    test_result(
        "0A.2 Non-trivial gradient (sin term)",
        rel_err < REL_TOL,
        f"autograd={autograd_grad:.6f}, FD={fd_grad:.6f}, rel_err={rel_err:.4f}"
    )

    # 0A.3: FD numerical stability across epsilon values
    def f3_numpy(theta):
        return theta ** 3 - 2 * theta ** 2 + theta

    # Analytical gradient: 3*theta^2 - 4*theta + 1
    theta_val = 0.6
    analytical_grad = 3 * theta_val ** 2 - 4 * theta_val + 1

    epsilons = [1e-4, 1e-5, 1e-6]
    fd_grads = [_finite_difference_gradient(f3_numpy, theta_val, eps) for eps in epsilons]

    # All FD gradients should be close to analytical
    max_rel_err = max(abs(fd - analytical_grad) / (abs(analytical_grad) + 1e-10) for fd in fd_grads)
    test_result(
        "0A.3 FD numerical stability (eps=1e-4,1e-5,1e-6)",
        max_rel_err < REL_TOL,
        f"analytical={analytical_grad:.6f}, FD grads={fd_grads}, max_rel_err={max_rel_err:.4f}"
    )

    # 0A.4: Chain rule g(h(theta)) where h(theta) = theta^2, g(x) = sin(x)
    def f4_numpy(theta):
        return np.sin(theta ** 2)

    def f4_torch(theta):
        return torch.sin(theta ** 2)

    theta_val = 0.8
    # Analytical: d/dtheta sin(theta^2) = cos(theta^2) * 2*theta
    analytical_grad = np.cos(theta_val ** 2) * 2 * theta_val

    theta_t = torch.tensor([theta_val], requires_grad=True, dtype=torch.float64)
    loss = f4_torch(theta_t)
    loss.backward()
    autograd_grad = theta_t.grad.item()
    fd_grad = _finite_difference_gradient(f4_numpy, theta_val)

    rel_err_autograd = abs(autograd_grad - analytical_grad) / (abs(analytical_grad) + 1e-10)
    rel_err_fd = abs(fd_grad - analytical_grad) / (abs(analytical_grad) + 1e-10)
    test_result(
        "0A.4 Chain rule composition",
        rel_err_autograd < REL_TOL and rel_err_fd < REL_TOL,
        f"analytical={analytical_grad:.6f}, autograd={autograd_grad:.6f}, FD={fd_grad:.6f}"
    )

    print(f"\nGradient Pipeline: 4 tests completed")


def test_0B_trust_region():
    """
    Test 0B: Trust Region Verification

    Verifies the trust region accept/reject logic and learning rate adaptation
    by testing various scenarios with a simple quadratic cost function.

    Cost: f(theta) = (theta - 1)^2 (minimize, optimum at theta=1)

    The trust region logic from hybrid_agent.py uses:
    - D = current_obj - candidate_obj
    - D > 0 means improvement (cost went down)
    - Accept if D >= -xi * |current_obj|
    """
    print("\n" + "="*50)
    print("Test 0B: Trust Region")
    print("="*50)

    def cost(theta):
        """Cost to minimize: (theta - 1)^2, optimum at theta=1, min value=0"""
        return (theta - 1) ** 2

    def cost_gradient(theta):
        """Gradient of cost: 2*(theta - 1)"""
        return 2 * (theta - 1)

    # 0B.1: Accept on improvement (D > 0)
    tr = MockTrustRegion(xi=0.1, lr_init=0.1)
    theta = 0.0
    current_cost = cost(theta)  # 1.0
    grad = cost_gradient(theta)  # -2.0
    theta_candidate = theta - tr.lr * grad  # 0.0 - 0.1 * (-2.0) = 0.2 (gradient descent)
    candidate_cost = cost(theta_candidate)  # 0.64

    accepted, D = tr.check_acceptance(current_cost, candidate_cost)
    # D = 1.0 - 0.64 = 0.36 (improvement, cost went down)
    # Accept if D >= -xi*|current| = -0.1*1.0 = -0.1
    # 0.36 >= -0.1 is TRUE, so accept

    test_result(
        "0B.1 Accept on improvement (D > 0)",
        accepted and D > 0,
        f"Expected accept with D>0 (cost reduction), got accepted={accepted}, D={D:.6f}"
    )

    # 0B.2: Reject on overshoot
    tr = MockTrustRegion(xi=0.1, lr_init=5.0)
    theta = 0.9
    current_cost = cost(theta)  # 0.01
    grad = cost_gradient(theta)  # -0.2
    theta_candidate = theta - tr.lr * grad  # 0.9 - 5.0 * (-0.2) = 1.9
    candidate_cost = cost(theta_candidate)  # 0.81

    accepted, D = tr.check_acceptance(current_cost, candidate_cost)
    # D = 0.01 - 0.81 = -0.8 (got worse)
    # Accept if D >= -xi*|current| = -0.1*0.01 = -0.001
    # -0.8 >= -0.001 is FALSE, so reject

    test_result(
        "0B.2 Reject on overshoot",
        not accepted and D < 0,
        f"Expected reject with D<0 (cost increased), got accepted={accepted}, D={D:.6f}"
    )

    # 0B.3: LR decays on reject
    tr = MockTrustRegion(xi=0.1, lr_init=0.5, decay_factor=0.5)
    initial_lr = tr.lr

    # Simulate a reject
    tr.update_lr(accepted=False, D=-0.5, current_obj=1.0)

    test_result(
        "0B.3 LR decays on reject",
        abs(tr.lr - initial_lr * 0.5) < 1e-10,
        f"Expected LR={initial_lr*0.5}, got LR={tr.lr}"
    )

    # 0B.4: LR grows on small improvement
    tr = MockTrustRegion(xi=0.1, lr_init=0.01, growth_factor=1.5)
    initial_lr = tr.lr

    # Simulate accept with small improvement (D < 1% of current)
    current_obj = 1.0
    small_D = 0.005  # 0.5% improvement
    tr.update_lr(accepted=True, D=small_D, current_obj=current_obj)

    test_result(
        "0B.4 LR grows on small improvement",
        abs(tr.lr - initial_lr * 1.5) < 1e-10,
        f"Expected LR={initial_lr*1.5}, got LR={tr.lr}"
    )

    # 0B.5: Convergence test - multiple steps should approach optimum
    tr = MockTrustRegion(xi=0.1, lr_init=0.1)
    theta = 0.0

    for _ in range(50):
        current_cost = cost(theta)
        grad = cost_gradient(theta)
        theta_candidate = theta - tr.lr * grad
        candidate_cost = cost(theta_candidate)

        accepted, D = tr.check_acceptance(current_cost, candidate_cost)
        tr.update_lr(accepted, D, current_cost)

        if accepted:
            theta = theta_candidate

    test_result(
        "0B.5 Converges to optimum",
        abs(theta - 1.0) < 0.01,
        f"Expected theta~1.0, got theta={theta:.4f}"
    )

    # 0B.6: Stay at optimum
    tr = MockTrustRegion(xi=0.1, lr_init=0.1)
    theta = 1.0  # Already at optimum

    for _ in range(10):
        current_cost = cost(theta)
        grad = cost_gradient(theta)
        theta_candidate = theta - tr.lr * grad
        candidate_cost = cost(theta_candidate)

        accepted, D = tr.check_acceptance(current_cost, candidate_cost)
        tr.update_lr(accepted, D, current_cost)

        if accepted:
            theta = theta_candidate

    test_result(
        "0B.6 Stays at optimum",
        abs(theta - 1.0) < 0.01,
        f"Expected theta~1.0, got theta={theta:.4f}"
    )

    print(f"\nTrust Region: 6 tests completed")


def test_0C_quaternion_chain_rule():
    """
    Test 0C: Quaternion Chain Rule Verification

    Verifies the quaternion math and chain rule coefficients used in
    hybrid_agent.py:610-665 for extracting design gradients.

    Quaternion for X-rotation: q = [cos(theta/2), sin(theta/2), 0, 0]

    Tolerance: Absolute error < 1e-6 (exact calculations, only numerical error)
    """
    print("\n" + "="*50)
    print("Test 0C: Quaternion Chain Rule")
    print("="*50)

    ABS_TOL = 1e-6

    # 0C.1: Identity quaternion at theta=0
    theta = 0.0
    half_theta = theta / 2.0
    w = math.cos(half_theta)
    x = math.sin(half_theta)
    q = [w, x, 0, 0]

    expected = [1, 0, 0, 0]
    test_result(
        "0C.1 Identity quaternion at theta=0",
        all(abs(q[i] - expected[i]) < ABS_TOL for i in range(4)),
        f"Expected {expected}, got {q}"
    )

    # 0C.2: d/dtheta cos(theta/2) = -sin(theta/2)/2
    theta = 0.5
    half_theta = theta / 2.0

    # Analytical derivative
    dw_dtheta_analytical = -math.sin(half_theta) / 2.0

    # Numerical derivative via finite difference
    eps = 1e-7
    w_plus = math.cos((theta + eps) / 2.0)
    w_minus = math.cos((theta - eps) / 2.0)
    dw_dtheta_numerical = (w_plus - w_minus) / (2 * eps)

    test_result(
        "0C.2 d/dtheta cos(theta/2) = -sin(theta/2)/2",
        abs(dw_dtheta_analytical - dw_dtheta_numerical) < ABS_TOL,
        f"analytical={dw_dtheta_analytical:.8f}, numerical={dw_dtheta_numerical:.8f}"
    )

    # 0C.3: d/dtheta sin(theta/2) = cos(theta/2)/2
    dx_dtheta_analytical = math.cos(half_theta) / 2.0

    x_plus = math.sin((theta + eps) / 2.0)
    x_minus = math.sin((theta - eps) / 2.0)
    dx_dtheta_numerical = (x_plus - x_minus) / (2 * eps)

    test_result(
        "0C.3 d/dtheta sin(theta/2) = cos(theta/2)/2",
        abs(dx_dtheta_analytical - dx_dtheta_numerical) < ABS_TOL,
        f"analytical={dx_dtheta_analytical:.8f}, numerical={dx_dtheta_numerical:.8f}"
    )

    # 0C.4: d/dtheta sin^2(theta/2) = sin(theta)/2
    # This tests the chain rule: d/dtheta [sin(theta/2)]^2 = 2*sin(theta/2)*cos(theta/2)/2 = sin(theta)/2
    f_analytical_derivative = math.sin(theta) / 2.0

    sin2_plus = math.sin((theta + eps) / 2.0) ** 2
    sin2_minus = math.sin((theta - eps) / 2.0) ** 2
    f_numerical_derivative = (sin2_plus - sin2_minus) / (2 * eps)

    test_result(
        "0C.4 d/dtheta sin^2(theta/2) = sin(theta)/2",
        abs(f_analytical_derivative - f_numerical_derivative) < ABS_TOL,
        f"analytical={f_analytical_derivative:.8f}, numerical={f_numerical_derivative:.8f}"
    )

    # 0C.5: Mock _extract_design_gradient chain rule
    # Simulates the chain rule from hybrid_agent.py:658-663
    # grad_theta = grad_w * dw_dtheta + grad_x * dx_dtheta

    theta = 0.5
    half_theta = theta / 2.0
    dw_dtheta = -math.sin(half_theta) / 2.0
    dx_dtheta = math.cos(half_theta) / 2.0

    # Mock quaternion gradients (as if from backprop)
    grad_w = 0.3
    grad_x = 0.7

    # Chain rule result
    grad_theta_computed = grad_w * dw_dtheta + grad_x * dx_dtheta

    # Verify by computing the same thing via torch autograd
    theta_t = torch.tensor([theta], requires_grad=True, dtype=torch.float64)
    half_theta_t = theta_t / 2.0
    w_t = torch.cos(half_theta_t)
    x_t = torch.sin(half_theta_t)

    # Compute "loss" = grad_w * w + grad_x * x (linear combination)
    # This simulates how the loss depends on quaternion components
    loss = grad_w * w_t + grad_x * x_t
    loss.backward()
    grad_theta_autograd = theta_t.grad.item()

    test_result(
        "0C.5 Mock _extract_design_gradient matches autograd",
        abs(grad_theta_computed - grad_theta_autograd) < ABS_TOL,
        f"manual={grad_theta_computed:.8f}, autograd={grad_theta_autograd:.8f}"
    )

    print(f"\nQuaternion Chain Rule: 5 tests completed")


def run_all_tests():
    """Run all Level 0 verification tests."""
    global TESTS_PASSED, TESTS_FAILED

    print("\n" + "#"*60)
    print("# LEVEL 0 VERIFICATION TESTS - PGHC Algorithm Components")
    print("#"*60)
    print("\nThese tests verify mathematical correctness WITHOUT physics simulation.")
    print("All tests must pass before proceeding to Level 1 (Pendulum).\n")

    # Run test suites
    test_0A_gradient_pipeline()
    test_0B_trust_region()
    test_0C_quaternion_chain_rule()

    # Summary
    print("\n" + "="*60)
    print("LEVEL 0 TEST SUMMARY")
    print("="*60)
    total = TESTS_PASSED + TESTS_FAILED
    print(f"Total:  {total}")
    print(f"Passed: {TESTS_PASSED}")
    print(f"Failed: {TESTS_FAILED}")

    if TESTS_FAILED == 0:
        print("\nAll Level 0 tests PASSED!")
        print("Ready to proceed to Level 1 (Pendulum) verification.")
        return 0
    else:
        print(f"\n{TESTS_FAILED} test(s) FAILED!")
        print("Fix identified issues before proceeding to Level 1.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
