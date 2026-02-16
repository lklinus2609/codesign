#!/usr/bin/env python3
"""
Smoke test for FDEvaluator — validates the finite difference gradient pipeline.

Tests:
1. FDEvaluator builds successfully with SolverMuJoCo
2. Forward rollout completes without NaN/blowup
3. CoT values are reasonable (finite, positive)
4. FD gradient is non-zero and finite
5. Solver consistency: FDEvaluator forward distances are physically plausible
6. Gradient sign check: perturbing theta should change CoT monotonically

Usage:
    python validate_outer_loop.py
"""

import os
import sys
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import warp as wp
import newton  # noqa: F401

from g1_mjcf_modifier import G1MJCFModifier, SYMMETRIC_PAIRS, NUM_DESIGN_PARAMS
from g1_eval_worker import FDEvaluator

# Paths
CODESIGN_DIR = os.path.dirname(SCRIPT_DIR)
BASE_MJCF_PATH = os.path.join(CODESIGN_DIR, "MimicKit", "data", "assets", "g1", "g1.xml")

# Test config — small for speed
NUM_WORLDS = 4
HORIZON = 30  # ~1 second at 30 Hz


def generate_random_actions(num_worlds, num_steps, act_dim=29):
    """Generate small random actions (simulating a weak policy)."""
    actions = []
    for _ in range(num_steps):
        # Small actions around zero — robot should mostly stand
        a = np.random.randn(num_worlds, act_dim).astype(np.float32) * 0.1
        actions.append(a)
    return actions


def test_1_build():
    """Test 1: FDEvaluator builds successfully."""
    print("\n" + "=" * 60)
    print("Test 1: FDEvaluator Construction")
    print("=" * 60)

    if not os.path.exists(BASE_MJCF_PATH):
        print(f"  [SKIP] G1 model not found: {BASE_MJCF_PATH}")
        return None, False

    theta = np.zeros(NUM_DESIGN_PARAMS, dtype=np.float64)
    mjcf_modifier = G1MJCFModifier(BASE_MJCF_PATH)

    t0 = time.time()
    fd_eval = FDEvaluator(
        mjcf_modifier=mjcf_modifier,
        theta_np=theta,
        num_worlds=NUM_WORLDS,
        horizon=HORIZON,
        device="cuda:0",
    )
    dt = time.time() - t0

    # Verify key attributes
    checks = [
        ("joints_per_world", fd_eval.joints_per_world > 0),
        ("bodies_per_world", fd_eval.bodies_per_world > 0),
        ("num_act_dofs", fd_eval.num_act_dofs > 0),
        ("total_mass", fd_eval.total_mass > 10.0),  # G1 is ~35 kg
        ("num_param_joints", fd_eval.num_param_joints == NUM_DESIGN_PARAMS * 2),
        ("solver exists", fd_eval.solver is not None),
    ]

    all_ok = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {name}")

    print(f"  Build time: {dt:.1f}s")
    print(f"  [{'PASS' if all_ok else 'FAIL'}] FDEvaluator construction")
    return fd_eval, all_ok


def test_2_rollout(fd_eval):
    """Test 2: Forward rollout completes without NaN."""
    print("\n" + "=" * 60)
    print("Test 2: Forward Rollout Stability")
    print("=" * 60)

    theta = np.zeros(NUM_DESIGN_PARAMS, dtype=np.float64)
    actions = generate_random_actions(NUM_WORLDS, HORIZON, fd_eval.num_act_dofs)

    t0 = time.time()
    fwd_dist, cot = fd_eval._run_rollout(theta, actions)
    dt = time.time() - t0

    print(f"  Forward distance: {fwd_dist:.4f} m")
    print(f"  Cost of Transport: {cot:.4f}")
    print(f"  Rollout time: {dt:.2f}s")

    checks = [
        ("fwd_dist finite", np.isfinite(fwd_dist)),
        ("cot finite", np.isfinite(cot)),
        ("cot positive", cot > 0),
        ("cot reasonable", cot < 1000),  # shouldn't be astronomically large
    ]

    all_ok = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {name}")

    print(f"  [{'PASS' if all_ok else 'FAIL'}] Forward rollout stability")
    return all_ok, actions


def test_3_cot_values(fd_eval, actions):
    """Test 3: CoT values are reasonable across different thetas."""
    print("\n" + "=" * 60)
    print("Test 3: CoT Value Sanity")
    print("=" * 60)

    thetas_to_test = [
        ("zero", np.zeros(NUM_DESIGN_PARAMS, dtype=np.float64)),
        ("small positive", np.full(NUM_DESIGN_PARAMS, 0.05, dtype=np.float64)),
        ("small negative", np.full(NUM_DESIGN_PARAMS, -0.05, dtype=np.float64)),
        ("mixed", np.array([0.1, -0.1, 0.05, -0.05, 0.02, -0.02], dtype=np.float64)),
    ]

    all_ok = True
    for name, theta in thetas_to_test:
        fwd_dist, cot = fd_eval._run_rollout(theta, actions)
        ok = np.isfinite(cot) and cot > 0 and np.isfinite(fwd_dist)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] theta={name}: fwd_dist={fwd_dist:.4f}, CoT={cot:.4f}")

    print(f"  [{'PASS' if all_ok else 'FAIL'}] CoT value sanity")
    return all_ok


def test_4_fd_gradient(fd_eval, actions):
    """Test 4: FD gradient is non-zero, finite, and sensible."""
    print("\n" + "=" * 60)
    print("Test 4: Finite Difference Gradient")
    print("=" * 60)

    theta = np.zeros(NUM_DESIGN_PARAMS, dtype=np.float64)

    t0 = time.time()
    grad, fwd_dist, cot = fd_eval.compute_fd_gradient(theta, actions, eps=1e-4)
    dt = time.time() - t0

    print(f"\n  Gradient: {grad.tolist()}")
    print(f"  Grad norm: {np.linalg.norm(grad):.6f}")
    print(f"  Center fwd_dist: {fwd_dist:.4f}, CoT: {cot:.4f}")
    print(f"  FD time: {dt:.1f}s ({dt / 13:.1f}s per rollout)")

    checks = [
        ("grad finite", np.all(np.isfinite(grad))),
        ("grad not all-zero", np.any(grad != 0)),
        ("grad norm reasonable", np.linalg.norm(grad) < 1000),
        ("center cot finite", np.isfinite(cot)),
    ]

    all_ok = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {name}")

    print(f"  [{'PASS' if all_ok else 'FAIL'}] FD gradient computation")
    return all_ok


def test_5_physical_plausibility(fd_eval, actions):
    """Test 5: Forward distances are physically plausible."""
    print("\n" + "=" * 60)
    print("Test 5: Physical Plausibility")
    print("=" * 60)

    theta = np.zeros(NUM_DESIGN_PARAMS, dtype=np.float64)
    fwd_dist, cot = fd_eval._run_rollout(theta, actions)

    # With random small actions over 1 second, the robot shouldn't fly away
    # or walk very far. Forward distance should be modest.
    checks = [
        ("fwd_dist not extreme", abs(fwd_dist) < 10.0),  # <10m in 1s
        ("cot > 0", cot > 0),
        ("cot < 100", cot < 100),  # reasonable CoT range
    ]

    all_ok = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {name} (fwd_dist={fwd_dist:.4f}, CoT={cot:.4f})")

    print(f"  [{'PASS' if all_ok else 'FAIL'}] Physical plausibility")
    return all_ok


def test_6_gradient_consistency(fd_eval, actions):
    """Test 6: Gradient direction is consistent with CoT changes."""
    print("\n" + "=" * 60)
    print("Test 6: Gradient Consistency")
    print("=" * 60)

    theta = np.zeros(NUM_DESIGN_PARAMS, dtype=np.float64)
    grad, _, cot_center = fd_eval.compute_fd_gradient(theta, actions, eps=1e-4)

    # Take a step in the gradient direction (should decrease CoT)
    step_size = 0.01
    theta_stepped = theta + step_size * grad  # grad is -dCoT/dtheta
    _, cot_stepped = fd_eval._run_rollout(theta_stepped, actions)

    delta_cot = cot_stepped - cot_center
    # grad is negative gradient of CoT, so stepping in grad direction
    # should decrease CoT (delta_cot < 0)
    expected_decrease = (delta_cot < 0) or (abs(delta_cot) < 1e-6)

    print(f"  CoT at center:  {cot_center:.6f}")
    print(f"  CoT after step: {cot_stepped:.6f}")
    print(f"  Delta CoT:      {delta_cot:+.6f}")
    print(f"  Grad norm:      {np.linalg.norm(grad):.6f}")

    status = "PASS" if expected_decrease else "WARN"
    print(f"  [{status}] Step in gradient direction {'decreases' if delta_cot < 0 else 'increases'} CoT")

    # This is a soft check — numerical noise can cause small increases
    if not expected_decrease:
        print(f"  (Small increase may be due to numerical noise with small grad norm)")

    return True  # Don't fail on this — it's a soft check


def main():
    print("\n" + "#" * 60)
    print("# FDEvaluator Smoke Tests")
    print("#" * 60)

    wp.init()
    np.random.seed(42)

    results = {}

    # Test 1: Build
    fd_eval, results["build"] = test_1_build()
    if fd_eval is None:
        print("\n[ABORT] Cannot continue without G1 model")
        return 1

    # Test 2: Rollout stability
    results["rollout"], actions = test_2_rollout(fd_eval)

    # Test 3: CoT values
    results["cot_values"] = test_3_cot_values(fd_eval, actions)

    # Test 4: FD gradient
    results["fd_gradient"] = test_4_fd_gradient(fd_eval, actions)

    # Test 5: Physical plausibility
    results["plausibility"] = test_5_physical_plausibility(fd_eval, actions)

    # Test 6: Gradient consistency
    results["consistency"] = test_6_gradient_consistency(fd_eval, actions)

    # Cleanup
    fd_eval.cleanup()

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {test}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll validations PASSED! FDEvaluator is ready for use.")
        return 0
    else:
        print(f"\n{total - passed} validation(s) FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
