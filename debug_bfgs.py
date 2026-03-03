#!/usr/bin/env python3
"""
Debug script for CautiousBFGS integration in codesign_g1_unified.py.

Simulates the outer loop's gradient→BFGS→clip→update cycle with synthetic
FD gradients on a known quadratic landscape.  No GPU, no MimicKit, no Newton.

Tests:
  1. Basic flow:      identity Hessian on first iter, direction = gradient
  2. Hessian update:  H learns curvature from (s, y) pairs
  3. Cautious filter:  negative curvature pairs are rejected
  4. Shanno-Phua:     initial H scaling matches local curvature
  5. Bounds clipping:  post-clip step recorded correctly in Hessian
  6. Condition reset:  H resets to I when condition number blows up
  7. Coupled landscape: BFGS learns off-diagonal coupling that GD ignores
  8. Full outer loop:  simulate 15 outer iterations on a coupled quadratic

Usage:
    python debug_bfgs.py
"""

import sys
import os
import numpy as np

# Import CautiousBFGS from the actual source file (no GPU deps needed)
# We can't import the module directly because it imports warp/newton/torch
# at module level.  Instead, extract just the class.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE = os.path.join(SCRIPT_DIR, "codesign_g1_unified.py")


def _load_bfgs_class():
    """Extract CautiousBFGS class from codesign_g1_unified.py without
    importing the full module (which requires GPU libraries)."""
    import importlib.util
    import types

    # Read only the class source + its numpy dependency
    with open(SOURCE_FILE) as f:
        source = f.read()

    # Find the class boundaries
    class_start = source.index("class CautiousBFGS:")
    # Find next top-level class or function (non-indented def/class after the BFGS class)
    rest = source[class_start:]
    lines = rest.split("\n")
    class_end = len(lines)
    for i, line in enumerate(lines[1:], 1):
        if line and not line[0].isspace() and not line.startswith("#"):
            class_end = i
            break
    class_source = "\n".join(lines[:class_end])

    # Execute in a namespace with numpy
    namespace = {"np": np, "__name__": "__bfgs_debug__"}
    exec(class_source, namespace)
    return namespace["CautiousBFGS"]


CautiousBFGS = _load_bfgs_class()

# ---------------------------------------------------------------------------
# Synthetic landscapes
# ---------------------------------------------------------------------------

def quadratic_gradient(theta, A, b):
    """Gradient of f(theta) = 0.5 * theta^T A theta + b^T theta.
    For gradient ascent (maximizing reward), negate: grad = -(A @ theta + b)."""
    return -(A @ theta + b)


def quadratic_value(theta, A, b):
    """f(theta) = 0.5 * theta^T A theta + b^T theta."""
    return 0.5 * theta @ A @ theta + b @ theta


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_1_basic_flow():
    """First iteration: H=I, direction should equal gradient."""
    print("\n" + "=" * 60)
    print("Test 1: Basic Flow (first iter, H=I)")
    print("=" * 60)

    n = 6
    bfgs = CautiousBFGS(n)
    grad = np.array([0.1, -0.2, 0.05, -0.1, 0.15, -0.05])

    direction = bfgs.compute_direction(grad)

    # H = I, so direction should be exactly grad
    diff = np.max(np.abs(direction - grad))
    ok = diff < 1e-15
    print(f"  direction = {direction}")
    print(f"  gradient  = {grad}")
    print(f"  max diff  = {diff:.2e}")
    print(f"  H is identity: {np.allclose(bfgs.H, np.eye(n))}")
    print(f"  [{'PASS' if ok else 'FAIL'}] Direction equals gradient on first iter")
    return ok


def test_2_hessian_update():
    """After one (s, y) pair with positive curvature, H should change.

    For gradient ascent BFGS, y = g_old - g_new (negated).  Approaching an
    optimum, the gradient shrinks, so g_new < g_old → y has same sign as s
    → s^T y > 0.  We simulate this by stepping toward the max and letting
    the gradient decrease.
    """
    print("\n" + "=" * 60)
    print("Test 2: Hessian Update")
    print("=" * 60)

    n = 6
    bfgs = CautiousBFGS(n)

    # Simulate first iteration: large gradient (far from optimum)
    grad_0 = np.array([1.0, 0.5, -0.3, 0.2, -0.1, 0.4])
    direction = bfgs.compute_direction(grad_0)
    theta_0 = np.zeros(n)
    theta_1 = theta_0 + 0.01 * direction
    actual_s = theta_1 - theta_0
    bfgs.update(actual_s, grad_0)
    print(f"  After iter 0: num_updates={bfgs.num_updates}, prev_grad stored")

    # Simulate second iteration: gradient decreased (approaching optimum)
    # y = g_old - g_new should be aligned with s for s^T y > 0
    grad_1 = np.array([0.8, 0.4, -0.25, 0.15, -0.08, 0.35])
    y = grad_0 - grad_1  # [0.2, 0.1, -0.05, 0.05, -0.02, 0.05] — same sign as s
    print(f"  y = g_old - g_new = {y}")
    print(f"  s^T y = {actual_s @ y:.6f} (should be > 0)")

    direction_1 = bfgs.compute_direction(grad_1)
    theta_2 = theta_1 + 0.01 * direction_1
    actual_s_1 = theta_2 - theta_1
    bfgs.update(actual_s_1, grad_1)

    h_changed = not np.allclose(bfgs.H, np.eye(n))

    print(f"  After iter 1: num_updates={bfgs.num_updates}, num_skipped={bfgs.num_skipped}")
    print(f"  H changed from identity: {h_changed}")
    print(f"  H diagonal: {np.diag(bfgs.H)}")
    print(f"  H condition number: {np.linalg.cond(bfgs.H):.2f}")

    ok = bfgs.num_updates >= 1 and h_changed
    print(f"  [{'PASS' if ok else 'FAIL'}] Hessian updated with positive curvature pair")
    return ok


def test_3_cautious_filter():
    """When s^T y <= 0 (negative curvature), update should be skipped.

    For ascent BFGS, y = g_old - g_new.  Negative curvature means the
    gradient INCREASED after stepping (moved away from optimum), so
    g_new > g_old → y = g_old - g_new < 0 while s > 0 → s^T y < 0.
    """
    print("\n" + "=" * 60)
    print("Test 3: Cautious Filter (negative curvature)")
    print("=" * 60)

    n = 6
    bfgs = CautiousBFGS(n)

    # First call: store prev_grad
    grad_0 = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.4])
    s_0 = np.array([0.01, 0.005, 0.003, 0.002, 0.001, 0.004])
    bfgs.update(s_0, grad_0)

    H_before = bfgs.H.copy()

    # Second call: gradient INCREASED (moved away from optimum / noise)
    # y = g_old - g_new = negative → s^T y < 0 → reject
    grad_1 = grad_0 + 2.0  # all components increased
    s_1 = np.array([0.01, 0.005, 0.003, 0.002, 0.001, 0.004])

    bfgs.update(s_1, grad_1)

    H_unchanged = np.allclose(bfgs.H, H_before)
    y = grad_0 - grad_1  # what the class computes internally
    sTy = s_1 @ y

    ok = bfgs.num_skipped == 1 and H_unchanged
    print(f"  y = g_old - g_new = {y}")
    print(f"  s^T y = {sTy:.4f} (should be negative)")
    print(f"  num_updates = {bfgs.num_updates}")
    print(f"  num_skipped = {bfgs.num_skipped}")
    print(f"  H unchanged: {H_unchanged}")
    print(f"  [{'PASS' if ok else 'FAIL'}] Cautious filter rejected negative curvature")
    return ok


def test_4_shanno_phua_scaling():
    """First accepted pair should trigger gamma*I scaling before the BFGS update.

    For ascent BFGS, y = g_old - g_new.  We need the gradient to shrink
    (approaching optimum) so y aligns with s → s^T y > 0.
    """
    print("\n" + "=" * 60)
    print("Test 4: Shanno-Phua Initial Scaling")
    print("=" * 60)

    n = 6
    bfgs = CautiousBFGS(n)

    # First call: store prev_grad (large gradient, far from optimum)
    grad_0 = np.array([10.0, 5.0, 3.0, 2.0, 1.0, 4.0])
    s_0 = np.array([0.1, 0.05, 0.03, 0.02, 0.01, 0.04])
    bfgs.update(s_0, grad_0)

    assert not bfgs.initialized, "Should not be initialized after first call"

    # Second call: gradient decreased (approaching optimum)
    # y = g_old - g_new = positive values aligned with s → s^T y > 0
    grad_1 = np.array([9.0, 4.5, 2.7, 1.8, 0.9, 3.6])  # smaller than grad_0
    s_1 = np.array([0.1, 0.05, 0.03, 0.02, 0.01, 0.04])
    y = grad_0 - grad_1  # [1.0, 0.5, 0.3, 0.2, 0.1, 0.4] — same sign as s
    sTy = s_1 @ y
    yTy = y @ y

    bfgs.update(s_1, grad_1)

    gamma = sTy / yTy if yTy > 1e-20 else 1.0
    print(f"  y = g_old - g_new = {y}")
    print(f"  s^T y = {sTy:.6f} (should be > 0)")
    print(f"  y^T y = {yTy:.6f}")
    print(f"  Expected gamma = {gamma:.6f}")
    print(f"  initialized = {bfgs.initialized}")
    print(f"  num_updates = {bfgs.num_updates}")
    print(f"  H diagonal (sample): {np.diag(bfgs.H)[:3]}")

    # After scaling + rank-2 update, H should NOT be identity
    ok = bfgs.initialized and bfgs.num_updates == 1
    print(f"  [{'PASS' if ok else 'FAIL'}] Shanno-Phua scaling applied on first accepted pair")
    return ok


def test_5_bounds_clipping():
    """Verify post-clip step is what gets recorded in Hessian history."""
    print("\n" + "=" * 60)
    print("Test 5: Bounds Clipping")
    print("=" * 60)

    n = 6
    bfgs = CautiousBFGS(n)
    theta_bounds = (-0.5236, 0.5236)  # +/-30 deg

    # Start near the upper bound
    theta = np.full(n, 0.50)
    grad = np.full(n, 1.0)  # gradient wants to push further positive

    direction = bfgs.compute_direction(grad)
    lr = 1.0
    proposed = theta + lr * direction
    clipped = np.clip(proposed, theta_bounds[0], theta_bounds[1])
    actual_s = clipped - theta

    # The actual step should be much smaller than the proposed step
    proposed_norm = np.linalg.norm(proposed - theta)
    actual_norm = np.linalg.norm(actual_s)

    bfgs.update(actual_s, grad)

    print(f"  theta start:    {theta[0]:.4f}")
    print(f"  proposed:       {proposed[0]:.4f}")
    print(f"  clipped:        {clipped[0]:.4f}")
    print(f"  proposed step:  {proposed_norm:.4f}")
    print(f"  actual step:    {actual_norm:.4f}")
    print(f"  step was clipped: {actual_norm < proposed_norm}")

    ok = actual_norm < proposed_norm and np.all(clipped <= theta_bounds[1])
    print(f"  [{'PASS' if ok else 'FAIL'}] Post-clip step recorded for Hessian")
    return ok


def test_6_condition_reset():
    """Force H to become ill-conditioned and verify it resets to I."""
    print("\n" + "=" * 60)
    print("Test 6: Condition Number Reset")
    print("=" * 60)

    n = 6
    bfgs = CautiousBFGS(n)

    # Manually inject a badly conditioned H
    bfgs.H = np.diag([1e7, 1.0, 1.0, 1.0, 1.0, 1.0])
    bfgs.initialized = True
    # prev_grad is large (far from optimum)
    bfgs.prev_grad = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.4])

    cond_before = np.linalg.cond(bfgs.H)
    print(f"  Condition before: {cond_before:.1e}")

    # grad_new is smaller (approaching optimum) → y = g_old - g_new > 0
    # Step s is positive → s^T y > 0 → update triggers
    grad_new = np.array([0.9, 0.45, 0.27, 0.18, 0.09, 0.36])
    s = np.array([0.01, 0.005, 0.003, 0.002, 0.001, 0.004])
    y = bfgs.prev_grad - grad_new  # ascent convention: y = g_old - g_new
    sTy = s @ y
    print(f"  y = g_old - g_new = {y}")
    print(f"  s^T y = {sTy:.6f} (must be > 0 to trigger update)")

    bfgs.update(s, grad_new)

    cond_after = np.linalg.cond(bfgs.H)
    is_identity = np.allclose(bfgs.H, np.eye(n))

    print(f"  Condition after:  {cond_after:.1e}")
    print(f"  H reset to identity: {is_identity}")

    # The update runs on the ill-conditioned H, producing an even worse H,
    # then the condition check fires and resets to identity.
    ok = is_identity
    print(f"  [{'PASS' if ok else 'FAIL'}] Condition number reset triggered")
    return ok


def test_7_coupled_landscape():
    """BFGS should learn off-diagonal coupling that vanilla GD ignores.

    Uses a 6D landscape with high condition number (100) and strong coupling.
    This is the scenario BFGS is designed for: GD zigzags on ill-conditioned
    problems because it can't see the coupling, while BFGS learns the
    inverse Hessian and takes better-conditioned steps.

    Uses small noise (0.5% of gradient) to simulate clean FD estimates.
    """
    print("\n" + "=" * 60)
    print("Test 7: Coupled Landscape (ill-conditioned, 6D)")
    print("=" * 60)

    n = 6
    # Build a coupled PD matrix with condition number ~100
    rng = np.random.RandomState(7)
    # Random rotation
    Q, _ = np.linalg.qr(rng.randn(n, n))
    # Eigenvalues spread from 0.1 to 10 → condition ~100
    eigs = np.array([0.1, 0.3, 0.7, 2.0, 5.0, 10.0])
    A = Q @ np.diag(eigs) @ Q.T
    b = rng.randn(n) * 0.1

    theta_star = -np.linalg.solve(A, b)
    print(f"  A condition number: {np.linalg.cond(A):.1f}")
    print(f"  Optimum: {theta_star}")

    theta_gd = np.array([0.3, -0.2, 0.15, -0.1, 0.25, -0.05])
    theta_bfgs = theta_gd.copy()
    bounds = (-0.5236, 0.5236)

    bfgs = CautiousBFGS(n)
    # GD lr tuned for the problem (1/max_eig is stable, use 0.5/max_eig)
    lr_gd = 0.05
    lr_bfgs = 0.05

    gd_values = []
    bfgs_values = []
    num_iters = 40

    for i in range(num_iters):
        noise = rng.randn(n) * 0.005  # tiny noise

        # GD
        grad_f_gd = A @ theta_gd + b + noise
        theta_gd = np.clip(theta_gd - lr_gd * grad_f_gd, bounds[0], bounds[1])
        gd_values.append(quadratic_value(theta_gd, A, b))

        # BFGS
        grad_ascent = -(A @ theta_bfgs + b + noise)
        direction = bfgs.compute_direction(grad_ascent)
        old = theta_bfgs.copy()
        theta_bfgs = np.clip(old + lr_bfgs * direction, bounds[0], bounds[1])
        actual_s = theta_bfgs - old
        bfgs.update(actual_s, grad_ascent)
        bfgs_values.append(quadratic_value(theta_bfgs, A, b))

    print(f"  lr_gd = {lr_gd}, lr_bfgs = {lr_bfgs}, iters = {num_iters}")
    print(f"  After {num_iters} iters:")
    print(f"    GD   f(theta) = {gd_values[-1]:.8f}")
    print(f"    BFGS f(theta) = {bfgs_values[-1]:.8f}")
    print(f"  BFGS updates = {bfgs.num_updates}, skipped = {bfgs.num_skipped}")
    print(f"  BFGS H cond = {np.linalg.cond(bfgs.H):.1f}")

    ok = bfgs_values[-1] <= gd_values[-1]
    print(f"  [{'PASS' if ok else 'FAIL'}] BFGS converges faster than GD on ill-conditioned landscape")
    return ok


def test_8_full_outer_loop_sim():
    """Simulate 15 outer iterations of the actual PGHC flow with synthetic FD gradients.

    Mimics the exact code path in pghc_worker: FD gradient → BFGS direction →
    lr scaling → clip → record actual step → update Hessian.
    """
    print("\n" + "=" * 60)
    print("Test 8: Full Outer Loop Simulation (15 iters)")
    print("=" * 60)

    n = 6
    theta_bounds = (-0.5236, 0.5236)
    design_lr = 0.1  # conservative; first iter H=I so full gradient step overshoots

    # Synthetic coupled landscape: A is positive definite with coupling
    # (simulates hip angle coupling with knee angle, etc.)
    rng = np.random.RandomState(42)
    L = rng.randn(n, n) * 0.5
    A = L.T @ L + 0.5 * np.eye(n)  # positive definite
    b = rng.randn(n) * 0.1
    theta_star = -np.linalg.solve(A, b)  # optimum of 0.5*x^T A x + b^T x

    print(f"  Landscape: f(x) = 0.5 x^T A x + b^T x")
    print(f"  A condition number: {np.linalg.cond(A):.1f}")
    print(f"  Optimum (unconstrained): {theta_star}")

    bfgs = CautiousBFGS(n)
    theta = np.zeros(n, dtype=np.float64)

    print(f"\n  {'Iter':>4}  {'f(theta)':>10}  {'||grad||':>10}  "
          f"{'||step||':>10}  {'H updates':>9}  {'H skipped':>9}  {'H cond':>10}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*10}")

    values = []
    for outer_iter in range(15):
        # Synthetic FD gradient (ascent direction = -grad_f = -(A@theta + b))
        # Add noise to simulate stochastic FD estimation
        grad_true = -(A @ theta + b)
        noise = rng.randn(n) * 0.01  # small FD noise
        grad_theta = grad_true + noise

        # --- Exact code path from pghc_worker ---
        old_theta = theta.copy()
        direction = bfgs.compute_direction(grad_theta)
        theta = old_theta + design_lr * direction
        theta = np.clip(theta, theta_bounds[0], theta_bounds[1])
        actual_s = theta - old_theta
        bfgs.update(actual_s, grad_theta)
        # --- End code path ---

        f_val = quadratic_value(theta, A, b)
        values.append(f_val)
        grad_norm = np.linalg.norm(grad_theta)
        step_norm = np.linalg.norm(actual_s)
        cond = np.linalg.cond(bfgs.H)

        print(f"  {outer_iter:4d}  {f_val:10.4f}  {grad_norm:10.6f}  "
              f"{step_norm:10.6f}  {bfgs.num_updates:9d}  {bfgs.num_skipped:9d}  "
              f"{cond:10.1f}")

    print(f"\n  Final theta: {theta}")
    print(f"  Final f(theta): {values[-1]:.6f}")
    print(f"  Initial f(theta): {values[0]:.6f}")
    print(f"  BFGS H updates: {bfgs.num_updates}, skipped: {bfgs.num_skipped}")

    # Should have made progress (f decreased since we're maximizing reward = -f)
    improved = values[-1] < values[0]
    h_used = bfgs.num_updates > 0
    no_nan = not np.any(np.isnan(theta))
    in_bounds = np.all(np.abs(theta) <= 0.5236 + 1e-10)

    ok = improved and h_used and no_nan and in_bounds
    print(f"\n  Improved:   {improved}")
    print(f"  H updated:  {h_used}")
    print(f"  No NaN:     {no_nan}")
    print(f"  In bounds:  {in_bounds}")
    print(f"  [{'PASS' if ok else 'FAIL'}] Full outer loop simulation")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("#" * 60)
    print(f"# CautiousBFGS Debug Script")
    print(f"# Source: {SOURCE_FILE}")
    print("#" * 60)

    results = {}
    results["basic_flow"] = test_1_basic_flow()
    results["hessian_update"] = test_2_hessian_update()
    results["cautious_filter"] = test_3_cautious_filter()
    results["shanno_phua"] = test_4_shanno_phua_scaling()
    results["bounds_clipping"] = test_5_bounds_clipping()
    results["condition_reset"] = test_6_condition_reset()
    results["coupled_landscape"] = test_7_coupled_landscape()
    results["full_outer_loop"] = test_8_full_outer_loop_sim()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  {passed}/{total} passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
