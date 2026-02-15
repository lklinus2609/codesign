"""Comprehensive diagnostics for SolverSemiImplicitStable.

Tests stability, identifies the ping-pong vs separate-states discrepancy,
validates gradient quality, and checks morphology parameter gradients.

Usage:
    python -u codesign/test_solver_semi_implicit.py
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

import warp as wp
import newton

MJCF_PATH = SCRIPT_DIR.parent / "MimicKit" / "data" / "assets" / "g1" / "g1.xml"

NUM_WORLDS = 1
DT = 1.0 / 30.0
NUM_SUBSTEPS = 8
SUB_DT = DT / NUM_SUBSTEPS
Z_HEIGHT = 0.8

# ===================================================================
# Helpers
# ===================================================================

def build_model(device="cuda:0", requires_grad=False):
    single = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(single)
    single.add_mjcf(
        str(MJCF_PATH),
        floating=True,
        ignore_inertial_definitions=False,
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        convert_3d_hinge_to_ball_joints=False,
        ignore_names=["floor", "ground"],
    )
    ground_cfg = single.ShapeConfig(mu=1.0, restitution=0)
    single.add_ground_plane(cfg=ground_cfg)

    builder = newton.ModelBuilder()
    builder.replicate(single, NUM_WORLDS, spacing=(5.0, 5.0, 0.0))
    model = builder.finalize(requires_grad=requires_grad)

    if hasattr(model, 'mujoco'):
        pd_ke = model.mujoco.dof_passive_stiffness.numpy().copy()
        pd_kd = model.mujoco.dof_passive_damping.numpy().copy()
        model.joint_target_ke.assign(pd_ke)
        model.joint_target_kd.assign(pd_kd)
        model.mujoco.dof_passive_stiffness.assign(np.zeros_like(pd_ke))
        model.mujoco.dof_passive_damping.assign(np.zeros_like(pd_kd))

    return model


def init_state(model, state, device="cuda:0"):
    """Set initial pose: standing at Z_HEIGHT with identity orientation."""
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    wp.synchronize()
    qpos = state.joint_q.numpy().copy()
    q_size = len(qpos) // NUM_WORLDS
    for w in range(NUM_WORLDS):
        s = w * q_size
        qpos[s + 2] = Z_HEIGHT
        qpos[s + 3] = 0.0; qpos[s + 4] = 0.0; qpos[s + 5] = 0.0; qpos[s + 6] = 1.0
    state.joint_q.assign(qpos)
    state.joint_qd.zero_()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    wp.synchronize()
    return qpos


# ===================================================================
# TEST A: Discrepancy investigation
# Run ke=10000/kd=100/8sub with DETAILED per-step state printout
# to understand exactly what triggers FAIL@step5 in ping-pong mode.
# ===================================================================

def test_a_discrepancy(device):
    print("\n" + "=" * 80)
    print("TEST A: Discrepancy investigation (ping-pong vs separate states)")
    print("        ke=10000, kd=100, 8 substeps, 30 steps")
    print("=" * 80)

    ke, kd = 1e4, 100
    num_steps = 30

    # --- Variant 1: Ping-pong (run_test style) ---
    print("\n  --- A1: Ping-pong buffers, requires_grad=False ---")
    m = build_model(device, requires_grad=False)
    solver = newton.solvers.SolverSemiImplicitStable(m, joint_attach_ke=ke, joint_attach_kd=kd)
    control = m.control()

    s0 = m.state(requires_grad=False)
    s1 = m.state(requires_grad=False)
    init_state(m, s0, device)

    src, dst = s0, s1
    print(f"  {'step':>4s}  {'root_z':>10s}  {'max|jq|':>10s}  {'max|jqd|':>10s}  "
          f"{'max|bqd|':>10s}  {'nan_jq':>6s}  {'nan_jqd':>7s}  status")

    for step in range(num_steps):
        src.clear_forces()
        contacts = m.collide(src)

        for sub in range(NUM_SUBSTEPS):
            src.clear_forces()
            solver.step(src, dst, control, contacts, SUB_DT)
            newton.eval_ik(m, dst, dst.joint_q, dst.joint_qd)
            src, dst = dst, src

        wp.synchronize()
        jq = src.joint_q.numpy()
        jqd = src.joint_qd.numpy()
        bqd = src.body_qd.numpy()
        root_z = jq[2] if not np.isnan(jq[2]) else float('nan')
        nan_jq = np.isnan(jq).any()
        nan_jqd = np.isnan(jqd).any()
        max_jq = np.nanmax(np.abs(jq))
        max_jqd = np.nanmax(np.abs(jqd))
        max_bqd = np.nanmax(np.abs(bqd))

        # Status using run_test criteria
        if nan_jq or nan_jqd or abs(root_z) > 100:
            status = "FAIL"
        else:
            status = "ok"

        print(f"  {step:4d}  {root_z:10.4f}  {max_jq:10.4f}  {max_jqd:10.4f}  "
              f"{max_bqd:10.4f}  {str(nan_jq):>6s}  {str(nan_jqd):>7s}  {status}")

        if nan_jq or nan_jqd:
            # Print which bodies have NaN
            bodies_per_world = m.body_count // NUM_WORLDS
            for i in range(bodies_per_world):
                bqd_i = bqd[i]
                if np.isnan(bqd_i).any():
                    name = m.body_key[i] if i < len(m.body_key) else f"body_{i}"
                    print(f"         NaN body {i} ({name}): {bqd_i}")
            break

        if abs(root_z) > 100:
            print(f"         root_z diverged to {root_z}")
            break

    del solver, m

    # --- Variant 2: Separate states (Test 5 style), requires_grad=False ---
    print("\n  --- A2: Separate states, requires_grad=False ---")
    m = build_model(device, requires_grad=False)
    solver = newton.solvers.SolverSemiImplicitStable(m, joint_attach_ke=ke, joint_attach_kd=kd)
    control = m.control()

    total_phys = num_steps * NUM_SUBSTEPS
    states = [m.state(requires_grad=False) for _ in range(total_phys + 1)]
    init_state(m, states[0], device)

    print(f"  {'step':>4s}  {'root_z':>10s}  {'max|jq|':>10s}  {'max|jqd|':>10s}  "
          f"{'max|bqd|':>10s}  {'nan_jq':>6s}  {'nan_jqd':>7s}  status")

    phys_step = 0
    for step in range(num_steps):
        states[phys_step].clear_forces()
        contacts = m.collide(states[phys_step])

        for sub in range(NUM_SUBSTEPS):
            s_in = states[phys_step]
            s_out = states[phys_step + 1]
            s_in.clear_forces()
            solver.step(s_in, s_out, control, contacts, SUB_DT)
            newton.eval_ik(m, s_out, s_out.joint_q, s_out.joint_qd)
            phys_step += 1

        wp.synchronize()
        jq = states[phys_step].joint_q.numpy()
        jqd = states[phys_step].joint_qd.numpy()
        bqd = states[phys_step].body_qd.numpy()
        root_z = jq[2] if not np.isnan(jq[2]) else float('nan')
        nan_jq = np.isnan(jq).any()
        nan_jqd = np.isnan(jqd).any()
        max_jq = np.nanmax(np.abs(jq))
        max_jqd = np.nanmax(np.abs(jqd))
        max_bqd = np.nanmax(np.abs(bqd))

        if nan_jq or nan_jqd or abs(root_z) > 100:
            status = "FAIL"
        else:
            status = "ok"

        print(f"  {step:4d}  {root_z:10.4f}  {max_jq:10.4f}  {max_jqd:10.4f}  "
              f"{max_bqd:10.4f}  {str(nan_jq):>6s}  {str(nan_jqd):>7s}  {status}")

        if nan_jq or nan_jqd:
            bodies_per_world = m.body_count // NUM_WORLDS
            for i in range(bodies_per_world):
                bqd_i = bqd[i]
                if np.isnan(bqd_i).any():
                    name = m.body_key[i] if i < len(m.body_key) else f"body_{i}"
                    print(f"         NaN body {i} ({name}): {bqd_i}")
            break

        if abs(root_z) > 100:
            print(f"         root_z diverged to {root_z}")
            break

    del states, solver, m

    # --- Variant 3: Ping-pong, requires_grad=True ---
    print("\n  --- A3: Ping-pong buffers, requires_grad=True ---")
    m = build_model(device, requires_grad=True)
    solver = newton.solvers.SolverSemiImplicitStable(m, joint_attach_ke=ke, joint_attach_kd=kd)
    control = m.control()

    s0 = m.state(requires_grad=True)
    s1 = m.state(requires_grad=True)
    init_state(m, s0, device)

    src, dst = s0, s1
    print(f"  {'step':>4s}  {'root_z':>10s}  {'max|jq|':>10s}  {'max|jqd|':>10s}  "
          f"{'max|bqd|':>10s}  {'nan_jq':>6s}  {'nan_jqd':>7s}  status")

    for step in range(num_steps):
        src.clear_forces()
        contacts = m.collide(src)

        for sub in range(NUM_SUBSTEPS):
            src.clear_forces()
            solver.step(src, dst, control, contacts, SUB_DT)
            newton.eval_ik(m, dst, dst.joint_q, dst.joint_qd)
            src, dst = dst, src

        wp.synchronize()
        jq = src.joint_q.numpy()
        jqd = src.joint_qd.numpy()
        bqd = src.body_qd.numpy()
        root_z = jq[2] if not np.isnan(jq[2]) else float('nan')
        nan_jq = np.isnan(jq).any()
        nan_jqd = np.isnan(jqd).any()
        max_jq = np.nanmax(np.abs(jq))
        max_jqd = np.nanmax(np.abs(jqd))
        max_bqd = np.nanmax(np.abs(bqd))

        if nan_jq or nan_jqd or abs(root_z) > 100:
            status = "FAIL"
        else:
            status = "ok"

        print(f"  {step:4d}  {root_z:10.4f}  {max_jq:10.4f}  {max_jqd:10.4f}  "
              f"{max_bqd:10.4f}  {str(nan_jq):>6s}  {str(nan_jqd):>7s}  {status}")

        if nan_jq or nan_jqd:
            bodies_per_world = m.body_count // NUM_WORLDS
            for i in range(bodies_per_world):
                bqd_i = bqd[i]
                if np.isnan(bqd_i).any():
                    name = m.body_key[i] if i < len(m.body_key) else f"body_{i}"
                    print(f"         NaN body {i} ({name}): {bqd_i}")
            break

        if abs(root_z) > 100:
            print(f"         root_z diverged to {root_z}")
            break

    del solver, m
    wp.synchronize()


# ===================================================================
# TEST B: Best config long-run validation
# Configs that passed Test 4b (100 steps): test them for 300 steps
# with per-step diagnostics to verify true stability.
# ===================================================================

def test_b_long_run(device):
    print("\n" + "=" * 80)
    print("TEST B: Long-run validation (300 steps = 10 seconds)")
    print("=" * 80)

    configs = [
        (1000, 100, 16),
        (1000, 10,  16),
        (10000, 100, 32),
        (10000, 10,  32),
    ]

    for ke, kd, n_sub in configs:
        sub_dt = DT / n_sub
        m = build_model(device, requires_grad=False)
        solver = newton.solvers.SolverSemiImplicitStable(
            m, joint_attach_ke=ke, joint_attach_kd=kd
        )
        control = m.control()
        s0 = m.state(requires_grad=False)
        s1 = m.state(requires_grad=False)
        init_state(m, s0, device)

        src, dst = s0, s1
        failed_step = None
        last_z = Z_HEIGHT
        max_bqd_seen = 0.0

        for step in range(300):
            src.clear_forces()
            contacts = m.collide(src)
            for sub in range(n_sub):
                src.clear_forces()
                solver.step(src, dst, control, contacts, sub_dt)
                newton.eval_ik(m, dst, dst.joint_q, dst.joint_qd)
                src, dst = dst, src

            wp.synchronize()
            jq = src.joint_q.numpy()
            jqd = src.joint_qd.numpy()
            bqd = src.body_qd.numpy()
            root_z = jq[2] if not np.isnan(jq[2]) else float('nan')
            max_bqd_now = np.nanmax(np.abs(bqd))
            max_bqd_seen = max(max_bqd_seen, max_bqd_now)
            last_z = root_z

            if np.isnan(jq).any() or np.isnan(jqd).any() or abs(root_z) > 100:
                failed_step = step
                break

        status = "PASS" if failed_step is None else f"FAIL@step{failed_step}"
        print(f"  ke={ke:5.0f} kd={kd:3d} sub={n_sub:2d}  "
              f"{status:12s}  z={last_z:.4f}  max|bqd|={max_bqd_seen:.4f}")

        del solver, m
    wp.synchronize()


# ===================================================================
# TEST C: Gradient quality check
# Run BPTT with a physically meaningful loss (root_z), verify
# gradients are non-zero, finite, and sensible.
# ===================================================================

@wp.kernel
def extract_root_z(joint_q: wp.array(dtype=float), out: wp.array(dtype=float)):
    """Extract root z position (index 2) as a differentiable scalar."""
    out[0] = joint_q[2]


def test_c_gradient_quality(device):
    print("\n" + "=" * 80)
    print("TEST C: Gradient quality (root_z loss, BPTT)")
    print("=" * 80)

    ke, kd = 1e4, 100
    num_tape_steps = 5  # Safe horizon (free-fall only)

    m = build_model(device, requires_grad=True)
    solver = newton.solvers.SolverSemiImplicitStable(m, joint_attach_ke=ke, joint_attach_kd=kd)
    control = m.control()

    total_phys = num_tape_steps * NUM_SUBSTEPS
    states = [m.state(requires_grad=True) for _ in range(total_phys + 1)]

    # Set initial joint_q OUTSIDE tape (this is the "input parameter")
    s_tmp = m.state(requires_grad=True)
    qpos = init_state(m, s_tmp, device)
    init_qpos = wp.array(qpos, dtype=float, device=device, requires_grad=True)
    wp.copy(states[0].joint_q, init_qpos)
    states[0].joint_qd.zero_()
    wp.synchronize()

    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        # eval_fk INSIDE tape so joint_q → body_q connection is recorded
        newton.eval_fk(m, states[0].joint_q, states[0].joint_qd, states[0])

        phys_step = 0
        for step in range(num_tape_steps):
            states[phys_step].clear_forces()
            contacts = m.collide(states[phys_step])

            for sub in range(NUM_SUBSTEPS):
                s_in = states[phys_step]
                s_out = states[phys_step + 1]
                s_in.clear_forces()
                solver.step(s_in, s_out, control, contacts, SUB_DT)
                newton.eval_ik(m, s_out, s_out.joint_q, s_out.joint_qd)
                phys_step += 1

        # Differentiable loss: final root_z
        wp.launch(extract_root_z, dim=1,
                  inputs=[states[phys_step].joint_q],
                  outputs=[loss], device=device)

    wp.synchronize()
    final_jq = states[phys_step].joint_q.numpy()
    loss_val = loss.numpy()[0]
    print(f"\n  Forward pass: {num_tape_steps} steps ({phys_step} physics steps)")
    print(f"  Loss (root_z): {loss_val:.6f}")
    print(f"  Final root_z:  {final_jq[2]:.6f}")

    # Backward
    tape.backward(loss)
    wp.synchronize()

    # Check gradients on initial joint_q
    grad_init_jq = tape.gradients.get(states[0].joint_q)
    if grad_init_jq is not None:
        g = grad_init_jq.numpy()
        print(f"\n  Gradient of root_z w.r.t. initial joint_q:")
        print(f"    shape: {g.shape}")
        print(f"    non-zero entries: {np.count_nonzero(g)} / {g.size}")
        print(f"    has NaN: {np.isnan(g).any()}")
        print(f"    has Inf: {np.isinf(g).any()}")
        print(f"    max |grad|: {np.nanmax(np.abs(g)):.6e}")
        print(f"    min |grad| (nonzero): {np.min(np.abs(g[g != 0])):.6e}" if np.any(g != 0) else "    (all zero)")

        q_size = len(g) // NUM_WORLDS
        labels = ["root_x", "root_y", "root_z",
                   "root_qx", "root_qy", "root_qz", "root_qw"]
        print(f"\n    Per-DOF gradients (first world):")
        for i in range(min(q_size, 40)):
            lbl = labels[i] if i < len(labels) else f"dof_{i}"
            if g[i] != 0.0:
                print(f"      [{i:3d}] {lbl:20s}: {g[i]:+.6e}")
    else:
        print("\n  WARNING: No gradient found for states[0].joint_q")
        # Fallback: check body_q gradient (gradient might stop at body level)
        grad_bq0 = tape.gradients.get(states[0].body_q)
        if grad_bq0 is not None:
            gb = grad_bq0.numpy()
            print(f"    BUT gradient found on states[0].body_q: {np.count_nonzero(gb)}/{gb.size} non-zero")
            print(f"    → eval_fk adjoint may not connect body_q → joint_q")
        else:
            print(f"    No gradient on states[0].body_q either — chain fully broken")

    # Check gradients on final state
    grad_final_jq = tape.gradients.get(states[phys_step].joint_q)
    if grad_final_jq is not None:
        gf = grad_final_jq.numpy()
        print(f"\n  Gradient of root_z w.r.t. final joint_q:")
        print(f"    non-zero entries: {np.count_nonzero(gf)} / {gf.size}")
        print(f"    grad[2] (d(root_z)/d(final_root_z)): {gf[2]:.6e}  (should be ~1.0)")
    else:
        print("\n  WARNING: No gradient found for final state joint_q")

    # Check intermediate body_q gradients to find where chain breaks
    print(f"\n  Gradient chain diagnosis (body_q at each macro step):")
    for s in range(min(num_tape_steps + 1, 6)):
        idx = s * NUM_SUBSTEPS
        gbq = tape.gradients.get(states[idx].body_q)
        gbqd = tape.gradients.get(states[idx].body_qd)
        gjq = tape.gradients.get(states[idx].joint_q)
        bq_nz = np.count_nonzero(gbq.numpy()) if gbq is not None else -1
        bqd_nz = np.count_nonzero(gbqd.numpy()) if gbqd is not None else -1
        jq_nz = np.count_nonzero(gjq.numpy()) if gjq is not None else -1
        print(f"    states[{idx:3d}]: body_q={bq_nz:4d}  body_qd={bqd_nz:4d}  joint_q={jq_nz:4d}  "
              f"(-1 = not tracked)")

    # Check body_mass gradient (should be non-zero if physics depends on mass)
    grad_mass = tape.gradients.get(m.body_mass)
    if grad_mass is not None:
        gm = grad_mass.numpy()
        print(f"\n  Gradient of root_z w.r.t. body_mass:")
        print(f"    non-zero entries: {np.count_nonzero(gm)} / {gm.size}")
        print(f"    has NaN: {np.isnan(gm).any()}")
        if np.any(gm != 0):
            print(f"    max |grad|: {np.nanmax(np.abs(gm)):.6e}")
            bodies_per_world = m.body_count // NUM_WORLDS
            for i in range(bodies_per_world):
                if gm[i] != 0.0:
                    name = m.body_key[i] if i < len(m.body_key) else f"body_{i}"
                    print(f"      body {i:2d} ({name:30s}): {gm[i]:+.6e}")
    else:
        print("\n  Gradient of root_z w.r.t. body_mass: NOT TRACKED (expected if mass is constant)")

    # Check body_inertia gradient
    grad_inertia = tape.gradients.get(m.body_inertia)
    if grad_inertia is not None:
        gi = grad_inertia.numpy()
        print(f"\n  Gradient of root_z w.r.t. body_inertia:")
        print(f"    non-zero entries: {np.count_nonzero(gi)} / {gi.size}")
        print(f"    has NaN: {np.isnan(gi).any()}")
    else:
        print("\n  Gradient of root_z w.r.t. body_inertia: NOT TRACKED")

    # Check joint_attach_ke/kd (these are floats, not arrays, so no gradient)
    print(f"\n  joint_attach_ke={solver.joint_attach_ke}, joint_attach_kd={solver.joint_attach_kd}")
    print(f"  (These are Python floats — not tracked by wp.Tape)")

    del states, tape, solver, m
    wp.synchronize()


# ===================================================================
# TEST D: Gradient finite-difference validation
# Compare tape gradient d(root_z)/d(init_root_z) against FD.
# ===================================================================

def run_forward_get_root_z(model, solver, init_qpos, num_steps, device):
    """Run forward sim and return final root_z."""
    control = model.control()
    total_phys = num_steps * NUM_SUBSTEPS
    s0 = model.state(requires_grad=False)
    s1 = model.state(requires_grad=False)

    s0.joint_q.assign(init_qpos)
    s0.joint_qd.zero_()
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    wp.synchronize()

    src, dst = s0, s1
    for step in range(num_steps):
        src.clear_forces()
        contacts = model.collide(src)
        for sub in range(NUM_SUBSTEPS):
            src.clear_forces()
            solver.step(src, dst, control, contacts, SUB_DT)
            newton.eval_ik(model, dst, dst.joint_q, dst.joint_qd)
            src, dst = dst, src
    wp.synchronize()
    jq = src.joint_q.numpy()
    return jq[2]


def test_d_finite_difference(device):
    print("\n" + "=" * 80)
    print("TEST D: Finite-difference gradient validation")
    print("=" * 80)

    ke, kd = 1e4, 100
    num_steps = 5
    eps = 1e-4

    m = build_model(device, requires_grad=False)
    solver = newton.solvers.SolverSemiImplicitStable(m, joint_attach_ke=ke, joint_attach_kd=kd)

    # Get baseline init_qpos
    s_tmp = m.state(requires_grad=False)
    base_qpos = init_state(m, s_tmp, device)

    # Baseline root_z
    z_base = run_forward_get_root_z(m, solver, base_qpos, num_steps, device)
    print(f"\n  Baseline: root_z = {z_base:.6f} after {num_steps} steps")

    # FD for a few DOFs
    dofs_to_test = [
        (0, "root_x"),
        (1, "root_y"),
        (2, "root_z"),
        (7, "dof_7 (first joint angle)"),
        (8, "dof_8"),
        (9, "dof_9"),
    ]

    print(f"\n  Finite-difference d(final_root_z)/d(init_qpos[i]) with eps={eps}:")
    for idx, name in dofs_to_test:
        if idx >= len(base_qpos):
            continue
        qpos_plus = base_qpos.copy()
        qpos_plus[idx] += eps
        z_plus = run_forward_get_root_z(m, solver, qpos_plus, num_steps, device)

        qpos_minus = base_qpos.copy()
        qpos_minus[idx] -= eps
        z_minus = run_forward_get_root_z(m, solver, qpos_minus, num_steps, device)

        fd_grad = (z_plus - z_minus) / (2.0 * eps)
        print(f"    [{idx:3d}] {name:25s}: FD = {fd_grad:+.6e}  "
              f"(z+={z_plus:.6f}, z-={z_minus:.6f})")

    # Now get tape gradient for comparison
    print(f"\n  Tape gradient d(final_root_z)/d(init_qpos[i]):")
    m2 = build_model(device, requires_grad=True)
    solver2 = newton.solvers.SolverSemiImplicitStable(m2, joint_attach_ke=ke, joint_attach_kd=kd)
    control2 = m2.control()
    total_phys = num_steps * NUM_SUBSTEPS
    states = [m2.state(requires_grad=True) for _ in range(total_phys + 1)]

    init_qpos_wp = wp.array(base_qpos, dtype=float, device=device, requires_grad=True)
    wp.copy(states[0].joint_q, init_qpos_wp)
    states[0].joint_qd.zero_()
    wp.synchronize()

    loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        # eval_fk INSIDE tape so joint_q → body_q is recorded
        newton.eval_fk(m2, states[0].joint_q, states[0].joint_qd, states[0])

        phys_step = 0
        for step in range(num_steps):
            states[phys_step].clear_forces()
            contacts = m2.collide(states[phys_step])
            for sub in range(NUM_SUBSTEPS):
                s_in = states[phys_step]
                s_out = states[phys_step + 1]
                s_in.clear_forces()
                solver2.step(s_in, s_out, control2, contacts, SUB_DT)
                newton.eval_ik(m2, s_out, s_out.joint_q, s_out.joint_qd)
                phys_step += 1

        wp.launch(extract_root_z, dim=1,
                  inputs=[states[phys_step].joint_q],
                  outputs=[loss], device=device)

    tape.backward(loss)
    wp.synchronize()

    grad_jq0 = tape.gradients.get(states[0].joint_q)
    if grad_jq0 is not None:
        g = grad_jq0.numpy()
        for idx, name in dofs_to_test:
            if idx < len(g):
                print(f"    [{idx:3d}] {name:25s}: tape = {g[idx]:+.6e}")
    else:
        print("    WARNING: No gradient found for states[0].joint_q")

    # Also check init_qpos_wp gradient (should propagate through wp.copy)
    grad_init = tape.gradients.get(init_qpos_wp)
    if grad_init is not None:
        gi = grad_init.numpy()
        print(f"\n  Gradient via init_qpos_wp (through wp.copy):")
        for idx, name in dofs_to_test:
            if idx < len(gi):
                print(f"    [{idx:3d}] {name:25s}: tape = {gi[idx]:+.6e}")
    else:
        print(f"\n  No gradient on init_qpos_wp (wp.copy may not propagate grads)")

    del states, tape, solver, solver2, m, m2
    wp.synchronize()


# ===================================================================
# TEST E: Contact gradient investigation
# Check what fraction of the computation graph has enable_backward=False.
# ===================================================================

def test_e_contact_gradients(device):
    print("\n" + "=" * 80)
    print("TEST E: Contact gradient analysis")
    print("=" * 80)

    ke, kd = 1e4, 100

    # Run two cases: above ground (no contact) and on ground (with contact)
    for scenario, z_init in [("free-fall (no contact)", 0.8), ("on-ground (with contact)", 0.15)]:
        print(f"\n  --- {scenario}, z_init={z_init} ---")
        m = build_model(device, requires_grad=True)
        solver = newton.solvers.SolverSemiImplicitStable(m, joint_attach_ke=ke, joint_attach_kd=kd)
        control = m.control()

        num_steps = 3
        total_phys = num_steps * NUM_SUBSTEPS
        states = [m.state(requires_grad=True) for _ in range(total_phys + 1)]

        s_tmp = m.state(requires_grad=True)
        qpos = init_state(m, s_tmp, device)
        qpos[2] = z_init  # override z
        init_qpos = wp.array(qpos, dtype=float, device=device, requires_grad=True)

        wp.copy(states[0].joint_q, init_qpos)
        states[0].joint_qd.zero_()
        wp.synchronize()

        loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)
        tape = wp.Tape()
        with tape:
            # eval_fk INSIDE tape so joint_q → body_q is recorded
            newton.eval_fk(m, states[0].joint_q, states[0].joint_qd, states[0])

            phys_step = 0
            for step in range(num_steps):
                states[phys_step].clear_forces()
                contacts = m.collide(states[phys_step])
                for sub in range(NUM_SUBSTEPS):
                    s_in = states[phys_step]
                    s_out = states[phys_step + 1]
                    s_in.clear_forces()
                    solver.step(s_in, s_out, control, contacts, SUB_DT)
                    newton.eval_ik(m, s_out, s_out.joint_q, s_out.joint_qd)
                    phys_step += 1

            wp.launch(extract_root_z, dim=1,
                      inputs=[states[phys_step].joint_q],
                      outputs=[loss], device=device)

        wp.synchronize()
        final_z = states[phys_step].joint_q.numpy()[2]
        print(f"    Final root_z: {final_z:.6f}")

        tape.backward(loss)
        wp.synchronize()

        grad_jq0 = tape.gradients.get(states[0].joint_q)
        if grad_jq0 is not None:
            g = grad_jq0.numpy()
            n_nonzero = np.count_nonzero(g)
            print(f"    Grad on initial joint_q: {n_nonzero}/{g.size} non-zero")
            print(f"    grad[0] (root_x): {g[0]:+.6e}")
            print(f"    grad[1] (root_y): {g[1]:+.6e}")
            print(f"    grad[2] (root_z): {g[2]:+.6e}")
            print(f"    max |grad|: {np.nanmax(np.abs(g)):.6e}")
            print(f"    has NaN: {np.isnan(g).any()}")
            print(f"    has Inf: {np.isinf(g).any()}")
        else:
            print(f"    WARNING: No gradient found")

        del states, tape, solver, m
        wp.synchronize()


# ===================================================================
# MAIN
# ===================================================================

def main():
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    wp.init()
    print(f"Device: {device}")
    print(f"Config: DT={DT:.4f}, NUM_SUBSTEPS={NUM_SUBSTEPS}, SUB_DT={SUB_DT:.6f}")

    test_a_discrepancy(device)
    test_b_long_run(device)
    test_c_gradient_quality(device)
    test_d_finite_difference(device)
    test_e_contact_gradients(device)

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
