"""Isolated test: G1 robot with SolverSemiImplicit vs SolverSemiImplicitStable.

Tests the new implicit joint attachment solver against the original solver.
The key test: SolverSemiImplicitStable should survive with REAL body inertias
(no clamping) where SolverSemiImplicit explodes.

Usage:
    python -u -m codesign.test_solver_semi_implicit
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
NUM_STEPS = 100
Z_HEIGHT = 0.8


def build_model(device="cuda:0"):
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
    model = builder.finalize(requires_grad=False)

    # Set PD gains from MuJoCo
    if hasattr(model, 'mujoco'):
        pd_ke = model.mujoco.dof_passive_stiffness.numpy().copy()
        pd_kd = model.mujoco.dof_passive_damping.numpy().copy()
        model.joint_target_ke.assign(pd_ke)
        model.joint_target_kd.assign(pd_kd)
        model.mujoco.dof_passive_stiffness.assign(np.zeros_like(pd_ke))
        model.mujoco.dof_passive_damping.assign(np.zeros_like(pd_kd))

    return model


def clamp_model(model, min_mass, min_I):
    """Clamp body mass/inertia for stability testing."""
    if min_mass > 0:
        mass = model.body_mass.numpy()
        inv_mass = model.body_inv_mass.numpy()
        light = mass < min_mass
        n = int(light.sum())
        if n > 0:
            mass[light] = min_mass
            inv_mass[light] = 1.0 / min_mass
            model.body_mass.assign(mass)
            model.body_inv_mass.assign(inv_mass)

    if min_I > 0:
        inertia = model.body_inertia.numpy()
        inv_inertia = model.body_inv_inertia.numpy()
        for i in range(len(inertia)):
            modified = False
            for d in range(3):
                if inertia[i, d, d] < min_I:
                    inertia[i, d, d] = min_I
                    modified = True
            if modified:
                inv_inertia[i] = np.linalg.inv(inertia[i])
        model.body_inertia.assign(inertia)
        model.body_inv_inertia.assign(inv_inertia)


def print_body_inertias(model):
    """Print all body masses and minimum inertia diagonals."""
    mass = model.body_mass.numpy()
    inertia = model.body_inertia.numpy()
    body_keys = model.body_key
    bodies_per_world = model.body_count // NUM_WORLDS

    print(f"\n  Body inertia diagnostics ({bodies_per_world} bodies/world):")
    for i in range(bodies_per_world):
        name = body_keys[i] if i < len(body_keys) else f"body_{i}"
        I_diag = [inertia[i, d, d] for d in range(3)]
        I_min = min(I_diag)
        kd_max = 2.0 * I_min / SUB_DT
        print(f"    body {i:2d} ({name:30s}): mass={mass[i]:.4f} kg, "
              f"I_min={I_min:.3e}, kd_max_stable={kd_max:.3f}")


def run_test(model, ke, kd, label, solver_type="standard", device="cuda:0"):
    """Run simulation with specified solver. Return (passed, final_root_z)."""
    if solver_type == "stable":
        solver = newton.solvers.SolverSemiImplicitStable(
            model, joint_attach_ke=ke, joint_attach_kd=kd
        )
    else:
        solver = newton.solvers.SolverSemiImplicit(
            model, joint_attach_ke=ke, joint_attach_kd=kd
        )

    control = model.control()

    s0 = model.state(requires_grad=False)
    s1 = model.state(requires_grad=False)

    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    wp.synchronize()

    qpos = s0.joint_q.numpy().copy()
    q_size = len(qpos) // NUM_WORLDS
    for w in range(NUM_WORLDS):
        s = w * q_size
        qpos[s + 2] = Z_HEIGHT
        qpos[s + 3] = 0.0; qpos[s + 4] = 0.0; qpos[s + 5] = 0.0; qpos[s + 6] = 1.0

    s0.joint_q.assign(qpos)
    s0.joint_qd.zero_()
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    wp.synchronize()

    src, dst = s0, s1
    failed_step = None

    for step in range(NUM_STEPS):
        src.clear_forces()
        contacts = model.collide(src)

        for sub in range(NUM_SUBSTEPS):
            src.clear_forces()
            solver.step(src, dst, control, contacts, SUB_DT)
            newton.eval_ik(model, dst, dst.joint_q, dst.joint_qd)
            src, dst = dst, src

        wp.synchronize()
        jq = src.joint_q.numpy()
        jqd = src.joint_qd.numpy()
        root_z = jq[2] if not np.isnan(jq[2]) else float('nan')

        if np.isnan(jq).any() or np.isnan(jqd).any() or abs(root_z) > 100:
            failed_step = step
            break

    status = "PASS" if failed_step is None else f"FAIL@step{failed_step}"
    final_z = root_z if failed_step is None else float('nan')
    print(f"  {label:65s} {status:12s} z={final_z:.4f}")
    return failed_step is None


def main():
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    wp.init()
    print(f"Device: {device}")
    print(f"Config: {NUM_STEPS} steps x {NUM_SUBSTEPS} substeps, sub_dt={SUB_DT:.6f}")

    # ---------------------------------------------------------------
    # Test 1: Show body inertias (no clamping)
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST 1: Body inertia diagnostics (real, unclamped)")
    print("=" * 80)
    m = build_model(device)
    print_body_inertias(m)

    # ---------------------------------------------------------------
    # Test 2: Standard solver with REAL inertias (expected: FAIL)
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST 2: Standard SolverSemiImplicit with REAL inertias (should FAIL)")
    print("=" * 80)
    for kd in [100, 10, 1]:
        m = build_model(device)
        label = f"[standard] ke=1e4 kd={kd} NO_CLAMP"
        run_test(m, ke=1e4, kd=kd, label=label, solver_type="standard", device=device)

    # ---------------------------------------------------------------
    # Test 3: Standard solver with clamped inertias (baseline: PASS)
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST 3: Standard SolverSemiImplicit with clamped inertias (baseline)")
    print("=" * 80)
    m = build_model(device)
    clamp_model(m, min_mass=1.0, min_I=1.0)
    label = f"[standard] ke=1e4 kd=10 min_I=1.0 min_mass=1.0"
    run_test(m, ke=1e4, kd=10, label=label, solver_type="standard", device=device)

    # ---------------------------------------------------------------
    # Test 4: DEBUG stable solver — single substep with diagnostics
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST 4: SolverSemiImplicitStable DEBUG (single substep)")
    print("=" * 80)

    m = build_model(device)
    ke_test, kd_test = 1e4, 100
    bodies_per_world = m.body_count // NUM_WORLDS
    body_keys = m.body_key

    # Manual step-by-step to find where NaN appears
    from newton.solvers import SolverSemiImplicitStable
    solver = SolverSemiImplicitStable(m, joint_attach_ke=ke_test, joint_attach_kd=kd_test)
    control = m.control()

    s0 = m.state(requires_grad=False)
    s1 = m.state(requires_grad=False)

    newton.eval_fk(m, s0.joint_q, s0.joint_qd, s0)
    wp.synchronize()

    # Set initial pose
    qpos = s0.joint_q.numpy().copy()
    q_size = len(qpos) // NUM_WORLDS
    for w in range(NUM_WORLDS):
        s = w * q_size
        qpos[s + 2] = Z_HEIGHT
        qpos[s + 3] = 0.0; qpos[s + 4] = 0.0; qpos[s + 5] = 0.0; qpos[s + 6] = 1.0
    s0.joint_q.assign(qpos)
    s0.joint_qd.zero_()
    newton.eval_fk(m, s0.joint_q, s0.joint_qd, s0)
    wp.synchronize()

    # Check initial state
    bq0 = s0.body_q.numpy()
    bqd0 = s0.body_qd.numpy()
    print(f"\n  Initial state:")
    print(f"    body_q  NaN: {np.isnan(bq0).any()}, inf: {np.isinf(bq0).any()}")
    print(f"    body_qd NaN: {np.isnan(bqd0).any()}, inf: {np.isinf(bqd0).any()}")
    print(f"    body_qd max: {np.abs(bqd0).max():.6f}")

    # Check body_f before step
    bf0 = s0.body_f.numpy()
    print(f"    body_f  NaN: {np.isnan(bf0).any()}, max: {np.abs(bf0).max():.6f}")

    # Print a few body positions/velocities
    for i in range(min(5, bodies_per_world)):
        name = body_keys[i] if i < len(body_keys) else f"body_{i}"
        print(f"    body {i} ({name}): q={bq0[i]}, qd={bqd0[i]}")

    # --- Run phases manually ---
    s0.clear_forces()
    contacts = m.collide(s0)
    wp.synchronize()

    # Phase 1: Force accumulation + integration (no joint forces)
    # We need to manually run the solver's step but check intermediate states.
    # Easiest: run the full step and check state_out
    s0.clear_forces()
    solver.step(s0, s1, control, contacts, SUB_DT)
    wp.synchronize()

    bq1 = s1.body_q.numpy()
    bqd1 = s1.body_qd.numpy()

    print(f"\n  After solver.step (1 substep):")
    print(f"    body_q  NaN: {np.isnan(bq1).any()}, inf: {np.isinf(bq1).any()}")
    print(f"    body_qd NaN: {np.isnan(bqd1).any()}, inf: {np.isinf(bqd1).any()}")
    print(f"    body_qd max: {np.abs(bqd1[~np.isnan(bqd1)]).max():.6f}" if not np.isnan(bqd1).all() else "    all NaN")

    # Find which bodies have NaN
    if np.isnan(bqd1).any():
        print(f"\n  NaN bodies after solver.step:")
        for i in range(bodies_per_world):
            q_nan = np.isnan(bq1[i]).any()
            qd_nan = np.isnan(bqd1[i]).any()
            if q_nan or qd_nan:
                name = body_keys[i] if i < len(body_keys) else f"body_{i}"
                print(f"    body {i:2d} ({name:30s}): q_nan={q_nan}, qd_nan={qd_nan}")
                if not np.isnan(bqd1[i]).all():
                    print(f"      qd = {bqd1[i]}")

    # Now test: run ONLY integration (no implicit correction) to isolate
    print(f"\n  --- Isolating: integration only (no implicit correction) ---")
    s0.clear_forces()
    # Manually replicate what the solver does minus the implicit kernel
    from newton._src.solvers.semi_implicit.solver_semi_implicit_stable import apply_free_joint_wrench
    s2 = m.state(requires_grad=False)

    # Apply free joint wrenches
    if m.joint_count:
        wp.launch(
            kernel=apply_free_joint_wrench,
            dim=m.joint_count,
            inputs=[m.joint_type, m.joint_enabled, m.joint_child, m.joint_qd_start,
                    control.joint_f, s0.body_f],
            device=m.device,
        )
    # Contact forces
    from newton._src.solvers.semi_implicit.kernels_contact import eval_body_contact_forces
    eval_body_contact_forces(m, s0, contacts, friction_smoothing=1.0)
    wp.synchronize()

    bf_pre = s0.body_f.numpy()
    print(f"  body_f before integration: NaN={np.isnan(bf_pre).any()}, max={np.abs(bf_pre).max():.6f}")

    # Integrate
    solver.integrate_bodies(m, s0, s2, SUB_DT, solver.angular_damping)
    wp.synchronize()

    bq2 = s2.body_q.numpy()
    bqd2 = s2.body_qd.numpy()
    print(f"  After integration only (no implicit correction):")
    print(f"    body_q  NaN: {np.isnan(bq2).any()}")
    print(f"    body_qd NaN: {np.isnan(bqd2).any()}")
    print(f"    body_qd max: {np.abs(bqd2[~np.isnan(bqd2)]).max():.6f}" if not np.isnan(bqd2).all() else "    all NaN")

    if not np.isnan(bqd2).any():
        print(f"  => Integration alone is STABLE. Bug is in the implicit correction kernel.")
    else:
        print(f"  => Integration itself produces NaN. Bug is in force accumulation or integration.")

    # ---------------------------------------------------------------
    # Test 4b: Full sweep (keeping for reference)
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST 4b: SolverSemiImplicitStable sweep (should PASS)")
    print("=" * 80)
    for ke in [1e4, 1e3]:
        for kd in [100, 10, 1]:
            m = build_model(device)
            label = f"[STABLE] ke={ke:.0f} kd={kd} NO_CLAMP"
            run_test(m, ke=ke, kd=kd, label=label, solver_type="stable", device=device)

    # ---------------------------------------------------------------
    # Test 5: Stable solver with gradient tracking (BPTT compatibility)
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST 5: SolverSemiImplicitStable with requires_grad=True")
    print("=" * 80)
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
    m_grad = builder.finalize(requires_grad=True)

    if hasattr(m_grad, 'mujoco'):
        pd_ke = m_grad.mujoco.dof_passive_stiffness.numpy().copy()
        pd_kd = m_grad.mujoco.dof_passive_damping.numpy().copy()
        m_grad.joint_target_ke.assign(pd_ke)
        m_grad.joint_target_kd.assign(pd_kd)
        m_grad.mujoco.dof_passive_stiffness.assign(np.zeros_like(pd_ke))
        m_grad.mujoco.dof_passive_damping.assign(np.zeros_like(pd_kd))

    solver = newton.solvers.SolverSemiImplicitStable(
        m_grad, joint_attach_ke=1e4, joint_attach_kd=100
    )
    control = m_grad.control()

    s0 = m_grad.state(requires_grad=True)
    s1 = m_grad.state(requires_grad=True)

    newton.eval_fk(m_grad, s0.joint_q, s0.joint_qd, s0)
    wp.synchronize()

    qpos = s0.joint_q.numpy().copy()
    q_size = len(qpos) // NUM_WORLDS
    for w in range(NUM_WORLDS):
        s = w * q_size
        qpos[s + 2] = Z_HEIGHT
        qpos[s + 3] = 0.0; qpos[s + 4] = 0.0; qpos[s + 5] = 0.0; qpos[s + 6] = 1.0

    init_qpos = wp.array(qpos, dtype=float, device=device)

    # Run a short forward pass on tape
    num_tape_steps = 5
    states = [m_grad.state(requires_grad=True) for _ in range(num_tape_steps * NUM_SUBSTEPS + 1)]

    # Initialize state 0
    total_q = m_grad.joint_q.numpy().shape[0]
    total_qd = m_grad.joint_qd.numpy().shape[0]
    wp.copy(states[0].joint_q, init_qpos)
    states[0].joint_qd.zero_()
    newton.eval_fk(m_grad, states[0].joint_q, states[0].joint_qd, states[0])
    wp.synchronize()

    tape = wp.Tape()
    failed = False
    with tape:
        phys_step = 0
        for step in range(num_tape_steps):
            src = states[phys_step]
            src.clear_forces()
            contacts = m_grad.collide(src)

            for sub in range(NUM_SUBSTEPS):
                s_in = states[phys_step]
                s_out = states[phys_step + 1]
                s_in.clear_forces()
                solver.step(s_in, s_out, control, contacts, SUB_DT)
                newton.eval_ik(m_grad, s_out, s_out.joint_q, s_out.joint_qd)
                phys_step += 1

            wp.synchronize()
            jq = states[phys_step].joint_q.numpy()
            if np.isnan(jq).any():
                print(f"  NaN at step {step}")
                failed = True
                break

    if not failed:
        # Check if final state is finite
        final_jq = states[phys_step].joint_q.numpy()
        root_z = final_jq[2]
        print(f"  Forward pass: {num_tape_steps} steps, root_z={root_z:.4f}, "
              f"nan={np.isnan(final_jq).any()}, inf={np.isinf(final_jq).any()}")

        # Try backward pass
        try:
            loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)
            # Simple loss: sum of final joint_q
            final_state = states[phys_step]
            wp.copy(loss, wp.array([float(final_jq.sum())], dtype=float, device=device))

            tape.backward(loss)
            print(f"  Backward pass: SUCCESS (tape recorded {phys_step} physics steps)")
        except Exception as e:
            print(f"  Backward pass: FAILED ({e})")
    else:
        print(f"  Forward pass: FAILED (NaN)")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
