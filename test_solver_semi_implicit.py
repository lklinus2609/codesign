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
    # Test 4: DEBUG stable solver — before/after implicit correction
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST 4: SolverSemiImplicitStable DEBUG (before/after implicit)")
    print("=" * 80)

    m = build_model(device)
    ke_test, kd_test = 1e4, 100
    bodies_per_world = m.body_count // NUM_WORLDS
    body_keys = m.body_key

    # Print body inertia info for reference
    mass = m.body_mass.numpy()
    inertia = m.body_inertia.numpy()
    print(f"\n  Body info ({bodies_per_world} bodies):")
    for i in range(min(bodies_per_world, 20)):
        name = body_keys[i] if i < len(body_keys) else f"body_{i}"
        I_diag = [inertia[i, d, d] for d in range(3)]
        I_ratio = max(I_diag) / max(min(I_diag), 1e-20)
        print(f"    {i:2d} {name:30s}: m={mass[i]:.4f}  I=({I_diag[0]:.3e}, {I_diag[1]:.3e}, {I_diag[2]:.3e})  ratio={I_ratio:.1f}")

    from newton.solvers import SolverSemiImplicitStable
    solver = SolverSemiImplicitStable(m, joint_attach_ke=ke_test, joint_attach_kd=kd_test)
    solver._debug = True  # Enable debug buffer
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

    print(f"\n  Running {NUM_SUBSTEPS} substeps with before/after implicit diagnostics:")
    print(f"  ke={ke_test}, kd={kd_test}, sub_dt={SUB_DT:.6f}")

    src, dst = s0, s1
    src.clear_forces()
    contacts = m.collide(src)
    wp.synchronize()

    for sub in range(NUM_SUBSTEPS):
        src.clear_forces()
        solver.step(src, dst, control, contacts, SUB_DT)
        wp.synchronize()

        bqd_pre = solver._debug_qd_buf.numpy()   # body_qd after integration, BEFORE implicit
        bqd_post = dst.body_qd.numpy()            # body_qd AFTER implicit + reintegration

        bqd_nan = np.isnan(bqd_post).any()
        bqd_max_pre = np.abs(bqd_pre[~np.isnan(bqd_pre)]).max() if not np.isnan(bqd_pre).all() else float('nan')
        bqd_max_post = np.abs(bqd_post[~np.isnan(bqd_post)]).max() if not np.isnan(bqd_post).all() else float('nan')

        # Compute per-body correction magnitude
        delta = bqd_post - bqd_pre
        delta_mag = np.linalg.norm(delta[:bodies_per_world], axis=1)
        pre_mag = np.linalg.norm(bqd_pre[:bodies_per_world], axis=1)
        post_mag = np.linalg.norm(bqd_post[:bodies_per_world], axis=1)

        print(f"\n    --- sub {sub} ---")
        print(f"    bqd_max: pre={bqd_max_pre:.6f}  post={bqd_max_post:.6f}")

        # Show top 10 bodies by correction magnitude
        top_idx = np.argsort(delta_mag)[::-1][:10]
        print(f"    Top 10 bodies by |correction|:")
        for idx in top_idx:
            name = body_keys[idx] if idx < len(body_keys) else f"body_{idx}"
            print(f"      {idx:2d} {name:30s}: |pre|={pre_mag[idx]:.6f}  |post|={post_mag[idx]:.6f}  "
                  f"|delta|={delta_mag[idx]:.6f}")
            if delta_mag[idx] > 0.001:  # Show components for significant corrections
                print(f"         pre ={bqd_pre[idx]}")
                print(f"         post={bqd_post[idx]}")
                print(f"         delta={delta[idx]}")

        if bqd_nan or bqd_max_post > 1e6:
            print(f"\n    STOPPING: {'NaN' if bqd_nan else 'velocity > 1e6'}")
            break

        # eval_ik for next substep
        newton.eval_ik(m, dst, dst.joint_q, dst.joint_qd)
        wp.synchronize()
        src, dst = dst, src

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
