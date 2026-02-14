"""Isolated test: G1 robot with SolverSemiImplicit.

Diagnoses NaN sources by dumping forces, contact stiffness, PD gains,
and body masses after the first substep.

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
NUM_STEPS = 10
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
    return model


def dump_model_info(model):
    """Print all the critical model parameters."""
    print("\n--- MODEL INFO ---")
    print(f"  Bodies: {model.body_count}, Joints: {model.joint_count}, Shapes: {model.shape_count}")

    # Body masses
    mass = model.body_mass.numpy()
    inv_mass = model.body_inv_mass.numpy()
    print(f"\n  Body masses (first world, {model.body_count // NUM_WORLDS} bodies):")
    n = model.body_count // NUM_WORLDS
    for i in range(n):
        print(f"    body {i:2d}: mass={mass[i]:.4f} kg, inv_mass={inv_mass[i]:.6g}")

    # Shape material properties (contact stiffness)
    ke = model.shape_material_ke.numpy()
    kd = model.shape_material_kd.numpy()
    kf = model.shape_material_kf.numpy()
    mu = model.shape_material_mu.numpy()
    print(f"\n  Shape materials ({model.shape_count} shapes):")
    for i in range(min(model.shape_count, 40)):
        print(f"    shape {i:2d}: ke={ke[i]:.1f} kd={kd[i]:.1f} kf={kf[i]:.1f} mu={mu[i]:.3f}")

    # PD gains
    target_ke = model.joint_target_ke.numpy()
    target_kd = model.joint_target_kd.numpy()
    nonzero_ke = target_ke[target_ke > 0]
    nonzero_kd = target_kd[target_kd > 0]
    print(f"\n  Joint target PD gains:")
    print(f"    ke: {len(nonzero_ke)} nonzero, range [{nonzero_ke.min():.1f}, {nonzero_ke.max():.1f}]" if len(nonzero_ke) else "    ke: all zero")
    print(f"    kd: {len(nonzero_kd)} nonzero, range [{nonzero_kd.min():.1f}, {nonzero_kd.max():.1f}]" if len(nonzero_kd) else "    kd: all zero")

    # MuJoCo passive stiffness if available
    if hasattr(model, 'mujoco'):
        ps = model.mujoco.dof_passive_stiffness.numpy()
        pd = model.mujoco.dof_passive_damping.numpy()
        nz_ps = ps[ps > 0]
        nz_pd = pd[pd > 0]
        print(f"\n  MuJoCo passive gains:")
        print(f"    stiffness: {len(nz_ps)} nonzero" + (f", range [{nz_ps.min():.1f}, {nz_ps.max():.1f}]" if len(nz_ps) else ""))
        print(f"    damping:   {len(nz_pd)} nonzero" + (f", range [{nz_pd.min():.1f}, {nz_pd.max():.1f}]" if len(nz_pd) else ""))

    # Gravity
    g = model.gravity.numpy().ravel()
    print(f"\n  Gravity: ({g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f})")


def run_test(model, solver, control, label, set_pd=True, enable_contacts=True, device="cuda:0"):
    """Run simulation and dump per-substep diagnostics."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    # Initialize state
    state0 = model.state(requires_grad=False)
    state1 = model.state(requires_grad=False)

    # Set initial joint_q from model defaults + override root
    newton.eval_fk(model, state0.joint_q, state0.joint_qd, state0)
    wp.synchronize()

    qpos = state0.joint_q.numpy().copy()
    joint_q_size = len(qpos) // NUM_WORLDS
    for w in range(NUM_WORLDS):
        s = w * joint_q_size
        qpos[s + 2] = Z_HEIGHT
        # Correct identity quaternion (xyzw)
        qpos[s + 3] = 0.0; qpos[s + 4] = 0.0; qpos[s + 5] = 0.0; qpos[s + 6] = 1.0

    state0.joint_q.assign(qpos)
    state0.joint_qd.zero_()
    newton.eval_fk(model, state0.joint_q, state0.joint_qd, state0)
    wp.synchronize()

    # Print initial state
    bq = state0.body_q.numpy().reshape(-1, 7)
    print(f"  Initial root body_q: pos={bq[0,:3]}, quat={bq[0,3:]}")
    print(f"  Initial joint_q[:7]: {qpos[:7]}")
    jq_default = qpos[7:joint_q_size]
    print(f"  Initial joint angles (non-root): min={jq_default.min():.4f} max={jq_default.max():.4f}")

    # Collide
    state0.clear_forces()
    if enable_contacts:
        contacts = model.collide(state0)
        wp.synchronize()
        n_contacts = int(contacts.rigid_contact_count.numpy()[0]) if contacts is not None and contacts.rigid_contact_max else 0
        print(f"  Contacts: {n_contacts}")
    else:
        contacts = None
        print(f"  Contacts: DISABLED")

    # Run substeps one at a time with diagnostics
    src, dst = state0, state1
    for sub in range(NUM_SUBSTEPS):
        src.clear_forces()
        solver.step(src, dst, control, contacts, SUB_DT)
        wp.synchronize()

        # Read forces AFTER step (body_f was accumulated, then integration happened)
        # Actually body_f is the accumulated force BEFORE integration
        bf = src.body_f.numpy().reshape(-1, 6)
        bf_lin = bf[:, :3]  # linear force
        bf_ang = bf[:, 3:]  # angular torque

        # Read integrated state
        dbq = dst.body_q.numpy().reshape(-1, 7)
        dbqd = dst.body_qd.numpy().reshape(-1, 6)

        # Root body stats
        root_f_lin = bf_lin[0]
        root_f_ang = bf_ang[0]
        root_pos = dbq[0, :3]
        root_vel_lin = dbqd[0, :3]

        # All-body force stats
        f_lin_mag = np.linalg.norm(bf_lin, axis=1)
        f_ang_mag = np.linalg.norm(bf_ang, axis=1)

        nan_f = int(np.isnan(bf).any(axis=1).sum())
        nan_bq = int(np.isnan(dbq).any(axis=1).sum())
        nan_bqd = int(np.isnan(dbqd).any(axis=1).sum())

        print(f"\n  --- Substep {sub} (dt={SUB_DT:.6f}) ---")
        print(f"  Root force:    lin=({root_f_lin[0]:.4g}, {root_f_lin[1]:.4g}, {root_f_lin[2]:.4g})  "
              f"|f|={np.linalg.norm(root_f_lin):.4g}")
        print(f"  Root torque:   ang=({root_f_ang[0]:.4g}, {root_f_ang[1]:.4g}, {root_f_ang[2]:.4g})")
        print(f"  Root pos:      ({root_pos[0]:.6g}, {root_pos[1]:.6g}, {root_pos[2]:.6g})")
        print(f"  Root lin vel:  ({root_vel_lin[0]:.4g}, {root_vel_lin[1]:.4g}, {root_vel_lin[2]:.4g})")
        print(f"  All bodies |f_lin|: max={f_lin_mag.max():.4g}  mean={f_lin_mag.mean():.4g}")
        print(f"  All bodies |f_ang|: max={f_ang_mag.max():.4g}  mean={f_ang_mag.mean():.4g}")
        if nan_f or nan_bq or nan_bqd:
            print(f"  *** NaN: {nan_f} body_f, {nan_bq} body_q, {nan_bqd} body_qd ***")

            # Find which bodies have NaN forces
            if nan_f > 0 and nan_f <= 10:
                nan_bodies = np.where(np.isnan(bf).any(axis=1))[0]
                for bi in nan_bodies[:5]:
                    print(f"    body {bi}: f_lin={bf_lin[bi]} f_ang={bf_ang[bi]}")
            break

        # Run IK to get joint coords
        newton.eval_ik(model, dst, dst.joint_q, dst.joint_qd)
        wp.synchronize()

        # Swap for next substep
        src, dst = dst, src

    # Run remaining macro-steps to check stability
    if not (nan_f or nan_bq or nan_bqd):
        print(f"\n  First macro-step OK, running {NUM_STEPS-1} more...")
        physics_step = NUM_SUBSTEPS
        states = [src]  # current state after first macro-step
        for _ in range(NUM_SUBSTEPS * (NUM_STEPS - 1)):
            states.append(model.state(requires_grad=False))

        for step in range(1, NUM_STEPS):
            macro_src = states[(step-1) * NUM_SUBSTEPS]
            macro_src.clear_forces()
            contacts_s = model.collide(macro_src) if enable_contacts else None
            for sub_i in range(NUM_SUBSTEPS):
                idx = (step-1) * NUM_SUBSTEPS + sub_i
                s = states[idx]
                d = states[idx + 1]
                s.clear_forces()
                solver.step(s, d, control, contacts_s, SUB_DT)
                newton.eval_ik(model, d, d.joint_q, d.joint_qd)

            wp.synchronize()
            last = states[step * NUM_SUBSTEPS]
            jq = last.joint_q.numpy()
            jqd = last.joint_qd.numpy()
            jq_nan = int(np.isnan(jq).sum())
            jqd_nan = int(np.isnan(jqd).sum())
            root_z = jq[2]
            status = "OK" if jq_nan == 0 and jqd_nan == 0 else "NaN!"
            print(f"  Step {step:2d}: {status}  root_z={root_z:.4f}  jq_nan={jq_nan} jqd_nan={jqd_nan}")
            if status == "NaN!":
                break


def main():
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    wp.init()
    print(f"Device: {device}")

    model = build_model(device)
    dump_model_info(model)

    # Test A: Full sim (PD gains from MuJoCo + contacts)
    if hasattr(model, 'mujoco'):
        ke = model.mujoco.dof_passive_stiffness.numpy().copy()
        kd = model.mujoco.dof_passive_damping.numpy().copy()
        model.joint_target_ke.assign(ke)
        model.joint_target_kd.assign(kd)
        model.mujoco.dof_passive_stiffness.assign(np.zeros_like(ke))
        model.mujoco.dof_passive_damping.assign(np.zeros_like(kd))

    solver = newton.solvers.SolverSemiImplicit(model)
    control = model.control()

    run_test(model, solver, control, "TEST A: Full sim (PD + contacts)", device=device)

    # Test B: No PD, no contacts — just gravity
    model.joint_target_ke.zero_()
    model.joint_target_kd.zero_()
    solver_b = newton.solvers.SolverSemiImplicit(model)
    run_test(model, solver_b, control, "TEST B: Gravity only (no PD, no contacts)",
             enable_contacts=False, device=device)

    # Test C: Gravity + contacts, no PD
    solver_c = newton.solvers.SolverSemiImplicit(model)
    run_test(model, solver_c, control, "TEST C: Gravity + contacts (no PD)", device=device)

    # Test D: Gravity + PD, no contacts
    if hasattr(model, 'mujoco'):
        model.joint_target_ke.assign(ke)
        model.joint_target_kd.assign(kd)
    solver_d = newton.solvers.SolverSemiImplicit(model)
    run_test(model, solver_d, control, "TEST D: Gravity + PD (no contacts)",
             enable_contacts=False, device=device)

    return 0


if __name__ == "__main__":
    sys.exit(main())
