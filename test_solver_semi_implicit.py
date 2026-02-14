"""Isolated test: G1 robot with SolverSemiImplicit.

Verifies that the robot stays stable (no NaN) with the correct identity
quaternion (0,0,0,1) in Warp's xyzw convention, and that the OLD wrong
quaternion (1,0,0,0) causes NaN — confirming the root cause of the BPTT bug.

Usage:
    python -m codesign.test_solver_semi_implicit
"""

import sys
import numpy as np
from pathlib import Path

# Ensure codesign package is importable
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

import warp as wp
import newton

MJCF_PATH = SCRIPT_DIR.parent / "MimicKit" / "data" / "assets" / "g1" / "g1.xml"

NUM_WORLDS = 2
DT = 1.0 / 30.0
NUM_SUBSTEPS = 8
SUB_DT = DT / NUM_SUBSTEPS
NUM_STEPS = 20  # macro-steps
Z_HEIGHT = 0.8


def build_model(device="cuda:0"):
    """Build G1 model + solver, matching the BPTT pipeline."""
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
    solver = newton.solvers.SolverSemiImplicit(model)
    control = model.control()

    # PD gains from MuJoCo attributes
    if hasattr(model, 'mujoco'):
        ke = model.mujoco.dof_passive_stiffness.numpy().copy()
        kd = model.mujoco.dof_passive_damping.numpy().copy()
        model.joint_target_ke.assign(ke)
        model.joint_target_kd.assign(kd)
        model.mujoco.dof_passive_stiffness.assign(np.zeros_like(ke))
        model.mujoco.dof_passive_damping.assign(np.zeros_like(kd))

    return model, solver, control


def init_joint_q(model, quat):
    """Initialize joint_q with given root quaternion (xyzw) for all worlds."""
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    wp.synchronize()

    qpos = state.joint_q.numpy().copy()
    joint_q_size = len(qpos) // NUM_WORLDS

    for w in range(NUM_WORLDS):
        s = w * joint_q_size
        qpos[s + 2] = Z_HEIGHT
        qpos[s + 3] = quat[0]  # qx
        qpos[s + 4] = quat[1]  # qy
        qpos[s + 5] = quat[2]  # qz
        qpos[s + 6] = quat[3]  # qw

    del state
    return qpos


def run_sim(model, solver, control, init_qpos, label, device="cuda:0"):
    """Run NUM_STEPS macro-steps and report NaN status at each step."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Root quat (xyzw): {init_qpos[3:7]}")
    print(f"{'='*60}")

    # Allocate state ring buffer (2 states for ping-pong)
    total_substeps = NUM_STEPS * NUM_SUBSTEPS
    states = [model.state(requires_grad=False) for _ in range(total_substeps + 1)]

    # Set initial joint_q
    qpos_wp = wp.array(init_qpos, dtype=float, device=device)
    wp.copy(states[0].joint_q, qpos_wp)
    # Zero velocities
    states[0].joint_qd.zero_()
    # Run FK to get consistent body_q
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    wp.synchronize()

    # Check initial body_q
    bq0 = states[0].body_q.numpy()
    print(f"  Initial body_q NaN: {np.isnan(bq0).sum()}/{bq0.size}")

    nan_step = None
    physics_step = 0

    for step in range(NUM_STEPS):
        # Collide on macro-step boundary
        macro_src = states[step * NUM_SUBSTEPS]
        macro_src.clear_forces()
        contacts = model.collide(macro_src)

        if step == 0:
            wp.synchronize()
            n_contacts = 0
            if contacts is not None and contacts.rigid_contact_max:
                n_contacts = int(contacts.rigid_contact_count.numpy()[0])
            print(f"  Step 0 contacts: {n_contacts}")

        for sub in range(NUM_SUBSTEPS):
            src = states[physics_step]
            dst = states[physics_step + 1]
            src.clear_forces()
            solver.step(src, dst, control, contacts, SUB_DT)

            # After integration, run IK to update joint_q/joint_qd
            newton.eval_ik(model, dst)

            physics_step += 1

        # Check for NaN after each macro-step
        wp.synchronize()
        jq = states[physics_step].joint_q.numpy()
        jqd = states[physics_step].joint_qd.numpy()
        bq = states[physics_step].body_q.numpy()

        jq_nan = int(np.isnan(jq).sum())
        jqd_nan = int(np.isnan(jqd).sum())
        bq_nan = int(np.isnan(bq).sum())

        status = "OK" if (jq_nan == 0 and jqd_nan == 0 and bq_nan == 0) else "NaN!"
        if status == "NaN!" and nan_step is None:
            nan_step = step

        # Print first few steps and any NaN steps
        if step < 5 or status == "NaN!" or step == NUM_STEPS - 1:
            # Get root position for world 0
            joint_q_size = len(jq) // NUM_WORLDS
            root_pos = jq[:3]
            root_z = root_pos[2] if not np.isnan(root_pos[2]) else float('nan')
            print(f"  Step {step:3d}: {status}  "
                  f"jq_nan={jq_nan} jqd_nan={jqd_nan} bq_nan={bq_nan}  "
                  f"root_z={root_z:.4f}")

    print(f"\n  Result: {'PASS - no NaN' if nan_step is None else f'FAIL - first NaN at step {nan_step}'}")
    return nan_step is None


def main():
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    print(f"Device: {device}")
    print(f"MJCF: {MJCF_PATH}")
    print(f"Config: {NUM_WORLDS} worlds, {NUM_STEPS} steps, {NUM_SUBSTEPS} substeps, dt={DT}")

    wp.init()

    model, solver, control = build_model(device)
    print(f"Bodies/world: {model.body_count // NUM_WORLDS}, "
          f"Joints/world: {model.joint_count // NUM_WORLDS}")

    # Test 1: CORRECT quaternion — identity in xyzw = (0, 0, 0, 1)
    correct_qpos = init_joint_q(model, quat=[0.0, 0.0, 0.0, 1.0])
    pass_correct = run_sim(model, solver, control, correct_qpos,
                           "TEST 1: Correct identity quat (0,0,0,1)", device)

    # Test 2: WRONG quaternion — old bug: (1, 0, 0, 0) = 180° x-rotation
    wrong_qpos = init_joint_q(model, quat=[1.0, 0.0, 0.0, 0.0])
    pass_wrong = run_sim(model, solver, control, wrong_qpos,
                         "TEST 2: Wrong quat (1,0,0,0) — robot upside down", device)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Correct quat (0,0,0,1): {'PASS' if pass_correct else 'FAIL'}")
    print(f"  Wrong   quat (1,0,0,0): {'PASS' if pass_wrong else 'FAIL'}")

    if pass_correct and not pass_wrong:
        print(f"\n  Quaternion convention bug CONFIRMED as root cause.")
    elif pass_correct and pass_wrong:
        print(f"\n  Both passed — wrong quat doesn't cause NaN alone.")
        print(f"  (NaN may need actions/torques to trigger.)")
    elif not pass_correct:
        print(f"\n  Even correct quat fails — there may be another issue.")

    return 0 if pass_correct else 1


if __name__ == "__main__":
    sys.exit(main())
