"""Isolated test: G1 robot with SolverSemiImplicit.

Sweeps joint_attach_ke/kd to find stable parameters.

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
NUM_STEPS = 30
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


def run_test(model, ke, kd, label, enable_contacts=True, device="cuda:0"):
    """Run simulation with given joint_attach params. Returns True if stable."""
    solver = newton.solvers.SolverSemiImplicit(model, joint_attach_ke=ke, joint_attach_kd=kd)
    control = model.control()

    # Initialize state
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
    max_force = 0.0

    for step in range(NUM_STEPS):
        if step == 0 or step % NUM_SUBSTEPS == 0:
            src.clear_forces()
            contacts = model.collide(src) if enable_contacts else None

        for sub in range(NUM_SUBSTEPS):
            src.clear_forces()
            solver.step(src, dst, control, contacts, SUB_DT)
            newton.eval_ik(model, dst, dst.joint_q, dst.joint_qd)
            src, dst = dst, src

        wp.synchronize()
        jq = src.joint_q.numpy()
        jqd = src.joint_qd.numpy()
        bf = src.body_f.numpy() if hasattr(src.body_f, 'numpy') else np.zeros(1)

        jq_nan = int(np.isnan(jq).sum())
        jqd_nan = int(np.isnan(jqd).sum())
        root_z = jq[2] if not np.isnan(jq[2]) else float('nan')

        finite_f = bf[np.isfinite(bf)]
        step_max_f = float(np.abs(finite_f).max()) if len(finite_f) > 0 else float('inf')
        max_force = max(max_force, step_max_f)

        if jq_nan > 0 or jqd_nan > 0 or np.isnan(root_z) or abs(root_z) > 100:
            failed_step = step
            break

    status = "PASS" if failed_step is None else f"FAIL@step{failed_step}"
    final_z = root_z if failed_step is None else float('nan')

    print(f"  {label:50s}  ke={ke:8.1f}  kd={kd:6.1f}  {status:12s}  "
          f"root_z={final_z:.4f}  max_|f|={max_force:.4g}")
    return failed_step is None


def main():
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    wp.init()
    print(f"Device: {device}")
    print(f"Config: {NUM_WORLDS} world, {NUM_STEPS} steps, {NUM_SUBSTEPS} substeps, "
          f"sub_dt={SUB_DT:.6f}")

    model = build_model(device)

    # Print stability limits for lightest bodies
    mass = model.body_mass.numpy()
    min_mass = mass.min()
    print(f"\nLightest body mass: {min_mass:.4f} kg")
    print(f"Stability limit: kd < {2.0 * min_mass / SUB_DT:.1f}")
    print(f"                 ke < {min_mass / (SUB_DT * SUB_DT):.0f}  (conservative)")

    # Set MuJoCo PD gains (matching BPTT pipeline)
    if hasattr(model, 'mujoco'):
        pd_ke = model.mujoco.dof_passive_stiffness.numpy().copy()
        pd_kd = model.mujoco.dof_passive_damping.numpy().copy()
        model.joint_target_ke.assign(pd_ke)
        model.joint_target_kd.assign(pd_kd)
        model.mujoco.dof_passive_stiffness.assign(np.zeros_like(pd_ke))
        model.mujoco.dof_passive_damping.assign(np.zeros_like(pd_kd))

    # Sweep joint_attach_kd with ke=1e4 (default)
    print(f"\n--- Sweep kd (ke=1e4, with contacts & PD) ---")
    for kd in [100, 50, 20, 10, 5, 2, 1]:
        run_test(model, ke=1e4, kd=kd, label=f"kd={kd}", device=device)

    # Sweep joint_attach_ke with best kd
    print(f"\n--- Sweep ke (kd=10, with contacts & PD) ---")
    for ke in [1e5, 1e4, 1e3, 500, 100]:
        run_test(model, ke=ke, kd=10, label=f"ke={ke}", device=device)

    # Best candidate without contacts
    print(f"\n--- Best candidate without contacts ---")
    run_test(model, ke=1e4, kd=10, label="no contacts", enable_contacts=False, device=device)

    # Try more substeps with default params
    print(f"\n--- More substeps (ke=1e4, kd=100) ---")
    global NUM_SUBSTEPS, SUB_DT
    for ns in [16, 32, 64]:
        NUM_SUBSTEPS = ns
        SUB_DT = DT / ns
        label = f"{ns} substeps (sub_dt={SUB_DT:.6f})"
        run_test(model, ke=1e4, kd=100, label=label, device=device)

    return 0


if __name__ == "__main__":
    sys.exit(main())
