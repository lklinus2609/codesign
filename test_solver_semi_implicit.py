"""Isolated test: G1 robot with SolverSemiImplicit.

Aggressive parameter sweep: very soft joint attachment + large inertia clamping.

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


def clamp_model(model, min_mass, min_I, contact_ke=None, contact_kd=None):
    """Clamp body mass/inertia and optionally contact stiffness."""
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

    if contact_ke is not None:
        ke = model.shape_material_ke.numpy()
        ke[:] = contact_ke
        model.shape_material_ke.assign(ke)

    if contact_kd is not None:
        kd = model.shape_material_kd.numpy()
        kd[:] = contact_kd
        model.shape_material_kd.assign(kd)


def run_test(model, ke, kd, label, device="cuda:0"):
    """Run simulation, return (passed, final_root_z)."""
    solver = newton.solvers.SolverSemiImplicit(model, joint_attach_ke=ke, joint_attach_kd=kd)
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
    print(f"  {label:60s} {status:12s} z={final_z:.4f}")
    return failed_step is None


def main():
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    wp.init()
    print(f"Device: {device}")
    print(f"Config: {NUM_STEPS} steps x {NUM_SUBSTEPS} substeps, sub_dt={SUB_DT:.6f}\n")

    # Sweep 1: Very soft attachment springs + large inertia clamp
    print("--- Sweep: joint_attach_ke/kd with min_I=1.0, min_mass=1.0 ---")
    for att_ke in [1e4, 1e3, 100, 10]:
        for att_kd in [10, 1, 0.1, 0.01]:
            m = build_model(device)
            clamp_model(m, min_mass=1.0, min_I=1.0)
            label = f"ke={att_ke:.0f} kd={att_kd}"
            run_test(m, ke=att_ke, kd=att_kd, label=label, device=device)

    # Sweep 2: Also soften contact stiffness
    print("\n--- Sweep: soft contacts (contact_ke=100, contact_kd=10) ---")
    for att_ke in [1e3, 100, 10]:
        for att_kd in [1, 0.1, 0.01]:
            m = build_model(device)
            clamp_model(m, min_mass=1.0, min_I=1.0, contact_ke=100, contact_kd=10)
            label = f"ke={att_ke:.0f} kd={att_kd} cke=100 ckd=10"
            run_test(m, ke=att_ke, kd=att_kd, label=label, device=device)

    # Sweep 3: Zero PD gains (eliminate PD torque source)
    print("\n--- Sweep: zero PD gains + soft contacts ---")
    for att_ke in [100, 10]:
        for att_kd in [1, 0.1]:
            m = build_model(device)
            m.joint_target_ke.zero_()
            m.joint_target_kd.zero_()
            clamp_model(m, min_mass=1.0, min_I=1.0, contact_ke=100, contact_kd=10)
            label = f"ke={att_ke:.0f} kd={att_kd} NO_PD cke=100 ckd=10"
            run_test(m, ke=att_ke, kd=att_kd, label=label, device=device)

    return 0


if __name__ == "__main__":
    sys.exit(main())
