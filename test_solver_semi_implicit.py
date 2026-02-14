"""Isolated test: G1 robot with SolverSemiImplicit.

Tests mass/inertia clamping to stabilize penalty-based joint attachment.

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
NUM_STEPS = 50
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


def clamp_body_properties(model, min_mass=0.0, min_inertia_diag=0.0):
    """Clamp minimum body mass and inertia for stability."""
    if min_mass > 0:
        mass = model.body_mass.numpy()
        inv_mass = model.body_inv_mass.numpy()
        light = mass < min_mass
        n_clamped = int(light.sum())
        if n_clamped > 0:
            print(f"    Clamping {n_clamped} bodies from mass "
                  f"[{mass[light].min():.4f}, {mass[light].max():.4f}] to {min_mass:.4f} kg")
            mass[light] = min_mass
            inv_mass[light] = 1.0 / min_mass
            model.body_mass.assign(mass)
            model.body_inv_mass.assign(inv_mass)

    if min_inertia_diag > 0:
        # body_inertia is wp.mat33 — numpy gives (N, 3, 3)
        inertia = model.body_inertia.numpy()
        inv_inertia = model.body_inv_inertia.numpy()
        n_clamped = 0
        for i in range(len(inertia)):
            modified = False
            for d in range(3):
                if inertia[i, d, d] < min_inertia_diag:
                    inertia[i, d, d] = min_inertia_diag
                    modified = True
            if modified:
                n_clamped += 1
                # Recompute inverse
                inv_inertia[i] = np.linalg.inv(inertia[i])
        if n_clamped > 0:
            print(f"    Clamping {n_clamped} bodies' inertia diagonal to >= {min_inertia_diag:.2e}")
            model.body_inertia.assign(inertia)
            model.body_inv_inertia.assign(inv_inertia)


def run_test(model, ke, kd, label, device="cuda:0"):
    """Run simulation and return (passed, last_root_z)."""
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
    print(f"  {label:55s} {status:12s} root_z={final_z:.4f}")
    return failed_step is None, final_z


def main():
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    wp.init()
    print(f"Device: {device}")
    print(f"Config: {NUM_WORLDS} world, {NUM_STEPS} steps, {NUM_SUBSTEPS} substeps, "
          f"sub_dt={SUB_DT:.6f}")

    # --- Show body inertia info ---
    model = build_model(device)
    mass = model.body_mass.numpy()
    inertia = model.body_inertia.numpy()  # (N, 3, 3)
    n = model.body_count // NUM_WORLDS
    print(f"\nBody mass & min inertia diagonal (first world, {n} bodies):")
    for i in range(n):
        I_diag = [inertia[i, d, d] for d in range(3)]
        I_min = min(I_diag)
        # Stability: kd * dt / I_min < 2  =>  kd_max = 2 * I_min / dt
        kd_max = 2.0 * I_min / SUB_DT if I_min > 0 else 0
        print(f"  body {i:2d}: mass={mass[i]:.4f}  I_diag=({I_diag[0]:.2e}, {I_diag[1]:.2e}, {I_diag[2]:.2e})  "
              f"I_min={I_min:.2e}  kd_max={kd_max:.1f}")

    I_all_min = min(inertia[i, d, d] for i in range(n) for d in range(3))
    print(f"\nSmallest inertia diagonal: {I_all_min:.2e}")
    print(f"Angular stability limit (kd < 2*I_min/dt): kd < {2*I_all_min/SUB_DT:.4f}")

    # --- Test 1: No clamping (baseline) ---
    print(f"\n--- Test 1: No clamping (ke=1e4, kd=100) ---")
    run_test(model, ke=1e4, kd=100, label="baseline", device=device)

    # --- Test 2: Sweep min_inertia_diag with reduced kd ---
    for min_I in [1e-4, 1e-3, 1e-2, 1e-1]:
        for att_kd in [10, 50, 100]:
            model2 = build_model(device)
            clamp_body_properties(model2, min_mass=0.5, min_inertia_diag=min_I)
            label = f"min_mass=0.5, min_I={min_I:.0e}, kd={att_kd}"
            run_test(model2, ke=1e4, kd=att_kd, label=label, device=device)

    # --- Test 3: Best candidate, longer run ---
    print(f"\n--- Test 3: Best candidate, {NUM_STEPS} steps ---")
    model3 = build_model(device)
    clamp_body_properties(model3, min_mass=0.5, min_inertia_diag=1e-2)
    run_test(model3, ke=1e4, kd=10, label="min_mass=0.5, min_I=1e-2, kd=10", device=device)

    # Also try lower ke
    model4 = build_model(device)
    clamp_body_properties(model4, min_mass=0.5, min_inertia_diag=1e-2)
    run_test(model4, ke=1e3, kd=10, label="min_mass=0.5, min_I=1e-2, ke=1e3, kd=10", device=device)

    model5 = build_model(device)
    clamp_body_properties(model5, min_mass=0.5, min_inertia_diag=1e-2)
    run_test(model5, ke=500, kd=5, label="min_mass=0.5, min_I=1e-2, ke=500, kd=5", device=device)

    return 0


if __name__ == "__main__":
    sys.exit(main())
