#!/usr/bin/env python3
"""Phase 0 probe -- can Newton model arrays be perturbed in-place?

Before committing to the design-space expansion (Phase 1: actuator params,
Phase 2: link lengths), this script answers two questions:

  1. Array audit -- which model attributes exist, what shape, what dtype?
     Specifically the ones we plan to use for SPSA perturbation:
         joint_target_ke / joint_target_kd / joint_effort_limit
         joint_armature
         body_mass / body_inertia / body_inv_mass / body_inv_inertia

  2. Propagation test -- if we write a value into one of those arrays via
     .assign() (the same mechanism that works for joint_X_p today), does
     the MuJoCo-Warp solver consume the new value on the next step()?

     For each candidate array, we run the simulator forward N steps with
     the default value and record a state trace, then write a 10x-scaled
     value, re-step from the same initial state, and compare traces.

       - If the traces DIFFER -> solver re-reads from the array each
         step. SPSA perturbation can use the same in-place pattern as
         joint_X_p. Phase 1 (actuator params) feasible from joint_target_*
         test; Phase 2 (link lengths) feasible from body_mass + body_inertia
         tests.

       - If the traces are IDENTICAL -> the solver baked the value at
         construction (MuJoCo caches CRB inertias). Phase 2 needs a
         different approach: per-perturbation env rebuild (slow), switch
         to SolverSemiImplicit (loses AMP support), or accept kinematic-
         only length changes (physically inconsistent).

Single-GPU, ~256-env, ~30 seconds total runtime.

Usage:
    python codesign/probe_newton_arrays.py
    python codesign/probe_newton_arrays.py --num-envs 64 --steps 30
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["PYGLET_HEADLESS"] = "1"

import numpy as np

# ---------------------------------------------------------------------------
# Paths (mirror codesign_g1_unified.py)
# ---------------------------------------------------------------------------
CODESIGN_DIR = Path(__file__).parent.resolve()
MIMICKIT_DIR = (CODESIGN_DIR / ".." / "MimicKit").resolve()
MIMICKIT_SRC_DIR = MIMICKIT_DIR / "mimickit"

BASE_MJCF_PATH = MIMICKIT_DIR / "data" / "assets" / "g1" / "g1.xml"
BASE_ENV_CONFIG = MIMICKIT_DIR / "data" / "envs" / "amp_g1_env.yaml"
BASE_ENGINE_CONFIG = MIMICKIT_DIR / "data" / "engines" / "newton_engine.yaml"

if str(MIMICKIT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(MIMICKIT_SRC_DIR))

import warp as wp        # noqa: E402
import newton            # noqa: E402, F401
import torch             # noqa: E402
import envs.env_builder as env_builder  # noqa: E402

from g1_mjcf_modifier import G1MJCFModifier  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_array_sample(attr, n_show=3):
    """Return shape, dtype, and first-few-elements sample as a string."""
    try:
        shape = tuple(attr.shape)
    except Exception:
        shape = "?"
    try:
        dtype = str(attr.dtype)
    except Exception:
        dtype = "?"
    sample = "?"
    try:
        arr_np = attr.numpy() if hasattr(attr, "numpy") else np.asarray(attr)
        n = min(n_show, arr_np.size)
        sample = np.array2string(arr_np.flatten()[:n], precision=4, suppress_small=True)
    except Exception:
        pass
    return shape, dtype, sample


def array_persistence_test(model, name, scale):
    """Write `scale * orig` into the named array, read back, compare.

    Tests only whether the array is mutable in-place via .assign(). Does
    NOT test whether the solver re-reads it -- that's the propagation
    test in main().
    """
    attr = getattr(model, name, None)
    if attr is None:
        return "missing"
    try:
        arr_np = attr.numpy()
        orig = arr_np.copy()
        new = (orig * scale).astype(orig.dtype)
        attr.assign(new)
        after = attr.numpy()
        ok = bool(np.allclose(after, new))
        # Restore original
        attr.assign(orig)
        return "persisted" if ok else "did_not_persist"
    except Exception as e:
        return f"err: {type(e).__name__}: {e}"


def run_trace(env, engine, num_envs, action_dim, device, num_steps,
              probe_joint_local, joints_per_env):
    """Reset env with fixed seed, step N times with zero action, record a
    scalar "dynamics signature" per step.

    Signature = L2 norm of the full dof_vel tensor + a focused per-step
    velocity at the probe joint. Two metrics so a propagation that only
    affects nearby joints is still visible even if the global norm masks it.
    """
    # Deterministic reset
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    obs, info = env.reset()

    zero_action = torch.zeros((num_envs, action_dim), dtype=torch.float32, device=device)

    global_l2 = []
    probe_vel_mean = []
    for _t in range(num_steps):
        obs, rew, done, info = env.step(zero_action)
        # engine.get_dof_vel(char_id=0) -> (num_envs, dofs_per_env)
        try:
            dof_vel = engine.get_dof_vel(0).cpu().numpy()
        except Exception:
            # Fall back to raw state
            dof_vel = engine._sim_state.joint_qd.numpy().reshape(num_envs, -1)
        global_l2.append(float(np.linalg.norm(dof_vel)))
        # Probe-joint column: use the index modulo dofs_per_env if shape allows
        if probe_joint_local < dof_vel.shape[1]:
            probe_vel_mean.append(float(np.mean(dof_vel[:, probe_joint_local])))
        else:
            probe_vel_mean.append(float("nan"))
    return np.array(global_l2), np.array(probe_vel_mean)


def trace_diff(t1, t2):
    """Return (L2 diff, relative diff) between two trace arrays."""
    diff = np.linalg.norm(t1 - t2)
    rel = diff / (np.linalg.norm(t1) + 1e-12)
    return float(diff), float(rel)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--num-envs", type=int, default=256,
                    help="Env count for probe (default 256 -- small for speed)")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out-dir", type=str,
                    default="output_g1_unified/probe_phase0",
                    help="Directory for the modified MJCF + result npz")
    ap.add_argument("--steps", type=int, default=50,
                    help="Timesteps per propagation trace (default 50)")
    ap.add_argument("--probe-body", type=str, default="left_knee_link")
    ap.add_argument("--probe-joint", type=str, default="left_knee_joint")
    ap.add_argument("--scale", type=float, default=10.0,
                    help="Multiplier applied to each probed array (default 10x)")
    ap.add_argument("--rel-threshold", type=float, default=0.01,
                    help="Relative trace diff considered evidence of "
                         "in-place propagation (default 0.01 = 1%%)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 0 NEWTON ARRAY PROBE")
    print("=" * 80)
    print(f"  num_envs={args.num_envs}  device={args.device}")
    print(f"  probe_body={args.probe_body}  probe_joint={args.probe_joint}")
    print(f"  steps={args.steps}  scale={args.scale}x  "
          f"rel-threshold={args.rel_threshold:.3f}")
    print(f"  out_dir={out_dir}")

    # =================================================================
    # 1. Build env (single GPU, theta=0)
    # =================================================================
    print("\n" + "=" * 80)
    print("[1/4] Building G1 env at theta=0 (default morphology)")
    print("=" * 80)

    mjcf_modifier = G1MJCFModifier(str(BASE_MJCF_PATH), scope="lower")
    modified_mjcf = out_dir / "g1_modified.xml"
    modified_env_config = out_dir / "env_config.yaml"
    theta_init = np.zeros(len(mjcf_modifier.groups), dtype=np.float64)
    mjcf_modifier.generate(theta_init, str(modified_mjcf))

    mesh_src = BASE_MJCF_PATH.parent / "meshes"
    mesh_dst = out_dir / "meshes"
    if not mesh_dst.exists():
        try:
            mesh_dst.symlink_to(mesh_src)
        except (OSError, NotImplementedError):
            import shutil
            shutil.copytree(str(mesh_src), str(mesh_dst))

    mjcf_modifier.generate_env_config(
        str(modified_mjcf), str(BASE_ENV_CONFIG), str(modified_env_config)
    )

    if "cuda" in args.device:
        torch.cuda.set_device(args.device)

    env = env_builder.build_env(
        str(modified_env_config), str(BASE_ENGINE_CONFIG),
        args.num_envs, args.device, visualize=False,
    )
    engine = env._engine
    model = engine._sim_model
    print(f"  [OK] env built")
    print(f"  model.body_count = {getattr(model, 'body_count', '?')}")
    print(f"  model.joint_count = {getattr(model, 'joint_count', '?')}")

    bodies_per_env = model.body_count // args.num_envs
    joints_per_env = model.joint_count // args.num_envs
    print(f"  inferred bodies_per_env = {bodies_per_env}")
    print(f"  inferred joints_per_env = {joints_per_env}")

    # Locate probe body / joint by name
    body_keys = list(model.body_key) if hasattr(model, "body_key") else []
    joint_keys = list(model.joint_key) if hasattr(model, "joint_key") else []
    try:
        body_local = body_keys[:bodies_per_env].index(args.probe_body)
    except ValueError:
        print(f"\n[ERROR] body '{args.probe_body}' not found in first env.")
        print(f"Available: {body_keys[:bodies_per_env]}")
        sys.exit(2)
    try:
        joint_local = joint_keys[:joints_per_env].index(args.probe_joint)
    except ValueError:
        print(f"\n[ERROR] joint '{args.probe_joint}' not found in first env.")
        print(f"Available: {joint_keys[:joints_per_env]}")
        sys.exit(2)
    print(f"  '{args.probe_body}'  -> local body idx {body_local}")
    print(f"  '{args.probe_joint}' -> local joint idx {joint_local}")

    # Per-world global indices for the probe body (across all envs)
    body_global_idx = np.array(
        [body_local + e * bodies_per_env for e in range(args.num_envs)],
        dtype=np.int64,
    )
    joint_global_idx = np.array(
        [joint_local + e * joints_per_env for e in range(args.num_envs)],
        dtype=np.int64,
    )

    # =================================================================
    # 2. Array audit
    # =================================================================
    print("\n" + "=" * 80)
    print("[2/4] ARRAY SHAPE + DTYPE AUDIT")
    print("=" * 80)

    audit = [
        # Joint angles (currently used)
        ("joint_X_p", "joint parent xform -- currently used for angle perturbation"),
        ("joint_X_c", "joint child xform"),
        ("joint_axis", "joint axis"),
        # Phase 1 candidates
        ("joint_target_ke", "PD Kp -- Phase 1 (actuator stiffness)"),
        ("joint_target_kd", "PD Kd -- Phase 1 (actuator damping)"),
        ("joint_effort_limit", "torque saturation -- Phase 1 (max torque)"),
        ("joint_armature", "rotor inertia -- Phase 1 (armature)"),
        ("joint_friction", "joint friction (informational)"),
        # Phase 2 candidates
        ("body_mass", "scalar body mass -- Phase 2 (link length scaling)"),
        ("body_inv_mass", "inverse mass"),
        ("body_inertia", "rotational inertia tensor -- Phase 2"),
        ("body_inv_inertia", "inverse inertia tensor"),
        ("body_com", "center of mass position"),
    ]
    for name, comment in audit:
        attr = getattr(model, name, None)
        if attr is None:
            print(f"  {name:<22} MISSING -- {comment}")
            continue
        shape, dtype, sample = fmt_array_sample(attr)
        print(f"  {name:<22} shape={str(shape):>22}  dtype={dtype:>12}  "
              f"sample={sample}  -- {comment}")

    # =================================================================
    # 3. In-place persistence (basic sanity -- does .assign() stick?)
    # =================================================================
    print("\n" + "=" * 80)
    print("[3/4] IN-PLACE WRITE PERSISTENCE")
    print("=" * 80)
    print("Confirms each array is mutable via .assign(). Required prerequisite")
    print("for propagation (Test 4); .assign() must at least persist for the")
    print("propagation question to make sense.\n")

    persistence_results = {}
    for name in [
        "joint_target_ke", "joint_target_kd", "joint_effort_limit",
        "joint_armature",
        "body_mass", "body_inv_mass", "body_inertia", "body_inv_inertia",
    ]:
        status = array_persistence_test(model, name, 2.0)
        persistence_results[name] = status
        flag = "[OK]  " if status == "persisted" else (
            "[FAIL]" if status == "did_not_persist" else "[SKIP]"
        )
        print(f"  {flag} {name:<22} {status}")

    # =================================================================
    # 4. Dynamics propagation -- the critical test
    # =================================================================
    print("\n" + "=" * 80)
    print("[4/4] DYNAMICS PROPAGATION TESTS")
    print("=" * 80)
    print("For each candidate array: snapshot a trace with default values,")
    print("modify the array in-place (scale by {0}x), restart from the same")
    print("seeded reset, snapshot a new trace, compare.\n".format(args.scale))

    action_space = env.get_action_space()
    action_dim = int(np.prod(action_space.shape))
    print(f"  action_dim = {action_dim}  (will use zero actions)")

    # ----- baseline trace -----
    print(f"\n  Trace 0 (baseline): no modification")
    base_g, base_p = run_trace(env, engine, args.num_envs, action_dim,
                               args.device, args.steps,
                               joint_local, joints_per_env)
    print(f"    global ||dof_vel|| trajectory: "
          f"start={base_g[0]:.4f} mid={base_g[len(base_g)//2]:.4f} "
          f"end={base_g[-1]:.4f}")
    print(f"    probe joint qd:                "
          f"start={base_p[0]:+.4f} mid={base_p[len(base_p)//2]:+.4f} "
          f"end={base_p[-1]:+.4f}")

    propagation_results = {}

    def run_one(name, get_attr_fn, scale_fn, restore_fn, label):
        attr = get_attr_fn()
        if attr is None:
            propagation_results[name] = ("missing", None, None)
            print(f"\n  Trace {label}: {name} MISSING on model -- skipped")
            return
        print(f"\n  Trace {label}: scale {name} by {args.scale}x at probe target")
        try:
            scale_fn()
        except Exception as e:
            propagation_results[name] = (f"scale_err: {e}", None, None)
            print(f"    [ERR] could not scale: {e}")
            return
        g, p = run_trace(env, engine, args.num_envs, action_dim,
                         args.device, args.steps,
                         joint_local, joints_per_env)
        restore_fn()
        d_g, r_g = trace_diff(base_g, g)
        d_p, r_p = trace_diff(base_p, p)
        verdict = ("PROPAGATES"
                   if (r_g > args.rel_threshold or r_p > args.rel_threshold)
                   else "DOES NOT PROPAGATE")
        propagation_results[name] = (verdict, r_g, r_p)
        print(f"    global ||dof_vel||: rel diff vs baseline = "
              f"{r_g*100:.3f}% (abs {d_g:.4f})")
        print(f"    probe joint qd:     rel diff vs baseline = "
              f"{r_p*100:.3f}% (abs {d_p:.4f})")
        print(f"    -> {name} {verdict}")

    # Phase 1a: joint_target_ke (Kp)
    def get_ke(): return getattr(model, "joint_target_ke", None)
    ke_orig = get_ke().numpy().copy() if get_ke() is not None else None
    def scale_ke():
        new = ke_orig.copy()
        new[joint_global_idx] *= args.scale
        get_ke().assign(new)
    def restore_ke(): get_ke().assign(ke_orig)
    run_one("joint_target_ke", get_ke, scale_ke, restore_ke, label="1a")

    # Phase 1b: joint_target_kd (Kd)
    def get_kd(): return getattr(model, "joint_target_kd", None)
    kd_orig = get_kd().numpy().copy() if get_kd() is not None else None
    def scale_kd():
        new = kd_orig.copy()
        new[joint_global_idx] *= args.scale
        get_kd().assign(new)
    def restore_kd(): get_kd().assign(kd_orig)
    run_one("joint_target_kd", get_kd, scale_kd, restore_kd, label="1b")

    # Phase 1c: joint_effort_limit
    def get_el(): return getattr(model, "joint_effort_limit", None)
    el_orig = get_el().numpy().copy() if get_el() is not None else None
    def scale_el():
        new = el_orig.copy()
        new[joint_global_idx] *= args.scale
        get_el().assign(new)
    def restore_el(): get_el().assign(el_orig)
    run_one("joint_effort_limit", get_el, scale_el, restore_el, label="1c")

    # Phase 2a: body_mass
    def get_mass(): return getattr(model, "body_mass", None)
    def get_inv_mass(): return getattr(model, "body_inv_mass", None)
    mass_orig = get_mass().numpy().copy() if get_mass() is not None else None
    inv_mass_orig = (get_inv_mass().numpy().copy()
                     if get_inv_mass() is not None else None)
    def scale_mass():
        new = mass_orig.copy()
        new[body_global_idx] *= args.scale
        get_mass().assign(new)
        if get_inv_mass() is not None:
            new_inv = inv_mass_orig.copy()
            new_inv[body_global_idx] /= args.scale
            get_inv_mass().assign(new_inv)
    def restore_mass():
        get_mass().assign(mass_orig)
        if get_inv_mass() is not None:
            get_inv_mass().assign(inv_mass_orig)
    run_one("body_mass", get_mass, scale_mass, restore_mass, label="2a")

    # Phase 2b: body_inertia
    def get_inertia(): return getattr(model, "body_inertia", None)
    def get_inv_inertia(): return getattr(model, "body_inv_inertia", None)
    inertia_orig = (get_inertia().numpy().copy()
                    if get_inertia() is not None else None)
    inv_inertia_orig = (get_inv_inertia().numpy().copy()
                        if get_inv_inertia() is not None else None)
    def scale_inertia():
        new = inertia_orig.copy()
        new[body_global_idx] *= args.scale
        get_inertia().assign(new)
        if get_inv_inertia() is not None:
            new_inv = inv_inertia_orig.copy()
            new_inv[body_global_idx] /= args.scale
            get_inv_inertia().assign(new_inv)
    def restore_inertia():
        get_inertia().assign(inertia_orig)
        if get_inv_inertia() is not None:
            get_inv_inertia().assign(inv_inertia_orig)
    run_one("body_inertia", get_inertia, scale_inertia, restore_inertia,
            label="2b")

    # =================================================================
    # Summary verdict
    # =================================================================
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    def status_str(name):
        p = persistence_results.get(name, "?")
        v = propagation_results.get(name)
        if v is None:
            return f"persistence={p}, propagation=not_tested"
        return f"persistence={p}, propagation={v[0]} (rel_diff={v[1]*100:.2f}% / probe={v[2]*100:.2f}%)" \
            if v[1] is not None else f"persistence={p}, propagation={v[0]}"

    print("\nArray-by-array results:")
    for name in ["joint_target_ke", "joint_target_kd", "joint_effort_limit",
                 "body_mass", "body_inertia"]:
        print(f"  {name:<22} -> {status_str(name)}")

    print()
    # Phase 1 verdict
    phase1_ok = (propagation_results.get("joint_target_ke", (None,))[0]
                 == "PROPAGATES")
    if phase1_ok:
        print("  PHASE 1 (actuator params): FEASIBLE via in-place writes.")
        print("                             Implement stiffness/damping/effort")
        print("                             perturbations like joint_X_p.")
    else:
        ke_v = propagation_results.get("joint_target_ke", (None,))[0]
        print(f"  PHASE 1 (actuator params): NEEDS INVESTIGATION")
        print(f"                             joint_target_ke verdict: {ke_v}")
        print(f"                             may require engine-level work.")

    # Phase 2 verdict
    mass_v = propagation_results.get("body_mass", (None,))[0]
    inertia_v = propagation_results.get("body_inertia", (None,))[0]
    if mass_v == "PROPAGATES" and inertia_v == "PROPAGATES":
        print()
        print("  PHASE 2 (link lengths):    FEASIBLE via in-place writes.")
        print("                             Mass + inertia both propagate.")
        print("                             Implement length perturbation by")
        print("                             coupled writes to joint_X_p (pos),")
        print("                             body_mass, body_inertia.")
    elif mass_v == "PROPAGATES" and inertia_v != "PROPAGATES":
        print()
        print("  PHASE 2 (link lengths):    PARTIAL.")
        print("                             body_mass propagates ({} for inertia)".format(inertia_v))
        print("                             Mass-only scaling is physically")
        print("                             inconsistent. Consider env rebuild")
        print("                             per perturbation, or skip Phase 2.")
    else:
        print()
        print(f"  PHASE 2 (link lengths):    BLOCKED.")
        print(f"                             body_mass verdict: {mass_v}")
        print(f"                             body_inertia verdict: {inertia_v}")
        print(f"                             Likely MuJoCo cached CRB at init.")
        print(f"                             Options:")
        print(f"                               - Rebuild env per perturbation")
        print(f"                                 (slow: ~30s x 163 perts x 30")
        print(f"                                 outer iters = ~41h overhead)")
        print(f"                               - Switch to SolverSemiImplicit")
        print(f"                                 (loses AMP support)")
        print(f"                               - Drop Phase 2, focus on")
        print(f"                                 actuator params only")

    # Save raw results
    npz_path = out_dir / "probe_results.npz"
    save_dict = {
        "baseline_global": base_g,
        "baseline_probe": base_p,
        "scale": args.scale,
        "steps": args.steps,
        "rel_threshold": args.rel_threshold,
        "body_local": body_local,
        "joint_local": joint_local,
    }
    for k, v in propagation_results.items():
        verdict, r_g, r_p = v
        save_dict[f"{k}__verdict"] = verdict
        save_dict[f"{k}__rel_diff_global"] = (r_g if r_g is not None else float("nan"))
        save_dict[f"{k}__rel_diff_probe"] = (r_p if r_p is not None else float("nan"))
    np.savez(str(npz_path), **save_dict)
    print(f"\nResults saved to {npz_path}")


if __name__ == "__main__":
    main()
