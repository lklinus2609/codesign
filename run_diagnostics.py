#!/usr/bin/env python3
"""GBC diagnostic suite -- one-shot launcher for failure-mode triage.

The current SPSA setup is failing: discovered morphology is genuinely worse
than theta=0 under fresh-PPO validation. Six candidate failure modes:
  F1  flat landscape (no gradient exists)
  F2  bilevel non-stationarity (fixed-policy and optimal-policy landscapes
      differ -- SPSA optimizes the wrong landscape)
  F3  SPSA gradient too noisy
  F4  trust region miscalibrated (collapses too aggressively)
  F5  AMP discriminator collapse during SPSA (confirmed already)
  F6  paired-seed eval too noisy

This script runs three tests in sequence to discriminate among F1-F4/F6:

  TEST A  eval-only single-axis sweep at hip_pitch (~30 min)
          5 paired-seed evaluations with the baseline-theta=0 policy at
          theta in {-10, -5, 0, +5, +10} degrees on the hip_pitch axis.
          Distinguishes F1 from "there is gradient" cases.

  TEST C  SPSA gradient cosine similarity at theta=0 (~10 min)
          Two independent SPSA gradient computations at theta=0 with
          different base seeds. Cosine similarity tells us whether the
          gradient direction is signal (>0.7), partial signal (0.3-0.7),
          or noise (<0.3). Distinguishes F3 from F4.

  TEST B  fresh-PPO single-axis sweep at hip_pitch (~12h, conditional)
          3 fresh-PPO runs at theta in {-5, 0, +5} degrees. Compares the
          optimal-policy CoT curve to TEST A's fixed-policy curve.
          Curves match -> envelope theorem holds, optimization is at fault.
          Curves differ -> F2 confirmed, problem is bilevel itself.
          Skipped if TEST A shows flat landscape (F1).

Outputs in <out_root>/:
  test_a_<deg>/early_stop_eval.npz        TEST A eval results
  test_c_seed_<seed>/diagnostic_gradient.npz  TEST C gradient samples
  test_b_<deg>/early_stop_eval.npz        TEST B fresh-PPO results
  diagnostic_report.txt                   text summary + decision

Usage (from codesign_workspace dir):
  python codesign/run_diagnostics.py
  python codesign/run_diagnostics.py --baseline-ckpt /path/to/baseline_model.pt
  python codesign/run_diagnostics.py --skip-test-b              # tests A+C only
  python codesign/run_diagnostics.py --axis 3                   # sweep knee instead
  python codesign/run_diagnostics.py --dry-run                  # preview, no runs
  python codesign/run_diagnostics.py --test-b-wallclock-min 240 # 4h fresh PPO

Auto-fetches baseline checkpoint from wandb run pti1yxba if --baseline-ckpt
not given (or downloads to baseline_model.pt in cwd).
"""

import argparse
import datetime
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent  # codesign/
SCRIPT = REPO_DIR / "codesign_g1_unified.py"

NUM_DESIGN_PARAMS = 16  # full-scope: 6 leg + 3 waist + 7 arm

# Default axis = 0 = left_hip_pitch (heaviest joint, most CoT-relevant)
DEFAULT_AXIS = 0
TEST_A_DEGREES = [-10, -5, 0, 5, 10]
TEST_B_DEGREES = [-5, 0, 5]


def make_theta_file(axis_idx, deg, path):
    theta = np.zeros(NUM_DESIGN_PARAMS, dtype=np.float64)
    theta[axis_idx] = np.radians(deg)
    np.save(str(path), theta)


def common_flags(seed, num_train_envs):
    """Paper-grade base flags shared across all diagnostic subprocess calls."""
    return [
        "--seed", str(seed),
        "--design-scope", "full",
        "--disc-morph-invariant",
        "--num-train-envs", str(num_train_envs),
        "--vel-cmd", "1",
        "--vel-tracking-sigma", "0.25",
        "--vel-reward-weight", "5",
        "--outer-task-weight", "0.5",
        "--outer-disc-weight", "0.5",
        "--outer-term-penalty", "-10",
        "--num-spsa-seeds", "30",
        "--eval-horizon", "300",
    ]


def run_subprocess(cmd, label, dry_run):
    print()
    print("=" * 80)
    print(f"=== {label} -- start {datetime.datetime.now().isoformat(timespec='seconds')}")
    print("=" * 80)
    print("Command: " + " ".join(cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    rc = proc.returncode
    print(f"=== {label} -- exit {rc} at {datetime.datetime.now().isoformat(timespec='seconds')}")
    return rc


def fetch_baseline_checkpoint(out_path):
    """Download the baseline-θ=0 checkpoint from wandb run pti1yxba."""
    import wandb
    api = wandb.Api()
    r = api.run("lklinus0926-the-university-of-texas-at-austin/gbc-codesign/pti1yxba")
    # The final model.pt lives in the iter outer_000 directory of the baseline run
    candidates = [f for f in r.files() if f.name.endswith("model.pt")]
    if not candidates:
        raise RuntimeError("No model.pt found in baseline run pti1yxba")
    # Pick the most-recent model.pt (the final one)
    candidates.sort(key=lambda f: f.updated_at, reverse=True)
    f = candidates[0]
    print(f"Downloading {f.name} from baseline run pti1yxba ({f.size/1e6:.1f} MB)...")
    f.download(replace=True)
    Path(f.name).rename(out_path)
    print(f"Saved {out_path}")


def test_a_analyze(out_root, degrees):
    """Read TEST A npz files, return CoT curve and F1 verdict."""
    cots = []
    rewards = []
    for deg in degrees:
        npz = out_root / f"test_a_{deg:+d}" / "early_stop_eval.npz"
        if not npz.exists():
            cots.append(None)
            rewards.append(None)
            continue
        d = np.load(str(npz))
        cots.append(float(d["cot"]))
        rewards.append(float(d["center_reward"]))
    return cots, rewards


def test_c_analyze(out_root, seeds):
    """Read TEST C gradient npz files, return cosine similarity."""
    grads = []
    for s in seeds:
        npz = out_root / f"test_c_seed_{s}" / "diagnostic_gradient.npz"
        if not npz.exists():
            return None, "missing"
        d = np.load(str(npz))
        grads.append(d["grad"])
    g1, g2 = grads
    n1, n2 = np.linalg.norm(g1), np.linalg.norm(g2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0, "zero gradient"
    cos = float((g1 @ g2) / (n1 * n2))
    return cos, "ok"


def write_report(out_root, axis_idx, axis_name,
                 a_degrees, a_cots, a_rewards,
                 c_cos, c_status,
                 b_degrees, b_cots, b_rewards,
                 skipped_b):
    """Write decision tree analysis to diagnostic_report.txt."""
    report = []
    report.append("=" * 80)
    report.append("GBC DIAGNOSTIC REPORT")
    report.append(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    report.append(f"Sweep axis: theta[{axis_idx}] = {axis_name}")
    report.append("=" * 80)

    # TEST A
    report.append("\nTEST A -- eval-only sweep (fixed-policy CoT vs theta):")
    report.append(f"  {'theta_deg':>10} {'CoT':>10} {'reward':>10}")
    for d, c, r in zip(a_degrees, a_cots, a_rewards):
        cot_s = f"{c:.5f}" if c is not None else "missing"
        rwd_s = f"{r:+.5f}" if r is not None else "missing"
        report.append(f"  {d:>+10d} {cot_s:>10} {rwd_s:>10}")
    valid_cots = [c for c in a_cots if c is not None]
    a_verdict = "unknown"
    if valid_cots:
        cot_min, cot_max = min(valid_cots), max(valid_cots)
        cot_range = cot_max - cot_min
        cot_range_rel = cot_range / np.mean(valid_cots) if np.mean(valid_cots) != 0 else 0
        report.append(f"  CoT range: {cot_range:.5f} ({cot_range_rel*100:.2f}% of mean)")
        zero_idx = a_degrees.index(0) if 0 in a_degrees else None
        if zero_idx is not None and a_cots[zero_idx] is not None:
            cot_at_zero = a_cots[zero_idx]
            if abs(cot_at_zero - cot_min) < 1e-6:
                report.append(f"  -> theta=0 is the MINIMUM in this sweep")
                if cot_range_rel < 0.01:
                    a_verdict = "F1_FLAT"
                    report.append(f"  -> F1 (FLAT landscape): range < 1% of mean.")
                    report.append(f"     Joint angles don't have exploitable gradient on this axis.")
                else:
                    a_verdict = "F1_LOCAL_MIN"
                    report.append(f"  -> theta=0 is a local minimum (range > 1%, but no better point found)")
            else:
                a_verdict = "STRUCTURE"
                min_deg = a_degrees[valid_cots.index(cot_min) if None not in a_cots else
                                    [i for i, c in enumerate(a_cots) if c == cot_min][0]]
                report.append(f"  -> theta={min_deg:+d} deg gives lower CoT than theta=0")
                report.append(f"     Landscape has structure; SPSA should find it.")

    # TEST C
    report.append("\nTEST C -- SPSA gradient cosine similarity:")
    if c_status != "ok":
        report.append(f"  status: {c_status}")
        c_verdict = "skipped"
    else:
        report.append(f"  cos(g1, g2) = {c_cos:.4f}")
        if c_cos > 0.7:
            c_verdict = "F4"
            report.append(f"  -> gradient direction is STABLE (>0.7). F4 trust-region miscalibration likely.")
            report.append(f"     Fix: --max-step-deg 5.0 and disable adaptive shrink, OR switch to CMA-ES.")
        elif c_cos > 0.3:
            c_verdict = "F3_MILD"
            report.append(f"  -> gradient is PARTIAL SIGNAL (0.3-0.7). F3 noise dominates.")
            report.append(f"     Fix: bump --num-spsa-seeds 60 or --spsa-sets 6.")
        else:
            c_verdict = "F3_SEVERE"
            report.append(f"  -> gradient is NOISE (<0.3). F3 severe.")
            report.append(f"     Fix: much more variance reduction or different gradient estimator.")

    # TEST B
    report.append("\nTEST B -- fresh-PPO sweep (optimal-policy CoT vs theta):")
    if skipped_b:
        report.append("  (skipped -- TEST A showed F1)")
        b_verdict = "skipped"
    else:
        report.append(f"  {'theta_deg':>10} {'CoT (opt-pi)':>14} {'CoT (fixed-pi)':>16}")
        valid_b = [c for c in b_cots if c is not None]
        for d, b_c in zip(b_degrees, b_cots):
            if d in a_degrees:
                a_c = a_cots[a_degrees.index(d)]
            else:
                a_c = None
            b_s = f"{b_c:.5f}" if b_c is not None else "missing"
            a_s = f"{a_c:.5f}" if a_c is not None else "-"
            report.append(f"  {d:>+10d} {b_s:>14} {a_s:>16}")
        # Compare curve shapes
        b_verdict = "STRUCTURE_OK"
        if all(c is not None for c in b_cots):
            b_min_deg = b_degrees[b_cots.index(min(b_cots))]
            report.append(f"  -> theta={b_min_deg:+d} deg has lowest CoT with fresh PPO")
            if b_min_deg != 0:
                report.append(f"     Fresh-PPO landscape has a better point than theta=0.")
            # Compare to TEST A
            a_curve = [a_cots[a_degrees.index(d)] for d in b_degrees if d in a_degrees]
            if all(c is not None for c in a_curve) and len(a_curve) == len(b_cots):
                # Check whether curves have same shape (correlate)
                a_arr = np.array(a_curve)
                b_arr = np.array(b_cots)
                if a_arr.std() > 1e-8 and b_arr.std() > 1e-8:
                    corr = float(np.corrcoef(a_arr, b_arr)[0, 1])
                    report.append(f"  Pearson correlation(fixed-pi, opt-pi) over sweep = {corr:.3f}")
                    if corr > 0.7:
                        b_verdict = "ENVELOPE_OK"
                        report.append(f"  -> envelope theorem holds; SPSA-style gradient is valid.")
                    else:
                        b_verdict = "F2"
                        report.append(f"  -> F2 confirmed: fixed-pi landscape != opt-pi landscape.")
                        report.append(f"     SPSA optimizes the wrong objective. Major reformulation needed.")

    # Overall verdict
    report.append("\n" + "=" * 80)
    report.append("OVERALL VERDICT")
    report.append("=" * 80)
    if a_verdict == "F1_FLAT":
        report.append("PRIMARY: F1 -- joint-angle morphology landscape is flat on this axis.")
        report.append("ACTION:  Pivot to a richer design space (link lengths, mass, actuator")
        report.append("         params), OR reframe paper as 'G1 is already locally optimal'.")
    elif a_verdict == "F1_LOCAL_MIN":
        report.append("PRIMARY: F1 -- theta=0 is a local min; no improvement on this axis.")
        report.append("ACTION:  Sweep other axes; if all show local min, pivot design space.")
    elif b_verdict == "F2":
        report.append("PRIMARY: F2 -- fixed-policy and optimal-policy landscapes differ.")
        report.append("ACTION:  Bilevel optimization in this formulation is fundamentally limited.")
        report.append("         Consider meta-policy or concurrent training approach.")
    elif c_verdict.startswith("F3"):
        report.append(f"PRIMARY: {c_verdict} -- SPSA gradient is too noisy.")
        report.append("ACTION:  Bump --num-spsa-seeds / --spsa-sets, or expand --num-train-envs,")
        report.append("         or switch to a less noise-prone estimator.")
    elif c_verdict == "F4":
        report.append("PRIMARY: F4 -- trust-region is mis-calibrated.")
        report.append("ACTION:  Try --max-step-deg 5.0 with --ablate-trust-region; if that")
        report.append("         improves, replace adaptive trust region with fixed bigger steps.")
    else:
        report.append("Verdict inconclusive. See per-test sections above.")

    report.append("\nSee per-test sections above for details.")
    report.append("=" * 80)

    report_path = out_root / "diagnostic_report.txt"
    report_path.write_text("\n".join(report))
    print()
    print("\n".join(report))
    print(f"\nReport saved to {report_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--baseline-ckpt", type=str, default=None,
                    help="Path to a baseline-theta=0 checkpoint. If not given, "
                         "auto-downloads from wandb run pti1yxba into "
                         "baseline_model.pt in cwd.")
    ap.add_argument("--out-root", type=str,
                    default="output_g1_unified/diagnostics",
                    help="Root output dir for all diagnostic subdirs. "
                         "Default: output_g1_unified/diagnostics")
    ap.add_argument("--axis", type=int, default=DEFAULT_AXIS,
                    help=f"Design-param axis to sweep (default {DEFAULT_AXIS} = "
                         f"left_hip_pitch). 0=hip_pitch, 1=hip_roll, 2=hip_yaw, "
                         f"3=knee, 4=ankle_pitch, 5=ankle_roll, 6=waist_yaw, "
                         f"7=waist_roll, 8=torso, 9-15=arm joints")
    ap.add_argument("--num-train-envs", type=int, default=32768,
                    help="Total training envs (must be divisible by GPU count). "
                         "Default 32768 matches the Tier 0a runs.")
    ap.add_argument("--test-b-wallclock-min", type=float, default=240,
                    help="Per-run wallclock cap for TEST B fresh-PPO runs in "
                         "minutes. Default 240 (4h). Smaller -> 3 runs fit in "
                         "fewer hours but PPO is less converged. Set to 820 "
                         "to fully match baseline conditions (will need ~42h "
                         "wallclock to run all 3 TEST B sweeps).")
    ap.add_argument("--skip-test-a", action="store_true",
                    help="Skip TEST A (eval-only sweep).")
    ap.add_argument("--skip-test-c", action="store_true",
                    help="Skip TEST C (gradient cosine similarity).")
    ap.add_argument("--skip-test-b", action="store_true",
                    help="Skip TEST B (fresh-PPO sweep). Useful for fast "
                         "TEST A + TEST C check (~40 min total) before "
                         "committing to the 12h TEST B.")
    ap.add_argument("--force-test-b", action="store_true",
                    help="Run TEST B even if TEST A shows F1 (flat). Useful "
                         "if you want the full picture regardless.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing.")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Axis name for the report
    axis_names = [
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
        "left_ankle_pitch", "left_ankle_roll",
        "waist_yaw", "waist_roll", "torso",
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
        "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    ]
    axis_name = axis_names[args.axis] if 0 <= args.axis < len(axis_names) else f"axis_{args.axis}"

    # ----- Resolve baseline checkpoint -----
    if args.baseline_ckpt is None:
        ckpt = Path("baseline_model.pt")
        if not ckpt.exists() and not args.dry_run:
            print(f"No --baseline-ckpt given; fetching from wandb run pti1yxba...")
            try:
                fetch_baseline_checkpoint(ckpt)
            except Exception as e:
                print(f"[ERROR] Could not fetch baseline checkpoint: {e}")
                print(f"        Pass --baseline-ckpt /path/to/model.pt explicitly.")
                sys.exit(2)
    else:
        ckpt = Path(args.baseline_ckpt).resolve()
        if not ckpt.exists() and not args.dry_run:
            print(f"[ERROR] --baseline-ckpt does not exist: {ckpt}")
            sys.exit(2)

    print(f"\nDiagnostic suite:")
    print(f"  axis           {args.axis} = {axis_name}")
    print(f"  baseline ckpt  {ckpt}")
    print(f"  out_root       {out_root}")
    print(f"  num_train_envs {args.num_train_envs}")
    print()

    # =========================================================
    # TEST A: eval-only sweep (5 evals, ~25 min)
    # =========================================================
    if not args.skip_test_a:
        for i, deg in enumerate(TEST_A_DEGREES):
            sub = out_root / f"test_a_{deg:+d}"
            sub.mkdir(parents=True, exist_ok=True)
            theta_file = sub / "theta.npy"
            if not args.dry_run:
                make_theta_file(args.axis, deg, theta_file)
            cmd = (
                ["python", str(SCRIPT), "--mode", "spsa",
                 "--eval-only",
                 "--init-theta", str(theta_file),
                 "--resume-checkpoint", str(ckpt),
                 "--out-dir", str(sub)]
                + common_flags(seed=7, num_train_envs=args.num_train_envs)
            )
            rc = run_subprocess(cmd, f"TEST A axis={axis_name} theta={deg:+d}deg", args.dry_run)
            if rc != 0:
                print(f"[WARN] TEST A theta={deg:+d} exited {rc}; continuing")
    # Always try to read existing TEST A results (lets a follow-up run that
    # passes --skip-test-a still include this test in the final report).
    a_cots, a_rewards = test_a_analyze(out_root, TEST_A_DEGREES) if not args.dry_run else ([None]*5, [None]*5)

    # =========================================================
    # TEST C: SPSA gradient cosine similarity (2 runs, ~10 min)
    # =========================================================
    c_seeds = [101, 202]   # arbitrary distinct seeds for independent estimates
    if not args.skip_test_c:
        for s in c_seeds:
            sub = out_root / f"test_c_seed_{s}"
            sub.mkdir(parents=True, exist_ok=True)
            cmd = (
                ["python", str(SCRIPT), "--mode", "spsa",
                 "--diagnostic-gradient",
                 "--resume-checkpoint", str(ckpt),
                 "--seed", str(s),
                 "--spsa-sets", "3", "--spsa-epsilon", "0.05",
                 "--out-dir", str(sub)]
                + [f for f in common_flags(seed=s, num_train_envs=args.num_train_envs)
                   if f not in ("--seed", str(s))]
            )
            rc = run_subprocess(cmd, f"TEST C gradient seed={s}", args.dry_run)
            if rc != 0:
                print(f"[WARN] TEST C seed={s} exited {rc}; continuing")
    # Always try to read existing TEST C gradients (lets follow-up runs that
    # pass --skip-test-c still include cosine similarity in the final report).
    if not args.dry_run:
        c_cos, c_status = test_c_analyze(out_root, c_seeds)
    elif args.skip_test_c:
        c_cos, c_status = None, "skipped"
    else:
        c_cos, c_status = None, "dry-run"

    # =========================================================
    # TEST B: fresh-PPO sweep (conditional, 3 runs, ~12h)
    # =========================================================
    skipped_b = args.skip_test_b
    if not args.skip_test_b and not args.dry_run:
        # Auto-skip if TEST A clearly showed F1
        if not args.force_test_b:
            valid = [c for c in a_cots if c is not None]
            if valid:
                cot_range_rel = (max(valid) - min(valid)) / (abs(np.mean(valid)) + 1e-8)
                if cot_range_rel < 0.01:
                    print(f"\n[INFO] TEST A range = {cot_range_rel*100:.2f}% of mean -- F1 flat. "
                          f"Skipping TEST B (use --force-test-b to override).")
                    skipped_b = True
    if not skipped_b:
        for deg in TEST_B_DEGREES:
            sub = out_root / f"test_b_{deg:+d}"
            sub.mkdir(parents=True, exist_ok=True)
            theta_file = sub / "theta.npy"
            if not args.dry_run:
                make_theta_file(args.axis, deg, theta_file)
            cmd = (
                ["python", str(SCRIPT), "--mode", "spsa", "--baseline",
                 "--init-theta", str(theta_file),
                 "--inner-cot-plateau-threshold", "0.0000001",
                 "--max-inner-iters", "40000",
                 "--max-wallclock-min", str(args.test_b_wallclock_min),
                 "--out-dir", str(sub)]
                + common_flags(seed=7, num_train_envs=args.num_train_envs)
            )
            rc = run_subprocess(cmd, f"TEST B axis={axis_name} theta={deg:+d}deg (fresh PPO)", args.dry_run)
            if rc != 0:
                print(f"[WARN] TEST B theta={deg:+d} exited {rc}; continuing")
    # Always try to read existing TEST B results (so follow-up runs that
    # pass --skip-test-b can still include B in the report if it ran prior).
    b_cots, b_rewards = [], []
    if not args.dry_run:
        for deg in TEST_B_DEGREES:
            npz = out_root / f"test_b_{deg:+d}" / "early_stop_eval.npz"
            if npz.exists():
                d = np.load(str(npz))
                b_cots.append(float(d["cot"]))
                b_rewards.append(float(d["center_reward"]))
            else:
                b_cots.append(None)
                b_rewards.append(None)
    else:
        b_cots = [None] * len(TEST_B_DEGREES)
        b_rewards = [None] * len(TEST_B_DEGREES)
    # Flag whether B is genuinely absent (so the report says "skipped" only
    # if no B data exists on disk).
    skipped_b = skipped_b and all(c is None for c in b_cots)

    # =========================================================
    # Report
    # =========================================================
    if not args.dry_run:
        write_report(out_root, args.axis, axis_name,
                     TEST_A_DEGREES, a_cots, a_rewards,
                     c_cos, c_status,
                     TEST_B_DEGREES, b_cots, b_rewards,
                     skipped_b)
    else:
        print("\n[dry-run] would write diagnostic_report.txt to", out_root)


if __name__ == "__main__":
    main()
