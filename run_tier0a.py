#!/usr/bin/env python3
"""Tier 0 + Tier 0a paper-runs launcher: 4 conditions x 3 seeds = 12 runs.

Sequential runs (one at a time on the local node's GPUs). Each run uses
fresh kickoff (no --resume-checkpoint) so the experimental claim is
end-to-end. Bounded by --max-wallclock-min so deterministic CoT eval lands
in <out_dir>/early_stop_eval.npz even if wallclock cap fires first.

The four conditions:
    spsa_full     - HEADLINE: full GBC method (SPSA + Adam + soft-drop ON)
    cmaes         - COMPARISON: derivative-free baseline (constrained CMA-ES)
    spsa_nodrop   - ABLATION:  GBC with soft-drop disabled (isolates the
                               soft-drop contribution from the gradient)
    baseline      - REFERENCE: PPO at theta=0, no co-design at all

Reading the result table:
    spsa_full vs baseline   -> "co-design beats no co-design"  (the claim)
    spsa_full vs cmaes      -> "gradient beats derivative-free" (the novelty)
    spsa_full vs spsa_nodrop-> "soft-drop is/isn't needed"     (the ablation)

Usage (from the codesign_workspace dir):
    python codesign/run_tier0a.py                          # all 12 runs
    python codesign/run_tier0a.py --only-cond spsa_full    # headline only (3 runs)
    python codesign/run_tier0a.py --only-cond spsa_full cmaes  # 6 runs
    python codesign/run_tier0a.py --only-seeds 7           # 4 runs (1 per cond)
    python codesign/run_tier0a.py --max-wallclock-min 240  # shorter per run
    python codesign/run_tier0a.py --dry-run                # preview commands

Per-run output dir: output_g1_unified/tier0a_fresh_<cond>_seed<seed>/
Each contains model.pt, theta_latest.npy, early_stop_eval.npz.

Ctrl-C behavior:
- First Ctrl-C propagates to the running child. Child's SIGINT handler
  creates the STOP file; child runs the deterministic eval and exits 0.
  Launcher then catches the resulting KeyboardInterrupt and stops the
  loop after the current run's eval finishes.
- Second Ctrl-C during eval: hard kill of child + launcher.

To stop a single in-progress run without killing the whole loop:
    touch output_g1_unified/tier0a_fresh_<cond>_seed<S>/STOP
"""

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent  # codesign/
SCRIPT = REPO_DIR / "codesign_g1_unified.py"

# Order matters: spsa_full first so a one-seed run gives the headline number.
CONDITIONS = ["spsa_full", "cmaes", "spsa_nodrop", "baseline"]
DEFAULT_SEEDS = [7, 5, 10]


def cond_flags(cond):
    """Mode-specific CLI args."""
    if cond == "spsa_full":
        # HEADLINE GBC arm: SPSA + Adam outer optimizer + soft-drop ON
        # (--snr-threshold 2.0 default + drop-window 3 default = permanent
        # axis drop kicks in when median SNR over 3 iters < 2.0). This is
        # the actual proposed method.
        return [
            "--mode", "spsa",
            "--spsa-sets", "3",
            "--spsa-epsilon", "0.05",
            "--max-step-deg", "0.5",
            "--design-optimizer", "adam",
            "--adam-lr", "0.05",
            # NOTE: no --ablate-soft-drop -> soft-drop is ON
        ]
    if cond == "cmaes":
        # Constrained CMA-ES (sigma0=0.0087 rad ~0.5deg, maxsigma=0.0175 rad
        # ~1deg per axis -- per-script defaults). Reuses --num-spsa-seeds for
        # paired-seed averaging on each candidate.
        return ["--mode", "cmaes"]
    if cond == "spsa_nodrop":
        # ABLATION arm: same as spsa_full but --ablate-soft-drop disables
        # permanent axis dropping (per-iter SNR mask still active). Tells us
        # whether the gain comes from the gradient signal or from soft-drop.
        return [
            "--mode", "spsa",
            "--spsa-sets", "3",
            "--spsa-epsilon", "0.05",
            "--max-step-deg", "0.5",
            "--design-optimizer", "adam",
            "--adam-lr", "0.05",
            "--ablate-soft-drop",
        ]
    if cond == "baseline":
        # REFERENCE: --baseline exits after the inner loop terminates at
        # theta=0. Outer loop never fires. Setting the plateau threshold
        # absurdly small (1e-7) prevents plateau-based exit so inner PPO
        # trains until the wallclock cap fires (--max-wallclock-min). This
        # is the compute-matched comparator for SPSA -- baseline gets the
        # same wallclock budget for inner training as the SPSA arm. The
        # graceful early-stop machinery then runs the deterministic eval
        # at theta=0 with paired seeds and saves early_stop_eval.npz.
        # SPSA flags below are ignored because the outer loop never runs.
        return [
            "--mode", "spsa",
            "--baseline",
            "--inner-cot-plateau-threshold", "0.0000001",
        ]
    raise ValueError(f"Unknown cond: {cond}")


def common_flags(seed, max_inner_iters, max_wallclock_min, out_dir):
    """Paper-grade SPSA config, shared by both arms."""
    return [
        "--seed", str(seed),
        "--design-scope", "full",
        "--disc-morph-invariant",
        "--num-train-envs", "32768",
        "--vel-cmd", "1",
        "--vel-tracking-sigma", "0.25",
        "--vel-reward-weight", "5",
        "--outer-task-weight", "0.5",
        "--outer-disc-weight", "0.5",
        "--outer-term-penalty", "-10",
        "--num-spsa-seeds", "30",
        "--outer-iters", "100",
        "--inner-cot-plateau-threshold", "0.002",
        "--max-inner-iters", str(max_inner_iters),
        "--max-wallclock-min", str(max_wallclock_min),
        "--out-dir", str(out_dir),
    ]


def run_one(cond, seed, max_inner_iters, max_wallclock_min, dry_run):
    out_dir = Path("output_g1_unified") / f"tier0a_fresh_{cond}_seed{seed}"
    # common_flags first, cond_flags last so cond can override common values
    # (e.g. baseline overrides --inner-cot-plateau-threshold to disable
    # plateau-based exit). argparse keeps the last value for duplicate flags.
    cmd = (
        ["python", str(SCRIPT)]
        + common_flags(seed, max_inner_iters, max_wallclock_min, out_dir)
        + cond_flags(cond)
    )
    print()
    print("=" * 70)
    print(f"=== {cond} seed={seed} -- start "
          f"{datetime.datetime.now().isoformat(timespec='seconds')}")
    print(f"=== out_dir: {out_dir}")
    print("=" * 70)
    print("Command:")
    print("  " + " ".join(cmd))
    print()
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    rc = proc.returncode
    print()
    print(f"=== {cond} seed={seed} -- exit {rc} at "
          f"{datetime.datetime.now().isoformat(timespec='seconds')}")
    return rc


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--only-cond", nargs="+", default=CONDITIONS,
                    choices=CONDITIONS,
                    help=f"Subset of conditions (default: {CONDITIONS})")
    ap.add_argument("--only-seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                    help=f"Subset of seeds (default: {DEFAULT_SEEDS})")
    ap.add_argument("--max-inner-iters", type=int, default=40000,
                    help="Per-run max inner iters (default: 40000 -- "
                         "essentially uncapped; plateau detector + wallclock "
                         "cap own termination)")
    ap.add_argument("--max-wallclock-min", type=float, default=820,
                    help="Per-run wallclock cap in minutes (default: 820 = "
                         "13h40m, leaving ~20m for graceful eval inside a "
                         "14h idev/SLURM allocation)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing")
    args = ap.parse_args()

    runs = [(c, s) for c in args.only_cond for s in args.only_seeds]
    print(f"\nLaunching {len(runs)} runs sequentially:")
    for c, s in runs:
        print(f"  - {c} seed={s}")
    print(f"Per-run config: max_inner_iters={args.max_inner_iters}, "
          f"max_wallclock_min={args.max_wallclock_min}")
    print(f"Estimated total wallclock: "
          f"~{len(runs) * args.max_wallclock_min / 60:.1f} hr "
          f"(if every run hits the cap)")

    failures = []
    try:
        for cond, seed in runs:
            rc = run_one(cond, seed, args.max_inner_iters,
                         args.max_wallclock_min, args.dry_run)
            if rc != 0:
                failures.append((cond, seed, rc))
    except KeyboardInterrupt:
        print("\n[INTERRUPT] User Ctrl-C -- stopping loop after current run.")
        sys.exit(130)

    print()
    print("=" * 70)
    if failures:
        print(f"DONE with {len(failures)} failure(s):")
        for cond, seed, rc in failures:
            print(f"  - {cond} seed={seed}: exit {rc}")
        sys.exit(1)
    print(f"DONE: all {len(runs)} runs completed.")


if __name__ == "__main__":
    main()
