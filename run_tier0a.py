#!/usr/bin/env python3
"""Tier 0a launcher: CMA-ES + SPSA-no-softdrop arms, 3 seeds each.

Sequential runs (one at a time on the local node's GPUs). Each run uses
fresh kickoff (no --resume-checkpoint) so the experimental claim is
end-to-end. Bounded by --max-wallclock-min so deterministic CoT eval lands
in <out_dir>/early_stop_eval.npz even if wallclock cap fires before the
outer loop completes all iters.

Usage (from the codesign_workspace dir):
    python codesign/run_tier0a.py                       # all 6 runs (default)
    python codesign/run_tier0a.py --only-cond cmaes     # 3 cmaes runs
    python codesign/run_tier0a.py --only-seeds 7 5      # 4 runs (2 cond x 2 seeds)
    python codesign/run_tier0a.py --max-wallclock-min 240
    python codesign/run_tier0a.py --dry-run             # print commands, run nothing

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

CONDITIONS = ["cmaes", "spsa_nodrop"]
DEFAULT_SEEDS = [7, 5, 10]


def cond_flags(cond):
    """Mode-specific CLI args."""
    if cond == "cmaes":
        # Constrained CMA-ES (sigma0=0.0087 rad ~0.5deg, maxsigma=0.0175 rad
        # ~1deg per axis -- per-script defaults). Reuses --num-spsa-seeds for
        # paired-seed averaging on each candidate.
        return ["--mode", "cmaes"]
    if cond == "spsa_nodrop":
        # Adam optimizer (winner of BFGS vs Adam A/B). --ablate-soft-drop
        # disables permanent axis dropping but keeps per-iter SNR mask, so
        # the gain is attributable to the gradient signal, not the drop
        # mechanism.
        return [
            "--mode", "spsa",
            "--spsa-sets", "3",
            "--spsa-epsilon", "0.05",
            "--max-step-deg", "0.5",
            "--design-optimizer", "adam",
            "--adam-lr", "0.05",
            "--ablate-soft-drop",
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
    cmd = (
        ["python", str(SCRIPT)]
        + cond_flags(cond)
        + common_flags(seed, max_inner_iters, max_wallclock_min, out_dir)
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
    ap.add_argument("--max-wallclock-min", type=float, default=460,
                    help="Per-run wallclock cap in minutes (default: 460 = "
                         "7h40m, leaving ~20m for graceful eval)")
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
