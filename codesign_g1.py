#!/usr/bin/env python3
"""
Level 3: PGHC Co-Design for G1 Humanoid with Backprop Design Gradients

CPU-only orchestrator. Never imports warp/newton/torch — all GPU work is
delegated to subprocesses to avoid CUDA context contention.

Architecture:
    1. Generate modified G1 MJCF from current theta (CPU, g1_mjcf_modifier)
    2. Subprocess: MimicKit AMP training → checkpoint (GPU, exits → GPU freed)
    3. Subprocess: g1_eval_worker.py — collect actions + BPTT gradient (GPU, exits → GPU freed)
    4. Parent reads eval_result.npz, Adam update on theta, project to ±30° bounds

Only one CUDA context exists at a time. Each subprocess exits before the next starts.

Run:
    python codesign_g1.py --wandb --num-train-envs 4096 --num-eval-worlds 8 --eval-horizon 100
"""

import os
import argparse
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from g1_mjcf_modifier import G1MJCFModifier, SYMMETRIC_PAIRS, NUM_DESIGN_PARAMS

# ---------------------------------------------------------------------------
# Paths (relative to codesign/ directory)
# ---------------------------------------------------------------------------
CODESIGN_DIR = Path(__file__).parent.resolve()
MIMICKIT_DIR = (CODESIGN_DIR / ".." / "MimicKit").resolve()

BASE_MJCF_PATH = MIMICKIT_DIR / "data" / "assets" / "g1" / "g1.xml"
BASE_ENV_CONFIG = MIMICKIT_DIR / "data" / "envs" / "amp_g1_env.yaml"
BASE_AGENT_CONFIG = MIMICKIT_DIR / "data" / "agents" / "amp_g1_agent.yaml"
BASE_ENGINE_CONFIG = MIMICKIT_DIR / "data" / "engines" / "newton_engine.yaml"

# G1 structural constants (with hinge joints — before ball conversion)
# Actual values are determined at runtime from the Newton model since
# convert_3d_hinge_to_ball_joints=True changes the DOF layout.
# Original: 31 bodies, 29 actuated DOFs, 36 qpos, 35 qvel


# ---------------------------------------------------------------------------
# MimicKit Inner Loop (subprocess)
# ---------------------------------------------------------------------------

class MimicKitInnerLoop:
    """Runs MimicKit AMP training as a subprocess with convergence detection."""

    # Regex to parse "| Test_Return  |     123.456 |" from MimicKit log tables
    _RETURN_RE = re.compile(r"\|\s*Test_Return\s*\|\s*([0-9eE.+-]+)\s*\|")

    def __init__(self, mimickit_dir, num_envs=4096,
                 plateau_threshold=0.02, plateau_window=5, min_outputs=10):
        """
        Args:
            mimickit_dir: path to MimicKit root
            plateau_threshold: relative change threshold for convergence (0.02 = 2%)
            plateau_window: number of consecutive log outputs to check for plateau
            min_outputs: minimum log outputs before allowing early stop
        """
        self.mimickit_dir = Path(mimickit_dir)
        self.mimickit_src = self.mimickit_dir / "mimickit"
        self.plateau_threshold = plateau_threshold
        self.plateau_window = plateau_window
        self.min_outputs = min_outputs

    def _check_plateau(self, returns):
        """Check if recent returns have plateaued.

        Returns True if the max relative change over the last plateau_window
        outputs is below plateau_threshold.
        """
        if len(returns) < max(self.plateau_window, self.min_outputs):
            return False

        recent = returns[-self.plateau_window:]
        mean_val = np.mean(recent)
        if abs(mean_val) < 1e-8:
            return False

        spread = max(recent) - min(recent)
        rel_change = spread / abs(mean_val)
        return rel_change < self.plateau_threshold

    def run(self, mjcf_modifier, theta_np, out_dir, num_envs=4096,
            max_samples=5_000_000, resume_from=None, logger_type="wandb"):
        """Run MimicKit AMP training with modified G1 morphology.

        Streams subprocess stdout, parses Test_Return, and terminates early
        when the reward plateaus (relative change < threshold over window).

        Args:
            mjcf_modifier: G1MJCFModifier instance
            theta_np: design parameters (6,)
            out_dir: output directory for this outer iteration
            num_envs: number of parallel training environments
            max_samples: max samples safety cap
            resume_from: checkpoint to resume from (speeds up convergence)
            logger_type: "wandb" or "tb"

        Returns:
            Path to the trained checkpoint (model.pt)
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate modified MJCF
        modified_mjcf = out_dir / "g1_modified.xml"
        mjcf_modifier.generate(theta_np, str(modified_mjcf))

        # Copy mesh directory reference — MJCF uses relative paths to meshes/
        # The modified XML needs to be in the same dir as the base or we symlink
        mesh_src = BASE_MJCF_PATH.parent / "meshes"
        mesh_dst = out_dir / "meshes"
        if not mesh_dst.exists():
            # Use symlink on Unix, copy on Windows
            try:
                mesh_dst.symlink_to(mesh_src)
            except (OSError, NotImplementedError):
                shutil.copytree(str(mesh_src), str(mesh_dst))

        # 2. Generate modified env config pointing to new MJCF
        modified_env_config = out_dir / "env_config.yaml"
        mjcf_modifier.generate_env_config(
            str(modified_mjcf), str(BASE_ENV_CONFIG), str(modified_env_config)
        )

        # 3. Build command
        cmd = [
            sys.executable,
            str(self.mimickit_src / "run.py"),
            "--engine_config", str(BASE_ENGINE_CONFIG),
            "--env_config", str(modified_env_config),
            "--agent_config", str(BASE_AGENT_CONFIG),
            "--num_envs", str(num_envs),
            "--max_samples", str(max_samples),
            "--out_dir", str(out_dir / "training"),
            "--logger", logger_type,
        ]

        if resume_from is not None and Path(resume_from).exists():
            cmd.extend(["--model_file", str(Path(resume_from).resolve())])

        print(f"    [MimicKit] Running: {' '.join(cmd[-8:])}")

        # 4. Run subprocess with stdout streaming for convergence detection
        #    cwd = MimicKit root so relative paths in configs resolve correctly
        env = os.environ.copy()
        env["PYGLET_HEADLESS"] = "1"
        env["PYTHONUNBUFFERED"] = "1"  # Force line-buffered stdout in subprocess

        test_returns = []
        converged = False

        proc = subprocess.Popen(
            cmd,
            cwd=str(self.mimickit_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        try:
            for line in proc.stdout:
                # Pass through to our stdout
                sys.stdout.write(line)
                sys.stdout.flush()

                # Parse Test_Return from MimicKit log tables
                m = self._RETURN_RE.search(line)
                if m:
                    try:
                        ret = float(m.group(1))
                        test_returns.append(ret)

                        if self._check_plateau(test_returns):
                            n = len(test_returns)
                            recent = test_returns[-self.plateau_window:]
                            mean_r = np.mean(recent)
                            spread = max(recent) - min(recent)
                            print(f"\n    [MimicKit] CONVERGED after {n} log outputs "
                                  f"(mean_return={mean_r:.2f}, spread={spread:.3f}, "
                                  f"rel_change={spread/abs(mean_r):.4f} < {self.plateau_threshold})")
                            converged = True
                            proc.terminate()
                            proc.wait(timeout=30)
                            break
                    except ValueError:
                        pass

        except Exception as e:
            print(f"    [MimicKit] Stream error: {e}")
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        rc = proc.returncode
        # -15 = SIGTERM (our termination), 0 = normal exit
        if rc not in (0, -15, -signal.SIGTERM, None) and not converged:
            print(f"    [MimicKit] WARNING: Training subprocess exited with code {rc}")

        if test_returns:
            print(f"    [MimicKit] Final Test_Return: {test_returns[-1]:.2f} "
                  f"(tracked {len(test_returns)} outputs)")
        else:
            print(f"    [MimicKit] WARNING: No Test_Return values parsed from output")

        # 5. Return checkpoint path
        checkpoint = out_dir / "training" / "model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Expected checkpoint at {checkpoint}")
        return str(checkpoint)


# ---------------------------------------------------------------------------
# Simple Adam optimizer for numpy arrays
# ---------------------------------------------------------------------------

class AdamOptimizer:
    """Adam optimizer for numpy design parameters."""

    def __init__(self, n_params, lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(n_params)
        self.v = np.zeros(n_params)
        self.t = 0

    def step(self, params, grad):
        """Update params using gradient (for gradient ASCENT, pass reward gradient)."""
        self.t += 1
        # For ASCENT: params += lr * adapted_grad
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return params + self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Main PGHC loop
# ---------------------------------------------------------------------------

def pghc_codesign_g1(
    n_outer_iterations=20,
    design_lr=0.005,
    num_train_envs=4096,
    num_eval_worlds=8,
    eval_horizon=100,
    max_inner_samples=5_000_000,
    use_wandb=False,
    compare_fd=False,
    out_dir="output_g1_codesign",
    inner_logger="tb",
    resume_checkpoint=None,
    plateau_threshold=0.02,
    plateau_window=5,
    min_plateau_outputs=10,
):
    """PGHC Co-Design for G1 Humanoid with backprop design gradients."""
    print("=" * 70)
    print("PGHC Co-Design for G1 Humanoid (Level 3 — BPTT Gradients)")
    print("=" * 70)

    out_dir = Path(out_dir).resolve()  # Must be absolute — MimicKit subprocess has different cwd
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="pghc-codesign",
            name=f"g1-bptt-{num_train_envs}env",
            config={
                "level": "3-g1-bptt",
                "num_train_envs": num_train_envs,
                "num_eval_worlds": num_eval_worlds,
                "eval_horizon": eval_horizon,
                "n_outer_iterations": n_outer_iterations,
                "design_lr": design_lr,
                "max_inner_samples": max_inner_samples,
                "num_design_params": NUM_DESIGN_PARAMS,
                "dt": 1.0/30.0,
                "num_substeps": 8,
                "gradient_method": "backprop",
                "eval_solver": "SolverSemiImplicit",
                "train_solver": "SolverMuJoCo (via MimicKit)",
                "plateau_threshold": plateau_threshold,
                "plateau_window": plateau_window,
                "min_plateau_outputs": min_plateau_outputs,
            },
        )
        print(f"  [wandb] Logging enabled")
    elif use_wandb:
        print("  [wandb] Not available")
        use_wandb = False

    # Initialize
    theta = np.zeros(NUM_DESIGN_PARAMS, dtype=np.float64)
    theta_bounds = (-0.5236, 0.5236)  # ±30 degrees

    mjcf_modifier = G1MJCFModifier(str(BASE_MJCF_PATH))
    inner_loop = MimicKitInnerLoop(
        str(MIMICKIT_DIR),
        plateau_threshold=plateau_threshold,
        plateau_window=plateau_window,
        min_outputs=min_plateau_outputs,
    )
    design_optimizer = AdamOptimizer(NUM_DESIGN_PARAMS, lr=design_lr)

    param_names = [f"theta_{i}_{SYMMETRIC_PAIRS[i][0].replace('_link','')}"
                   for i in range(NUM_DESIGN_PARAMS)]

    history = {
        "theta": [theta.copy()],
        "forward_dist": [],
        "gradients": [],
        "inner_times": [],
    }
    theta_history = deque(maxlen=5)

    last_checkpoint = resume_checkpoint

    print(f"\nConfiguration:")
    print(f"  Num parallel envs (training): {num_train_envs}")
    print(f"  Num eval worlds (backprop):   {num_eval_worlds}")
    print(f"  Eval horizon:                 {eval_horizon} steps ({eval_horizon/30.0:.1f}s)")
    print(f"  Max inner samples:            {max_inner_samples:,}")
    print(f"  Design optimizer:             Adam (lr={design_lr})")
    print(f"  Design params:                {NUM_DESIGN_PARAMS} (symmetric lower-body pairs)")
    print(f"  Theta bounds:                 ±30° (±0.5236 rad)")
    print(f"  Inner convergence:            plateau <{plateau_threshold*100:.0f}% over {plateau_window} outputs "
          f"(min {min_plateau_outputs} outputs)")
    print(f"  Initial theta:                all zeros")
    if compare_fd:
        print(f"  FD comparison:                ENABLED")

    for outer_iter in range(n_outer_iterations):
        print(f"\n{'='*70}")
        print(f"Outer Iteration {outer_iter + 1}/{n_outer_iterations}")
        print(f"{'='*70}")
        theta_deg = np.degrees(theta)
        for i, name in enumerate(param_names):
            print(f"  {name}: {theta[i]:+.4f} rad ({theta_deg[i]:+.2f}°)")

        if use_wandb:
            log_dict = {"outer/iteration": outer_iter + 1}
            for i, name in enumerate(param_names):
                log_dict[f"outer/{name}_rad"] = theta[i]
                log_dict[f"outer/{name}_deg"] = theta_deg[i]
            wandb.log(log_dict)

        iter_dir = out_dir / f"outer_{outer_iter:03d}"

        # =============================================
        # INNER LOOP (MimicKit subprocess)
        # =============================================
        print(f"\n  [Inner Loop] Training MimicKit AMP ({num_train_envs} envs)...")
        t0 = time.time()

        try:
            checkpoint = inner_loop.run(
                mjcf_modifier=mjcf_modifier,
                theta_np=theta,
                out_dir=str(iter_dir),
                num_envs=num_train_envs,
                max_samples=max_inner_samples,
                resume_from=last_checkpoint,
                logger_type=inner_logger,
            )
            last_checkpoint = checkpoint
            inner_time = time.time() - t0
            print(f"  [Inner Loop] Done in {inner_time/60:.1f} min. Checkpoint: {checkpoint}")
        except Exception as e:
            print(f"  [Inner Loop] FAILED: {e}")
            history["inner_times"].append(time.time() - t0)
            continue

        history["inner_times"].append(inner_time)

        # =============================================
        # Phase 1+2: Eval worker subprocess (GPU-isolated)
        # =============================================
        print(f"\n  [Eval] Launching eval worker subprocess "
              f"({num_eval_worlds} worlds, {eval_horizon} steps)...")

        # Save theta for eval worker
        theta_file = iter_dir / "theta_for_eval.npy"
        np.save(str(theta_file), theta)

        env_config_path = iter_dir / "env_config.yaml"
        result_file = iter_dir / "eval_result.npz"
        eval_cmd = [
            sys.executable, str(CODESIGN_DIR / "g1_eval_worker.py"),
            "--theta-file", str(theta_file),
            "--checkpoint", checkpoint,
            "--env-config", str(env_config_path),
            "--engine-config", str(BASE_ENGINE_CONFIG),
            "--agent-config", str(BASE_AGENT_CONFIG),
            "--output-file", str(result_file),
            "--num-eval-worlds", str(num_eval_worlds),
            "--eval-horizon", str(eval_horizon),
        ]
        eval_env = os.environ.copy()
        eval_env["PYGLET_HEADLESS"] = "1"
        eval_env["PYTHONUNBUFFERED"] = "1"

        t_eval = time.time()
        eval_result = subprocess.run(eval_cmd, env=eval_env)
        eval_time = time.time() - t_eval

        if eval_result.returncode != 0 or not result_file.exists():
            print(f"  [Eval] FAILED (rc={eval_result.returncode}, "
                  f"result_exists={result_file.exists()})")
            continue

        # Read results from eval worker
        data = np.load(str(result_file))
        grad_theta = data["grad_theta"]
        eval_fwd_dist = float(data["fwd_dist"])
        print(f"  [Eval] Done in {eval_time:.1f}s")

        print(f"    BPTT gradients:")
        for i, name in enumerate(param_names):
            print(f"      ∂reward/∂{name} = {grad_theta[i]:+.6f}")
        print(f"    Eval forward distance = {eval_fwd_dist:.3f} m")

        history["forward_dist"].append(eval_fwd_dist)
        history["gradients"].append(grad_theta.copy())

        # =============================================
        # Update design parameters
        # =============================================
        old_theta = theta.copy()
        theta = design_optimizer.step(theta, grad_theta)
        theta = np.clip(theta, theta_bounds[0], theta_bounds[1])

        print(f"\n  Design update:")
        for i, name in enumerate(param_names):
            delta = theta[i] - old_theta[i]
            print(f"    {name}: {old_theta[i]:+.4f} → {theta[i]:+.4f} "
                  f"(Δ={delta:+.5f}, {np.degrees(delta):+.3f}°)")

        history["theta"].append(theta.copy())

        if use_wandb:
            log_dict = {
                "outer/eval_forward_distance": eval_fwd_dist,
                "outer/inner_time_min": inner_time / 60.0,
                "outer/eval_time_s": eval_time,
                "outer/grad_norm": np.linalg.norm(grad_theta),
            }
            for i, name in enumerate(param_names):
                log_dict[f"outer/grad_{name}"] = grad_theta[i]
                log_dict[f"outer/{name}_new_rad"] = theta[i]
                log_dict[f"outer/{name}_new_deg"] = np.degrees(theta[i])
            wandb.log(log_dict)

        # Save theta checkpoint
        np.save(str(out_dir / "theta_latest.npy"), theta)
        np.save(str(iter_dir / "theta.npy"), theta)
        np.save(str(iter_dir / "grad.npy"), grad_theta)

        # Check outer convergence (theta stable over last 5 iters)
        theta_history.append(theta.copy())
        if len(theta_history) >= 5:
            theta_stack = np.array(list(theta_history))
            ranges = theta_stack.max(axis=0) - theta_stack.min(axis=0)
            max_range = ranges.max()
            if max_range < np.radians(0.5):  # 0.5 degree stability
                print(f"\n  OUTER CONVERGED: All theta stable over last 5 iters "
                      f"(max range = {np.degrees(max_range):.3f}°)")
                break

    # =============================================
    # Final Results
    # =============================================
    print("\n" + "=" * 70)
    print("PGHC Co-Design Complete (Level 3 — G1 Humanoid)!")
    print("=" * 70)

    print(f"\nDesign parameter evolution:")
    initial = history["theta"][0]
    final = history["theta"][-1]
    for i, name in enumerate(param_names):
        print(f"  {name}: {initial[i]:+.4f} → {final[i]:+.4f} "
              f"({np.degrees(initial[i]):+.2f}° → {np.degrees(final[i]):+.2f}°)")

    if history["forward_dist"]:
        print(f"\nForward distance: {history['forward_dist'][0]:.3f} → "
              f"{history['forward_dist'][-1]:.3f} m")

    total_inner_time = sum(history["inner_times"])
    print(f"\nTotal inner loop time: {total_inner_time/3600:.1f} hours")

    if use_wandb:
        wandb.log({
            "summary/total_inner_time_hours": total_inner_time / 3600,
            "summary/num_outer_iters": len(history["theta"]) - 1,
        })
        wandb.finish()

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PGHC Co-Design for G1 Humanoid (Level 3 — BPTT)"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--outer-iters", type=int, default=20)
    parser.add_argument("--design-lr", type=float, default=0.005)
    parser.add_argument("--num-train-envs", type=int, default=4096)
    parser.add_argument("--num-eval-worlds", type=int, default=8)
    parser.add_argument("--eval-horizon", type=int, default=100)
    parser.add_argument("--max-inner-samples", type=int, default=200_000_000_000)
    parser.add_argument("--out-dir", type=str, default="output_g1_codesign")
    parser.add_argument("--inner-logger", type=str, default="tb",
                        choices=["tb", "wandb"])
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="MimicKit checkpoint to resume from")
    parser.add_argument("--compare-fd", action="store_true",
                        help="Also compute FD gradient for comparison (very slow)")
    parser.add_argument("--plateau-threshold", type=float, default=0.02,
                        help="Inner convergence: relative change threshold (default: 0.02 = 2%%)")
    parser.add_argument("--plateau-window", type=int, default=5,
                        help="Inner convergence: number of log outputs to check (default: 5)")
    parser.add_argument("--min-plateau-outputs", type=int, default=10,
                        help="Inner convergence: min log outputs before early stop (default: 10)")
    args = parser.parse_args()

    history = pghc_codesign_g1(
        n_outer_iterations=args.outer_iters,
        design_lr=args.design_lr,
        num_train_envs=args.num_train_envs,
        num_eval_worlds=args.num_eval_worlds,
        eval_horizon=args.eval_horizon,
        max_inner_samples=args.max_inner_samples,
        use_wandb=args.wandb,
        compare_fd=args.compare_fd,
        out_dir=args.out_dir,
        inner_logger=args.inner_logger,
        resume_checkpoint=args.resume_checkpoint,
        plateau_threshold=args.plateau_threshold,
        plateau_window=args.plateau_window,
        min_plateau_outputs=args.min_plateau_outputs,
    )
