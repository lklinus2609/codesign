#!/usr/bin/env python3
"""
Level 3: Unified Single-Process PGHC Co-Design for G1 Humanoid

Eliminates subprocess overhead by importing MimicKit as a library and keeping
all GPU resources persistent across outer iterations.

Architecture:
    Single process, single CUDA context:
    1. Build MimicKit AMPEnv + AMPAgent ONCE (N training worlds)
    2. Build DiffG1Eval ONCE (M eval worlds, requires_grad=True)
    3. OUTER LOOP:
        a. Update training model joint_X_p in-place from theta
        b. Inner loop: agent._train_iter() until convergence
        c. Collect actions (frozen policy, slice M worlds from training env)
        d. BPTT gradient via DiffG1Eval.compute_gradient()
        e. Adam update theta, clip to +/-30 deg
        f. Check outer convergence (theta stable over last 5 iters)

Run:
    python codesign_g1_unified.py --wandb --num-train-envs 4096 --num-eval-worlds 32
"""

import os
os.environ["PYGLET_HEADLESS"] = "1"

import argparse
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODESIGN_DIR = Path(__file__).parent.resolve()
MIMICKIT_DIR = (CODESIGN_DIR / ".." / "MimicKit").resolve()
MIMICKIT_SRC_DIR = MIMICKIT_DIR / "mimickit"

BASE_MJCF_PATH = MIMICKIT_DIR / "data" / "assets" / "g1" / "g1.xml"
BASE_ENV_CONFIG = MIMICKIT_DIR / "data" / "envs" / "amp_g1_env.yaml"
BASE_AGENT_CONFIG = MIMICKIT_DIR / "data" / "agents" / "amp_g1_agent.yaml"
BASE_ENGINE_CONFIG = MIMICKIT_DIR / "data" / "engines" / "newton_engine.yaml"

# ---------------------------------------------------------------------------
# MimicKit + GPU imports (single process, single CUDA context)
# ---------------------------------------------------------------------------
if str(MIMICKIT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(MIMICKIT_SRC_DIR))

import warp as wp
import newton  # noqa: F401
import torch

# MimicKit modules (newton_engine.py sets wp.config.enable_backward = False)
import util.mp_util as mp_util
import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import learning.base_agent as base_agent_mod

# CRITICAL: Restore backward mode for BPTT kernels.
# Warp kernel compilation is lazy (happens on first launch, not on
# @wp.kernel decoration), so all kernels will compile with backward enabled.
wp.config.enable_backward = True

# Co-design modules
from g1_mjcf_modifier import (
    G1MJCFModifier, SYMMETRIC_PAIRS, NUM_DESIGN_PARAMS,
    quat_from_x_rotation, quat_multiply, quat_normalize,
)
from g1_eval_worker import DiffG1Eval

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Adam optimizer for design parameters
# ---------------------------------------------------------------------------

class AdamOptimizer:
    """Adam optimizer for numpy arrays (gradient ascent)."""

    def __init__(self, n_params, lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(n_params)
        self.v = np.zeros(n_params)
        self.t = 0

    def step(self, params, grad):
        """Gradient ascent: params += lr * adapted_grad."""
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return params + self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Joint info extraction + in-place update
# ---------------------------------------------------------------------------

def extract_joint_info(model, num_worlds):
    """Extract base quaternions and joint index map from Newton model.

    Returns:
        base_quats_dict: {body_name: (w,x,y,z)} for parameterized bodies
        joint_idx_map:   {body_name: local_joint_index}
    """
    body_keys = model.body_key
    joint_keys = model.joint_key
    bodies_per_world = model.body_count // num_worlds
    joints_per_world = model.joint_count // num_worlds

    body_name_to_idx = {body_keys[i]: i for i in range(bodies_per_world)}
    joint_name_to_idx = {joint_keys[i]: i for i in range(joints_per_world)}

    joint_X_p_np = model.joint_X_p.numpy()
    base_quats_dict = {}
    joint_idx_map = {}

    for left_body, right_body in SYMMETRIC_PAIRS:
        for body_name in (left_body, right_body):
            if body_name in body_name_to_idx:
                idx = body_name_to_idx[body_name]
            else:
                joint_name = body_name.replace("_link", "_joint")
                if joint_name in joint_name_to_idx:
                    idx = joint_name_to_idx[joint_name]
                else:
                    print(f"  [WARN] Body/joint not found: {body_name}")
                    continue
            base_quats_dict[body_name] = tuple(joint_X_p_np[idx, 3:7].tolist())
            joint_idx_map[body_name] = idx

    print(f"  [JointInfo] Found {len(joint_idx_map)} parameterized joints "
          f"(expected {NUM_DESIGN_PARAMS * 2})")
    return base_quats_dict, joint_idx_map


def update_training_joint_X_p(model, theta_np, base_quats_dict, joint_idx_map,
                               num_worlds):
    """Update training model joint_X_p in-place for new theta.

    Uses numpy round-trip (one per outer iter, negligible cost).
    model.joint_X_p.assign() writes to the same device pointer, so the
    CUDA graph picks up the new values without recapture.
    """
    joints_per_world = model.joint_count // num_worlds
    joint_X_p_np = model.joint_X_p.numpy()

    for i, (left_body, right_body) in enumerate(SYMMETRIC_PAIRS):
        angle = float(theta_np[i])
        delta_q = quat_from_x_rotation(angle)
        for body_name in (left_body, right_body):
            if body_name not in joint_idx_map:
                continue
            base_q = base_quats_dict[body_name]
            new_q = quat_normalize(quat_multiply(delta_q, base_q))
            local_idx = joint_idx_map[body_name]
            for w in range(num_worlds):
                global_idx = w * joints_per_world + local_idx
                joint_X_p_np[global_idx, 3:7] = new_q

    model.joint_X_p.assign(joint_X_p_np)
    wp.synchronize()


# ---------------------------------------------------------------------------
# Inner loop controller
# ---------------------------------------------------------------------------

class InnerLoopController:
    """Drives MimicKit agent._train_iter() with convergence detection."""

    def __init__(self, agent, plateau_threshold=0.02, plateau_window=5,
                 min_plateau_outputs=10, max_samples=200_000_000):
        self.agent = agent
        self.plateau_threshold = plateau_threshold
        self.plateau_window = plateau_window
        self.min_plateau_outputs = min_plateau_outputs
        self.max_samples = max_samples

    def _check_plateau(self, returns):
        n_required = max(self.plateau_window, self.min_plateau_outputs)
        if len(returns) < n_required:
            return False
        recent = returns[-self.plateau_window:]
        mean_val = np.mean(recent)
        if abs(mean_val) < 1e-8:
            return False
        spread = max(recent) - min(recent)
        return (spread / abs(mean_val)) < self.plateau_threshold

    def train_until_converged(self, out_dir):
        """Run inner training loop until convergence or sample cap.

        Returns (converged: bool, test_returns: list[float]).
        """
        agent = self.agent
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(out_dir / "model.pt")

        # Reset counters and buffers (preserves network weights = warm-start)
        agent._curr_obs, agent._curr_info = agent._reset_envs()
        agent._init_train()

        test_returns = []
        converged = False

        while agent._sample_count < self.max_samples:
            agent._train_iter()
            agent._sample_count = agent._update_sample_count()

            if agent._iter % agent._iters_per_output == 0:
                test_info = agent.test_model(agent._test_episodes)
                mean_ret = test_info["mean_return"]
                test_returns.append(mean_ret)

                print(f"    [Inner] iter={agent._iter:4d}  "
                      f"samples={agent._sample_count:>10,}  "
                      f"test_return={mean_ret:.2f}")

                agent.save(checkpoint_path)

                if self._check_plateau(test_returns):
                    recent = test_returns[-self.plateau_window:]
                    spread = max(recent) - min(recent)
                    print(f"    [Inner] CONVERGED ({len(test_returns)} outputs, "
                          f"mean={np.mean(recent):.2f}, spread={spread:.3f})")
                    converged = True
                    break

                # Reset after test (matches MimicKit train_model behavior)
                agent._train_return_tracker.reset()
                agent._curr_obs, agent._curr_info = agent._reset_envs()

            agent._iter += 1

        if not converged:
            agent.save(checkpoint_path)
            print(f"    [Inner] Sample cap reached ({agent._sample_count:,})")

        return converged, test_returns


# ---------------------------------------------------------------------------
# Action collection
# ---------------------------------------------------------------------------

def collect_actions_from_agent(agent, env, num_eval_worlds, horizon):
    """Collect actions from frozen policy using the training env.

    Runs the full training env but only keeps first num_eval_worlds actions.
    """
    agent.eval()
    agent.set_mode(base_agent_mod.AgentMode.TEST)

    obs, info = env.reset()
    actions_list = []

    with torch.no_grad():
        for _ in range(horizon):
            action, _ = agent._decide_action(obs, info)
            actions_list.append(action[:num_eval_worlds].cpu().numpy().copy())
            obs, reward, done, info = env.step(action)

    agent.set_mode(base_agent_mod.AgentMode.TRAIN)
    return actions_list


# ---------------------------------------------------------------------------
# Main PGHC loop
# ---------------------------------------------------------------------------

def pghc_codesign_g1_unified(args):
    """Unified single-process PGHC co-design for G1 humanoid."""
    print("=" * 70)
    print("PGHC Co-Design for G1 Humanoid (Level 3 — Unified Single-Process)")
    print("=" * 70)

    device = "cuda:0"
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- wandb ---
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project="pghc-codesign",
            name=f"g1-unified-{args.num_train_envs}env",
            config=vars(args),
        )
    elif args.wandb:
        print("  [wandb] Not available, continuing without")

    # =========================================================
    # One-time initialization
    # =========================================================
    print("\n[1/4] Initializing MimicKit...")
    try:
        mp_util.init(0, 1, device, np.random.randint(6000, 7000))
    except (AssertionError, Exception):
        pass  # Already initialized

    # Generate initial MJCF (theta=0 = unmodified base G1)
    theta = np.zeros(NUM_DESIGN_PARAMS, dtype=np.float64)
    mjcf_modifier = G1MJCFModifier(str(BASE_MJCF_PATH))

    modified_mjcf = out_dir / "g1_modified.xml"
    mjcf_modifier.generate(theta, str(modified_mjcf))

    # Mesh directory symlink for MJCF relative paths
    mesh_src = BASE_MJCF_PATH.parent / "meshes"
    mesh_dst = out_dir / "meshes"
    if not mesh_dst.exists():
        try:
            mesh_dst.symlink_to(mesh_src)
        except (OSError, NotImplementedError):
            shutil.copytree(str(mesh_src), str(mesh_dst))

    # Env config pointing to modified MJCF
    modified_env_config = out_dir / "env_config.yaml"
    mjcf_modifier.generate_env_config(
        str(modified_mjcf), str(BASE_ENV_CONFIG), str(modified_env_config)
    )

    print("[2/4] Building training env and agent...")
    env = env_builder.build_env(
        str(modified_env_config), str(BASE_ENGINE_CONFIG),
        args.num_train_envs, device, visualize=False,
    )
    agent = agent_builder.build_agent(str(BASE_AGENT_CONFIG), env, device)

    if args.resume_checkpoint:
        agent.load(args.resume_checkpoint)
        print(f"  Resumed from {args.resume_checkpoint}")

    # Logger for MimicKit internals (test_model() logs via agent._logger)
    log_file = str(out_dir / "training_log.txt")
    agent._logger = agent._build_logger("tb", log_file, agent._config)

    # Extract joint info from training Newton model
    engine = env._engine
    sim_model = engine._sim_model
    base_quats_dict, joint_idx_map = extract_joint_info(
        sim_model, args.num_train_envs
    )

    print("[3/4] Building differentiable eval model...")
    diff_eval = DiffG1Eval(
        mjcf_modifier=mjcf_modifier,
        theta_np=theta,
        num_worlds=args.num_eval_worlds,
        horizon=args.eval_horizon,
        dt=1.0 / 30.0,
        num_substeps=8,
        device=device,
    )

    # =========================================================
    # Outer loop setup
    # =========================================================
    print("[4/4] Starting PGHC outer loop...\n")

    design_optimizer = AdamOptimizer(NUM_DESIGN_PARAMS, lr=args.design_lr)
    theta_bounds = (-0.5236, 0.5236)  # +/-30 deg

    param_names = [
        f"theta_{i}_{SYMMETRIC_PAIRS[i][0].replace('_link', '')}"
        for i in range(NUM_DESIGN_PARAMS)
    ]

    inner_ctrl = InnerLoopController(
        agent,
        plateau_threshold=args.plateau_threshold,
        plateau_window=args.plateau_window,
        min_plateau_outputs=args.min_plateau_outputs,
        max_samples=args.max_inner_samples,
    )

    history = {
        "theta": [theta.copy()],
        "forward_dist": [],
        "gradients": [],
        "inner_times": [],
    }
    theta_history = deque(maxlen=5)

    print(f"Configuration:")
    print(f"  Training envs:     {args.num_train_envs}")
    print(f"  Eval worlds:       {args.num_eval_worlds}")
    print(f"  Eval horizon:      {args.eval_horizon} steps "
          f"({args.eval_horizon / 30:.1f}s)")
    print(f"  Max inner samples: {args.max_inner_samples:,}")
    print(f"  Design optimizer:  Adam (lr={args.design_lr})")
    print(f"  Design params:     {NUM_DESIGN_PARAMS} (symmetric lower-body)")
    print(f"  Theta bounds:      +/-30 deg (+/-0.5236 rad)")

    # =========================================================
    # Outer loop
    # =========================================================
    for outer_iter in range(args.outer_iters):
        print(f"\n{'=' * 70}")
        print(f"Outer Iteration {outer_iter + 1}/{args.outer_iters}")
        print(f"{'=' * 70}")

        theta_deg = np.degrees(theta)
        for i, name in enumerate(param_names):
            print(f"  {name}: {theta[i]:+.4f} rad ({theta_deg[i]:+.2f} deg)")

        iter_dir = out_dir / f"outer_{outer_iter:03d}"

        # ----- Inner Loop -----
        print(f"\n  [Inner Loop] Training ({args.num_train_envs} envs)...")
        t0 = time.time()

        try:
            converged, test_returns = inner_ctrl.train_until_converged(
                iter_dir
            )
            inner_time = time.time() - t0
            print(f"  [Inner Loop] Done in {inner_time / 60:.1f} min")
        except Exception as e:
            print(f"  [Inner Loop] FAILED: {e}")
            import traceback
            traceback.print_exc()
            history["inner_times"].append(time.time() - t0)
            continue

        history["inner_times"].append(inner_time)

        # ----- Collect Actions -----
        print(f"\n  [Actions] Collecting ({args.num_eval_worlds} worlds x "
              f"{args.eval_horizon} steps)...")

        actions_list = collect_actions_from_agent(
            agent, env, args.num_eval_worlds, args.eval_horizon
        )
        print(f"    Got {len(actions_list)} steps x {actions_list[0].shape}")

        # ----- BPTT Gradient -----
        print(f"\n  [BPTT] Computing gradient...")

        diff_eval.theta_np = theta.copy()
        diff_eval.theta_wp.assign(theta.astype(np.float64))

        grad_theta, fwd_dist = diff_eval.compute_gradient(actions_list)

        print(f"    BPTT gradients:")
        for i, name in enumerate(param_names):
            print(f"      d_reward/d_{name} = {grad_theta[i]:+.6f}")
        print(f"    Forward distance = {fwd_dist:.3f} m")

        history["forward_dist"].append(fwd_dist)
        history["gradients"].append(grad_theta.copy())

        # ----- Design Update -----
        old_theta = theta.copy()
        theta = design_optimizer.step(theta, grad_theta)
        theta = np.clip(theta, theta_bounds[0], theta_bounds[1])

        print(f"\n  Design update:")
        for i, name in enumerate(param_names):
            delta = theta[i] - old_theta[i]
            print(f"    {name}: {old_theta[i]:+.4f} -> {theta[i]:+.4f} "
                  f"(delta={delta:+.5f}, {np.degrees(delta):+.3f} deg)")

        history["theta"].append(theta.copy())

        # Update training model for next outer iteration
        update_training_joint_X_p(
            sim_model, theta, base_quats_dict, joint_idx_map,
            args.num_train_envs
        )

        # Save checkpoints
        np.save(str(out_dir / "theta_latest.npy"), theta)
        np.save(str(iter_dir / "theta.npy"), theta)
        np.save(str(iter_dir / "grad.npy"), grad_theta)

        # wandb logging
        if use_wandb:
            log_dict = {
                "outer/iteration": outer_iter + 1,
                "outer/eval_forward_distance": fwd_dist,
                "outer/inner_time_min": inner_time / 60.0,
                "outer/grad_norm": np.linalg.norm(grad_theta),
            }
            for i, name in enumerate(param_names):
                log_dict[f"outer/{name}_rad"] = theta[i]
                log_dict[f"outer/{name}_deg"] = np.degrees(theta[i])
                log_dict[f"outer/grad_{name}"] = grad_theta[i]
            if test_returns:
                log_dict["outer/final_test_return"] = test_returns[-1]
            wandb.log(log_dict)

        # Outer convergence check
        theta_history.append(theta.copy())
        if len(theta_history) >= 5:
            theta_stack = np.array(list(theta_history))
            ranges = theta_stack.max(axis=0) - theta_stack.min(axis=0)
            max_range = ranges.max()
            if max_range < np.radians(0.5):
                print(f"\n  OUTER CONVERGED: theta stable "
                      f"(max range = {np.degrees(max_range):.3f} deg)")
                break

    # =========================================================
    # Final results
    # =========================================================
    print("\n" + "=" * 70)
    print("PGHC Co-Design Complete!")
    print("=" * 70)

    initial = history["theta"][0]
    final = history["theta"][-1]
    for i, name in enumerate(param_names):
        print(f"  {name}: {initial[i]:+.4f} -> {final[i]:+.4f} "
              f"({np.degrees(initial[i]):+.2f} -> "
              f"{np.degrees(final[i]):+.2f} deg)")

    if history["forward_dist"]:
        print(f"\nForward distance: {history['forward_dist'][0]:.3f} -> "
              f"{history['forward_dist'][-1]:.3f} m")

    total_time = sum(history["inner_times"])
    print(f"Total inner loop time: {total_time / 3600:.1f} hours")

    # Cleanup
    diff_eval.cleanup()
    if use_wandb:
        wandb.finish()

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PGHC Co-Design for G1 Humanoid (Unified Single-Process)"
    )
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging for outer loop")
    parser.add_argument("--outer-iters", type=int, default=20)
    parser.add_argument("--design-lr", type=float, default=0.005)
    parser.add_argument("--num-train-envs", type=int, default=4096)
    parser.add_argument("--num-eval-worlds", type=int, default=32)
    parser.add_argument("--eval-horizon", type=int, default=100)
    parser.add_argument("--max-inner-samples", type=int, default=200_000_000)
    parser.add_argument("--out-dir", type=str, default="output_g1_unified")
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="MimicKit checkpoint to resume from")
    parser.add_argument("--plateau-threshold", type=float, default=0.02,
                        help="Inner convergence: relative change threshold")
    parser.add_argument("--plateau-window", type=int, default=5,
                        help="Inner convergence: window size")
    parser.add_argument("--min-plateau-outputs", type=int, default=10,
                        help="Inner convergence: min outputs before early stop")
    args = parser.parse_args()

    pghc_codesign_g1_unified(args)
