#!/usr/bin/env python3
"""
Level 3: Unified Single-Process PGHC Co-Design for G1 Humanoid

Eliminates subprocess overhead by importing MimicKit as a library and keeping
all GPU resources persistent across outer iterations.

Architecture:
    Single process per GPU, single CUDA context per rank:
    1. Build MimicKit AMPEnv + AMPAgent ONCE (N/num_gpus training worlds per GPU)
    2. OUTER LOOP:
        a. Update training model joint_X_p in-place from theta
        b. Inner loop: agent._train_iter() until convergence (all GPUs)
        c. Collect actions (frozen policy, rank 0 only)
        d. Build DiffG1Eval lazily (rank 0 only, requires_grad=True)
        e. BPTT gradient via DiffG1Eval.compute_gradient() (rank 0 only)
        f. Free DiffG1Eval (reclaim 2-4 GiB VRAM)
        g. Broadcast theta to all ranks
        h. Adam update theta, clip to +/-30 deg
        i. Check outer convergence (theta stable over last 5 iters)

Run:
    # Single GPU (unchanged)
    python codesign_g1_unified.py --wandb --num-train-envs 4096

    # Multi-GPU (4x)
    python codesign_g1_unified.py --wandb --num-train-envs 4096 \
        --devices cuda:0 cuda:1 cuda:2 cuda:3
"""

import os
os.environ["PYGLET_HEADLESS"] = "1"

import argparse
import random
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import gc

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
from newton.viewer import ViewerGL
import torch

# MimicKit modules (newton_engine.py sets wp.config.enable_backward = False)
import util.mp_util as mp_util
import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import learning.base_agent as base_agent_mod

# NOTE: MimicKit's newton_engine.py sets wp.config.enable_backward = False.
# We keep it False during training (matching pure MimicKit behavior) and only
# set it True right before building DiffG1Eval for BPTT. Warp kernel compilation
# is lazy (first launch), so training kernels compile WITHOUT adjoint code,
# and BPTT kernels compile WITH it — each set only compiles once.

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
            # joint_X_p numpy layout: [px,py,pz, qx,qy,qz,qw] (xyzw)
            # Reorder to (qw,qx,qy,qz) for g1_mjcf_modifier's wxyz convention
            raw = joint_X_p_np[idx, 3:7].tolist()  # [qx,qy,qz,qw]
            base_quats_dict[body_name] = (raw[3], raw[0], raw[1], raw[2])  # (qw,qx,qy,qz)
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
            new_q = quat_normalize(quat_multiply(delta_q, base_q))  # (w,x,y,z)
            local_idx = joint_idx_map[body_name]
            for w in range(num_worlds):
                global_idx = w * joints_per_world + local_idx
                # Convert wxyz → xyzw for joint_X_p numpy layout
                joint_X_p_np[global_idx, 3:7] = [new_q[1], new_q[2], new_q[3], new_q[0]]

    model.joint_X_p.assign(joint_X_p_np)
    wp.synchronize()


# ---------------------------------------------------------------------------
# Inner loop controller
# ---------------------------------------------------------------------------

class InnerLoopController:
    """Drives MimicKit agent._train_iter() with convergence detection.

    Multi-GPU aware: all ranks call _train_iter() and _log_train_info()
    (which contain collective ops). Root decides convergence and broadcasts.
    """

    def __init__(self, agent, plateau_threshold=0.02, plateau_window=5,
                 min_plateau_outputs=10, max_samples=200_000_000,
                 min_inner_iters=0,
                 use_wandb=False, viewer=None, engine=None,
                 video_interval=100,
                 is_root=True, device="cuda:0"):
        self.agent = agent
        self.plateau_threshold = plateau_threshold
        self.plateau_window = plateau_window
        self.min_plateau_outputs = min_plateau_outputs
        self.max_samples = max_samples
        self.min_inner_iters = min_inner_iters
        self.use_wandb = use_wandb
        self.viewer = viewer
        self.engine = engine
        self.video_interval = video_interval
        self.is_root = is_root
        self.device = device

    def _check_plateau(self, values):
        n_required = max(self.plateau_window, self.min_plateau_outputs)
        if len(values) < n_required:
            return False
        recent = values[-self.plateau_window:]
        mean_val = np.mean(recent)
        if abs(mean_val) < 1e-8:
            return False
        spread = max(recent) - min(recent)
        return (spread / abs(mean_val)) < self.plateau_threshold

    def train_until_converged(self, out_dir):
        """Run inner training loop until convergence or sample cap.

        Convergence is detected when disc_reward_mean (sampled at output
        iterations, before env reset) plateaus. AMP's env reward is 0 —
        the discriminator reward is the real quality signal.

        All ranks participate in training (collective ops in _train_iter,
        _update_sample_count, _log_train_info). Root decides convergence
        and broadcasts the decision to prevent deadlocks.

        Returns (converged: bool, disc_rewards: list[float]).
        """
        agent = self.agent
        env = agent._env
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(out_dir / "model.pt")

        # Reset counters and buffers (preserves network weights = warm-start)
        agent._curr_obs, agent._curr_info = agent._reset_envs()
        saved_iter = agent._iter  # preserve continuous iter count across outer loops
        agent._init_train()
        agent._iter = saved_iter  # restore so wandb x-axis is monotonic

        disc_rewards = []
        converged = False
        start_time = time.time()
        test_info = None
        inner_iters = 0

        while agent._sample_count < self.max_samples:
            train_info = agent._train_iter()                      # COLLECTIVE
            agent._sample_count = agent._update_sample_count()    # COLLECTIVE

            output_iter = (agent._iter % agent._iters_per_output == 0)

            if output_iter:
                # Sample disc_reward_mean here — before env reset, so no
                # sawtooth spike (spike happens on the iter AFTER reset)
                disc_r = train_info.get("disc_reward_mean")
                if disc_r is not None:
                    if torch.is_tensor(disc_r):
                        disc_r = disc_r.item()
                    disc_rewards.append(disc_r)

                test_info = agent.test_model(agent._test_episodes)  # ALL ranks

            # MimicKit native logging: populate logger (contains collective ops)
            env_diag_info = env.get_diagnostics()
            agent._log_train_info(train_info, test_info, env_diag_info,
                                  start_time)                     # COLLECTIVE

            # Only root prints/writes logs
            if self.is_root:
                agent._logger.print_log()

            if output_iter:
                if self.is_root:
                    agent._logger.write_log()
                    agent.save(checkpoint_path)

                    # wandb: forward inner loop metrics
                    if self.use_wandb:
                        wlog = {}
                        for k, entry in agent._logger.log_current_row.items():
                            v = entry.val
                            try:
                                wlog[f"inner/{k}"] = float(v)
                            except (TypeError, ValueError):
                                pass
                        if wlog:
                            wandb.log(wlog)

                    # Video capture every video_interval iterations
                    if (self.use_wandb and self.viewer is not None
                            and agent._iter % self.video_interval == 0):
                        try:
                            video = capture_video(
                                self.viewer, agent, env, self.engine,
                                num_steps=200,
                            )
                            if video is not None:
                                wandb.log({"inner/video": wandb.Video(
                                    video, fps=30, format="mp4",
                                )})
                        except Exception as e:
                            print(f"  [Inner Loop] Video logging failed: {e}")

                # Convergence check: root decides, broadcast to all ranks
                should_stop = False
                if self.is_root:
                    should_stop = (inner_iters >= self.min_inner_iters
                                   and self._check_plateau(disc_rewards))

                if mp_util.enable_mp():
                    flag = torch.tensor([int(should_stop)], dtype=torch.int32,
                                        device=self.device)
                    flag = mp_util.broadcast(flag)
                    should_stop = flag.item() == 1

                if should_stop:
                    if self.is_root:
                        recent = disc_rewards[-self.plateau_window:]
                        spread = max(recent) - min(recent)
                        print(f"    [Inner] CONVERGED ({len(disc_rewards)} outputs, "
                              f"{inner_iters} iters, "
                              f"disc_reward={np.mean(recent):.4f}, spread={spread:.4f})")
                    converged = True
                    break

                # Reset after test (matches MimicKit train_model behavior)
                agent._train_return_tracker.reset()
                agent._curr_obs, agent._curr_info = agent._reset_envs()

            agent._iter += 1
            inner_iters += 1

        if not converged:
            if self.is_root:
                agent.save(checkpoint_path)
                print(f"    [Inner] Sample cap reached ({agent._sample_count:,})")

        return converged, disc_rewards


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
# Video capture
# ---------------------------------------------------------------------------

def capture_video(viewer, agent, env, engine, num_steps=200, fps=30):
    """Capture a short evaluation video for wandb logging.

    Returns (T, C, H, W) uint8 numpy array, or None on failure.
    """
    try:
        agent.eval()
        agent.set_mode(base_agent_mod.AgentMode.TEST)

        obs, info = env.reset()
        wp.synchronize()

        frames = []
        sim_time = 0.0
        dt = 1.0 / fps

        with torch.no_grad():
            for step in range(num_steps):
                action, _ = agent._decide_action(obs, info)
                obs, reward, done, info = env.step(action)
                sim_time += dt

                viewer.begin_frame(sim_time)
                viewer.log_state(engine._sim_state.raw_state)
                viewer.end_frame()

                frame_wp = viewer.get_frame()
                if frame_wp is not None:
                    frames.append(frame_wp.numpy().copy())

        agent.set_mode(base_agent_mod.AgentMode.TRAIN)

        if not frames:
            return None

        # wandb.Video expects (T, C, H, W)
        video = np.stack(frames)            # (T, H, W, 3)
        video = video.transpose(0, 3, 1, 2)  # (T, 3, H, W)
        return video

    except Exception as e:
        print(f"    [Video] Capture failed: {e}")
        agent.set_mode(base_agent_mod.AgentMode.TRAIN)
        return None


# ---------------------------------------------------------------------------
# Main PGHC loop (one per GPU rank)
# ---------------------------------------------------------------------------

def pghc_worker(rank, num_procs, device, master_port, args):
    """Unified single-process PGHC co-design for G1 humanoid.

    Each GPU rank runs this function. The inner loop (MimicKit training) is
    distributed across all ranks via torch.distributed collective ops. BPTT
    gradient computation runs on rank 0 only; theta is broadcast after.
    """
    is_root = (rank == 0)
    per_rank_envs = args.num_train_envs // num_procs

    # Seed per rank for diverse rollouts
    seed = int(np.uint64(42) + np.uint64(41 * rank))
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    # Initialize torch.distributed (NCCL on Linux, gloo on Windows)
    mp_util.init(rank, num_procs, device, master_port)

    if is_root:
        print("=" * 70)
        print("PGHC Co-Design for G1 Humanoid (Level 3 — Unified Single-Process)")
        if num_procs > 1:
            print(f"  Multi-GPU: {num_procs} GPUs, {per_rank_envs} envs/GPU "
                  f"({args.num_train_envs} total)")
        print("=" * 70)

    out_dir = Path(args.out_dir).resolve()
    if is_root:
        out_dir.mkdir(parents=True, exist_ok=True)
    if mp_util.enable_mp():
        torch.distributed.barrier()

    # --- wandb (root only) ---
    use_wandb = is_root and args.wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project="pghc-codesign",
            name=f"g1-unified-{args.num_train_envs}env-{num_procs}gpu",
            config=vars(args),
        )
    elif is_root and args.wandb:
        print("  [wandb] Not available, continuing without")

    # =========================================================
    # One-time initialization
    # =========================================================
    if is_root:
        print("\n[1/4] Initializing MimicKit...")

    # Generate initial MJCF (theta=0 = unmodified base G1)
    theta = np.zeros(NUM_DESIGN_PARAMS, dtype=np.float64)
    mjcf_modifier = G1MJCFModifier(str(BASE_MJCF_PATH))

    modified_mjcf = out_dir / "g1_modified.xml"
    modified_env_config = out_dir / "env_config.yaml"

    if is_root:
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
        mjcf_modifier.generate_env_config(
            str(modified_mjcf), str(BASE_ENV_CONFIG), str(modified_env_config)
        )

    if mp_util.enable_mp():
        torch.distributed.barrier()  # all ranks wait for MJCF + config files

    if is_root:
        print("[2/4] Building training env and agent...")
    env = env_builder.build_env(
        str(modified_env_config), str(BASE_ENGINE_CONFIG),
        per_rank_envs, device, visualize=False,
    )
    agent = agent_builder.build_agent(str(BASE_AGENT_CONFIG), env, device)

    if args.resume_checkpoint:
        agent.load(args.resume_checkpoint)
        if is_root:
            print(f"  Resumed from {args.resume_checkpoint}")

    # Logger for MimicKit internals (all ranks need it for _log_train_info)
    log_file = str(out_dir / f"training_log_rank{rank}.txt")
    agent._logger = agent._build_logger("tb", log_file, agent._config)

    # Extract joint info from training Newton model (per-rank model)
    engine = env._engine
    sim_model = engine._sim_model
    base_quats_dict, joint_idx_map = extract_joint_info(
        sim_model, per_rank_envs
    )

    # Headless viewer for video capture (root only)
    viewer = None
    if use_wandb:
        try:
            viewer = ViewerGL(headless=True, width=640, height=480)
            viewer.set_model(sim_model, max_worlds=1)
            viewer.set_camera(
                pos=wp.vec3(3.0, -3.0, 1.5),
                pitch=-10.0,
                yaw=90.0,
            )
            print("  [Viewer] Headless viewer created for video capture")
        except Exception as e:
            print(f"  [Viewer] Failed to create viewer: {e}")
            viewer = None

    # DiffG1Eval is built LAZILY in the outer loop (right before BPTT) and
    # freed immediately after. Only rank 0 builds it.
    diff_eval = None
    if is_root:
        print("[3/4] DiffG1Eval will be built lazily before each BPTT phase")

    # =========================================================
    # Outer loop setup
    # =========================================================
    if is_root:
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
        min_inner_iters=args.kickoff_min_iters,  # first outer iter; reset to 0 after
        use_wandb=use_wandb,
        viewer=viewer,
        engine=engine,
        video_interval=args.video_interval,
        is_root=is_root,
        device=device,
    )

    history = {
        "theta": [theta.copy()],
        "forward_dist": [],
        "cot": [],
        "gradients": [],
        "inner_times": [],
    }
    theta_history = deque(maxlen=5)

    if is_root:
        print(f"Configuration:")
        print(f"  Training envs:     {args.num_train_envs} total "
              f"({per_rank_envs}/GPU x {num_procs} GPUs)")
        print(f"  Eval worlds:       {args.num_eval_worlds}")
        print(f"  Eval horizon:      {args.eval_horizon} steps "
              f"({args.eval_horizon / 30:.1f}s)")
        print(f"  Max inner samples: {args.max_inner_samples:,}")
        print(f"  Design optimizer:  Adam (lr={args.design_lr})")
        print(f"  Design params:     {NUM_DESIGN_PARAMS} (symmetric lower-body)")
        print(f"  Theta bounds:      +/-30 deg (+/-0.5236 rad)")
        if args.kickoff_min_iters > 0:
            print(f"  Kickoff min iters: {args.kickoff_min_iters} (first outer iter only)")

    # =========================================================
    # Outer loop
    # =========================================================
    for outer_iter in range(args.outer_iters):
        if is_root:
            print(f"\n{'=' * 70}")
            print(f"Outer Iteration {outer_iter + 1}/{args.outer_iters}")
            print(f"{'=' * 70}")

            theta_deg = np.degrees(theta)
            for i, name in enumerate(param_names):
                print(f"  {name}: {theta[i]:+.4f} rad ({theta_deg[i]:+.2f} deg)")

        iter_dir = out_dir / f"outer_{outer_iter:03d}"

        # ----- Inner Loop (ALL ranks) -----
        if is_root:
            print(f"\n  [Inner Loop] Training ({per_rank_envs} envs/GPU "
                  f"x {num_procs} GPUs)...")
        t0 = time.time()

        try:
            converged, disc_rewards = inner_ctrl.train_until_converged(
                iter_dir
            )
            inner_time = time.time() - t0
            if is_root:
                print(f"  [Inner Loop] Done in {inner_time / 60:.1f} min")
        except Exception as e:
            if is_root:
                print(f"  [Inner Loop] FAILED: {e}")
                import traceback
                traceback.print_exc()
            history["inner_times"].append(time.time() - t0)
            continue

        history["inner_times"].append(inner_time)

        # After kickoff, allow convergence without minimum iter gate
        if outer_iter == 0 and inner_ctrl.min_inner_iters > 0:
            if is_root:
                print(f"  [Kickoff] Disabling min_inner_iters "
                      f"({inner_ctrl.min_inner_iters}) for subsequent iters")
            inner_ctrl.min_inner_iters = 0

        # ----- BPTT Phase (rank 0 only) -----
        # Non-root ranks wait at the theta broadcast below.
        grad_theta = None
        fwd_dist = None
        cot = None

        if is_root:
            # Collect actions from frozen policy
            print(f"\n  [Actions] Collecting ({args.num_eval_worlds} worlds x "
                  f"{args.eval_horizon} steps)...")

            actions_list = collect_actions_from_agent(
                agent, env, args.num_eval_worlds, args.eval_horizon
            )
            print(f"    Got {len(actions_list)} steps x {actions_list[0].shape}")

            # Build DiffG1Eval lazily
            print(f"\n  [BPTT] Building DiffG1Eval lazily...")

            # Enable backward ONLY for BPTT kernel compilation
            wp.config.enable_backward = True

            diff_eval = DiffG1Eval(
                mjcf_modifier=mjcf_modifier,
                theta_np=theta.copy(),
                num_worlds=args.num_eval_worlds,
                horizon=args.eval_horizon,
                dt=1.0 / 30.0,
                num_substeps=8,
                device=device,
            )

            print(f"  [BPTT] Computing gradient...")
            grad_theta, fwd_dist, cot = diff_eval.compute_gradient(actions_list)

            # Free DiffG1Eval immediately (reclaim 2-4 GiB VRAM)
            diff_eval.cleanup()
            diff_eval = None
            gc.collect()
            torch.cuda.empty_cache()

            # Restore enable_backward = False for next inner loop
            wp.config.enable_backward = False
            print(f"  [BPTT] DiffG1Eval freed, VRAM reclaimed")

            print(f"    BPTT gradients:")
            for i, name in enumerate(param_names):
                print(f"      d_reward/d_{name} = {grad_theta[i]:+.6f}")
            print(f"    Forward distance = {fwd_dist:.3f} m")
            print(f"    Cost of Transport = {cot:.4f}")

            history["forward_dist"].append(fwd_dist)
            history["cot"].append(cot)
            history["gradients"].append(grad_theta.copy())

            # ----- Design Update -----
            if np.any(np.isnan(grad_theta)) or np.all(grad_theta == 0):
                print(f"\n  [FATAL] Degenerate gradient (NaN or all-zero) — "
                      f"signaling all ranks to stop.")
                # Signal via NaN theta — all ranks will detect and break
                theta[:] = np.nan
            else:
                old_theta = theta.copy()
                theta = design_optimizer.step(theta, grad_theta)
                theta = np.clip(theta, theta_bounds[0], theta_bounds[1])

                print(f"\n  Design update:")
                for i, name in enumerate(param_names):
                    delta = theta[i] - old_theta[i]
                    print(f"    {name}: {old_theta[i]:+.4f} -> {theta[i]:+.4f} "
                          f"(delta={delta:+.5f}, {np.degrees(delta):+.3f} deg)")

        # ----- Broadcast theta to all ranks -----
        # Non-root ranks block here until root finishes BPTT
        theta_tensor = torch.from_numpy(theta).float().to(device)
        if mp_util.enable_mp():
            theta_tensor = mp_util.broadcast(theta_tensor)
        theta = theta_tensor.cpu().double().numpy()

        # Check for NaN signal (degenerate gradient on root)
        if np.any(np.isnan(theta)):
            break

        history["theta"].append(theta.copy())

        # ALL ranks: update their local Newton model
        update_training_joint_X_p(
            sim_model, theta, base_quats_dict, joint_idx_map,
            per_rank_envs
        )

        # Root: save checkpoints + wandb
        if is_root:
            np.save(str(out_dir / "theta_latest.npy"), theta)
            np.save(str(iter_dir / "theta.npy"), theta)
            np.save(str(iter_dir / "grad.npy"), grad_theta)

            if use_wandb:
                log_dict = {
                    "outer/iteration": outer_iter + 1,
                    "outer/eval_forward_distance": fwd_dist,
                    "outer/cot": cot,
                    "outer/inner_time_min": inner_time / 60.0,
                    "outer/grad_norm": np.linalg.norm(grad_theta),
                }
                for i, name in enumerate(param_names):
                    log_dict[f"outer/{name}_rad"] = theta[i]
                    log_dict[f"outer/{name}_deg"] = np.degrees(theta[i])
                    log_dict[f"outer/grad_{name}"] = grad_theta[i]
                if disc_rewards:
                    log_dict["outer/final_disc_reward"] = disc_rewards[-1]
                wandb.log(log_dict)

                # Save artifacts to wandb files
                wandb.save(str(iter_dir / "model.pt"), base_path=str(out_dir))
                wandb.save(str(iter_dir / "theta.npy"), base_path=str(out_dir))
                wandb.save(str(iter_dir / "grad.npy"), base_path=str(out_dir))
                wandb.save(str(out_dir / "theta_latest.npy"), base_path=str(out_dir))

        # ----- Outer convergence check -----
        outer_converged = False
        if is_root:
            theta_history.append(theta.copy())
            if len(theta_history) >= 5:
                theta_stack = np.array(list(theta_history))
                ranges = theta_stack.max(axis=0) - theta_stack.min(axis=0)
                max_range = ranges.max()
                if max_range < np.radians(0.5):
                    print(f"\n  OUTER CONVERGED: theta stable "
                          f"(max range = {np.degrees(max_range):.3f} deg)")
                    outer_converged = True

        if mp_util.enable_mp():
            flag = torch.tensor([int(outer_converged)], dtype=torch.int32,
                                device=device)
            flag = mp_util.broadcast(flag)
            outer_converged = flag.item() == 1

        if outer_converged:
            break

    # =========================================================
    # Final results
    # =========================================================
    if is_root:
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
        if history["cot"]:
            print(f"Cost of Transport: {history['cot'][0]:.4f} -> "
                  f"{history['cot'][-1]:.4f}")

        total_time = sum(history["inner_times"])
        print(f"Total inner loop time: {total_time / 3600:.1f} hours")

    # Cleanup
    if diff_eval is not None:
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
    parser.add_argument("--num-train-envs", type=int, default=4096,
                        help="Total training envs (divided across GPUs)")
    parser.add_argument("--num-eval-worlds", type=int, default=32)
    parser.add_argument("--eval-horizon", type=int, default=100)
    parser.add_argument("--max-inner-samples", type=int, default=2_000_000_000)
    parser.add_argument("--out-dir", type=str, default="output_g1_unified")
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="MimicKit checkpoint to resume from")
    parser.add_argument("--plateau-threshold", type=float, default=0.02,
                        help="Inner convergence: relative change threshold")
    parser.add_argument("--plateau-window", type=int, default=50,
                        help="Inner convergence: window size (in output intervals)")
    parser.add_argument("--min-plateau-outputs", type=int, default=10,
                        help="Inner convergence: min output intervals before early stop")
    parser.add_argument("--kickoff-min-iters", type=int, default=2000,
                        help="Min inner iters before convergence on first outer iter (0=disabled)")
    parser.add_argument("--video-interval", type=int, default=100,
                        help="Log video to wandb every N inner iterations")
    # Multi-GPU arguments
    parser.add_argument("--devices", nargs="+", default=["cuda:0"],
                        help="CUDA devices for multi-GPU training "
                             "(e.g., cuda:0 cuda:1 cuda:2 cuda:3)")
    parser.add_argument("--master-port", type=int, default=None,
                        help="Port for torch.distributed (default: random 6000-7000)")
    args = parser.parse_args()

    num_workers = len(args.devices)
    assert args.num_train_envs % num_workers == 0, (
        f"--num-train-envs ({args.num_train_envs}) must be divisible by "
        f"number of devices ({num_workers})"
    )

    master_port = args.master_port if args.master_port is not None else random.randint(6000, 7000)

    if num_workers > 1:
        torch.multiprocessing.set_start_method("spawn")
        processes = []
        for r in range(1, num_workers):
            p = torch.multiprocessing.Process(
                target=pghc_worker,
                args=[r, num_workers, args.devices[r], master_port, args],
            )
            p.start()
            processes.append(p)

        # Root process runs in main thread
        pghc_worker(0, num_workers, args.devices[0], master_port, args)

        for p in processes:
            p.join()
    else:
        # Single GPU — backward compatible, no spawn overhead
        pghc_worker(0, 1, args.devices[0], master_port, args)
