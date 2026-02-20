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
        c. Closed-loop FD gradient: run frozen policy on perturbed morphologies
           with K paired seeds, distributed round-robin across GPUs, all_reduce
        d. SGD update theta on rank 0, clip to +/-30 deg
        e. Broadcast theta to all ranks
        f. Check outer convergence (theta stable over last 5 iters)

Run:
    # Single GPU (unchanged)
    python codesign_g1_unified.py --wandb --num-train-envs 4096

    # Multi-GPU (4x)
    python codesign_g1_unified.py --wandb --num-train-envs 4096 \
        --devices cuda:0 cuda:1 cuda:2 cuda:3
"""

import os
os.environ["PYGLET_HEADLESS"] = "1"

# ---------------------------------------------------------------------------
# Multi-GPU device isolation (must run BEFORE any GPU library imports)
#
# Spawned worker processes inherit _PGHC_GPU_INDEX from the parent.  We read
# it here — at module import time — and restrict CUDA_VISIBLE_DEVICES to that
# single physical GPU.  This ensures warp/torch/newton only see one device,
# avoiding cross-device allocation failures on HPC nodes.
# ---------------------------------------------------------------------------
_PGHC_GPU_INDEX = os.environ.pop("_PGHC_GPU_INDEX", None)
if _PGHC_GPU_INDEX is not None:
    _orig_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if _orig_cvd:
        _gpu_list = [g.strip() for g in _orig_cvd.split(",")]
        _idx = int(_PGHC_GPU_INDEX)
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            _gpu_list[_idx] if _idx < len(_gpu_list) else _PGHC_GPU_INDEX
        )
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _PGHC_GPU_INDEX

import argparse
import platform
import random
import shutil
import sys
import time
import types
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
# MimicKit + GPU imports (single process, single CUDA context per rank)
# ---------------------------------------------------------------------------
if str(MIMICKIT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(MIMICKIT_SRC_DIR))

import warp as wp

# Per-rank Warp cache to prevent NVRTC race conditions on multi-GPU nodes.
# Without this, all ranks share ~/.cache/warp/ and corrupt PCH files when
# compiling kernels simultaneously (e.g. during FD evaluation phase).
if _PGHC_GPU_INDEX is not None:
    _warp_cache = os.path.join(
        os.path.expanduser("~"), ".cache", "warp_per_rank", f"rank_{_PGHC_GPU_INDEX}"
    )
    os.makedirs(_warp_cache, exist_ok=True)
    wp.config.kernel_cache_dir = _warp_cache

import newton  # noqa: F401
from newton.viewer import ViewerGL
import torch

# MimicKit modules (newton_engine.py sets wp.config.enable_backward = False)
import util.mp_util as mp_util
import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import learning.base_agent as base_agent_mod
import envs.base_env as base_env_mod
import util.torch_util as torch_util

# NOTE: MimicKit's newton_engine.py sets wp.config.enable_backward = False.
# With closed-loop FD (no BPTT), we never need backward mode enabled.

# Co-design modules
from g1_mjcf_modifier import (
    G1MJCFModifier, SYMMETRIC_PAIRS, NUM_DESIGN_PARAMS,
    quat_from_x_rotation, quat_multiply, quat_normalize,
)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Episode-averaged CoT tracker
# ---------------------------------------------------------------------------

class EpisodeCoTTracker:
    """Tracks per-episode Cost of Transport by accumulating every step.

    Accumulates mechanical power and forward velocity per env.  When an
    episode ends (done_buf > 0), finalises CoT = mean_power / (m*g*|mean_v|)
    and stores it.  ``get_stats()`` returns the mean over completed episodes
    since the last call, giving a much smoother signal than an instantaneous
    snapshot.

    Only counts lower-body + torso DoFs (indices 0-14) for power, excluding
    arm joints whose energy use is not relevant to locomotion co-design.

    Must be called via ``accumulate()`` + ``finalize_episodes()`` every env
    step (hooked into ``_post_physics_step``).
    """

    # G1 DoF layout: 0-5 left leg, 6-11 right leg, 12-14 torso, 15-28 arms
    LOCOMOTION_DOFS = 15  # first 15 DoFs = legs + torso

    @staticmethod
    def get_dof_forces_safe(engine, char_id):
        """Get dof forces as (num_envs, num_dofs).

        Workaround for Newton position-control bug: get_dof_forces(obj_id)
        does tensor indexing on _dof_forces which is (num_envs, num_dofs, 2),
        so [obj_id] selects one *env* instead of one *object*, returning
        (num_dofs, 2) → sum → (num_dofs,) which is wrong.
        We bypass it by reading _dof_forces directly.
        """
        raw = engine._dof_forces
        if isinstance(raw, torch.Tensor):
            if raw.dim() == 3:
                # Position control: (num_envs, num_dofs, 2) → sum P+D
                return torch.sum(raw, dim=-1)
            return raw  # Already (num_envs, num_dofs)
        # List indexed by char_id (none/torque/pd_explicit modes)
        return raw[char_id]

    def __init__(self, num_envs, total_mass, device):
        self.total_mass = total_mass
        self.g = 9.81
        self.power_sum = torch.zeros(num_envs, device=device)
        self.vel_sum = torch.zeros(num_envs, device=device)
        self.step_count = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.episode_cots = []
        self.episode_powers = []
        self.episode_vels = []

    def reset_accumulators(self):
        """Reset per-env accumulators after env reset to avoid stale data."""
        self.power_sum.zero_()
        self.vel_sum.zero_()
        self.step_count.zero_()

    def accumulate(self, engine, char_id):
        """Accumulate instantaneous power and forward velocity."""
        dof_forces = self.get_dof_forces_safe(engine, char_id)
        dof_vel = engine.get_dof_vel(char_id)
        root_vel = engine.get_root_vel(char_id)
        dof_forces = dof_forces[:, :self.LOCOMOTION_DOFS]
        dof_vel = dof_vel[:, :self.LOCOMOTION_DOFS]
        power = torch.sum(torch.abs(dof_forces * dof_vel), dim=-1)
        self.power_sum += power
        self.vel_sum += root_vel[:, 0]
        self.step_count += 1

    def finalize_episodes(self, done_buf):
        """Finalize CoT for episodes that just ended."""
        done_mask = done_buf > 0
        if not done_mask.any():
            return
        done_idx = done_mask.nonzero(as_tuple=True)[0]
        steps = self.step_count[done_idx].float()
        valid = steps > 0
        if valid.any():
            vi = done_idx[valid]
            vs = steps[valid]
            mp = self.power_sum[vi] / vs
            mv = self.vel_sum[vi] / vs
            safe_v = torch.sqrt(mv ** 2 + 0.01)
            cots = mp / (self.total_mass * self.g * safe_v)
            self.episode_cots.extend(cots.tolist())
            self.episode_powers.extend(mp.tolist())
            self.episode_vels.extend(mv.tolist())
        self.power_sum[done_idx] = 0
        self.vel_sum[done_idx] = 0
        self.step_count[done_idx] = 0

    def get_stats(self):
        """Return (mean_cot, mean_power, mean_fwd_vel, n_episodes) and reset."""
        if not self.episode_cots:
            return None, None, None, 0
        n = len(self.episode_cots)
        cot = sum(self.episode_cots) / n
        pwr = sum(self.episode_powers) / n
        vel = sum(self.episode_vels) / n
        self.episode_cots.clear()
        self.episode_powers.clear()
        self.episode_vels.clear()
        return cot, pwr, vel, n


# ---------------------------------------------------------------------------
# Morphology-agnostic task reward (monkey-patched onto AMPEnv)
# ---------------------------------------------------------------------------

def codesign_task_reward(self):
    """Morphology-agnostic task reward for co-design inner loop.

    Replaces AMPEnv._update_reward (which is empty by default) so that
    the agent's reward blending picks up a non-zero task_r:
        r = task_reward_weight * task_r + disc_reward_weight * disc_r
    """
    char_id = self._get_char_id()
    root_vel = self._engine.get_root_vel(char_id)   # (num_envs, 3)
    root_rot = self._engine.get_root_rot(char_id)   # (num_envs, 4) xyzw

    # 1. Velocity tracking: exp(-|v_x - 1.0|^2 / 0.25)
    fwd_vel = root_vel[:, 0]
    vel_reward = torch.exp(-torch.square(fwd_vel - 1.0) / 0.25)

    # 2. Orientation penalty: -||projected_gravity_xy||^2
    #    Rotate world gravity (0,0,-1) into body frame via inverse rotation
    gravity = torch.zeros(3, device=root_rot.device, dtype=root_rot.dtype)
    gravity[2] = -1.0
    gravity = gravity.unsqueeze(0).expand(root_rot.shape[0], -1)
    inv_rot = torch_util.quat_conjugate(root_rot)
    projected_gravity = torch_util.quat_rotate(inv_rot, gravity)
    orientation_penalty = -torch.sum(torch.square(projected_gravity[:, :2]), dim=-1)

    # 3. Alive bonus
    alive = 0.1

    # 4. Termination penalty
    term_penalty = -10.0 * (self._done_buf == base_env_mod.DoneFlags.FAIL.value).float()

    # 5. Mechanical power penalty: aligns inner loop with outer loop CoT objective
    #    power = sum(|tau * dof_vel|) over legs + torso only (DoFs 0-14)
    _LOCO_DOFS = EpisodeCoTTracker.LOCOMOTION_DOFS
    dof_forces = EpisodeCoTTracker.get_dof_forces_safe(self._engine, char_id)
    dof_vel = self._engine.get_dof_vel(char_id)
    dof_forces = dof_forces[:, :_LOCO_DOFS]
    dof_vel = dof_vel[:, :_LOCO_DOFS]
    mech_power = torch.sum(torch.abs(dof_forces * dof_vel), dim=-1)
    power_penalty = -self._power_penalty_weight * mech_power

    self._reward_buf[:] = (vel_reward + orientation_penalty + alive
                           + term_penalty + power_penalty)


def _make_post_physics_hook(original_fn, char_id):
    """Create a _post_physics_step wrapper that drives the CoT tracker.

    Called after the original _post_physics_step (which runs _update_reward
    then _update_done), so both _reward_buf and _done_buf are current.
    """
    def hooked_post_physics_step(self):
        original_fn(self)
        self._cot_tracker.accumulate(self._engine, char_id)
        self._cot_tracker.finalize_episodes(self._done_buf)
    return hooked_post_physics_step


def _make_compute_rewards_hook(agent):
    """Wrap _compute_rewards to also log raw task reward statistics.

    AMP's _compute_rewards reads task_r from the exp buffer, blends it with
    disc_r, and overwrites the buffer.  We capture task_r before the blend
    so it can be logged alongside disc_reward_mean/std.
    """
    original = agent._compute_rewards

    def hooked():
        task_r = agent._exp_buffer.get_data_flat("reward")
        task_reward_std, task_reward_mean = torch.std_mean(task_r)
        info = original()
        info["task_reward_mean"] = task_reward_mean
        info["task_reward_std"] = task_reward_std
        return info

    return hooked


# ---------------------------------------------------------------------------
# Design parameter optimizer
# ---------------------------------------------------------------------------

class AdamOptimizer:
    """Adam optimizer for numpy arrays (gradient ascent).

    Normalizes each parameter independently so effective step size is ~lr
    regardless of raw gradient scale. First-moment averaging (beta1=0.9)
    filters out sign flips from noisy per-iteration FD estimates.
    """

    def __init__(self, n_params, lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(n_params, dtype=np.float64)
        self.v = np.zeros(n_params, dtype=np.float64)
        self.t = 0

    def step(self, params, grad):
        """Gradient ascent: params += lr * adam_normalized(grad)."""
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
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

    world_indices = np.arange(num_worlds)
    for i, (left_body, right_body) in enumerate(SYMMETRIC_PAIRS):
        angle = float(theta_np[i])
        delta_q = quat_from_x_rotation(angle)
        for body_name in (left_body, right_body):
            if body_name not in joint_idx_map:
                continue
            base_q = base_quats_dict[body_name]
            new_q = quat_normalize(quat_multiply(delta_q, base_q))  # (w,x,y,z)
            local_idx = joint_idx_map[body_name]
            global_indices = world_indices * joints_per_world + local_idx
            # Convert wxyz → xyzw for joint_X_p numpy layout
            joint_X_p_np[global_indices, 3:7] = [new_q[1], new_q[2], new_q[3], new_q[0]]

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
                 convergence_signal='task_reward',
                 ramp_start_iter=1000, ramp_end_iter=2500,
                 use_wandb=False, viewer=None, engine=None,
                 video_interval=100, save_interval=100,
                 log_file=None,
                 is_root=True, device="cuda:0", rank=0):
        self.agent = agent
        self.plateau_threshold = plateau_threshold
        self.plateau_window = plateau_window
        self.min_plateau_outputs = min_plateau_outputs
        self.max_samples = max_samples
        self.min_inner_iters = min_inner_iters
        self.convergence_signal = convergence_signal
        self.ramp_start_iter = ramp_start_iter
        self.ramp_end_iter = ramp_end_iter
        self.ramp_completed = False  # set True after first ramp finishes
        self.use_wandb = use_wandb
        self.viewer = viewer
        self.engine = engine
        self.video_interval = video_interval
        self.save_interval = save_interval
        self.log_file = log_file
        self.is_root = is_root
        self.device = device
        self.rank = rank

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

        Convergence is detected when the configured signal (task_reward or
        disc_reward) plateaus at output iterations.  Default is task_reward
        (Train_Return), which plateaus cleanly.  disc_reward can decline
        indefinitely and produce false plateau triggers.

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
        convergence_values = []
        converged = False
        start_time = time.time()
        test_info = {"mean_return": 0.0, "mean_ep_len": 0.0, "num_eps": 0}
        inner_iters = 0

        while agent._sample_count < self.max_samples:
            train_info = agent._train_iter()                      # COLLECTIVE
            agent._sample_count = agent._update_sample_count()    # COLLECTIVE

            # Reward weight ramp: disc 1.0→0.0, task 0.0→1.0 over [ramp_start, ramp_end]
            # Only runs during kickoff; after ramp completes, stays at task=1.0/disc=0.0
            if not self.ramp_completed:
                it = inner_iters
                if it <= self.ramp_start_iter:
                    agent._task_reward_weight = 0.0
                    agent._disc_reward_weight = 1.0
                elif it >= self.ramp_end_iter:
                    agent._task_reward_weight = 1.0
                    agent._disc_reward_weight = 0.0
                    self.ramp_completed = True
                else:
                    t = (it - self.ramp_start_iter) / (self.ramp_end_iter - self.ramp_start_iter)
                    agent._task_reward_weight = t
                    agent._disc_reward_weight = 1.0 - t

            output_iter = (agent._iter % agent._iters_per_output == 0)

            if output_iter:
                # Track disc_reward (always, for diagnostics/return value)
                disc_r = train_info.get("disc_reward_mean")
                if disc_r is not None:
                    if torch.is_tensor(disc_r):
                        disc_r = disc_r.item()
                    disc_rewards.append(disc_r)

                # Track convergence signal (only after reward ramp completes,
                # since the changing reward landscape invalidates plateau detection)
                if self.ramp_completed:
                    if self.convergence_signal == 'task_reward':
                        cv = train_info.get("mean_return")
                        if cv is not None:
                            convergence_values.append(float(cv))
                    elif self.convergence_signal == 'disc_reward':
                        if disc_r is not None:
                            convergence_values.append(disc_r)

                test_info = {
                    "mean_return": 0.0,
                    "mean_ep_len": 0.0,
                    "num_eps": 0,
                }

            env_diag_info = env.get_diagnostics()
            agent._log_train_info(train_info, test_info, env_diag_info,
                                  start_time)                     # COLLECTIVE

            # All ranks must call print_log() — it contains a collective
            # reduce_inplace_mean. Actual printing is gated by Logger.is_root().
            agent._logger.print_log()

            if output_iter:
                # Write log at output intervals only — Train_Return accumulator
                # resets each output_iter, so per-iter writes create sawtooth.
                agent._logger.write_log()
                # Convergence check: root decides, broadcast to all ranks.
                # This MUST happen before root-only I/O (save, wandb, video)
                # to prevent non-root ranks from reaching the broadcast first
                # and hanging while root does slow I/O.
                should_stop = False
                if self.is_root:
                    should_stop = (inner_iters >= self.min_inner_iters
                                   and self._check_plateau(convergence_values))

                if mp_util.enable_mp():
                    flag = torch.tensor([int(should_stop)], dtype=torch.int32,
                                        device=self.device)
                    flag = mp_util.broadcast(flag)
                    should_stop = flag.item() == 1

                # Root-only I/O: save, wandb, video (after broadcast)
                if self.is_root:
                    # Save numbered checkpoint every save_interval iterations
                    if agent._iter % self.save_interval == 0:
                        numbered_path = str(out_dir / f"model_{agent._iter:06d}.pt")
                        agent.save(numbered_path)
                        if self.use_wandb:
                            wandb.save(numbered_path, base_path=str(out_dir))
                            if self.log_file:
                                wandb.save(self.log_file, policy="live")

                    # wandb: forward inner loop metrics + CoT
                    if self.use_wandb:
                        wlog = {}
                        for k, entry in agent._logger.log_current_row.items():
                            v = entry.val
                            try:
                                wlog[f"inner/{k}"] = float(v)
                            except (TypeError, ValueError):
                                pass

                        # Episode-averaged CoT from tracker
                        cot_tracker = env._cot_tracker
                        cot, mech_power, fwd_vel, n_eps = cot_tracker.get_stats()
                        if cot is not None:
                            wlog["inner/cot"] = cot
                            wlog["inner/mechanical_power"] = mech_power
                            wlog["inner/forward_velocity"] = fwd_vel
                            wlog["inner/cot_episodes"] = n_eps

                        # Reward weights so we can see actual contributions
                        wlog["inner/task_reward_weight"] = agent._task_reward_weight
                        wlog["inner/disc_reward_weight"] = agent._disc_reward_weight

                        if wlog:
                            wandb.log(wlog, step=agent._iter)

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
                                )}, step=agent._iter)
                        except Exception as e:
                            print(f"  [Inner Loop] Video logging failed: {e}")

                if should_stop:
                    if self.is_root:
                        recent = convergence_values[-self.plateau_window:]
                        spread = max(recent) - min(recent)
                        print(f"    [Inner] CONVERGED ({len(convergence_values)} outputs, "
                              f"{inner_iters} iters, "
                              f"{self.convergence_signal}={np.mean(recent):.4f}, "
                              f"spread={spread:.4f})")
                    converged = True
                    break

                # Reset after test (matches MimicKit train_model behavior)
                agent._train_return_tracker.reset()
                agent._curr_obs, agent._curr_info = agent._reset_envs()
                env._cot_tracker.reset_accumulators()

            agent._iter += 1
            inner_iters += 1

        # Save final checkpoint on convergence or sample cap
        if self.is_root:
            agent.save(checkpoint_path)
            if not converged:
                print(f"    [Inner] Sample cap reached ({agent._sample_count:,})")

        return converged, disc_rewards


# ---------------------------------------------------------------------------
# World-partitioned FD gradient
# ---------------------------------------------------------------------------

def apply_partitioned_morphologies(model, theta_np, eps, n_params,
                                   base_quats_dict, joint_idx_map,
                                   num_worlds, worlds_per_pert):
    """Apply all 13 perturbed morphologies to world partitions in one GPU round-trip.

    Partition layout (n_pert = 1 + 2*n_params = 13 for 6 params):
        partition 0:              center (theta)
        partition 2*i+1:          param i + eps
        partition 2*i+2:          param i - eps

    Each partition spans worlds [p * wpp, (p+1) * wpp).
    Worlds beyond n_pert * wpp are left untouched (they won't be read).
    """
    n_pert = 1 + 2 * n_params
    joints_per_world = model.joint_count // num_worlds
    joint_X_p_np = model.joint_X_p.numpy()

    for p in range(n_pert):
        # Compute theta for this perturbation
        if p == 0:
            theta_pert = theta_np  # center — no copy needed (read-only)
        else:
            param_i = (p - 1) // 2
            is_plus = (p % 2 == 1)
            theta_pert = theta_np.copy()
            theta_pert[param_i] += eps if is_plus else -eps

        w_start = p * worlds_per_pert
        w_end = (p + 1) * worlds_per_pert

        # Write quaternions for this partition's world range
        world_indices = np.arange(w_start, w_end)
        for i, (left_body, right_body) in enumerate(SYMMETRIC_PAIRS):
            angle = float(theta_pert[i])
            delta_q = quat_from_x_rotation(angle)
            for body_name in (left_body, right_body):
                if body_name not in joint_idx_map:
                    continue
                base_q = base_quats_dict[body_name]
                new_q = quat_normalize(quat_multiply(delta_q, base_q))  # (w,x,y,z)
                local_idx = joint_idx_map[body_name]
                global_indices = world_indices * joints_per_world + local_idx
                # Convert wxyz → xyzw for joint_X_p numpy layout
                joint_X_p_np[global_indices, 3:7] = [new_q[1], new_q[2], new_q[3], new_q[0]]

    model.joint_X_p.assign(joint_X_p_np)
    wp.synchronize()


def compute_fd_gradient_parallel(agent, env, engine, char_id, total_mass,
                                  theta_np, base_quats_dict, joint_idx_map,
                                  num_worlds, horizon,
                                  rank, num_procs, device, param_names,
                                  num_seeds=10, base_seed=42,
                                  eps=0.05, vel_reward_weight=0.1):
    """Compute closed-loop FD gradient with world-partitioned perturbations.

    Instead of running 130 sequential rollouts (13 perturbations x 10 seeds),
    partitions worlds into 13 groups — each running a different morphology
    simultaneously. The seed loop (10 iterations) remains sequential.
    Total: 10 rollouts instead of 130 (13x speedup).

    Each GPU partitions its own per_rank_envs worlds independently. Multi-GPU
    results are averaged via all_reduce(SUM) / num_procs.

    Returns (grad, fwd_dist_center, cot_center) on all ranks.
    """
    n_params = len(theta_np)
    n_pert = 1 + 2 * n_params  # 13 for 6 params
    wpp = num_worlds // n_pert  # worlds per perturbation
    assert num_worlds >= n_pert, f"Need >= {n_pert} worlds, got {num_worlds}"

    if rank == 0:
        print(f"    [FD-parallel] {n_pert} perturbations x {num_seeds} seeds, "
              f"{wpp} worlds/perturbation ({num_worlds} total), "
              f"{num_procs} GPU(s)")

    # Result tensors: (n_pert, num_seeds)
    cot_all = torch.zeros(n_pert, num_seeds, dtype=torch.float64, device=device)
    dist_all = torch.zeros(n_pert, num_seeds, dtype=torch.float64, device=device)

    sim_model = engine._sim_model
    _LOCO_DOFS = EpisodeCoTTracker.LOCOMOTION_DOFS

    # Apply all 13 morphologies to their world partitions (1 GPU<->CPU round-trip)
    apply_partitioned_morphologies(sim_model, theta_np, eps, n_params,
                                   base_quats_dict, joint_idx_map,
                                   num_worlds, wpp)

    agent.eval()
    agent.set_mode(base_agent_mod.AgentMode.TEST)

    for seed_idx in range(num_seeds):
        seed = base_seed + seed_idx
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        obs, info = env.reset()
        initial_root_x = engine.get_root_pos(char_id)[:, 0].clone()

        # Per-world episode-aware accumulators.
        # On done, we finalize the current episode's CoT and distance,
        # then reset accumulators so the next episode starts clean.
        dt_ctrl = 1.0 / 30.0  # control timestep
        power_sum = torch.zeros(num_worlds, device=device)
        vel_sum = torch.zeros(num_worlds, device=device)
        step_count = torch.zeros(num_worlds, device=device, dtype=torch.long)
        ep_cot_sum = torch.zeros(num_worlds, device=device)
        ep_dist_sum = torch.zeros(num_worlds, device=device)
        ep_count = torch.zeros(num_worlds, device=device, dtype=torch.long)
        seg_start_x = initial_root_x.clone()

        with torch.no_grad():
            for _ in range(horizon):
                action, _ = agent._decide_action(obs, info)
                obs, reward, done, info = env.step(action)

                dof_forces = EpisodeCoTTracker.get_dof_forces_safe(engine, char_id)
                dof_vel = engine.get_dof_vel(char_id)
                root_vel = engine.get_root_vel(char_id)

                power = torch.sum(torch.abs(
                    dof_forces[:num_worlds, :_LOCO_DOFS] *
                    dof_vel[:num_worlds, :_LOCO_DOFS]), dim=-1)
                power_sum += power
                vel_sum += root_vel[:num_worlds, 0]
                step_count += 1

                # Finalize episodes that just ended (env already auto-reset)
                done_mask = (done[:num_worlds] > 0)
                if done_mask.any():
                    di = done_mask.nonzero(as_tuple=True)[0]
                    vs = step_count[di].float()
                    valid = vs > 0
                    if valid.any():
                        vi = di[valid]
                        vsc = step_count[vi].float()
                        mp = power_sum[vi] / vsc
                        mv = vel_sum[vi] / vsc
                        safe_v = torch.sqrt(mv ** 2 + 0.01)
                        ep_cot_sum[vi] += mp / (total_mass * 9.81 * safe_v)
                        # Can't read pre-reset root_pos (env already reset),
                        # so integrate velocity: dist = sum(v_i) * dt_ctrl
                        ep_dist_sum[vi] += vel_sum[vi] * dt_ctrl
                        ep_count[vi] += 1
                    power_sum[di] = 0
                    vel_sum[di] = 0
                    step_count[di] = 0
                    seg_start_x[di] = engine.get_root_pos(char_id)[di, 0]

        # Finalize trailing partial episodes (worlds still alive at horizon)
        alive = step_count > 0
        if alive.any():
            ai = alive.nonzero(as_tuple=True)[0]
            vsc = step_count[ai].float()
            mp = power_sum[ai] / vsc
            mv = vel_sum[ai] / vsc
            safe_v = torch.sqrt(mv ** 2 + 0.01)
            ep_cot_sum[ai] += mp / (total_mass * 9.81 * safe_v)
            final_root_x = engine.get_root_pos(char_id)[:, 0]
            ep_dist_sum[ai] += final_root_x[ai] - seg_start_x[ai]
            ep_count[ai] += 1

        # Extract per-partition metrics
        for p in range(n_pert):
            s, e = p * wpp, (p + 1) * wpp
            counts = ep_count[s:e].float().clamp(min=1)
            cot = (ep_cot_sum[s:e] / counts).mean().item()
            fwd_dist = (ep_dist_sum[s:e] / counts).mean().item()
            cot_all[p, seed_idx] = cot
            dist_all[p, seed_idx] = fwd_dist

    agent.set_mode(base_agent_mod.AgentMode.TRAIN)

    # Multi-GPU: average across ranks
    if num_procs > 1:
        torch.distributed.all_reduce(cot_all, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(dist_all, op=torch.distributed.ReduceOp.SUM)
        cot_all /= num_procs
        dist_all /= num_procs

    # Restore original morphology
    update_training_joint_X_p(sim_model, theta_np, base_quats_dict,
                               joint_idx_map, num_worlds)

    # Gradient computation (same math as sequential version)
    cot_np = cot_all.cpu().numpy()
    dist_np = dist_all.cpu().numpy()

    # Center metrics (mean over seeds)
    cot_center = float(np.mean(cot_np[0]))
    fwd_dist_center = float(np.mean(dist_np[0]))

    # reward = -CoT + vel_reward_weight * fwd_dist
    reward_all = -cot_np + vel_reward_weight * dist_np  # (n_pert, num_seeds)

    grad = np.zeros(n_params, dtype=np.float64)
    grad_stderr = np.zeros(n_params, dtype=np.float64)
    for i in range(n_params):
        reward_plus = reward_all[2 * i + 1]   # (num_seeds,)
        reward_minus = reward_all[2 * i + 2]  # (num_seeds,)
        diffs = (reward_plus - reward_minus) / (2 * eps)  # (num_seeds,)

        valid = ~np.isnan(diffs)
        if valid.sum() > 0:
            grad[i] = np.mean(diffs[valid])
            if valid.sum() > 1:
                grad_stderr[i] = np.std(diffs[valid], ddof=1) / np.sqrt(valid.sum())
        else:
            grad[i] = 0.0

    grad = np.clip(grad, -10.0, 10.0)

    if rank == 0:
        print(f"    [FD] Center (mean over {num_seeds} seeds): "
              f"fwd_dist={fwd_dist_center:.4f} m, CoT={cot_center:.4f}")
        for i, name in enumerate(param_names):
            cot_plus_mean = float(np.mean(cot_np[2 * i + 1]))
            cot_minus_mean = float(np.mean(cot_np[2 * i + 2]))
            dist_plus_mean = float(np.mean(dist_np[2 * i + 1]))
            dist_minus_mean = float(np.mean(dist_np[2 * i + 2]))
            print(f"    [FD] param {i} ({name}): "
                  f"grad={grad[i]:+.6f} +/- {grad_stderr[i]:.6f}, "
                  f"CoT+={cot_plus_mean:.6f}, CoT-={cot_minus_mean:.6f}, "
                  f"dist+={dist_plus_mean:.3f}, dist-={dist_minus_mean:.3f}")

    return grad, fwd_dist_center, cot_center


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

    Each GPU rank runs this function. The inner loop (MimicKit training) and
    FD gradient rollouts are both distributed across all ranks via
    torch.distributed collective ops. Adam update runs on rank 0; theta is
    broadcast after.
    """
    is_root = (rank == 0)
    per_rank_envs = args.num_train_envs // num_procs

    # Explicitly set the CUDA device before any GPU operations.
    # With CUDA_VISIBLE_DEVICES isolation, each process sees one GPU as cuda:0.
    if "cuda" in device:
        torch.cuda.set_device(device)

    # Seed per rank for diverse rollouts
    seed = int(np.uint64(42) + np.uint64(41 * rank))
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if "cuda" in device:
        torch.cuda.manual_seed(seed)

    # Initialize torch.distributed with backend override
    # (bypasses mp_util.init so we can control the backend via --dist-backend)
    mp_util.global_mp_device = device
    mp_util.global_num_procs = num_procs

    if num_procs > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)

        backend = getattr(args, "dist_backend", "nccl")
        if backend == "auto":
            if "cuda" in device:
                backend = "nccl"
                os_platform = platform.system()
                if os_platform == "Windows":
                    backend = "gloo"
            else:
                backend = "gloo"

        print(f"  [Rank {rank}] init_process_group(backend={backend}, "
              f"rank={rank}, world_size={num_procs})", flush=True)
        torch.distributed.init_process_group(backend, rank=rank,
                                             world_size=num_procs)
        print(f"  [Rank {rank}] init_process_group DONE", flush=True)

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
        print("\n[1/3] Initializing MimicKit...")

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
        print("[2/3] Building training env and agent...")

    # Build env/agent with staggered startup to avoid Warp NVRTC cache race.
    # When the Newton submodule is updated, all Warp kernels need recompilation.
    # If all workers build simultaneously, they race on the shared disk cache.
    # Rank 0 builds first (compiling + caching kernels), then the rest build
    # and find the warm cache.
    if mp_util.enable_mp():
        if is_root:
            env = env_builder.build_env(
                str(modified_env_config), str(BASE_ENGINE_CONFIG),
                per_rank_envs, device, visualize=False,
            )
            wp.synchronize()
        torch.distributed.barrier()  # non-root waits for rank 0's cache to be warm
        if not is_root:
            env = env_builder.build_env(
                str(modified_env_config), str(BASE_ENGINE_CONFIG),
                per_rank_envs, device, visualize=False,
            )
    else:
        env = env_builder.build_env(
            str(modified_env_config), str(BASE_ENGINE_CONFIG),
            per_rank_envs, device, visualize=False,
        )
    # Monkey-patch task reward onto env (replaces empty _update_reward)
    env._power_penalty_weight = args.power_penalty_weight
    env._update_reward = types.MethodType(codesign_task_reward, env)

    agent = agent_builder.build_agent(str(BASE_AGENT_CONFIG), env, device)

    # Ensure disc replay buffer can hold a full rollout batch.
    # With 16384 envs x 32 steps_per_iter = 524k samples, but default buffer is 200k.
    import learning.experience_buffer as experience_buffer_mod
    min_disc_buf = per_rank_envs * agent._config["steps_per_iter"]
    if agent._disc_buffer._buffer_length < min_disc_buf:
        new_size = min_disc_buf * 2  # 2x headroom
        if is_root:
            print(f"  [Fix] Expanding disc_buffer: {agent._disc_buffer._buffer_length} -> {new_size} "
                  f"(need >= {min_disc_buf} for {per_rank_envs} envs x {agent._config['steps_per_iter']} steps)")
        agent._disc_buffer = experience_buffer_mod.ExperienceBuffer(
            buffer_length=new_size, batch_size=1, device=device)

    # Initial reward weights: pure style learning (ramp handles transition)
    agent._task_reward_weight = 0.0
    agent._disc_reward_weight = 1.0

    # Monkey-patch _compute_rewards to also log raw task reward stats
    agent._compute_rewards = _make_compute_rewards_hook(agent)

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
    char_id = env._get_char_id()
    total_mass = engine.calc_obj_mass(0, char_id)
    base_quats_dict, joint_idx_map = extract_joint_info(
        sim_model, per_rank_envs
    )
    if is_root:
        print(f"  Robot mass: {total_mass:.2f} kg")

    # Episode-averaged CoT tracker (hooked into _post_physics_step)
    env._cot_tracker = EpisodeCoTTracker(per_rank_envs, total_mass, device)
    _orig_post_physics = type(env)._post_physics_step
    env._post_physics_step = types.MethodType(
        _make_post_physics_hook(_orig_post_physics, char_id), env
    )

    # Headless viewer for video capture (root only, skip on HPC / --no-video)
    viewer = None
    if use_wandb and not args.no_video:
        try:
            viewer = ViewerGL(headless=True, width=640, height=480)
            viewer.set_model(sim_model, max_worlds=1)
            viewer.set_camera(
                pos=wp.vec3(3.0, -3.0, 1.5),
                pitch=-10.0,
                yaw=90.0,
            )
            print("  [Viewer] Headless viewer created for video capture")
        except BaseException as e:
            # Catch SystemExit too — ViewerGL may call exit() on framebuffer failure
            print(f"  [Viewer] Failed to create viewer: {e}")
            viewer = None

    # =========================================================
    # Outer loop setup
    # =========================================================
    if is_root:
        print("[3/3] Starting PGHC outer loop...\n")

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
        ramp_start_iter=args.ramp_start_iter,
        ramp_end_iter=args.ramp_end_iter,
        use_wandb=use_wandb,
        viewer=viewer,
        engine=engine,
        video_interval=args.video_interval,
        save_interval=args.save_interval,
        log_file=log_file,
        is_root=is_root,
        device=device,
        rank=rank,
    )

    # Skip kickoff ramp when resuming from a checkpoint (the resumed policy
    # was already trained with task reward; re-ramping would destroy it).
    if args.resume_checkpoint:
        inner_ctrl.ramp_completed = True
        inner_ctrl.min_inner_iters = args.post_kickoff_min_iters
        agent._task_reward_weight = 1.0
        agent._disc_reward_weight = 0.0
        if is_root:
            print(f"  [Resume] Skipping kickoff ramp (ramp_completed=True, "
                  f"min_inner_iters={inner_ctrl.min_inner_iters})")

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
        print(f"  Eval horizon:      {args.eval_horizon} steps "
              f"({args.eval_horizon / 30:.1f}s)")
        print(f"  Max inner samples: {args.max_inner_samples:,}")
        print(f"  Design optimizer:  Adam (lr={args.design_lr})")
        print(f"  Design params:     {NUM_DESIGN_PARAMS} (symmetric lower-body)")
        print(f"  Theta bounds:      +/-30 deg (+/-0.5236 rad)")
        if args.kickoff_min_iters > 0:
            print(f"  Kickoff min iters: {args.kickoff_min_iters} (first outer iter only)")
        print(f"  Reward ramp:       disc 1.0→0.0, task 0.0→1.0 "
              f"(iter {args.ramp_start_iter}→{args.ramp_end_iter})")
        print(f"  FD seeds:          {args.num_fd_seeds} (paired-seed averaging)")
        print(f"  FD epsilon:        {args.fd_epsilon} rad "
              f"({np.degrees(args.fd_epsilon):.1f} deg)")
        print(f"  FD vel weight:     {args.vel_reward_weight} "
              f"(reward = -CoT + {args.vel_reward_weight}*dist)")
        print(f"  FD mode:           closed-loop (policy reacts to observations)")
        print(f"  Power penalty:     {args.power_penalty_weight}")

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

        if use_wandb:
            wandb.log({
                "outer/iteration": outer_iter + 1,
                "outer/boundary": 1,
            }, step=agent._iter)

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

        # After kickoff, disable min_inner_iters gate for subsequent iters.
        # Reward weights are now handled by the ramp inside train_until_converged
        # (disc 1.0→0.0, task 0.0→1.0 over ramp_start..ramp_end iters).
        if outer_iter == 0 and inner_ctrl.min_inner_iters > 0:
            if is_root:
                print(f"  [Kickoff] Disabling min_inner_iters "
                      f"({inner_ctrl.min_inner_iters}) for subsequent iters")
            inner_ctrl.min_inner_iters = args.post_kickoff_min_iters

        # ----- FD Gradient Phase (ALL ranks, closed-loop) -----
        # Clear CoT tracker before FD to prevent cross-contamination
        env._cot_tracker.get_stats()  # drain any accumulated episodes
        env._cot_tracker.reset_accumulators()

        if is_root:
            print(f"\n  [FD] Computing closed-loop FD gradient "
                  f"({args.num_fd_seeds} seeds, {args.eval_horizon} steps)...")

        fd_base_seed = 42 + outer_iter * 1000

        grad_theta, fwd_dist, cot = compute_fd_gradient_parallel(
            agent, env, engine, char_id, total_mass,
            theta.copy(), base_quats_dict, joint_idx_map,
            per_rank_envs, args.eval_horizon,
            rank, num_procs, device, param_names,
            num_seeds=args.num_fd_seeds, base_seed=fd_base_seed,
            eps=args.fd_epsilon, vel_reward_weight=args.vel_reward_weight,
        )

        # Clear CoT tracker after FD to prevent pollution of inner loop
        env._cot_tracker.get_stats()
        env._cot_tracker.reset_accumulators()

        if is_root:
            print(f"    FD gradients:")
            for i, name in enumerate(param_names):
                print(f"      d_reward/d_{name} = {grad_theta[i]:+.6f}")
            print(f"    Forward distance = {fwd_dist:.3f} m")
            print(f"    Cost of Transport = {cot:.4f}")

            history["forward_dist"].append(fwd_dist)
            history["cot"].append(cot)
            history["gradients"].append(grad_theta.copy())

            # ----- Design Update (rank 0 only) -----
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
        # Non-root ranks block here until root finishes FD gradient
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

        # ----- Outer convergence check -----
        # Broadcast BEFORE root-only saves to prevent non-root ranks from
        # reaching the broadcast first and hanging while root does slow I/O.
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

        # Root: save checkpoints + wandb (after broadcast, non-root ranks
        # proceed to next iteration or break without waiting)
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
                wandb.log(log_dict, step=agent._iter)

                # Save artifacts to wandb files
                wandb.save(str(iter_dir / "model.pt"), base_path=str(out_dir))
                wandb.save(str(iter_dir / "theta.npy"), base_path=str(out_dir))
                wandb.save(str(iter_dir / "grad.npy"), base_path=str(out_dir))
                wandb.save(str(out_dir / "theta_latest.npy"), base_path=str(out_dir))

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
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--outer-iters", type=int, default=20)
    parser.add_argument("--design-lr", type=float, default=0.005)
    parser.add_argument("--num-train-envs", type=int, default=4096,
                        help="Total training envs (divided across GPUs)")
    parser.add_argument("--eval-horizon", type=int, default=300,
                        help="Eval rollout length in control steps (300 = 10s full episode)")
    parser.add_argument("--num-fd-seeds", type=int, default=10,
                        help="Number of paired seeds per FD perturbation (K)")
    parser.add_argument("--fd-epsilon", type=float, default=0.05,
                        help="FD perturbation size in radians (default: 0.05 ~ 2.9 deg)")
    parser.add_argument("--max-inner-samples", type=int, default=2_000_000_000)
    parser.add_argument("--out-dir", type=str, default="output_g1_unified")
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="MimicKit checkpoint to resume from")
    parser.add_argument("--plateau-threshold", type=float, default=0.002,
                        help="Inner convergence: spread/mean threshold")
    parser.add_argument("--plateau-window", type=int, default=10,
                        help="Inner convergence: window size (in output intervals)")
    parser.add_argument("--min-plateau-outputs", type=int, default=20,
                        help="Inner convergence: min output intervals before early stop")
    parser.add_argument("--save-interval", type=int, default=500,
                        help="Save numbered model checkpoint every N inner iterations")
    parser.add_argument("--kickoff-min-iters", type=int, default=2000,
                        help="Min inner iters before convergence on first outer iter (0=disabled)")
    parser.add_argument("--ramp-start-iter", type=int, default=1000,
                        help="Inner iter to start ramping disc→task reward weights")
    parser.add_argument("--ramp-end-iter", type=int, default=2500,
                        help="Inner iter to finish ramping (disc=0, task=1)")
    parser.add_argument("--post-kickoff-min-iters", type=int, default=100,
                        help="Min inner iters for convergence after first outer iter")
    parser.add_argument("--vel-reward-weight", type=float, default=0.1,
                        help="Weight for forward distance in outer loop FD objective "
                             "(reward = -CoT + w * fwd_dist)")
    parser.add_argument("--no-video", action="store_true",
                        help="Disable headless viewer (use on HPC nodes without EGL)")
    parser.add_argument("--video-interval", type=int, default=100,
                        help="Log video to wandb every N inner iterations")
    parser.add_argument("--power-penalty-weight", type=float, default=0.0005,
                        help="Weight for mechanical power penalty in task reward "
                             "(aligns inner loop with outer loop CoT objective)")
    # Multi-GPU arguments
    parser.add_argument("--devices", nargs="+", default=None,
                        help="CUDA devices (default: auto-detect all GPUs)")
    parser.add_argument("--master-port", type=int, default=None,
                        help="Port for torch.distributed (default: random 6000-7000)")
    parser.add_argument("--dist-backend", type=str, default="nccl",
                        choices=["nccl", "gloo", "auto"],
                        help="torch.distributed backend "
                             "(try 'gloo' if nccl hangs, 'auto' for platform default)")
    args = parser.parse_args()

    # Auto-detect GPUs if --devices not specified
    if args.devices is None:
        n_gpus = torch.cuda.device_count()
        args.devices = [f"cuda:{i}" for i in range(n_gpus)] if n_gpus > 0 else ["cuda:0"]

    # Flip --no-wandb to args.wandb for downstream use
    args.wandb = not args.no_wandb

    num_workers = len(args.devices)
    assert args.num_train_envs % num_workers == 0, (
        f"--num-train-envs ({args.num_train_envs}) must be divisible by "
        f"number of devices ({num_workers})"
    )

    # Parse logical GPU indices from device strings (e.g., "cuda:2" -> 2)
    gpu_indices = []
    for d in args.devices:
        assert "cuda" in d, f"Expected CUDA device, got {d}"
        gpu_indices.append(int(d.split(":")[1]))

    master_port = args.master_port if args.master_port is not None else random.randint(6000, 7000)

    if num_workers > 1:
        # Spawn ALL ranks (including rank 0) as subprocesses so that every
        # worker re-imports the module after CUDA_VISIBLE_DEVICES is set.
        # This avoids the root-process CUDA visibility asymmetry that causes
        # NCCL topology mismatches and all_reduce deadlocks.
        torch.multiprocessing.set_start_method("spawn")
        processes = []
        for r in range(num_workers):
            os.environ["_PGHC_GPU_INDEX"] = str(gpu_indices[r])
            p = torch.multiprocessing.Process(
                target=pghc_worker,
                args=[r, num_workers, "cuda:0", master_port, args],
            )
            p.start()
            processes.append(p)
        os.environ.pop("_PGHC_GPU_INDEX", None)

        # Wait for all workers and check exit codes
        failed = []
        for r, p in enumerate(processes):
            p.join()
            if p.exitcode != 0:
                failed.append((r, p.exitcode))
        if failed:
            print(f"\n[ERROR] {len(failed)} worker(s) failed:")
            for r, code in failed:
                print(f"  Rank {r}: exit code {code}")
            sys.exit(1)
    else:
        # Single GPU — backward compatible, no spawn overhead
        pghc_worker(0, 1, args.devices[0], master_port, args)
