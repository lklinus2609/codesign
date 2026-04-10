#!/usr/bin/env python3
"""
Unified Single-Process GBC (Gradient-Based Co-Design) for G1 Humanoid

Eliminates subprocess overhead by importing MimicKit as a library and keeping
all GPU resources persistent across outer iterations.

Architecture:
    Single process per GPU, single CUDA context per rank:
    1. Build MimicKit AMPEnv + AMPAgent ONCE (N/num_gpus training worlds per GPU)
    2. OUTER LOOP:
        a. Update training model joint_X_p in-place from theta
        b. Inner loop: agent._train_iter() until convergence (all GPUs)
        c. Orthogonal SPSA gradient: run frozen policy on S*N direction-perturbed
           morphologies with K paired seeds, reconstruct via lstsq, all_reduce
        d. BFGS update theta on rank 0, clip to +/-30 deg
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
# Spawned worker processes inherit _GBC_GPU_INDEX from the parent.  We read it
# here — at module import time — and restrict CUDA_VISIBLE_DEVICES to that
# single physical GPU.  This ensures warp/torch/newton only see one device,
# avoiding cross-device allocation failures on HPC nodes.
# ---------------------------------------------------------------------------
_GBC_GPU_INDEX = os.environ.pop("_GBC_GPU_INDEX", None)
if _GBC_GPU_INDEX is not None:
    _orig_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if _orig_cvd:
        _gpu_list = [g.strip() for g in _orig_cvd.split(",")]
        _idx = int(_GBC_GPU_INDEX)
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            _gpu_list[_idx] if _idx < len(_gpu_list) else _GBC_GPU_INDEX
        )
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _GBC_GPU_INDEX

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
# compiling kernels simultaneously (e.g. during SPSA evaluation phase).
if _GBC_GPU_INDEX is not None:
    _warp_cache = os.path.join(
        os.path.expanduser("~"), ".cache", "warp_per_rank", f"rank_{_GBC_GPU_INDEX}"
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

# NOTE: MimicKit's newton_engine.py sets wp.config.enable_backward = False.
# With closed-loop SPSA (no BPTT), we never need backward mode enabled.

# Co-design modules
from g1_mjcf_modifier import (
    G1MJCFModifier, SYMMETRIC_PAIRS, NUM_DESIGN_PARAMS,
    quat_from_x_rotation, quat_multiply, quat_normalize,
)
from convergence_gate import CompositeConvergenceGate
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------
GRAVITY = 9.81
VELOCITY_EPSILON = 0.01  # smoothing for safe_v = sqrt(v^2 + eps)
TERM_PENALTY = -10.0
THETA_BOUNDS = (-np.radians(30), np.radians(30))  # +/-30 deg

_EMPTY_TEST_INFO = {"mean_return": 0.0, "mean_ep_len": 0.0, "num_eps": 0}


def compute_cot(mean_power, mean_fwd_vel, total_mass):
    """Cost of Transport: power / (m * g * |v|), with velocity smoothing."""
    safe_v = torch.sqrt(mean_fwd_vel ** 2 + VELOCITY_EPSILON)
    return mean_power / (total_mass * GRAVITY * safe_v)


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

    Counts all DoFs (legs, torso, and arms) for power.

    Must be called via ``accumulate()`` + ``finalize_episodes()`` every env
    step (hooked into ``_post_physics_step``).
    """

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
            cots = compute_cot(mp, mv, self.total_mass)
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
    """Task reward aligned with outer loop objective: -CoT + w * vel_tracking.

    Replaces AMPEnv._update_reward (which is empty by default) so that
    the agent's reward blending picks up a non-zero task_r:
        r = task_reward_weight * task_r + disc_reward_weight * disc_r

    Uses the same cost function as the outer loop SPSA objective:
        reward = -0.1*CoT + vel_reward_weight * exp(-|v_x - v_cmd|^2 / sigma)
    """
    char_id = self._get_char_id()
    root_vel = self._engine.get_root_vel(char_id)   # (num_envs, 3)
    fwd_vel = root_vel[:, 0]

    # 1. Per-step CoT: power / (m * g * safe_v)
    dof_forces = EpisodeCoTTracker.get_dof_forces_safe(self._engine, char_id)
    dof_vel = self._engine.get_dof_vel(char_id)
    mech_power = torch.sum(torch.abs(dof_forces * dof_vel), dim=-1)
    cot = compute_cot(mech_power, fwd_vel, self._total_mass)

    # 2. Velocity tracking: exp(-|v_x - v_cmd|^2 / sigma)
    vel_tracking = torch.exp(-torch.square(fwd_vel - self._vel_cmd) / self._vel_tracking_sigma)
    vel_reward = self._vel_reward_weight * vel_tracking

    # 3. Termination penalty
    term_penalty = TERM_PENALTY * (self._done_buf == base_env_mod.DoneFlags.FAIL.value).float()

    self._reward_buf[:] = -0.1 * cot + vel_reward + term_penalty


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

class CautiousBFGS:
    """Cautious BFGS for morphology optimization (gradient ascent).

    Maintains an explicit n x n inverse Hessian approximation H_k, updated
    each iteration via the BFGS formula.  At 6-20 params this is trivial
    (6x6 = 288 bytes).  Full BFGS uses the entire history of accepted pairs,
    giving a better Hessian estimate than limited-memory L-BFGS.

    Cautious: skips the Hessian update when s^T y <= 0 (negative or zero
    curvature from stochastic SPSA noise).  H_k stays at its previous value
    rather than being corrupted by a bad update.

    API is split into compute_direction / update so that bounds clipping
    (and future trust region) can intervene between proposing and recording
    the actual step.
    """

    COND_RESET_THRESHOLD = 1e6

    def __init__(self, n_params):
        self.n = n_params
        self.H = np.eye(n_params, dtype=np.float64)
        self.prev_grad = None
        self.initialized = False  # first scaling not yet applied
        self.num_updates = 0
        self.num_skipped = 0

    def compute_direction(self, grad):
        """Return H_k @ grad (inverse-Hessian-scaled ascent direction)."""
        return self.H @ grad

    def update(self, actual_s, grad_new):
        """Update inverse Hessian with the actual step taken (post-clip).

        Since we do gradient ascent (maximizing reward f), the equivalent
        minimization problem is h = -f with gradient -g.  The BFGS y vector
        for minimization is: y = (-g_new) - (-g_old) = g_old - g_new.
        This ensures s^T y > 0 when approaching a maximum (gradient shrinks).

        Applies cautious filter: only updates H when s^T y > 0.
        Resets H to identity if condition number exceeds threshold.

        Must be called every outer iteration (even on the first, to store
        prev_grad for the next iteration's y computation).
        """
        if self.prev_grad is not None:
            y = self.prev_grad - grad_new  # negated for ascent→minimization
            sTy = actual_s @ y

            if sTy > 1e-10:
                # One-time initial scaling (Shanno-Phua): H_0 <- gamma * I
                # before the very first BFGS update, so the Hessian scale
                # matches local curvature rather than being identity.
                if not self.initialized:
                    yTy = y @ y
                    if yTy > 1e-20:
                        gamma = sTy / yTy
                        self.H = gamma * np.eye(self.n, dtype=np.float64)
                    self.initialized = True

                # Standard BFGS inverse Hessian update (rank-2)
                rho = 1.0 / sTy
                I_n = np.eye(self.n, dtype=np.float64)
                V = I_n - rho * np.outer(actual_s, y)
                W = I_n - rho * np.outer(y, actual_s)
                self.H = V @ self.H @ W + rho * np.outer(actual_s, actual_s)
                self.num_updates += 1

                # Safety: reset if H becomes poorly conditioned
                cond = np.linalg.cond(self.H)
                if cond > self.COND_RESET_THRESHOLD:
                    print(f"    [BFGS] Condition number {cond:.1e} exceeds "
                          f"threshold — resetting H to identity")
                    self.H = np.eye(self.n, dtype=np.float64)
                    self.initialized = False
            else:
                self.num_skipped += 1

        self.prev_grad = grad_new.copy()


# ---------------------------------------------------------------------------
# Joint info extraction + in-place update
# ---------------------------------------------------------------------------

def extract_joint_info(model, num_worlds):
    """Extract base quaternions and joint index map from Newton model.

    Returns:
        base_quats_dict: {body_name: (w,x,y,z)} for parameterized bodies
        joint_idx_map:   {body_name: local_joint_index}
    """
    joint_keys = model.joint_key
    joints_per_world = model.joint_count // num_worlds

    joint_name_to_idx = {joint_keys[i]: i for i in range(joints_per_world)}

    joint_X_p_np = model.joint_X_p.numpy()
    base_quats_dict = {}
    joint_idx_map = {}

    for left_body, right_body in SYMMETRIC_PAIRS:
        for body_name in (left_body, right_body):
            joint_name = body_name.replace("_link", "_joint")
            if joint_name in joint_name_to_idx:
                idx = joint_name_to_idx[joint_name]
            else:
                print(f"  [WARN] Joint not found: {joint_name} (body: {body_name})")
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
                 min_plateau_outputs=10, max_inner_iters=10000,
                 min_inner_iters=0,
                 convergence_signal='task_reward',
                 convergence_mode='codesign',
                 convergence_window=10,
                 critic_loss_plateau_threshold=0.05,
                 adv_std_threshold=0.1,
                 actor_loss_plateau_threshold=0.05,
                 kl_conv_threshold=1e-4, kl_window=10,
                 use_kl_convergence=True,
                 ep_len_threshold_frac=0.9,
                 task_plateau_threshold=0.02,
                 task_plateau_window=10,
                 ramp_start_iter=1000, ramp_end_iter=2500,
                 use_wandb=False, viewer=None, engine=None,
                 video_interval=100, save_interval=100,
                 log_file=None,
                 is_root=True, device="cuda:0", rank=0):
        self.agent = agent
        self.plateau_threshold = plateau_threshold
        self.plateau_window = plateau_window
        self.min_plateau_outputs = min_plateau_outputs
        self.max_inner_iters = max_inner_iters
        self.min_inner_iters = min_inner_iters
        self.convergence_signal = convergence_signal
        self.convergence_mode = convergence_mode
        self.kl_conv_threshold = kl_conv_threshold
        self.kl_window = kl_window
        self.use_kl_convergence = use_kl_convergence
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

        # Codesign convergence: ep_len gate + task_reward plateau
        self.ep_len_threshold_frac = ep_len_threshold_frac
        self.task_plateau_threshold = task_plateau_threshold
        self.task_plateau_window = task_plateau_window
        self._ep_len_gate_passed = False
        self._task_plateau_values = []
        self._last_ep_len = 0.0

        # Composite convergence gate (MimicKit signals)
        self.convergence_gate = CompositeConvergenceGate(window=convergence_window)
        self.convergence_gate.add_signal(
            'critic_loss', 'plateau', critic_loss_plateau_threshold)
        self.convergence_gate.add_signal(
            'adv_std', 'below', adv_std_threshold)
        self.convergence_gate.add_signal(
            'actor_loss', 'plateau', actor_loss_plateau_threshold)

    # MimicKit logger keys forwarded to wandb (allowlist)
    _INNER_WANDB_KEYS = {
        "Train_Return", "Train_Episode_Length",
        "Critic_Loss", "Actor_Loss",
        "Disc_Reward_Mean",
    }

    def _log_wandb_inner(self):
        """Forward essential inner loop metrics to wandb."""
        agent = self.agent
        env = agent._env
        wlog = {}
        for k, entry in agent._logger.log_current_row.items():
            if k not in self._INNER_WANDB_KEYS:
                continue
            v = entry.val
            try:
                wlog[f"inner/{k}"] = float(v)
            except (TypeError, ValueError):
                pass
        cot_tracker = env._cot_tracker
        cot, mech_power, fwd_vel, n_eps = cot_tracker.get_stats()
        if cot is not None:
            wlog["inner/cot"] = cot
            wlog["inner/forward_velocity"] = fwd_vel
        wlog["inner/task_reward_weight"] = agent._task_reward_weight
        if wlog:
            wandb.log(wlog, step=agent._iter)

    def _check_plateau(self, values):
        n_required = max(self.plateau_window, self.min_plateau_outputs)
        if len(values) < n_required:
            return False
        recent = values[-self.plateau_window:]
        mean_val = np.mean(recent)
        spread = max(recent) - min(recent)
        if abs(mean_val) < 1.0:
            # Near zero: use absolute spread threshold
            return spread < self.plateau_threshold
        return (spread / abs(mean_val)) < self.plateau_threshold

    def _check_kl_converged(self, kl_values):
        if len(kl_values) < self.kl_window:
            return False
        recent = list(kl_values)[-self.kl_window:]
        return np.mean(recent) < self.kl_conv_threshold

    def _check_task_plateau(self):
        """Check if task_reward has plateaued (codesign convergence stage 2)."""
        if len(self._task_plateau_values) < self.task_plateau_window:
            return False
        recent = self._task_plateau_values[-self.task_plateau_window:]
        mean_val = np.mean(recent)
        spread = max(recent) - min(recent)
        if abs(mean_val) < 1.0:
            return spread < self.task_plateau_threshold
        return (spread / abs(mean_val)) < self.task_plateau_threshold

    def train_until_converged(self, out_dir):
        """Run inner training loop until convergence or iteration cap.

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

        # Restore reward weights after _init_train() which may reset them.
        # When the ramp has completed, we must stay at the final blend;
        # otherwise the ramp logic below will set them on the first iteration.
        if self.ramp_completed:
            agent._task_reward_weight = 0.5
            agent._disc_reward_weight = 0.5

        disc_rewards = []
        convergence_values = []
        kl_values = []
        self.convergence_gate.reset()
        self._ep_len_gate_passed = False
        self._task_plateau_values = []
        self._last_ep_len = 0.0
        # Max episode length in steps (episode_length_s * control_freq)
        self._max_ep_steps = env._episode_length * 30
        converged = False
        start_time = time.time()
        test_info = _EMPTY_TEST_INFO
        inner_iters = 0
        last_actor_loss = None

        while inner_iters < self.max_inner_iters:
            train_info = agent._train_iter()                      # COLLECTIVE
            agent._sample_count = agent._update_sample_count()    # COLLECTIVE

            # Reward weight ramp: disc 1.0→0.1, task 0.0→0.9 over [ramp_start, ramp_end]
            # Only runs during kickoff; after ramp completes, stays at task=0.9/disc=0.1
            if not self.ramp_completed:
                it = inner_iters
                if it <= self.ramp_start_iter:
                    agent._task_reward_weight = 0.0
                    agent._disc_reward_weight = 1.0
                elif it >= self.ramp_end_iter:
                    agent._task_reward_weight = 0.5
                    agent._disc_reward_weight = 0.5
                    self.ramp_completed = True
                else:
                    t = (it - self.ramp_start_iter) / (self.ramp_end_iter - self.ramp_start_iter)
                    agent._task_reward_weight = 0.5 * t
                    agent._disc_reward_weight = 1.0 - 0.5 * t

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
                        cv = train_info.get("task_reward_mean")
                        if cv is not None:
                            convergence_values.append(float(cv))
                    elif self.convergence_signal == 'disc_reward':
                        if disc_r is not None:
                            convergence_values.append(disc_r)

                    # Track approx_kl for KL-based convergence
                    akl = train_info.get("approx_kl")
                    if akl is not None:
                        if torch.is_tensor(akl):
                            akl = akl.item()
                        kl_values.append(abs(akl))

                # Track actor loss as envelope theorem quality proxy
                al = train_info.get("actor_loss")
                if al is not None:
                    if torch.is_tensor(al):
                        al = al.item()
                    last_actor_loss = al

                # Feed composite convergence gate (only after ramp completes)
                if self.ramp_completed:
                    gate_metrics = {}
                    for key in ('critic_loss', 'adv_std', 'actor_loss'):
                        v = train_info.get(key)
                        if v is not None:
                            gate_metrics[key] = v.item() if torch.is_tensor(v) else float(v)
                    self.convergence_gate.update(gate_metrics)

                # Codesign convergence: ep_len gate + task_reward plateau
                # (extract mean_ep_len before _log_train_info pops it)
                if self.ramp_completed and self.convergence_mode == 'codesign':
                    ep_len = train_info.get("mean_ep_len")
                    if ep_len is not None:
                        ep_len = ep_len.item() if torch.is_tensor(ep_len) else float(ep_len)
                        self._last_ep_len = ep_len
                        if ep_len >= self.ep_len_threshold_frac * self._max_ep_steps:
                            self._ep_len_gate_passed = True
                        if self._ep_len_gate_passed:
                            tv = train_info.get("task_reward_mean")
                            if tv is not None:
                                self._task_plateau_values.append(
                                    tv.item() if torch.is_tensor(tv) else float(tv))

                test_info = _EMPTY_TEST_INFO

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
                if self.is_root and inner_iters >= self.min_inner_iters:
                    if self.convergence_mode == 'codesign':
                        should_stop = (self._ep_len_gate_passed
                                       and self._check_task_plateau())
                    elif self.convergence_mode == 'composite':
                        should_stop = self.convergence_gate.check_converged()
                    elif self.use_kl_convergence:
                        should_stop = self._check_kl_converged(kl_values)
                    else:
                        should_stop = self._check_plateau(convergence_values)

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
                                wandb.save(self.log_file, base_path=str(Path(self.log_file).parent), policy="live")

                    if self.use_wandb:
                        self._log_wandb_inner()
                        # Log gate diagnostics for active convergence mode
                        if self.convergence_mode == 'codesign':
                            wandb.log({
                                "gate/ep_len_passed": int(self._ep_len_gate_passed),
                                "gate/mean_ep_len": self._last_ep_len,
                                "gate/task_plateau_passed": int(self._check_task_plateau()),
                            }, step=agent._iter)
                        elif self.convergence_mode == 'composite':
                            gate_diag = self.convergence_gate.get_diagnostics()
                            wandb.log(gate_diag, step=agent._iter)

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

                # Barrier: resync all ranks after root-only I/O (wandb, video,
                # checkpoint) to prevent NCCL collective timeout from drift.
                if mp_util.enable_mp():
                    torch.distributed.barrier()

                if should_stop:
                    if self.is_root:
                        actor_loss_str = (f", actor_loss={last_actor_loss:.6f}"
                                          if last_actor_loss is not None else "")
                        if self.convergence_mode == 'codesign':
                            recent_task = self._task_plateau_values[-self.task_plateau_window:]
                            task_mean = np.mean(recent_task) if recent_task else 0.0
                            print(f"    [Inner] CONVERGED ({inner_iters} iters, "
                                  f"mode=codesign{actor_loss_str}, "
                                  f"ep_len={self._last_ep_len:.0f}/{self._max_ep_steps:.0f}, "
                                  f"task_reward={task_mean:.4f})")
                        else:
                            diag = self.convergence_gate.get_diagnostics()
                            print(f"    [Inner] CONVERGED ({inner_iters} iters, "
                                  f"mode={self.convergence_mode}{actor_loss_str}, "
                                  f"gate={diag})")
                    converged = True
                    agent._iter += 1
                    inner_iters += 1
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
                print(f"    [Inner] Iteration cap reached ({inner_iters} iters)")

        return converged, disc_rewards, last_actor_loss


# ---------------------------------------------------------------------------
# World-partitioned SPSA gradient
# ---------------------------------------------------------------------------

def apply_partitioned_morphologies(model, theta_np, eps, n_params,
                                   base_quats_dict, joint_idx_map,
                                   num_worlds, worlds_per_pert,
                                   directions=None):
    """Apply perturbed morphologies to world partitions in one GPU round-trip.

    Partition layout (n_pert = 1 + 2*M, where M = number of directions):
        partition 0:              center (theta)
        partition 2*m+1:          theta + eps * directions[m]
        partition 2*m+2:          theta - eps * directions[m]

    When directions is None, falls back to axis-aligned perturbations
    (M=n_params).

    Each partition spans worlds [p * wpp, (p+1) * wpp).
    Worlds beyond n_pert * wpp are left untouched (they won't be read).
    """
    if directions is None:
        directions = np.eye(n_params)
    M = directions.shape[0]
    n_pert = 1 + 2 * M
    joints_per_world = model.joint_count // num_worlds
    joint_X_p_np = model.joint_X_p.numpy()

    for p in range(n_pert):
        # Compute theta for this perturbation
        if p == 0:
            theta_pert = theta_np  # center — no copy needed (read-only)
        else:
            m = (p - 1) // 2       # direction index
            is_plus = (p % 2 == 1)
            sign = 1.0 if is_plus else -1.0
            theta_pert = theta_np + sign * eps * directions[m]

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


def generate_orthogonal_directions(n_params, num_sets, seed):
    """Generate S orthogonal bases as perturbation directions.

    Returns an (S*N, N) matrix where each block of N rows is a
    Haar-random orthogonal matrix (via QR of Gaussian).  Rows are
    unit vectors, so eps scaling gives consistent perturbation magnitude.

    Args:
        n_params: N, number of design parameters (6)
        num_sets: S, number of orthogonal basis sets
        seed: RNG seed (should vary per outer iteration)

    Returns:
        directions: (S*N, N) numpy array of perturbation directions
    """
    rng = np.random.RandomState(seed)
    blocks = []
    for _ in range(num_sets):
        A = rng.standard_normal((n_params, n_params))
        Q, R = np.linalg.qr(A)
        Q *= np.sign(np.diag(R))  # Haar-uniform over O(N)
        blocks.append(Q)
    return np.vstack(blocks)  # (S*N, N)


def compute_spsa_gradient_parallel(agent, env, engine, char_id, total_mass,
                                  theta_np, base_quats_dict, joint_idx_map,
                                  num_worlds, horizon,
                                  rank, num_procs, device, param_names,
                                  num_seeds=10, base_seed=42,
                                  eps=0.05, vel_reward_weight=0.1,
                                  vel_cmd=1.0, vel_tracking_sigma=0.25,
                                  num_spsa_sets=3):
    """Compute closed-loop gradient with orthogonal SPSA world-partitioned perturbations.

    Generates S random orthogonal bases (S*N directions total), partitions
    worlds into 1+2*S*N groups — each running a different morphology
    simultaneously. The seed loop (K iterations) remains sequential.
    Gradient reconstructed via least-squares solve per seed.

    Cost function matches inner loop task reward:
        reward = -0.1*CoT + vel_reward_weight * exp(-|v_x - v_cmd|^2 / sigma)

    Each GPU partitions its own per_rank_envs worlds independently. Multi-GPU
    results are averaged via all_reduce(SUM) / num_procs.

    Returns (grad, grad_stderr, snr, vel_tracking_center, cot_center) on all ranks.
    """
    n_params = len(theta_np)
    assert num_spsa_sets >= 1, f"num_spsa_sets must be >= 1, got {num_spsa_sets}"

    # Generate orthogonal perturbation directions (deterministic across ranks)
    directions = generate_orthogonal_directions(n_params, num_spsa_sets,
                                                 seed=base_seed + 9999)
    M = directions.shape[0]  # S*N total directions
    n_pert = 1 + 2 * M
    wpp = num_worlds // n_pert  # worlds per perturbation
    assert num_worlds >= n_pert, (
        f"Need >= {n_pert} worlds for S={num_spsa_sets} orthogonal sets "
        f"(M={M} directions), got {num_worlds}")

    if rank == 0:
        print(f"    [SPSA] {num_spsa_sets} orthogonal sets x {n_params} params "
              f"= {M} directions, n_pert={n_pert}, "
              f"{wpp} worlds/perturbation ({num_worlds} total), "
              f"{num_procs} GPU(s), {num_seeds} seeds")

    # Result tensors: (n_pert, num_seeds)
    cot_all = torch.zeros(n_pert, num_seeds, dtype=torch.float64, device=device)
    vel_track_all = torch.zeros(n_pert, num_seeds, dtype=torch.float64, device=device)

    sim_model = engine._sim_model

    # Apply all perturbed morphologies to their world partitions (1 GPU<->CPU round-trip)
    apply_partitioned_morphologies(sim_model, theta_np, eps, n_params,
                                   base_quats_dict, joint_idx_map,
                                   num_worlds, wpp,
                                   directions=directions)

    agent.eval()
    agent.set_mode(base_agent_mod.AgentMode.TEST)

    for seed_idx in range(num_seeds):
        seed = base_seed + seed_idx
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        obs, info = env.reset()

        # Per-world episode-aware accumulators.
        # On done, we finalize the current episode's CoT and vel tracking,
        # then reset accumulators so the next episode starts clean.
        power_sum = torch.zeros(num_worlds, device=device)
        vel_sum = torch.zeros(num_worlds, device=device)
        vel_track_sum = torch.zeros(num_worlds, device=device)
        step_count = torch.zeros(num_worlds, device=device, dtype=torch.long)
        ep_cot_sum = torch.zeros(num_worlds, device=device)
        ep_vel_track_sum = torch.zeros(num_worlds, device=device)
        ep_count = torch.zeros(num_worlds, device=device, dtype=torch.long)

        with torch.no_grad():
            for _ in range(horizon):
                action, _ = agent._decide_action(obs, info)
                obs, reward, done, info = env.step(action)

                dof_forces = EpisodeCoTTracker.get_dof_forces_safe(engine, char_id)
                dof_vel = engine.get_dof_vel(char_id)
                root_vel = engine.get_root_vel(char_id)

                power = torch.sum(torch.abs(
                    dof_forces[:num_worlds] *
                    dof_vel[:num_worlds]), dim=-1)
                fwd_vel = root_vel[:num_worlds, 0]
                power_sum += power
                vel_sum += fwd_vel
                vel_track_sum += torch.exp(-torch.square(fwd_vel - vel_cmd) / vel_tracking_sigma)
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
                        ep_cot_sum[vi] += compute_cot(mp, mv, total_mass)
                        ep_vel_track_sum[vi] += vel_track_sum[vi] / vsc
                        ep_count[vi] += 1
                    power_sum[di] = 0
                    vel_sum[di] = 0
                    vel_track_sum[di] = 0
                    step_count[di] = 0

        # Finalize trailing partial episodes (worlds still alive at horizon)
        alive = step_count > 0
        if alive.any():
            ai = alive.nonzero(as_tuple=True)[0]
            vsc = step_count[ai].float()
            mp = power_sum[ai] / vsc
            mv = vel_sum[ai] / vsc
            ep_cot_sum[ai] += compute_cot(mp, mv, total_mass)
            ep_vel_track_sum[ai] += vel_track_sum[ai] / vsc
            ep_count[ai] += 1

        # Extract per-partition metrics
        for p in range(n_pert):
            s, e = p * wpp, (p + 1) * wpp
            counts = ep_count[s:e].float().clamp(min=1)
            cot = (ep_cot_sum[s:e] / counts).mean().item()
            vt = (ep_vel_track_sum[s:e] / counts).mean().item()
            cot_all[p, seed_idx] = cot
            vel_track_all[p, seed_idx] = vt

    agent.set_mode(base_agent_mod.AgentMode.TRAIN)
    agent.train()

    # Multi-GPU: average across ranks
    if num_procs > 1:
        torch.distributed.all_reduce(cot_all, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(vel_track_all, op=torch.distributed.ReduceOp.SUM)
        cot_all /= num_procs
        vel_track_all /= num_procs

    # Restore original morphology
    update_training_joint_X_p(sim_model, theta_np, base_quats_dict,
                               joint_idx_map, num_worlds)

    # Gradient computation via least-squares on orthogonal directions
    cot_np = cot_all.cpu().numpy()
    vt_np = vel_track_all.cpu().numpy()

    # Center metrics (mean over seeds)
    cot_center = float(np.mean(cot_np[0]))
    vel_track_center = float(np.mean(vt_np[0]))

    # reward = -0.1*CoT + vel_reward_weight * vel_tracking
    reward_all = -0.1 * cot_np + vel_reward_weight * vt_np  # (n_pert, num_seeds)

    # Per-seed: compute directional finite differences, then solve for gradient
    grad_per_seed = []
    for k in range(num_seeds):
        dJ = np.array([
            (reward_all[2 * m + 1, k] - reward_all[2 * m + 2, k]) / (2 * eps)
            for m in range(M)
        ])  # (M,) directional derivatives along each direction

        # Skip seeds where any dJ is NaN
        if np.any(np.isnan(dJ)):
            continue

        # Solve: directions @ grad_k = dJ  (M equations, N unknowns, M >= N)
        grad_k, _, _, _ = np.linalg.lstsq(directions, dJ, rcond=None)
        grad_per_seed.append(grad_k)

    if len(grad_per_seed) == 0:
        # All seeds produced NaN — degenerate
        grad = np.zeros(n_params, dtype=np.float64)
        grad_stderr = np.full(n_params, float('inf'), dtype=np.float64)
    else:
        grad_samples = np.stack(grad_per_seed)  # (K_valid, N)
        grad = grad_samples.mean(axis=0)
        if len(grad_per_seed) > 1:
            grad_stderr = grad_samples.std(axis=0, ddof=1) / np.sqrt(len(grad_per_seed))
        else:
            grad_stderr = np.full(n_params, float('inf'), dtype=np.float64)

    # Per-parameter SNR: |gradient| / standard_error (computed before clipping)
    snr = np.zeros(n_params, dtype=np.float64)
    for i in range(n_params):
        if grad_stderr[i] > 0 and np.isfinite(grad_stderr[i]):
            snr[i] = abs(grad[i]) / grad_stderr[i]
        else:
            snr[i] = float('inf') if grad[i] != 0 else 0.0

    grad = np.clip(grad, -10.0, 10.0)

    if rank == 0:
        print(f"    [SPSA] Center (mean over {num_seeds} seeds): "
              f"vel_track={vel_track_center:.4f}, CoT={cot_center:.4f}")
        print(f"    [SPSA] Gradient ({len(grad_per_seed)}/{num_seeds} valid seeds):")
        for i, name in enumerate(param_names):
            print(f"      d_reward/d_{name} = {grad[i]:+.6f} "
                  f"+/- {grad_stderr[i]:.6f} (SNR={snr[i]:.2f})")
        finite_snr = snr[np.isfinite(snr)]
        if len(finite_snr) > 0:
            print(f"    [SPSA] Gradient SNR: min={np.min(finite_snr):.2f}, "
                  f"mean={np.mean(finite_snr):.2f}, "
                  f"max={np.max(finite_snr):.2f}")

    return grad, grad_stderr, snr, vel_track_center, cot_center


# ---------------------------------------------------------------------------
# Envelope Theorem validation: policy gradient norm at convergence
# ---------------------------------------------------------------------------

def compute_envelope_gradient_norm(agent, env, device, num_steps=None):
    """Measure ||grad_theta J_task|| at policy convergence.

    Collects a fresh rollout with the converged policy, computes Monte Carlo
    returns from task rewards, then backprops the REINFORCE objective through
    the actor network to get the un-clipped policy gradient norm.

    A small value (relative to mid-training) empirically validates the
    Envelope Theorem precondition: grad_theta J ≈ 0 at the converged policy.

    Single-rank diagnostic — no collective ops. Call on root only.
    """
    if num_steps is None:
        num_steps = agent._steps_per_iter

    # 1. Collect rollout (no grad) — task reward is in _reward_buf via
    #    codesign_task_reward monkey-patch
    agent.eval()
    agent.set_mode(base_agent_mod.AgentMode.TRAIN)
    obs, info = env.reset()

    all_obs, all_norm_actions, all_rewards, all_dones = [], [], [], []
    with torch.no_grad():
        for _ in range(num_steps):
            action, action_info = agent._decide_action(obs, info)
            next_obs, reward, done, info = env.step(action)
            all_obs.append(obs)
            all_norm_actions.append(agent._a_norm.normalize(action))
            all_rewards.append(reward)
            all_dones.append(done)
            obs = next_obs

    obs_t = torch.stack(all_obs)           # [T, N, obs_dim]
    act_t = torch.stack(all_norm_actions)   # [T, N, act_dim]
    rew_t = torch.stack(all_rewards)        # [T, N]
    done_t = torch.stack(all_dones)         # [T, N]
    T, N = rew_t.shape

    # 2. Monte Carlo returns
    returns = torch.zeros_like(rew_t)
    G = torch.zeros(N, device=device)
    gamma = agent._discount
    for t in reversed(range(T)):
        alive = (done_t[t] == 0).float()
        G = rew_t[t] + gamma * G * alive
        returns[t] = G

    flat_ret = returns.reshape(-1)
    ret_mean = flat_ret.mean()
    ret_std = flat_ret.std().clamp(min=1e-5)
    norm_ret = ((returns - ret_mean) / ret_std).reshape(-1).detach()

    # 3. Forward pass WITH gradient — REINFORCE loss
    agent.train()
    for p in agent._model.get_actor_params():
        if p.grad is not None:
            p.grad.zero_()

    flat_obs = obs_t.reshape(T * N, -1)
    flat_act = act_t.reshape(T * N, -1)

    norm_obs = agent._obs_norm.normalize(flat_obs)
    a_dist = agent._model.eval_actor(norm_obs)
    logp = a_dist.log_prob(flat_act)

    loss = -torch.mean(logp * norm_ret)
    loss.backward()

    # 4. Read actor gradient L2 norm
    total_norm_sq = 0.0
    for p in agent._model.get_actor_params():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    grad_norm = total_norm_sq ** 0.5

    # Clean up — don't leave stale gradients
    for p in agent._model.get_actor_params():
        if p.grad is not None:
            p.grad.zero_()
    agent.eval()

    # Reset env so SPSA starts from clean state
    env.reset()

    return grad_norm


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
        agent.train()

        if not frames:
            return None

        # wandb.Video expects (T, C, H, W)
        video = np.stack(frames)            # (T, H, W, 3)
        video = video.transpose(0, 3, 1, 2)  # (T, 3, H, W)
        return video

    except Exception as e:
        print(f"    [Video] Capture failed: {e}")
        agent.set_mode(base_agent_mod.AgentMode.TRAIN)
        agent.train()
        return None


# ---------------------------------------------------------------------------
# Worker initialization
# ---------------------------------------------------------------------------

def _init_gbc_worker(rank, num_procs, device, master_port, args):
    """One-time worker setup: distributed init, env/agent build, joint extraction.

    Returns a SimpleNamespace with all state needed by the outer loop.
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
        print("GBC (Gradient-Based Co-Design) for G1 Humanoid")
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
            project="gbc-codesign",
            name=f"g1-unified-{args.num_train_envs}env-{num_procs}gpu",
            config=vars(args),
        )
    elif is_root and args.wandb:
        print("  [wandb] Not available, continuing without")

    # ---- MimicKit initialization ----
    if is_root:
        print("\n[1/3] Initializing MimicKit...")

    mjcf_modifier = G1MJCFModifier(str(BASE_MJCF_PATH))
    modified_mjcf = out_dir / "g1_modified.xml"
    modified_env_config = out_dir / "env_config.yaml"

    theta_init = np.zeros(NUM_DESIGN_PARAMS, dtype=np.float64)
    if is_root:
        mjcf_modifier.generate(theta_init, str(modified_mjcf))

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
    env._update_reward = types.MethodType(codesign_task_reward, env)

    # Placeholder mass (build_agent may trigger env steps before real mass is known)
    env._total_mass = 1.0
    env._vel_reward_weight = args.vel_reward_weight
    env._vel_cmd = args.vel_cmd
    env._vel_tracking_sigma = args.vel_tracking_sigma

    agent = agent_builder.build_agent(str(BASE_AGENT_CONFIG), env, device)

    # Ensure disc replay buffer can hold a full rollout batch.
    # With 16384 envs x 32 steps_per_iter = 524k samples, but default buffer is 200k.
    import learning.experience_buffer as experience_buffer_mod
    min_disc_buf = per_rank_envs * agent._config["steps_per_iter"]
    if agent._disc_buffer._buffer_length < min_disc_buf:
        new_size = min_disc_buf * 2  # 2x headroom
        orig_batch_size = agent._disc_buffer._batch_size
        if is_root:
            print(f"  [Fix] Expanding disc_buffer: {agent._disc_buffer._buffer_length} -> {new_size} "
                  f"(need >= {min_disc_buf} for {per_rank_envs} envs x {agent._config['steps_per_iter']} steps)")
        agent._disc_buffer = experience_buffer_mod.ExperienceBuffer(
            buffer_length=new_size, batch_size=orig_batch_size, device=device)

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

    # Overwrite placeholder mass with real value
    env._total_mass = total_mass

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

    return types.SimpleNamespace(
        is_root=is_root, per_rank_envs=per_rank_envs,
        env=env, agent=agent, engine=engine, sim_model=sim_model,
        char_id=char_id, total_mass=total_mass,
        base_quats_dict=base_quats_dict, joint_idx_map=joint_idx_map,
        viewer=viewer, use_wandb=use_wandb, out_dir=out_dir, log_file=log_file,
    )


# ---------------------------------------------------------------------------
# Main GBC loop (one per GPU rank)
# ---------------------------------------------------------------------------

def gbc_worker(rank, num_procs, device, master_port, args):
    """Unified single-process GBC (Gradient-Based Co-Design) for G1 humanoid.

    Each GPU rank runs this function. The inner loop (MimicKit training) and
    SPSA gradient rollouts are both distributed across all ranks via
    torch.distributed collective ops. BFGS update runs on rank 0; theta is
    broadcast after.
    """
    ws = _init_gbc_worker(rank, num_procs, device, master_port, args)
    is_root, per_rank_envs = ws.is_root, ws.per_rank_envs
    env, agent, engine = ws.env, ws.agent, ws.engine
    sim_model, char_id, total_mass = ws.sim_model, ws.char_id, ws.total_mass
    base_quats_dict, joint_idx_map = ws.base_quats_dict, ws.joint_idx_map
    viewer, use_wandb = ws.viewer, ws.use_wandb
    out_dir, log_file = ws.out_dir, ws.log_file

    # =========================================================
    # Outer loop setup
    # =========================================================
    if is_root:
        print("[3/3] Starting GBC outer loop...\n")

    theta = np.zeros(NUM_DESIGN_PARAMS, dtype=np.float64)
    bfgs = CautiousBFGS(NUM_DESIGN_PARAMS)
    theta_bounds = THETA_BOUNDS

    param_names = [
        f"theta_{i}_{SYMMETRIC_PAIRS[i][0].replace('_link', '')}"
        for i in range(NUM_DESIGN_PARAMS)
    ]

    # Handle deprecated --use-plateau-convergence flag
    convergence_mode = args.convergence_mode
    if args.use_plateau_convergence and convergence_mode == 'composite':
        convergence_mode = 'plateau'
        if is_root:
            print("  [WARN] --use-plateau-convergence is deprecated, "
                  "use --convergence-mode=plateau instead")

    inner_ctrl = InnerLoopController(
        agent,
        plateau_threshold=args.plateau_threshold,
        plateau_window=args.plateau_window,
        min_plateau_outputs=args.min_plateau_outputs,
        max_inner_iters=args.max_inner_iters,
        min_inner_iters=args.kickoff_min_iters,  # first outer iter; reset to 0 after
        convergence_mode=convergence_mode,
        convergence_window=args.convergence_window,
        critic_loss_plateau_threshold=args.critic_loss_plateau_threshold,
        adv_std_threshold=args.adv_std_threshold,
        actor_loss_plateau_threshold=args.actor_loss_plateau_threshold,
        kl_conv_threshold=args.kl_conv_threshold,
        kl_window=args.kl_window,
        use_kl_convergence=not args.use_plateau_convergence,
        ep_len_threshold_frac=args.ep_len_threshold_frac,
        task_plateau_threshold=args.task_plateau_threshold,
        task_plateau_window=args.task_plateau_window,
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
        agent._task_reward_weight = 0.5
        agent._disc_reward_weight = 0.5
        if is_root:
            print(f"  [Resume] Skipping kickoff ramp (ramp_completed=True, "
                  f"min_inner_iters={inner_ctrl.min_inner_iters})")

    history = {
        "theta": [theta.copy()],
        "vel_tracking": [],
        "cot": [],
        "gradients": [],
        "grad_snr": [],
        "inner_times": [],
    }

    # Adaptive trust region state (1b)
    trust_radius = np.radians(args.max_step_deg)
    TRUST_RADIUS_MIN = np.radians(0.05)
    TRUST_RADIUS_MAX = np.radians(5.0)
    prev_center_reward = None
    predicted_improvement = None
    theta_history = deque(maxlen=5)

    if is_root:
        print(f"Configuration:")
        print(f"  Training envs:     {args.num_train_envs} total "
              f"({per_rank_envs}/GPU x {num_procs} GPUs)")
        print(f"  Eval horizon:      {args.eval_horizon} steps "
              f"({args.eval_horizon / 30:.1f}s)")
        print(f"  Max inner iters:   {args.max_inner_iters}")
        print(f"  Design optimizer:  Cautious BFGS (n={NUM_DESIGN_PARAMS}, lr={args.design_lr})")
        print(f"  Max step size:     {args.max_step_deg} deg/iter initial "
              f"(adaptive trust region, floor={np.degrees(TRUST_RADIUS_MIN):.2f} deg, "
              f"ceil={np.degrees(TRUST_RADIUS_MAX):.1f} deg)")
        print(f"  Design params:     {NUM_DESIGN_PARAMS} (symmetric lower-body)")
        print(f"  Theta bounds:      +/-{np.degrees(THETA_BOUNDS[1]):.0f} deg "
              f"(+/-{THETA_BOUNDS[1]:.4f} rad)")
        if args.kickoff_min_iters > 0:
            print(f"  Kickoff min iters: {args.kickoff_min_iters} (first outer iter only)")
        print(f"  Reward ramp:       disc 1.0→0.5, task 0.0→0.5 "
              f"(iter {args.ramp_start_iter}→{args.ramp_end_iter})")
        print(f"  Convergence mode:  {convergence_mode}")
        if convergence_mode == 'codesign':
            print(f"    ep_len gate:     >= {args.ep_len_threshold_frac:.0%} of max episode length")
            print(f"    task plateau:    {args.task_plateau_threshold:.0%} spread over "
                  f"window={args.task_plateau_window}")
        _M = args.spsa_sets * NUM_DESIGN_PARAMS
        _n_pert = 1 + 2 * _M
        _wpp = per_rank_envs // _n_pert
        print(f"  SPSA sets:         {args.spsa_sets} ({_M} directions, "
              f"{_n_pert} perturbations, {_wpp} worlds/pert/GPU)")
        print(f"  SPSA seeds:        {args.num_spsa_seeds} (paired-seed averaging)")
        print(f"  SPSA epsilon:      {args.spsa_epsilon} rad "
              f"({np.degrees(args.spsa_epsilon):.1f} deg)")
        if args.snr_threshold > 0:
            print(f"  SNR threshold:     {args.snr_threshold}")
        print(f"  Vel reward weight: {args.vel_reward_weight} "
              f"(reward = -CoT + {args.vel_reward_weight}*exp(-|v-{args.vel_cmd}|^2/{args.vel_tracking_sigma}))")
        print(f"  Gradient mode:     orthogonal SPSA (closed-loop, lstsq reconstruction)")

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

        converged, disc_rewards, final_actor_loss = inner_ctrl.train_until_converged(
            iter_dir
        )
        inner_time = time.time() - t0
        if is_root:
            actor_loss_str = (f", actor_loss={final_actor_loss:.6f}"
                              if final_actor_loss is not None else "")
            print(f"  [Inner Loop] Done in {inner_time / 60:.1f} min{actor_loss_str}")

        history["inner_times"].append(inner_time)

        # After kickoff, disable min_inner_iters gate for subsequent iters.
        # Reward weights are now handled by the ramp inside train_until_converged
        # (disc 1.0→0.0, task 0.0→1.0 over ramp_start..ramp_end iters).
        if outer_iter == 0 and inner_ctrl.min_inner_iters > 0:
            if is_root:
                print(f"  [Kickoff] Disabling min_inner_iters "
                      f"({inner_ctrl.min_inner_iters}) for subsequent iters")
            inner_ctrl.min_inner_iters = args.post_kickoff_min_iters

        # ----- Envelope Theorem validation (1a, root only) -----
        envelope_grad_norm = None
        if is_root and inner_ctrl.ramp_completed:
            envelope_grad_norm = compute_envelope_gradient_norm(
                agent, env, device)
            print(f"  [Envelope] Policy gradient norm at convergence: "
                  f"{envelope_grad_norm:.6f}")

        # ----- SPSA Gradient Phase (ALL ranks, closed-loop) -----
        # Clear CoT tracker before SPSA to prevent cross-contamination
        env._cot_tracker.get_stats()  # drain any accumulated episodes
        env._cot_tracker.reset_accumulators()

        if is_root:
            print(f"\n  [SPSA] Computing orthogonal SPSA gradient "
                  f"(S={args.spsa_sets}, {args.num_spsa_seeds} seeds, "
                  f"{args.eval_horizon} steps)...")

        spsa_base_seed = 42 + outer_iter * 1000

        grad_theta, grad_stderr, grad_snr, vel_track, cot = compute_spsa_gradient_parallel(
            agent, env, engine, char_id, total_mass,
            theta.copy(), base_quats_dict, joint_idx_map,
            per_rank_envs, args.eval_horizon,
            rank, num_procs, device, param_names,
            num_seeds=args.num_spsa_seeds, base_seed=spsa_base_seed,
            eps=args.spsa_epsilon, vel_reward_weight=args.vel_reward_weight,
            vel_cmd=args.vel_cmd, vel_tracking_sigma=args.vel_tracking_sigma,
            num_spsa_sets=args.spsa_sets,
        )

        # Clear CoT tracker after SPSA to prevent pollution of inner loop
        env._cot_tracker.get_stats()
        env._cot_tracker.reset_accumulators()

        if is_root:
            print(f"    SPSA gradients:")
            for i, name in enumerate(param_names):
                print(f"      d_reward/d_{name} = {grad_theta[i]:+.6f}")
            print(f"    Vel tracking = {vel_track:.4f}")
            print(f"    Cost of Transport = {cot:.4f}")

            history["vel_tracking"].append(vel_track)
            history["cot"].append(cot)
            history["gradients"].append(grad_theta.copy())
            history["grad_snr"].append(grad_snr.copy())

            # ----- Adaptive trust region ratio test (1b) -----
            center_reward = -0.1 * cot + args.vel_reward_weight * vel_track
            rho = None
            if prev_center_reward is not None and predicted_improvement is not None:
                actual_improvement = center_reward - prev_center_reward
                if abs(predicted_improvement) > 1e-12:
                    rho = actual_improvement / predicted_improvement
                else:
                    rho = 0.0

                if rho > 0.75:
                    trust_radius = min(trust_radius * 1.5, TRUST_RADIUS_MAX)
                elif rho < 0.25:
                    trust_radius = max(trust_radius * 0.5, TRUST_RADIUS_MIN)

                print(f"    [Trust Region] rho={rho:.3f}, "
                      f"actual={actual_improvement:+.6f}, "
                      f"predicted={predicted_improvement:+.6f}, "
                      f"radius={np.degrees(trust_radius):.3f} deg")

            prev_center_reward = center_reward

            # ----- Design Update (rank 0 only) -----
            n_clamped = 0
            raw_step = np.zeros(NUM_DESIGN_PARAMS, dtype=np.float64)

            if np.any(np.isnan(grad_theta)) or np.all(grad_theta == 0):
                print(f"\n  [FATAL] Degenerate gradient (NaN or all-zero) — "
                      f"signaling all ranks to stop.")
                theta[:] = np.nan
                predicted_improvement = None
            else:
                # Save raw gradient before SNR masking (Bug B fix: BFGS
                # must receive the true gradient for correct y-vector)
                grad_theta_raw = grad_theta.copy()

                # SNR gating: zero out gradient components with low confidence
                # (inf SNR = high confidence from single seed, kept through)
                skip_update = False
                if args.snr_threshold > 0:
                    snr_mask = (grad_snr > args.snr_threshold) | ~np.isfinite(grad_snr)
                    n_masked = NUM_DESIGN_PARAMS - snr_mask.sum()
                    if n_masked == NUM_DESIGN_PARAMS:
                        # Bug A fix: actually skip the update instead of
                        # proceeding with all-zero gradients
                        print(f"    [SNR gate] WARNING: All {NUM_DESIGN_PARAMS} params "
                              f"masked (SNR < {args.snr_threshold}), skipping update")
                        skip_update = True
                        predicted_improvement = 0.0
                    else:
                        if n_masked > 0:
                            print(f"    [SNR gate] Masking {n_masked}/{NUM_DESIGN_PARAMS} params "
                                  f"with SNR < {args.snr_threshold}")
                        grad_theta = grad_theta * snr_mask.astype(float)

                if not skip_update:
                    old_theta = theta.copy()
                    direction = bfgs.compute_direction(grad_theta)
                    step = args.design_lr * direction

                    # Adaptive trust-region clamp (1b)
                    max_step_rad = trust_radius
                    raw_step = step.copy()
                    step = np.clip(step, -max_step_rad, max_step_rad)
                    n_clamped = np.sum(np.abs(raw_step) > max_step_rad)
                    if n_clamped > 0:
                        print(f"    [Trust region] Clamped {n_clamped}/{NUM_DESIGN_PARAMS} "
                              f"params to +/-{np.degrees(max_step_rad):.3f} deg")

                    theta = old_theta + step
                    theta = np.clip(theta, theta_bounds[0], theta_bounds[1])

                    # Record actual step (post-clip) for Hessian update
                    actual_s = theta - old_theta
                    # Bug B fix: pass raw (un-masked) gradient to BFGS so
                    # prev_grad reflects the true gradient for y-vector
                    bfgs.update(actual_s, grad_theta_raw)

                    # Predicted improvement for next iteration's ratio test (1b)
                    predicted_improvement = grad_theta_raw @ actual_s

                    print(f"\n  Design update (BFGS, {bfgs.num_updates} H updates, "
                          f"{bfgs.num_skipped} skipped):")
                    for i, name in enumerate(param_names):
                        delta = theta[i] - old_theta[i]
                        print(f"    {name}: {old_theta[i]:+.4f} -> {theta[i]:+.4f} "
                              f"(delta={delta:+.5f}, {np.degrees(delta):+.3f} deg)")

        # ----- Broadcast theta to all ranks -----
        # Non-root ranks block here until root finishes SPSA gradient
        # Bug C fix: use .double() to preserve float64 precision
        theta_tensor = torch.from_numpy(theta).double().to(device)
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
                finite_snr = grad_snr[np.isfinite(grad_snr)]
                log_dict = {
                    "outer/iteration": outer_iter + 1,
                    "outer/vel_tracking": vel_track,
                    "outer/cot": cot,
                    "outer/center_reward": center_reward,
                    "outer/inner_time_min": inner_time / 60.0,
                    "outer/grad_norm": np.linalg.norm(grad_theta),
                    "outer/grad_snr_mean": float(np.mean(finite_snr)) if len(finite_snr) > 0 else 0.0,
                    "outer/trust_radius_deg": np.degrees(trust_radius),
                }
                if rho is not None:
                    log_dict["outer/trust_ratio_rho"] = rho
                for i, name in enumerate(param_names):
                    log_dict[f"outer/{name}_deg"] = np.degrees(theta[i])
                    log_dict[f"outer/grad_{name}"] = grad_theta[i]
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
        print("GBC Co-Design Complete!")
        print("=" * 70)

        initial = history["theta"][0]
        final = history["theta"][-1]
        for i, name in enumerate(param_names):
            print(f"  {name}: {initial[i]:+.4f} -> {final[i]:+.4f} "
                  f"({np.degrees(initial[i]):+.2f} -> "
                  f"{np.degrees(final[i]):+.2f} deg)")

        if history["vel_tracking"]:
            print(f"\nVel tracking: {history['vel_tracking'][0]:.4f} -> "
                  f"{history['vel_tracking'][-1]:.4f}")
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
        description="GBC (Gradient-Based Co-Design) for G1 Humanoid"
    )
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--outer-iters", type=int, default=20)
    parser.add_argument("--design-lr", type=float, default=1.0,
                        help="Step size scaling on BFGS direction (default: 1.0, BFGS self-scales)")
    parser.add_argument("--num-train-envs", type=int, default=4096,
                        help="Total training envs (divided across GPUs)")
    parser.add_argument("--eval-horizon", type=int, default=300,
                        help="Eval rollout length in control steps (300 = 10s full episode)")
    parser.add_argument("--num-spsa-seeds", type=int, default=30,
                        help="Number of paired seeds per SPSA evaluation (K)")
    parser.add_argument("--spsa-epsilon", type=float, default=0.05,
                        help="SPSA perturbation size in radians (default: 0.05 ~ 2.9 deg)")
    parser.add_argument("--spsa-sets", type=int, default=3,
                        help="Number of orthogonal basis sets S for SPSA gradient "
                             "(M=S*N directions, n_pert=1+2*S*N). Default: 3")
    parser.add_argument("--snr-threshold", type=float, default=0.0,
                        help="SNR gating threshold (0=disabled). Mask gradient components "
                             "where SNR < threshold before BFGS update. Recommended: 2.0")
    parser.add_argument("--max-step-deg", type=float, default=0.5,
                        help="Max per-parameter step size in degrees per outer iteration. "
                             "Ensures policy stability across morphology updates. Default: 0.5")
    parser.add_argument("--max-inner-iters", type=int, default=10000,
                        help="Max inner iterations per outer loop (env-count-invariant safety cap)")
    parser.add_argument("--out-dir", type=str, default="output_g1_unified")
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="MimicKit checkpoint to resume from")
    parser.add_argument("--convergence-mode", type=str, default="codesign",
                        choices=["codesign", "composite", "kl", "plateau"],
                        help="Inner convergence mode: 'codesign' (ep_len gate + task_reward plateau), "
                             "'composite' (multi-signal gate), "
                             "'kl' (legacy KL), 'plateau' (legacy reward plateau)")
    parser.add_argument("--convergence-window", type=int, default=10,
                        help="Window size for composite convergence gate signals")
    parser.add_argument("--ep-len-threshold-frac", type=float, default=0.9,
                        help="Codesign mode: fraction of max episode length for stage-1 gate "
                             "(default: 0.9 = 90%% of max)")
    parser.add_argument("--task-plateau-threshold", type=float, default=0.02,
                        help="Codesign mode: relative spread threshold for task_reward plateau "
                             "(default: 0.02 = 2%%)")
    parser.add_argument("--task-plateau-window", type=int, default=10,
                        help="Codesign mode: window size for task_reward plateau check")
    parser.add_argument("--critic-loss-plateau-threshold", type=float, default=0.05,
                        help="Composite gate: critic_loss plateau threshold")
    parser.add_argument("--adv-std-threshold", type=float, default=0.1,
                        help="Composite gate: adv_std below threshold")
    parser.add_argument("--actor-loss-plateau-threshold", type=float, default=0.05,
                        help="Composite gate: actor_loss plateau threshold")
    parser.add_argument("--kl-conv-threshold", type=float, default=1e-4,
                        help="Legacy KL convergence threshold (--convergence-mode=kl)")
    parser.add_argument("--kl-window", type=int, default=10,
                        help="Legacy KL convergence window (--convergence-mode=kl)")
    parser.add_argument("--use-plateau-convergence", action="store_true",
                        help="[DEPRECATED] Use --convergence-mode=plateau instead")
    parser.add_argument("--plateau-threshold", type=float, default=0.0002,
                        help="Legacy plateau threshold (--convergence-mode=plateau)")
    parser.add_argument("--plateau-window", type=int, default=10,
                        help="Legacy plateau window (--convergence-mode=plateau)")
    parser.add_argument("--min-plateau-outputs", type=int, default=10,
                        help="Legacy plateau min outputs (--convergence-mode=plateau)")
    parser.add_argument("--save-interval", type=int, default=500,
                        help="Save numbered model checkpoint every N inner iterations")
    parser.add_argument("--kickoff-min-iters", type=int, default=2500,
                        help="Min inner iters before convergence on first outer iter "
                             "(must be >= ramp-end-iter so outer loop starts with pure task reward)")
    parser.add_argument("--ramp-start-iter", type=int, default=1000,
                        help="Inner iter to start ramping disc→task reward weights")
    parser.add_argument("--ramp-end-iter", type=int, default=2500,
                        help="Inner iter to finish ramping (disc=0, task=1)")
    parser.add_argument("--post-kickoff-min-iters", type=int, default=100,
                        help="Min inner iters for convergence after first outer iter")
    parser.add_argument("--vel-reward-weight", type=float, default=5.0,
                        help="Weight for velocity tracking in cost function "
                             "(reward = -CoT + w * exp(-|v-v_cmd|^2/sigma))")
    parser.add_argument("--vel-cmd", type=float, default=1.0,
                        help="Target forward velocity for velocity tracking (m/s)")
    parser.add_argument("--vel-tracking-sigma", type=float, default=0.25,
                        help="Sigma for Gaussian velocity tracking reward")
    parser.add_argument("--no-video", action="store_true",
                        help="Disable headless viewer (use on HPC nodes without EGL)")
    parser.add_argument("--video-interval", type=int, default=100,
                        help="Log video to wandb every N inner iterations")
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

    # Validate: outer loop must not fire before reward ramp completes
    if args.kickoff_min_iters < args.ramp_end_iter:
        parser.error(
            f"--kickoff-min-iters ({args.kickoff_min_iters}) must be >= "
            f"--ramp-end-iter ({args.ramp_end_iter}) so the outer loop "
            f"only starts after the reward ramp completes"
        )

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
            os.environ["_GBC_GPU_INDEX"] = str(gpu_indices[r])
            p = torch.multiprocessing.Process(
                target=gbc_worker,
                args=[r, num_workers, "cuda:0", master_port, args],
            )
            p.start()
            processes.append(p)
        os.environ.pop("_GBC_GPU_INDEX", None)

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
        gbc_worker(0, 1, args.devices[0], master_port, args)
