#!/usr/bin/env python3
"""
Level 3: PGHC Co-Design for G1 Humanoid with Backprop Design Gradients

Outer loop optimizes 6 joint oblique angle parameters (symmetric lower-body pairs)
using BPTT through SolverSemiImplicit.

Inner loop trains AMP walking policy via MimicKit subprocess (unchanged).

Architecture:
    1. Generate modified G1 MJCF from current theta
    2. Run MimicKit AMP training (subprocess)
    3. Load trained policy from checkpoint
    4. Pre-collect actions using MimicKit env programmatically (Phase 1)
    5. Replay actions on Warp tape with SolverSemiImplicit (Phase 2)
    6. tape.backward() → joint_X_p.grad → theta grad (Phase 3)
    7. Adam update on theta, project to ±30° bounds

Run:
    python codesign_g1.py --wandb --num-train-envs 4096 --num-eval-worlds 8 --eval-horizon 100
"""

import os
os.environ["PYGLET_HEADLESS"] = "1"

import argparse
import gc
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import warp as wp
import newton

from g1_mjcf_modifier import G1MJCFModifier, SYMMETRIC_PAIRS, NUM_DESIGN_PARAMS

# ---------------------------------------------------------------------------
# Paths (relative to codesign/ directory)
# ---------------------------------------------------------------------------
CODESIGN_DIR = Path(__file__).parent.resolve()
MIMICKIT_DIR = (CODESIGN_DIR / ".." / "MimicKit").resolve()
MIMICKIT_SRC_DIR = MIMICKIT_DIR / "mimickit"

BASE_MJCF_PATH = MIMICKIT_DIR / "data" / "assets" / "g1" / "g1.xml"
BASE_ENV_CONFIG = MIMICKIT_DIR / "data" / "envs" / "amp_g1_env.yaml"
BASE_AGENT_CONFIG = MIMICKIT_DIR / "data" / "agents" / "amp_g1_agent.yaml"
BASE_ENGINE_CONFIG = MIMICKIT_DIR / "data" / "engines" / "newton_engine.yaml"

# G1 structural constants (with hinge joints — before ball conversion)
# Actual values are determined at runtime from the Newton model since
# convert_3d_hinge_to_ball_joints=True changes the DOF layout.
# Original: 31 bodies, 29 actuated DOFs, 36 qpos, 35 qvel


# ---------------------------------------------------------------------------
# Warp kernels for differentiable evaluation
# ---------------------------------------------------------------------------

@wp.kernel
def update_joint_X_p_from_theta_kernel(
    theta: wp.array(dtype=float),
    base_quats: wp.array2d(dtype=float),
    joint_local_indices: wp.array(dtype=int),
    param_for_joint: wp.array(dtype=int),
    joint_X_p: wp.array2d(dtype=float),
    joints_per_world: int,
    num_worlds: int,
    num_param_joints: int,
):
    """Map theta → joint_X_p quaternions on tape for BPTT.

    Launched with dim = num_param_joints * num_worlds.
    Each thread handles one (parameterized joint, world) pair.
    """
    tid = wp.tid()
    j = tid // num_worlds   # parameterized joint index (0..11)
    w = tid % num_worlds    # world index
    if j >= num_param_joints:
        return

    p_idx = param_for_joint[j]
    angle = theta[p_idx]

    # Delta quat from X-rotation: (cos(a/2), sin(a/2), 0, 0)
    half = angle * 0.5
    dw = wp.cos(half)
    dx = wp.sin(half)

    # Base quat (w, x, y, z)
    bw = base_quats[j, 0]
    bx = base_quats[j, 1]
    by = base_quats[j, 2]
    bz = base_quats[j, 3]

    # Hamilton product: delta * base
    nw = dw * bw - dx * bx
    nx = dw * bx + dx * bw
    ny = dw * by - dx * bz
    nz = dw * bz + dx * by

    # Normalize
    norm = wp.sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
    inv_norm = 1.0 / wp.max(norm, 1.0e-12)
    nw = nw * inv_norm
    nx = nx * inv_norm
    ny = ny * inv_norm
    nz = nz * inv_norm

    # Write to joint_X_p[global_idx, 3:7]
    global_idx = w * joints_per_world + joint_local_indices[j]
    joint_X_p[global_idx, 3] = nw
    joint_X_p[global_idx, 4] = nx
    joint_X_p[global_idx, 5] = ny
    joint_X_p[global_idx, 6] = nz


@wp.kernel
def diff_set_target_pos_kernel(
    actions: wp.array2d(dtype=float),
    joint_target_pos: wp.array(dtype=float),
    qd_dof_start: int,
    num_act_dofs: int,
    total_qd_per_world: int,
):
    """Set joint target positions from pre-collected actions.

    G1 uses position control mode: actions → target joint positions.
    joint_target_pos is flat: (num_worlds * total_qd_per_world,).
    DOFs for the character start at qd_dof_start within each world's qd slice.
    """
    tid = wp.tid()
    base = tid * total_qd_per_world + qd_dof_start
    for i in range(num_act_dofs):
        joint_target_pos[base + i] = actions[tid, i]


@wp.kernel
def diff_init_worlds_kernel(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    init_q_full: wp.array(dtype=float),
    total_q_size: int,
    total_qd_size: int,
):
    """Initialize joint_q from pre-computed init array and zero joint_qd.

    init_q_full is the full flat array (all worlds), copied directly.
    """
    tid = wp.tid()
    if tid < total_q_size:
        joint_q[tid] = init_q_full[tid]
    if tid < total_qd_size:
        joint_qd[tid] = 0.0


@wp.kernel
def compute_forward_loss_kernel(
    body_q_final: wp.array(dtype=float),
    body_q_initial: wp.array(dtype=float),
    num_bodies_per_world: int,
    loss: wp.array(dtype=float),
    num_worlds: int,
):
    """loss = -mean(x_final - x_initial) over all worlds.

    body_q is flat: (num_worlds * num_bodies_per_world * 7,).
    Root body x position is at body_q[w * num_bodies * 7 + 0].
    """
    tid = wp.tid()
    if tid == 0:
        total = float(0.0)
        for w in range(num_worlds):
            stride = w * num_bodies_per_world * 7
            x_final = body_q_final[stride]
            x_initial = body_q_initial[stride]
            total = total + (x_final - x_initial)
        loss[0] = -total / float(num_worlds)


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
# Action collection via MimicKit (programmatic import)
# ---------------------------------------------------------------------------

def collect_actions_mimickit(
    mimickit_src_dir,
    env_config_path,
    engine_config_path,
    agent_config_path,
    checkpoint_path,
    num_worlds,
    horizon,
    device="cuda:0",
):
    """Collect deterministic actions from trained MimicKit policy.

    Imports MimicKit programmatically to use its exact env/agent.
    Returns list of (num_worlds, 29) numpy arrays — one per timestep.
    """
    # Add MimicKit to sys.path temporarily
    mimickit_src = str(mimickit_src_dir)
    added_path = False
    if mimickit_src not in sys.path:
        sys.path.insert(0, mimickit_src)
        added_path = True

    try:
        # Import MimicKit modules
        import util.arg_parser as arg_parser
        import envs.env_builder as env_builder
        import learning.agent_builder as agent_builder_mod
        import learning.base_agent as base_agent_mod
        import util.mp_util as mp_util

        # Initialize mp_util for single-process
        try:
            mp_util.init(0, 1, device, np.random.randint(6000, 7000))
        except Exception:
            pass  # May already be initialized

        # Build env
        env = env_builder.build_env(
            str(env_config_path),
            str(engine_config_path),
            num_worlds,
            device,
            visualize=False,
        )

        # Build agent
        agent = agent_builder_mod.build_agent(
            str(agent_config_path),
            env,
            device,
        )

        # Load checkpoint
        agent.load(str(checkpoint_path))
        agent.eval()
        agent.set_mode(base_agent_mod.AgentMode.TEST)

        # Collect actions
        obs, info = env.reset()
        actions_list = []

        with torch.no_grad():
            for step in range(horizon):
                # Use agent's own action pipeline (handles obs normalization)
                action, action_info = agent._decide_action(obs, info)
                actions_list.append(action.cpu().numpy().copy())
                obs, reward, done, info = env.step(action)

        # Cleanup env
        try:
            env.close()
        except (AttributeError, Exception):
            pass

    finally:
        if added_path and mimickit_src in sys.path:
            sys.path.remove(mimickit_src)

    return actions_list


# ---------------------------------------------------------------------------
# Differentiable G1 Evaluation Environment
# ---------------------------------------------------------------------------

class DiffG1Eval:
    """Differentiable G1 eval for computing design gradients via BPTT.

    Builds a small eval env with SolverSemiImplicit, runs a frozen-policy
    episode on wp.Tape(), then tape.backward() to get ∂reward/∂joint_X_p,
    and chains that to ∂reward/∂theta via the Warp kernel.
    """

    def __init__(
        self,
        mjcf_modifier,
        theta_np,
        num_worlds=8,
        horizon=100,
        dt=1.0/30.0,
        num_substeps=8,
        device="cuda:0",
    ):
        self.mjcf_modifier = mjcf_modifier
        self.theta_np = theta_np.copy()
        self.num_worlds = num_worlds
        self.horizon = horizon
        self.dt = dt
        self.num_substeps = num_substeps
        self.sub_dt = dt / num_substeps
        self.device = device

        self._build_model()
        self._build_theta_buffers()
        self._alloc_buffers()

    def _build_model(self):
        """Build Newton model with SolverSemiImplicit for BPTT."""
        # Generate modified MJCF
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.xml', delete=False, dir=str(BASE_MJCF_PATH.parent)
        ) as f:
            self.mjcf_modifier.generate(self.theta_np, f.name)
            mjcf_path = f.name

        try:
            single_builder = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(single_builder)

            single_builder.add_mjcf(
                mjcf_path,
                floating=True,
                ignore_inertial_definitions=False,
                collapse_fixed_joints=False,
                enable_self_collisions=False,
                convert_3d_hinge_to_ball_joints=True,
                ignore_names=["floor", "ground"],
            )

            single_builder.add_ground_plane()

            # Replicate for parallel worlds
            builder = newton.ModelBuilder()
            builder.replicate(single_builder, self.num_worlds, spacing=(5.0, 5.0, 0.0))

            self.model = builder.finalize(requires_grad=True)
            self.solver = newton.solvers.SolverSemiImplicit(self.model)
            self.control = self.model.control()

            # Store joint structure info
            self.joints_per_world = self.model.joint_count // self.num_worlds
            self.bodies_per_world = self.model.body_count // self.num_worlds
            total_joint_q = self.model.joint_q.numpy().shape[0]
            total_joint_qd = self.model.joint_qd.numpy().shape[0]
            self.joint_q_size = total_joint_q // self.num_worlds
            self.joint_qd_size = total_joint_qd // self.num_worlds

            print(f"    [DiffG1Eval] Model built: {self.joints_per_world} joints/world, "
                  f"{self.bodies_per_world} bodies/world, "
                  f"joint_q_size={self.joint_q_size}, joint_qd_size={self.joint_qd_size}")

        finally:
            os.unlink(mjcf_path)

    def _build_theta_buffers(self):
        """Build Warp buffers for theta → joint_X_p mapping."""
        # theta as wp.array (requires_grad for tape)
        self.theta_wp = wp.array(
            self.theta_np.astype(np.float64),
            dtype=float, requires_grad=True, device=self.device,
        )

        # Build body_name → joint_local_index mapping
        # In Newton, each body has exactly one joint. Body i has joint i.
        # But with ball-joint conversion, some bodies get merged.
        # We use body_key to find body indices, then use the same index for joints
        # (Newton convention: body_key[i] corresponds to joint i).
        body_keys = self.model.body_key
        joint_keys = self.model.joint_key
        bodies_per_world = self.bodies_per_world
        joints_per_world = self.joints_per_world

        # Map body name → local index (first world only)
        body_name_to_idx = {}
        for i in range(bodies_per_world):
            body_name_to_idx[body_keys[i]] = i

        # Map joint name → local index (first world only)
        joint_name_to_idx = {}
        for i in range(joints_per_world):
            joint_name_to_idx[joint_keys[i]] = i

        # For each body in SYMMETRIC_PAIRS, find its joint index.
        # The body name is like "left_hip_pitch_link" and the associated
        # joint name is "left_hip_pitch_joint" (same prefix, different suffix).
        # But with ball-joint conversion, hip pitch/roll/yaw may be merged.
        # Strategy: look up by body name first; if not found, try joint name.
        param_joint_local_indices = []
        param_for_joint = []
        base_quats = []

        joint_X_p_np = self.model.joint_X_p.numpy()

        for param_idx, (left_body, right_body) in enumerate(SYMMETRIC_PAIRS):
            for body_name in (left_body, right_body):
                # Try body_key lookup (body index == joint index)
                if body_name in body_name_to_idx:
                    joint_local_idx = body_name_to_idx[body_name]
                    param_joint_local_indices.append(joint_local_idx)
                    param_for_joint.append(param_idx)
                    base_quats.append(joint_X_p_np[joint_local_idx, 3:7].copy())
                else:
                    # Try joint name (replace _link with _joint)
                    joint_name = body_name.replace("_link", "_joint")
                    if joint_name in joint_name_to_idx:
                        joint_local_idx = joint_name_to_idx[joint_name]
                        param_joint_local_indices.append(joint_local_idx)
                        param_for_joint.append(param_idx)
                        base_quats.append(joint_X_p_np[joint_local_idx, 3:7].copy())
                    else:
                        print(f"    [WARN] Body/joint not found: {body_name}")

        self.num_param_joints = len(param_joint_local_indices)
        expected = NUM_DESIGN_PARAMS * 2
        print(f"    [DiffG1Eval] Found {self.num_param_joints} parameterized joints "
              f"(expected {expected})")
        if self.num_param_joints != expected:
            print(f"    [WARN] Mismatch! Available body keys (first world): "
                  f"{body_keys[:bodies_per_world]}")
            print(f"    [WARN] Available joint keys (first world): "
                  f"{joint_keys[:joints_per_world]}")

        # Convert to warp arrays
        self.joint_local_indices_wp = wp.array(
            np.array(param_joint_local_indices, dtype=np.int32),
            dtype=int, device=self.device,
        )
        self.param_for_joint_wp = wp.array(
            np.array(param_for_joint, dtype=np.int32),
            dtype=int, device=self.device,
        )
        self.base_quats_wp = wp.array(
            np.array(base_quats, dtype=np.float64),
            dtype=float, device=self.device,
        )

    def _alloc_buffers(self):
        """Pre-allocate all GPU buffers."""
        total_physics_steps = self.horizon * self.num_substeps
        self.states = [
            self.model.state(requires_grad=True)
            for _ in range(total_physics_steps + 1)
        ]

        # Figure out the DOF structure from the model
        # joint_qd_start gives us the qd-space layout
        joint_qd_start = self.model.joint_qd_start.numpy()
        articulation_start = self.model.articulation_start.numpy()

        # For the first world/articulation:
        body_start = articulation_start[0]
        body_end = articulation_start[1]
        # Root DOFs: qd_start[body_start] to qd_start[body_start+1]
        # Actuated DOFs: qd_start[body_start+1] to qd_start[body_end]
        self.qd_dof_start = int(joint_qd_start[body_start + 1])
        qd_dof_end = int(joint_qd_start[body_end])
        self.num_act_dofs = qd_dof_end - self.qd_dof_start
        self.total_qd_per_world = self.joint_qd_size

        print(f"    [DiffG1Eval] DOF structure: qd_dof_start={self.qd_dof_start}, "
              f"num_act_dofs={self.num_act_dofs}, total_qd/world={self.total_qd_per_world}")

        # Pre-collected actions buffer
        self.pre_actions = [
            wp.zeros((self.num_worlds, self.num_act_dofs), dtype=float, device=self.device)
            for _ in range(self.horizon)
        ]

        # Loss buffer
        self.loss_buf = wp.zeros(1, dtype=float, requires_grad=True, device=self.device)

        # Initial body_q snapshot (flat, matching model.body_q layout)
        body_q_size = self.model.body_count * 7
        self.body_q_initial = wp.zeros(
            body_q_size, dtype=float,
            requires_grad=True, device=self.device,
        )

        # Build init qpos from env config
        with open(str(BASE_ENV_CONFIG), "r") as f:
            env_cfg = yaml.safe_load(f)
        init_pose_list = env_cfg.get("init_pose", [])

        # The init_pose in the YAML has format:
        # [x, y, z, dof0, dof1, ..., dof28] (3 + 29 = 32 entries for hinge joints)
        # But with ball joint conversion, joint_q layout changes.
        # We'll use Newton's default state as init and only override z height.
        # The model.state() already has sensible defaults from MJCF.
        init_state = self.model.state()
        newton.eval_fk(self.model, init_state.joint_q, init_state.joint_qd, init_state)
        wp.synchronize()
        init_qpos = init_state.joint_q.numpy().copy()

        # Override: set z-height for each world's root
        z_height = float(init_pose_list[2]) if len(init_pose_list) > 2 else 0.8
        for w in range(self.num_worlds):
            q_start = w * self.joint_q_size
            init_qpos[q_start + 2] = z_height  # z position
            init_qpos[q_start + 3] = 1.0       # qw = 1

        self.init_qpos_wp = wp.array(init_qpos, dtype=float, device=self.device)

        # Also store the init_qpos per-world size for the kernel
        # (the init kernel will just copy the full flat array)
        del init_state

    def compute_gradient(self, actions_list):
        """Compute ∂reward/∂theta via BPTT.

        Args:
            actions_list: list of (num_worlds, act_dim) numpy arrays, one per timestep.
                          act_dim should match self.num_act_dofs.

        Returns:
            (grad_theta_np, mean_forward_dist)
            grad_theta_np: (6,) numpy array of reward gradients w.r.t. theta
            mean_forward_dist: scalar, mean forward distance achieved
        """
        wp.synchronize()

        # ---------------------------------------------------------------
        # Prepare: Initialize state and load actions
        # ---------------------------------------------------------------
        state_0 = self.states[0]
        total_q = self.model.joint_q.numpy().shape[0]
        total_qd = self.model.joint_qd.numpy().shape[0]

        # Initialize joint_q from pre-computed init, zero joint_qd
        launch_dim = max(total_q, total_qd)
        wp.launch(diff_init_worlds_kernel, dim=launch_dim,
                  inputs=[state_0.joint_q, state_0.joint_qd,
                          self.init_qpos_wp, total_q, total_qd])
        wp.synchronize()

        # Store actions into pre-allocated warp buffers
        for step in range(min(self.horizon, len(actions_list))):
            actions_np = actions_list[step].astype(np.float64)
            act_dim = actions_np.shape[1] if actions_np.ndim > 1 else actions_np.shape[0]

            # Handle dimension mismatch: MimicKit actions may differ from
            # ball-joint DOF space. Truncate or pad as needed.
            if act_dim != self.num_act_dofs:
                adapted = np.zeros((self.num_worlds, self.num_act_dofs), dtype=np.float64)
                copy_dim = min(act_dim, self.num_act_dofs)
                adapted[:actions_np.shape[0], :copy_dim] = actions_np[:self.num_worlds, :copy_dim]
                actions_np = adapted
            else:
                # Pad/truncate worlds
                if actions_np.shape[0] < self.num_worlds:
                    padded = np.zeros((self.num_worlds, self.num_act_dofs), dtype=np.float64)
                    padded[:actions_np.shape[0]] = actions_np
                    actions_np = padded
                elif actions_np.shape[0] > self.num_worlds:
                    actions_np = actions_np[:self.num_worlds]

            self.pre_actions[step].assign(actions_np)

        # ---------------------------------------------------------------
        # Phase 2: Replay on tape for gradients
        # ---------------------------------------------------------------
        self.loss_buf.zero_()

        tape = wp.Tape()
        with tape:
            # Apply theta → joint_X_p on tape (differentiable)
            wp.launch(
                update_joint_X_p_from_theta_kernel,
                dim=self.num_param_joints * self.num_worlds,
                inputs=[
                    self.theta_wp,
                    self.base_quats_wp,
                    self.joint_local_indices_wp,
                    self.param_for_joint_wp,
                    self.model.joint_X_p,
                    self.joints_per_world,
                    self.num_worlds,
                    self.num_param_joints,
                ],
            )

            # FK to populate body_q from joint_q
            newton.eval_fk(self.model, state_0.joint_q, state_0.joint_qd, state_0)

            # Store initial body_q
            wp.copy(self.body_q_initial, state_0.body_q)

            # Physics loop
            physics_step = 0
            for step in range(self.horizon):
                # Set target positions (position control)
                wp.launch(
                    diff_set_target_pos_kernel,
                    dim=self.num_worlds,
                    inputs=[
                        self.pre_actions[step],
                        self.control.joint_target_pos,
                        self.qd_dof_start,
                        self.num_act_dofs,
                        self.total_qd_per_world,
                    ],
                )

                # Substeps
                for sub in range(self.num_substeps):
                    src = self.states[physics_step]
                    dst = self.states[physics_step + 1]

                    src.clear_forces()
                    contacts = self.model.collide(src)
                    self.solver.step(src, dst, self.control, contacts, self.sub_dt)

                    physics_step += 1

            # Compute loss: -mean(forward distance)
            final_state = self.states[physics_step]
            wp.launch(
                compute_forward_loss_kernel,
                dim=1,
                inputs=[
                    final_state.body_q,
                    self.body_q_initial,
                    self.bodies_per_world,
                    self.loss_buf,
                    self.num_worlds,
                ],
            )

        # Get loss value
        wp.synchronize()
        loss_val = self.loss_buf.numpy()[0]
        mean_forward_dist = -loss_val  # loss = -mean(fwd_dist)

        if np.isnan(loss_val) or np.isinf(loss_val):
            print("    [WARN] Loss is NaN/Inf, skipping backward")
            tape.zero()
            return np.zeros(NUM_DESIGN_PARAMS), 0.0

        # Backward
        tape.backward(self.loss_buf)
        wp.synchronize()

        # ---------------------------------------------------------------
        # Phase 3: Extract gradients
        # ---------------------------------------------------------------
        theta_grad = self.theta_wp.grad
        if theta_grad is None:
            print("    [WARN] No theta gradient available")
            tape.zero()
            return np.zeros(NUM_DESIGN_PARAMS), mean_forward_dist

        grad_np = theta_grad.numpy().copy()

        # Since loss = -mean(fwd_dist), and we want gradient ASCENT on fwd_dist,
        # ∂fwd_dist/∂theta = -∂loss/∂theta = -grad_np
        # But Adam uses gradient descent, and we negate in the optimizer step,
        # so we return grad_np as-is (the caller will negate for Adam).
        # Actually: tape.backward on loss gives ∂loss/∂theta.
        # reward = -loss, so ∂reward/∂theta = -∂loss/∂theta.
        # We want to ASCEND reward, so the gradient for ascent = ∂reward/∂theta = -grad_np
        # The caller uses Adam with negated gradient, so we return the reward gradient.
        reward_grad = -grad_np

        # Clip gradients
        grad_clip = 10.0
        reward_grad = np.clip(reward_grad, -grad_clip, grad_clip)

        tape.zero()

        return reward_grad.astype(np.float64), float(mean_forward_dist)

    def cleanup(self):
        """Free GPU resources."""
        for attr in ('states', 'model', 'solver', 'control',
                     'pre_actions', 'loss_buf', 'body_q_initial',
                     'theta_wp', 'base_quats_wp', 'joint_local_indices_wp',
                     'param_for_joint_wp', 'init_qpos_wp'):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass
        gc.collect()
        wp.synchronize()


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
        # Phase 1: Collect actions via MimicKit env
        # =============================================
        print(f"\n  [Phase 1] Collecting actions ({num_eval_worlds} worlds, {eval_horizon} steps)...")

        # Use the env config from this iteration
        env_config_path = iter_dir / "env_config.yaml"
        try:
            actions_list = collect_actions_mimickit(
                mimickit_src_dir=str(MIMICKIT_SRC_DIR),
                env_config_path=str(env_config_path),
                engine_config_path=str(BASE_ENGINE_CONFIG),
                agent_config_path=str(BASE_AGENT_CONFIG),
                checkpoint_path=checkpoint,
                num_worlds=num_eval_worlds,
                horizon=eval_horizon,
            )
            print(f"    Collected {len(actions_list)} steps × {actions_list[0].shape} actions")
        except Exception as e:
            print(f"    [Phase 1] FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

        # =============================================
        # Phase 2+3: BPTT gradient
        # =============================================
        print(f"\n  [Phase 2] Computing BPTT gradient...")

        try:
            diff_eval = DiffG1Eval(
                mjcf_modifier=mjcf_modifier,
                theta_np=theta,
                num_worlds=num_eval_worlds,
                horizon=eval_horizon,
                dt=1.0/30.0,
                num_substeps=8,
            )

            grad_theta, eval_fwd_dist = diff_eval.compute_gradient(actions_list)

            diff_eval.cleanup()
            wp.synchronize()

        except Exception as e:
            print(f"    [Phase 2] FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

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
