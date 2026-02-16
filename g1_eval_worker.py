#!/usr/bin/env python3
"""
G1 Eval Worker — GPU subprocess for PGHC co-design gradient computation.

Provides two evaluator classes:
  - DiffG1Eval: Legacy BPTT via SolverSemiImplicitStable + wp.Tape()
  - FDEvaluator: Central finite differences via SolverMuJoCo (preferred)

FDEvaluator uses the same solver as training (SolverMuJoCo), eliminating
the solver mismatch and contact instability issues of the BPTT approach.

Called by codesign_g1.py as:
    python g1_eval_worker.py --theta-file theta.npy --checkpoint model.pt \
        --env-config env_config.yaml --engine-config newton_engine.yaml \
        --agent-config amp_g1_agent.yaml --output-file eval_result.npz \
        --num-eval-worlds 8 --eval-horizon 100
"""

import os
os.environ["PYGLET_HEADLESS"] = "1"

import argparse
import gc
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

import warp as wp
import newton

from g1_mjcf_modifier import G1MJCFModifier, SYMMETRIC_PAIRS, NUM_DESIGN_PARAMS

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
# Warp kernels for differentiable evaluation
# ---------------------------------------------------------------------------

@wp.kernel
def update_joint_X_p_from_theta_kernel(
    theta: wp.array(dtype=float),
    base_quats: wp.array2d(dtype=float),
    joint_local_indices: wp.array(dtype=int),
    param_for_joint: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transformf),
    joints_per_world: int,
    num_worlds: int,
    num_param_joints: int,
):
    """Map theta → joint_X_p quaternions on tape for BPTT.

    Launched with dim = num_param_joints * num_worlds.
    Each thread handles one (parameterized joint, world) pair.
    joint_X_p has dtype=transformf (pos: vec3f, rot: quatf).
    base_quats stores the 4 quaternion components in (w, x, y, z) order,
    reordered from joint_X_p's native (x, y, z, w) layout at extraction.
    """
    tid = wp.tid()
    j = tid // num_worlds   # parameterized joint index (0..11)
    w = tid % num_worlds    # world index
    if j >= num_param_joints:
        return

    p_idx = param_for_joint[j]
    angle = theta[p_idx]

    # Delta quat from X-rotation
    half = angle * 0.5
    dw = wp.cos(half)
    dx = wp.sin(half)

    # Base quat components in (w, x, y, z) order (reordered at extraction)
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

    # Read current transform (preserves position), write new rotation
    global_idx = w * joints_per_world + joint_local_indices[j]
    pos = wp.transform_get_translation(joint_X_p[global_idx])
    # wp.quatf() takes (x, y, z, w); Hamilton product gives (nw, nx, ny, nz)
    # in wxyz order, so pass as quatf(nx, ny, nz, nw) to convert to xyzw.
    joint_X_p[global_idx] = wp.transformf(pos, wp.quatf(nx, ny, nz, nw))


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
    joint_q_final: wp.array(dtype=float),
    joint_q_initial: wp.array(dtype=float),
    joint_q_size: int,
    loss: wp.array(dtype=float),
    num_worlds: int,
):
    """loss = -mean(x_final - x_initial) over all worlds.

    Uses joint_q (always flat float) instead of body_q.
    Root x position is at joint_q[w * joint_q_size + 0].
    """
    tid = wp.tid()
    if tid == 0:
        total = float(0.0)
        for w in range(num_worlds):
            idx = w * joint_q_size
            x_final = joint_q_final[idx]
            x_initial = joint_q_initial[idx]
            total = total + (x_final - x_initial)
        loss[0] = -total / float(num_worlds)


@wp.kernel
def accumulate_qd_energy_kernel(
    joint_qd: wp.array(dtype=float),
    energy_buf: wp.array(dtype=float),
    qd_dof_start: int,
    num_act_dofs: int,
    total_qd_per_world: int,
    num_worlds: int,
    sub_dt: float,
):
    """Accumulate ||qd_actuated||^2 * sub_dt / num_worlds into energy_buf[0].

    Called once per substep (inside tape). Single-threaded — 32 worlds x ~30 DOFs
    is trivial.
    """
    tid = wp.tid()
    if tid == 0:
        total = float(0.0)
        for w in range(num_worlds):
            base = w * total_qd_per_world + qd_dof_start
            for d in range(num_act_dofs):
                qd_val = joint_qd[base + d]
                total = total + qd_val * qd_val
        energy_buf[0] = energy_buf[0] + total * sub_dt / float(num_worlds)


@wp.kernel
def accumulate_mechanical_power_kernel(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_target_pos: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    energy_buf: wp.array(dtype=float),
    qd_dof_start: int,
    q_dof_start: int,
    num_act_dofs: int,
    total_qd_per_world: int,
    total_q_per_world: int,
    num_worlds: int,
    sub_dt: float,
):
    """Accumulate Σ|τ·qd| * sub_dt / num_worlds into energy_buf[0].

    Mechanical power: P = |τ_i * qd_i| where τ_i = ke_i*(target_i - q_i) - kd_i*qd_i.
    More sensitive to morphology changes than ||qd||^2 because torques depend
    directly on joint positions which shift with body frame orientation.
    """
    tid = wp.tid()
    if tid == 0:
        total = float(0.0)
        for w in range(num_worlds):
            qd_base = w * total_qd_per_world + qd_dof_start
            q_base = w * total_q_per_world + q_dof_start
            for d in range(num_act_dofs):
                qd_val = joint_qd[qd_base + d]
                q_val = joint_q[q_base + d]
                target = joint_target_pos[qd_base + d]
                ke = joint_target_ke[qd_base + d]
                kd = joint_target_kd[qd_base + d]
                torque = ke * (target - q_val) - kd * qd_val
                total = total + wp.abs(torque * qd_val)
        energy_buf[0] = energy_buf[0] + total * sub_dt / float(num_worlds)


@wp.kernel
def compute_cot_loss_kernel(
    joint_q_final: wp.array(dtype=float),
    joint_q_initial: wp.array(dtype=float),
    joint_q_size: int,
    energy_buf: wp.array(dtype=float),
    total_mass: float,
    loss: wp.array(dtype=float),
    fwd_dist_buf: wp.array(dtype=float),
    num_worlds: int,
):
    """loss = CoT = energy / (m * g * d).

    Uses joint_q (always flat float) instead of body_q.
    Root x position is at joint_q[w * joint_q_size + 0].
    Also writes mean forward distance to fwd_dist_buf for logging.
    """
    tid = wp.tid()
    if tid == 0:
        total_dist = float(0.0)
        for w in range(num_worlds):
            idx = w * joint_q_size
            x_final = joint_q_final[idx]
            x_initial = joint_q_initial[idx]
            total_dist = total_dist + (x_final - x_initial)
        mean_dist = total_dist / float(num_worlds)
        fwd_dist_buf[0] = mean_dist

        # Clamp distance to avoid div-by-zero
        safe_dist = wp.max(mean_dist, 0.1)
        loss[0] = energy_buf[0] / (total_mass * 9.81 * safe_dist)


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

    Builds a small eval env with SolverSemiImplicitStable, runs a frozen-policy
    episode on wp.Tape(), then tape.backward() to get ∂reward/∂joint_X_p,
    and chains that to ∂reward/∂theta via the Warp kernel.

    SolverSemiImplicitStable uses implicit joint attachment forces that are
    unconditionally stable — no mass/inertia clamping needed, preserving
    correct physics for meaningful BPTT gradients.
    """

    def __init__(
        self,
        mjcf_modifier,
        theta_np,
        num_worlds=8,
        horizon=100,
        dt=1.0/30.0,
        num_substeps=16,
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
        """Build Newton model with SolverSemiImplicitStable for BPTT."""
        # Generate modified MJCF
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.xml', delete=False, dir=str(BASE_MJCF_PATH.parent)
        ) as f:
            self.mjcf_modifier.generate(self.theta_np, f.name)
            mjcf_path = f.name

        try:
            single_builder = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(single_builder)

            # Match MimicKit's newton_engine.py add_mjcf flags where possible.
            # enable_self_collisions=False because penalty-based contacts in
            # SolverSemiImplicitStable can't handle inter-link self-contacts
            # (causes immediate blowup). MuJoCo's implicit solver handles them
            # fine, but this is an inherent solver limitation.
            single_builder.add_mjcf(
                mjcf_path,
                floating=True,
                ignore_inertial_definitions=False,
                collapse_fixed_joints=False,
                enable_self_collisions=False,
                convert_3d_hinge_to_ball_joints=True,
            )

            # Match MimicKit ground friction (newton_engine.py _build_ground)
            ground_cfg = single_builder.ShapeConfig(mu=1.0, restitution=0)
            single_builder.add_ground_plane(cfg=ground_cfg)

            # Replicate for parallel worlds
            builder = newton.ModelBuilder()
            builder.replicate(single_builder, self.num_worlds, spacing=(5.0, 5.0, 0.0))

            self.model = builder.finalize(requires_grad=True)

            # No mass/inertia clamping needed — SolverSemiImplicitStable uses
            # implicit joint forces that are unconditionally stable regardless
            # of body mass or inertia (denominator m + dt²*ke + dt*kd > 0 always).
            mass_np = self.model.body_mass.numpy()
            min_mass = float(mass_np.min())
            inertia_np = self.model.body_inertia.numpy()
            min_inertia = float(min(inertia_np[bi, d, d]
                                    for bi in range(len(inertia_np))
                                    for d in range(3)))
            print(f"    [DiffG1Eval] Body stats (unmodified): "
                  f"min_mass={min_mass:.4g} kg, min_inertia_diag={min_inertia:.4g} kg·m²")

            self.solver = newton.solvers.SolverSemiImplicitStable(
                self.model, joint_attach_ke=1.0e3, joint_attach_kd=100.0
            )
            self.control = self.model.control()

            # Set PD gains matching MimicKit's ControlMode.pos behavior.
            # Without this, joint_target_ke/kd are 0 and position targets
            # produce zero torques — the robot just stands still.
            if hasattr(self.model, 'mujoco'):
                ke_np = self.model.mujoco.dof_passive_stiffness.numpy().copy()
                kd_np = self.model.mujoco.dof_passive_damping.numpy().copy()
                self.model.joint_target_ke.assign(ke_np)
                self.model.joint_target_kd.assign(kd_np)
                # Zero passive gains to avoid double-counting
                self.model.mujoco.dof_passive_stiffness.assign(np.zeros_like(ke_np))
                self.model.mujoco.dof_passive_damping.assign(np.zeros_like(kd_np))
                nonzero_ke = ke_np[ke_np > 0]
                nonzero_kd = kd_np[kd_np > 0]
                print(f"    [DiffG1Eval] PD gains set: "
                      f"ke={nonzero_ke.mean():.1f} (n={len(nonzero_ke)}), "
                      f"kd={nonzero_kd.mean():.1f} (n={len(nonzero_kd)})")
            else:
                print("    [WARN] No mujoco attributes — setting default PD gains (ke=100, kd=10)")
                ke_np = self.model.joint_target_ke.numpy()
                kd_np = self.model.joint_target_kd.numpy()
                ke_np[:] = 100.0
                kd_np[:] = 10.0
                self.model.joint_target_ke.assign(ke_np)
                self.model.joint_target_kd.assign(kd_np)

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
                    # joint_X_p numpy layout is [px,py,pz, qx,qy,qz,qw] (xyzw)
                    # Reorder to (w,x,y,z) for the Hamilton product in the kernel
                    raw = joint_X_p_np[joint_local_idx, 3:7].copy()  # [qx,qy,qz,qw]
                    base_quats.append([raw[3], raw[0], raw[1], raw[2]])  # [qw,qx,qy,qz]
                else:
                    # Try joint name (replace _link with _joint)
                    joint_name = body_name.replace("_link", "_joint")
                    if joint_name in joint_name_to_idx:
                        joint_local_idx = joint_name_to_idx[joint_name]
                        param_joint_local_indices.append(joint_local_idx)
                        param_for_joint.append(param_idx)
                        raw = joint_X_p_np[joint_local_idx, 3:7].copy()
                        base_quats.append([raw[3], raw[0], raw[1], raw[2]])
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
        joint_qd_start = self.model.joint_qd_start.numpy()
        articulation_start = self.model.articulation_start.numpy()

        # For the first world/articulation:
        body_start = articulation_start[0]
        body_end = articulation_start[1]
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

        # Energy accumulation buffer (CoT numerator)
        self.energy_buf = wp.zeros(1, dtype=float, requires_grad=True, device=self.device)

        # Forward distance buffer (for logging, not differentiated)
        self.fwd_dist_buf = wp.zeros(1, dtype=float, device=self.device)

        # Total mass of one robot (for CoT denominator)
        body_mass_np = self.model.body_mass.numpy()
        self.total_mass = float(body_mass_np[:self.bodies_per_world].sum())
        print(f"    [DiffG1Eval] Total robot mass = {self.total_mass:.2f} kg")

        # Initial joint_q snapshot buffer (for forward distance computation)
        total_q = self.model.joint_q.numpy().shape[0]
        self.joint_q_initial = wp.zeros(
            total_q, dtype=float,
            requires_grad=True, device=self.device,
        )

        # Build init qpos from env config.
        # MimicKit init_pose format: [x,y,z, rx,ry,rz, dof0,dof1,...] where
        # rotation is exp-map. For G1, root rot is [0,0,0] (identity) and
        # the only non-zero DOFs are shoulder pitch (1.57 rad) at arms, which
        # don't affect lower-body locomotion evaluation.
        # Full DOF mapping is complex (ball-joint conversion changes indices),
        # so we use MJCF defaults for joint angles and only override root pose.
        with open(str(BASE_ENV_CONFIG), "r") as f:
            env_cfg = yaml.safe_load(f)
        init_pose_list = env_cfg.get("init_pose", [])

        init_state = self.model.state()
        newton.eval_fk(self.model, init_state.joint_q, init_state.joint_qd, init_state)
        wp.synchronize()
        init_qpos = init_state.joint_q.numpy().copy()

        # Override root pose: z-height and identity quaternion
        z_height = float(init_pose_list[2]) if len(init_pose_list) > 2 else 0.8
        for w in range(self.num_worlds):
            q_start = w * self.joint_q_size
            init_qpos[q_start + 2] = z_height  # z position
            # joint_q root quat: [qx, qy, qz, qw] at indices 3:7 (Warp xyzw convention)
            init_qpos[q_start + 3] = 0.0       # qx = 0
            init_qpos[q_start + 4] = 0.0       # qy = 0
            init_qpos[q_start + 5] = 0.0       # qz = 0
            init_qpos[q_start + 6] = 1.0       # qw = 1

        self.init_qpos_wp = wp.array(init_qpos, dtype=float, device=self.device)

        del init_state

    @staticmethod
    def _dbg_wp(name, arr, max_print=8):
        """Debug helper: print stats for a warp array, flag NaN/Inf."""
        a = arr.numpy()
        has_nan = np.any(np.isnan(a))
        has_inf = np.any(np.isinf(a))
        tag = ""
        if has_nan:
            nan_count = int(np.isnan(a).sum())
            tag += f" *** NaN({nan_count}/{a.size}) ***"
        if has_inf:
            inf_count = int(np.isinf(a).sum())
            tag += f" *** Inf({inf_count}/{a.size}) ***"
        finite = a[np.isfinite(a)]
        if finite.size > 0:
            print(f"      [{name}] shape={a.shape} min={finite.min():.6g} "
                  f"max={finite.max():.6g} mean={finite.mean():.6g} "
                  f"absmax={np.abs(finite).max():.6g}{tag}")
        else:
            print(f"      [{name}] shape={a.shape} ALL NaN/Inf{tag}")
        return has_nan or has_inf

    def compute_gradient(self, actions_list):
        """Compute ∂CoT/∂theta via BPTT and return negative gradient for ascent.

        CoT = Energy / (m * g * d), where Energy = mean(Σ ||qd_actuated||² * dt).

        Args:
            actions_list: list of (num_worlds, act_dim) numpy arrays, one per timestep.
                          act_dim should match self.num_act_dofs.

        Returns:
            (grad_theta_np, mean_forward_dist, cot_value)
            grad_theta_np: (6,) numpy array of -∂CoT/∂theta (for gradient ascent)
            mean_forward_dist: scalar, mean forward distance achieved
            cot_value: scalar, Cost of Transport
        """
        wp.synchronize()
        DBG = "    [BPTT-DBG]"

        # ---------------------------------------------------------------
        # Prepare: Initialize state and load actions
        # ---------------------------------------------------------------
        state_0 = self.states[0]
        total_q = self.model.joint_q.numpy().shape[0]
        total_qd = self.model.joint_qd.numpy().shape[0]

        print(f"{DBG} === PHASE 0: Initialization ===")
        print(f"{DBG} total_q={total_q}, total_qd={total_qd}, "
              f"num_worlds={self.num_worlds}, horizon={self.horizon}, "
              f"num_substeps={self.num_substeps}")
        print(f"{DBG} joint_q_size/world={self.joint_q_size}, "
              f"joint_qd_size/world={self.joint_qd_size}")
        print(f"{DBG} qd_dof_start={self.qd_dof_start}, "
              f"num_act_dofs={self.num_act_dofs}")

        # Check theta input
        print(f"{DBG} theta_wp:")
        self._dbg_wp("theta", self.theta_wp)

        # Check init qpos
        print(f"{DBG} init_qpos_wp:")
        self._dbg_wp("init_qpos", self.init_qpos_wp)

        # Initialize joint_q from pre-computed init, zero joint_qd
        launch_dim = max(total_q, total_qd)
        wp.launch(diff_init_worlds_kernel, dim=launch_dim,
                  inputs=[state_0.joint_q, state_0.joint_qd,
                          self.init_qpos_wp, total_q, total_qd])
        wp.synchronize()

        print(f"{DBG} After init_worlds:")
        self._dbg_wp("state_0.joint_q", state_0.joint_q)
        self._dbg_wp("state_0.joint_qd", state_0.joint_qd)

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

        # Diagnostic: verify actions are non-zero
        print(f"{DBG} === Actions summary ===")
        for step_i in [0, self.horizon // 2, self.horizon - 1]:
            if step_i < len(self.pre_actions):
                a = self.pre_actions[step_i].numpy()
                print(f"{DBG}   step {step_i}: mean|a|={np.abs(a).mean():.4f}, "
                      f"max|a|={np.abs(a).max():.4f}, "
                      f"nan={np.isnan(a).sum()}, inf={np.isinf(a).sum()}")

        # Check PD gains
        print(f"{DBG} === PD gains ===")
        self._dbg_wp("joint_target_ke", self.model.joint_target_ke)
        self._dbg_wp("joint_target_kd", self.model.joint_target_kd)

        # Check joint_X_p before tape
        print(f"{DBG} === joint_X_p (before tape) ===")
        self._dbg_wp("joint_X_p", self.model.joint_X_p)

        # ---------------------------------------------------------------
        # Phase 2: Replay on tape for gradients
        # ---------------------------------------------------------------
        print(f"{DBG} === PHASE 2: Forward pass on tape ===")
        self.loss_buf.zero_()
        self.energy_buf.zero_()

        tape = wp.Tape()
        first_nan_step = None
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

            wp.synchronize()
            print(f"{DBG} After theta→joint_X_p kernel:")
            self._dbg_wp("joint_X_p", self.model.joint_X_p)

            # FK to populate body_q from joint_q
            newton.eval_fk(self.model, state_0.joint_q, state_0.joint_qd, state_0)
            wp.synchronize()

            print(f"{DBG} After eval_fk:")
            self._dbg_wp("state_0.joint_q", state_0.joint_q)
            self._dbg_wp("state_0.body_q", state_0.body_q)

            # Root positions for world 0 (x,y,z)
            jq0 = state_0.joint_q.numpy()
            print(f"{DBG} World 0 root pos: x={jq0[0]:.4f}, y={jq0[1]:.4f}, z={jq0[2]:.4f}")
            print(f"{DBG} World 0 root quat: {jq0[3:7]}")

            # Store initial joint_q (for forward distance computation)
            wp.copy(self.joint_q_initial, state_0.joint_q)

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

                # Collide once per macro-step (not per substep) to avoid
                # GPU memory accumulation — same fix as Known Issue #15.
                macro_src = self.states[physics_step]
                macro_src.clear_forces()
                contacts = self.model.collide(macro_src)

                # Substeps
                for sub in range(self.num_substeps):
                    src = self.states[physics_step]
                    dst = self.states[physics_step + 1]

                    src.clear_forces()
                    self.solver.step(src, dst, self.control, contacts, self.sub_dt)

                    # SolverSemiImplicitStable only updates body_q/body_qd (maximal
                    # coords). Our loss/energy kernels read joint_q/joint_qd
                    # (generalized coords), so we must run eval_ik to map
                    # body-space → joint-space after each integration step.
                    newton.eval_ik(self.model, dst, dst.joint_q, dst.joint_qd)

                    # Accumulate joint velocity energy on tape
                    wp.launch(
                        accumulate_qd_energy_kernel,
                        dim=1,
                        inputs=[
                            dst.joint_qd,
                            self.energy_buf,
                            self.qd_dof_start,
                            self.num_act_dofs,
                            self.total_qd_per_world,
                            self.num_worlds,
                            self.sub_dt,
                        ],
                    )

                    physics_step += 1

                # Periodic state health check (every 10 macro-steps + first + last)
                if step == 0 or step == self.horizon - 1 or step % 10 == 0:
                    wp.synchronize()
                    dst_check = self.states[physics_step]
                    jq = dst_check.joint_q.numpy()
                    jqd = dst_check.joint_qd.numpy()
                    bq = dst_check.body_q.numpy()
                    has_nan_q = np.any(np.isnan(jq))
                    has_nan_qd = np.any(np.isnan(jqd))
                    has_nan_bq = np.any(np.isnan(bq))
                    has_inf_q = np.any(np.isinf(jq))
                    has_inf_qd = np.any(np.isinf(jqd))

                    # Per-world root x position
                    root_xs = [jq[w * self.joint_q_size] for w in range(self.num_worlds)]
                    root_zs = [jq[w * self.joint_q_size + 2] for w in range(self.num_worlds)]

                    energy_so_far = float(self.energy_buf.numpy()[0])

                    tag = ""
                    if has_nan_q or has_nan_qd or has_nan_bq:
                        tag = " *** NaN DETECTED ***"
                        if first_nan_step is None:
                            first_nan_step = step
                    if has_inf_q or has_inf_qd:
                        tag += " *** Inf DETECTED ***"

                    print(f"{DBG} Step {step}/{self.horizon} (phys={physics_step}): "
                          f"root_x=[{min(root_xs):.3f}..{max(root_xs):.3f}] "
                          f"root_z=[{min(root_zs):.3f}..{max(root_zs):.3f}] "
                          f"|jq|max={np.abs(jq[np.isfinite(jq)]).max():.3g} "
                          f"|jqd|max={np.abs(jqd[np.isfinite(jqd)]).max() if np.any(np.isfinite(jqd)) else float('nan'):.3g} "
                          f"energy={energy_so_far:.4f}{tag}")

                    if has_nan_q or has_nan_qd:
                        nan_q_idx = np.where(np.isnan(jq))[0]
                        nan_qd_idx = np.where(np.isnan(jqd))[0]
                        if len(nan_q_idx) > 0:
                            # Map to world and local index
                            worlds = nan_q_idx // self.joint_q_size
                            locals_ = nan_q_idx % self.joint_q_size
                            print(f"{DBG}   NaN joint_q at {len(nan_q_idx)} indices, "
                                  f"worlds={np.unique(worlds).tolist()}, "
                                  f"local_idx={np.unique(locals_).tolist()[:20]}")
                        if len(nan_qd_idx) > 0:
                            worlds = nan_qd_idx // self.joint_qd_size
                            locals_ = nan_qd_idx % self.joint_qd_size
                            print(f"{DBG}   NaN joint_qd at {len(nan_qd_idx)} indices, "
                                  f"worlds={np.unique(worlds).tolist()}, "
                                  f"local_idx={np.unique(locals_).tolist()[:20]}")

                    if has_nan_bq:
                        nan_bq_idx = np.where(np.isnan(bq.ravel()))[0]
                        print(f"{DBG}   NaN body_q at {len(nan_bq_idx)} elements")

            # Compute loss: CoT = energy / (m * g * d)
            final_state = self.states[physics_step]
            wp.launch(
                compute_cot_loss_kernel,
                dim=1,
                inputs=[
                    final_state.joint_q,
                    self.joint_q_initial,
                    self.joint_q_size,
                    self.energy_buf,
                    self.total_mass,
                    self.loss_buf,
                    self.fwd_dist_buf,
                    self.num_worlds,
                ],
            )

        # Get loss value (CoT) and forward distance
        wp.synchronize()
        cot_val = self.loss_buf.numpy()[0]
        mean_forward_dist = float(self.fwd_dist_buf.numpy()[0])
        energy_val = float(self.energy_buf.numpy()[0])

        print(f"{DBG} === Forward pass summary ===")
        print(f"{DBG} energy={energy_val:.6g}, fwd_dist={mean_forward_dist:.6g}, "
              f"CoT={cot_val:.6g}, total_mass={self.total_mass:.2f}")
        print(f"{DBG} loss_buf={cot_val:.6g} "
              f"(nan={np.isnan(cot_val)}, inf={np.isinf(cot_val)})")
        if first_nan_step is not None:
            print(f"{DBG} *** First NaN appeared at macro-step {first_nan_step} ***")

        # Per-world forward distances
        jq_final = self.states[physics_step].joint_q.numpy()
        jq_init = self.joint_q_initial.numpy()
        per_world_fwd = []
        for w in range(self.num_worlds):
            idx = w * self.joint_q_size
            per_world_fwd.append(jq_final[idx] - jq_init[idx])
        per_world_fwd = np.array(per_world_fwd)
        print(f"{DBG} Per-world fwd_dist: min={per_world_fwd.min():.4f}, "
              f"max={per_world_fwd.max():.4f}, mean={per_world_fwd.mean():.4f}, "
              f"std={per_world_fwd.std():.4f}")
        print(f"{DBG} Per-world fwd_dist values: {per_world_fwd.tolist()}")

        if np.isnan(cot_val) or np.isinf(cot_val):
            print(f"{DBG} *** CoT is NaN/Inf — skipping backward ***")
            print(f"{DBG}   energy={energy_val}, fwd_dist={mean_forward_dist}")
            print(f"{DBG}   Denominator would be: m*g*d = "
                  f"{self.total_mass}*9.81*{max(mean_forward_dist,0.1):.4f} = "
                  f"{self.total_mass * 9.81 * max(mean_forward_dist, 0.1):.4f}")
            tape.zero()
            return np.zeros(NUM_DESIGN_PARAMS), 0.0, float('inf')

        # Backward: minimize CoT
        print(f"{DBG} === PHASE 3: Backward pass ===")
        tape.backward(self.loss_buf)
        wp.synchronize()

        # ---------------------------------------------------------------
        # Phase 3: Extract gradients
        # ---------------------------------------------------------------
        # Check intermediate gradients
        print(f"{DBG} Checking intermediate gradients...")

        # joint_X_p gradient
        if self.model.joint_X_p.grad is not None:
            self._dbg_wp("joint_X_p.grad", self.model.joint_X_p.grad)
        else:
            print(f"{DBG}   joint_X_p.grad = None")

        # base_quats gradient
        if self.base_quats_wp.grad is not None:
            self._dbg_wp("base_quats.grad", self.base_quats_wp.grad)
        else:
            print(f"{DBG}   base_quats.grad = None")

        # energy_buf gradient
        if self.energy_buf.grad is not None:
            self._dbg_wp("energy_buf.grad", self.energy_buf.grad)
        else:
            print(f"{DBG}   energy_buf.grad = None")

        # loss_buf gradient
        if self.loss_buf.grad is not None:
            self._dbg_wp("loss_buf.grad", self.loss_buf.grad)
        else:
            print(f"{DBG}   loss_buf.grad = None")

        # Spot-check a few state gradients (first, middle, last physics step)
        total_phys = self.horizon * self.num_substeps
        for si in [0, total_phys // 2, total_phys]:
            if si < len(self.states):
                s = self.states[si]
                label = f"states[{si}]"
                if s.joint_q.grad is not None:
                    self._dbg_wp(f"{label}.joint_q.grad", s.joint_q.grad)
                else:
                    print(f"{DBG}   {label}.joint_q.grad = None")
                if s.joint_qd.grad is not None:
                    self._dbg_wp(f"{label}.joint_qd.grad", s.joint_qd.grad)
                else:
                    print(f"{DBG}   {label}.joint_qd.grad = None")
                if s.body_q.grad is not None:
                    self._dbg_wp(f"{label}.body_q.grad", s.body_q.grad)
                else:
                    print(f"{DBG}   {label}.body_q.grad = None")

        # theta gradient
        theta_grad = self.theta_wp.grad
        if theta_grad is None:
            print(f"{DBG} *** theta_wp.grad = None — gradient chain broken ***")
            tape.zero()
            return np.zeros(NUM_DESIGN_PARAMS), mean_forward_dist, float(cot_val)

        grad_np = theta_grad.numpy().copy()
        print(f"{DBG} theta_wp.grad = {grad_np.tolist()}")

        # Guard against NaN/Inf gradients (degenerate backward pass)
        if np.any(np.isnan(grad_np)) or np.any(np.isinf(grad_np)):
            print(f"{DBG} *** NaN/Inf in theta gradient ***")
            print(f"{DBG}   grad values: {grad_np.tolist()}")
            print(f"{DBG}   nan mask: {np.isnan(grad_np).tolist()}")
            print(f"{DBG}   inf mask: {np.isinf(grad_np).tolist()}")
            tape.zero()
            return np.zeros(NUM_DESIGN_PARAMS), float(mean_forward_dist), float(cot_val)

        # loss = CoT → minimize CoT
        # reward_grad = -∂loss/∂theta (gradient ascent on -CoT = descent on CoT)
        reward_grad = -grad_np
        print(f"{DBG} reward_grad (before clip) = {reward_grad.tolist()}")

        # Clip gradients
        grad_clip = 10.0
        reward_grad = np.clip(reward_grad, -grad_clip, grad_clip)
        print(f"{DBG} reward_grad (after clip)  = {reward_grad.tolist()}")

        tape.zero()

        return reward_grad.astype(np.float64), float(mean_forward_dist), float(cot_val)

    def cleanup(self):
        """Free GPU resources."""
        for attr in ('states', 'model', 'solver', 'control',
                     'pre_actions', 'loss_buf', 'energy_buf', 'fwd_dist_buf',
                     'joint_q_initial', 'theta_wp', 'base_quats_wp',
                     'joint_local_indices_wp', 'param_for_joint_wp',
                     'init_qpos_wp'):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass
        gc.collect()
        wp.synchronize()


# ---------------------------------------------------------------------------
# FDEvaluator — Finite Difference gradient via SolverMuJoCo
# ---------------------------------------------------------------------------

class FDEvaluator:
    """Compute design gradients via central finite differences.

    Uses SolverMuJoCo (same solver as training) to eliminate solver mismatch.
    Only 2 state buffers needed (ping-pong), no adjoint/tape memory.
    """

    def __init__(
        self,
        mjcf_modifier,
        theta_np,
        num_worlds=32,
        horizon=100,
        device="cuda:0",
    ):
        self.mjcf_modifier = mjcf_modifier
        self.theta_np = theta_np.copy()
        self.num_worlds = num_worlds
        self.horizon = horizon
        self.device = device

        # Sim params matching training (newton_engine.yaml)
        self.sim_dt = 1.0 / 240.0      # 240 Hz sim
        self.control_dt = 1.0 / 30.0    # 30 Hz control
        self.num_substeps = 8           # 240 / 30

        self._build_model()
        self._build_theta_buffers()
        self._alloc_buffers()

    def _build_model(self):
        """Build Newton model with SolverMuJoCo (matching training physics)."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.xml', delete=False, dir=str(BASE_MJCF_PATH.parent)
        ) as f:
            self.mjcf_modifier.generate(self.theta_np, f.name)
            mjcf_path = f.name

        try:
            single_builder = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(single_builder)

            # Match MimicKit's newton_engine.py add_mjcf flags exactly,
            # but with enable_self_collisions=True (SolverMuJoCo handles them)
            single_builder.add_mjcf(
                mjcf_path,
                floating=True,
                ignore_inertial_definitions=False,
                collapse_fixed_joints=False,
                enable_self_collisions=True,
                convert_3d_hinge_to_ball_joints=True,
            )

            # Match MimicKit ground friction
            ground_cfg = single_builder.ShapeConfig(mu=1.0, restitution=0)
            single_builder.add_ground_plane(cfg=ground_cfg)

            # Replicate for parallel worlds
            builder = newton.ModelBuilder()
            builder.replicate(single_builder, self.num_worlds, spacing=(5.0, 5.0, 0.0))

            self.model = builder.finalize(requires_grad=False)

            # SolverMuJoCo matching training exactly
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                solver="newton",
                njmax=450,
                nconmax=150,
                impratio=10,
                iterations=100,
                ls_iterations=50,
            )
            self.control = self.model.control()

            # PD gains: transfer from MuJoCo passive stiffness/damping
            if hasattr(self.model, 'mujoco'):
                ke_np = self.model.mujoco.dof_passive_stiffness.numpy().copy()
                kd_np = self.model.mujoco.dof_passive_damping.numpy().copy()
                self.model.joint_target_ke.assign(ke_np)
                self.model.joint_target_kd.assign(kd_np)
                self.model.mujoco.dof_passive_stiffness.assign(np.zeros_like(ke_np))
                self.model.mujoco.dof_passive_damping.assign(np.zeros_like(kd_np))
                nonzero_ke = ke_np[ke_np > 0]
                nonzero_kd = kd_np[kd_np > 0]
                print(f"    [FDEval] PD gains set: "
                      f"ke={nonzero_ke.mean():.1f} (n={len(nonzero_ke)}), "
                      f"kd={nonzero_kd.mean():.1f} (n={len(nonzero_kd)})")
            else:
                print("    [WARN] No mujoco attributes — setting default PD gains")
                ke_np = self.model.joint_target_ke.numpy()
                kd_np = self.model.joint_target_kd.numpy()
                ke_np[:] = 100.0
                kd_np[:] = 10.0
                self.model.joint_target_ke.assign(ke_np)
                self.model.joint_target_kd.assign(kd_np)

            # Joint structure info
            self.joints_per_world = self.model.joint_count // self.num_worlds
            self.bodies_per_world = self.model.body_count // self.num_worlds
            total_joint_q = self.model.joint_q.numpy().shape[0]
            total_joint_qd = self.model.joint_qd.numpy().shape[0]
            self.joint_q_size = total_joint_q // self.num_worlds
            self.joint_qd_size = total_joint_qd // self.num_worlds

            print(f"    [FDEval] Model built: {self.joints_per_world} joints/world, "
                  f"{self.bodies_per_world} bodies/world, "
                  f"joint_q_size={self.joint_q_size}, joint_qd_size={self.joint_qd_size}")

        finally:
            os.unlink(mjcf_path)

    def _build_theta_buffers(self):
        """Build buffers for theta → joint_X_p mapping (no requires_grad)."""
        # Build body_name → joint_local_index mapping
        body_keys = self.model.body_key
        joint_keys = self.model.joint_key
        bodies_per_world = self.bodies_per_world
        joints_per_world = self.joints_per_world

        body_name_to_idx = {}
        for i in range(bodies_per_world):
            body_name_to_idx[body_keys[i]] = i

        joint_name_to_idx = {}
        for i in range(joints_per_world):
            joint_name_to_idx[joint_keys[i]] = i

        self.param_joint_local_indices = []
        self.param_for_joint = []
        self.base_quats = []  # (w, x, y, z) order

        joint_X_p_np = self.model.joint_X_p.numpy()

        for param_idx, (left_body, right_body) in enumerate(SYMMETRIC_PAIRS):
            for body_name in (left_body, right_body):
                if body_name in body_name_to_idx:
                    joint_local_idx = body_name_to_idx[body_name]
                elif body_name.replace("_link", "_joint") in joint_name_to_idx:
                    joint_local_idx = joint_name_to_idx[body_name.replace("_link", "_joint")]
                else:
                    print(f"    [WARN] Body/joint not found: {body_name}")
                    continue

                self.param_joint_local_indices.append(joint_local_idx)
                self.param_for_joint.append(param_idx)
                # joint_X_p numpy: [px,py,pz, qx,qy,qz,qw] → reorder to (w,x,y,z)
                raw = joint_X_p_np[joint_local_idx, 3:7].copy()  # [qx,qy,qz,qw]
                self.base_quats.append([raw[3], raw[0], raw[1], raw[2]])

        self.num_param_joints = len(self.param_joint_local_indices)
        expected = NUM_DESIGN_PARAMS * 2
        print(f"    [FDEval] Found {self.num_param_joints} parameterized joints "
              f"(expected {expected})")

        # Convert to warp arrays for the kernel
        self.joint_local_indices_wp = wp.array(
            np.array(self.param_joint_local_indices, dtype=np.int32),
            dtype=int, device=self.device,
        )
        self.param_for_joint_wp = wp.array(
            np.array(self.param_for_joint, dtype=np.int32),
            dtype=int, device=self.device,
        )
        self.base_quats_wp = wp.array(
            np.array(self.base_quats, dtype=np.float64),
            dtype=float, device=self.device,
        )

    def _alloc_buffers(self):
        """Pre-allocate GPU buffers (only 2 states for ping-pong)."""
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # DOF structure
        joint_qd_start = self.model.joint_qd_start.numpy()
        articulation_start = self.model.articulation_start.numpy()

        body_start = articulation_start[0]
        body_end = articulation_start[1]
        self.qd_dof_start = int(joint_qd_start[body_start + 1])
        qd_dof_end = int(joint_qd_start[body_end])
        self.num_act_dofs = qd_dof_end - self.qd_dof_start
        self.total_qd_per_world = self.joint_qd_size
        # q-space offset: free joint has 7 q-DOFs (pos+quat) vs 6 qd-DOFs,
        # so all hinge joints after are shifted by +1 in q-space.
        self.q_dof_start = self.qd_dof_start + 1
        self.total_q_per_world = self.joint_q_size

        print(f"    [FDEval] DOF structure: qd_dof_start={self.qd_dof_start}, "
              f"q_dof_start={self.q_dof_start}, "
              f"num_act_dofs={self.num_act_dofs}, total_qd/world={self.total_qd_per_world}")

        # Pre-collected actions buffer
        self.pre_actions = [
            wp.zeros((self.num_worlds, self.num_act_dofs), dtype=float, device=self.device)
            for _ in range(self.horizon)
        ]

        # Total mass for CoT denominator
        body_mass_np = self.model.body_mass.numpy()
        self.total_mass = float(body_mass_np[:self.bodies_per_world].sum())
        print(f"    [FDEval] Total robot mass = {self.total_mass:.2f} kg")

        # Build init qpos from env config
        with open(str(BASE_ENV_CONFIG), "r") as f:
            env_cfg = yaml.safe_load(f)
        init_pose_list = env_cfg.get("init_pose", [])

        init_state = self.model.state()
        newton.eval_fk(self.model, init_state.joint_q, init_state.joint_qd, init_state)
        wp.synchronize()
        self.init_qpos = init_state.joint_q.numpy().copy()

        # Override root pose: z-height and identity quaternion
        z_height = float(init_pose_list[2]) if len(init_pose_list) > 2 else 0.8
        for w in range(self.num_worlds):
            q_start = w * self.joint_q_size
            self.init_qpos[q_start + 2] = z_height
            self.init_qpos[q_start + 3] = 0.0  # qx
            self.init_qpos[q_start + 4] = 0.0  # qy
            self.init_qpos[q_start + 5] = 0.0  # qz
            self.init_qpos[q_start + 6] = 1.0  # qw

        self.init_qpos_wp = wp.array(self.init_qpos, dtype=float, device=self.device)

        del init_state

    def _apply_theta(self, theta_np):
        """Apply theta → joint_X_p in-place (numpy round-trip, no tape)."""
        joint_X_p_np = self.model.joint_X_p.numpy()

        for j in range(self.num_param_joints):
            p_idx = self.param_for_joint[j]
            angle = float(theta_np[p_idx])

            # Delta quat from X-rotation (w, x, y, z)
            half = angle * 0.5
            dw = np.cos(half)
            dx = np.sin(half)

            # Base quat (w, x, y, z)
            bw, bx, by, bz = self.base_quats[j]

            # Hamilton product: delta * base
            nw = dw * bw - dx * bx
            nx = dw * bx + dx * bw
            ny = dw * by - dx * bz
            nz = dw * bz + dx * by

            # Normalize
            norm = np.sqrt(nw*nw + nx*nx + ny*ny + nz*nz)
            inv_norm = 1.0 / max(norm, 1e-12)
            nw *= inv_norm
            nx *= inv_norm
            ny *= inv_norm
            nz *= inv_norm

            local_idx = self.param_joint_local_indices[j]
            for w in range(self.num_worlds):
                global_idx = w * self.joints_per_world + local_idx
                # Write xyzw to joint_X_p numpy layout
                joint_X_p_np[global_idx, 3] = nx
                joint_X_p_np[global_idx, 4] = ny
                joint_X_p_np[global_idx, 5] = nz
                joint_X_p_np[global_idx, 6] = nw

        self.model.joint_X_p.assign(joint_X_p_np)
        wp.synchronize()

    def _run_rollout(self, theta_np, actions_list):
        """Run a forward-only rollout and return (mean_fwd_dist, cot).

        Args:
            theta_np: (6,) design parameters
            actions_list: list of (num_worlds, act_dim) numpy arrays

        Returns:
            (mean_forward_dist, cot_value)
        """
        # Apply theta → joint_X_p
        self._apply_theta(theta_np)

        # Reset state: init qpos, zero qd
        total_q = self.model.joint_q.numpy().shape[0]
        total_qd = self.model.joint_qd.numpy().shape[0]
        launch_dim = max(total_q, total_qd)
        wp.launch(diff_init_worlds_kernel, dim=launch_dim,
                  inputs=[self.state_0.joint_q, self.state_0.joint_qd,
                          self.init_qpos_wp,
                          total_q, total_qd])

        # FK to populate body transforms from joint state
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        wp.synchronize()

        # Store initial root x positions
        jq_init = self.state_0.joint_q.numpy().copy()

        # Load actions into warp buffers
        for step in range(min(self.horizon, len(actions_list))):
            actions_np = actions_list[step].astype(np.float64)
            act_dim = actions_np.shape[1] if actions_np.ndim > 1 else actions_np.shape[0]

            if act_dim != self.num_act_dofs:
                adapted = np.zeros((self.num_worlds, self.num_act_dofs), dtype=np.float64)
                copy_dim = min(act_dim, self.num_act_dofs)
                adapted[:actions_np.shape[0], :copy_dim] = actions_np[:self.num_worlds, :copy_dim]
                actions_np = adapted
            else:
                if actions_np.shape[0] < self.num_worlds:
                    padded = np.zeros((self.num_worlds, self.num_act_dofs), dtype=np.float64)
                    padded[:actions_np.shape[0]] = actions_np
                    actions_np = padded
                elif actions_np.shape[0] > self.num_worlds:
                    actions_np = actions_np[:self.num_worlds]

            self.pre_actions[step].assign(actions_np)

        # Energy buffer on GPU (reuse existing kernel for speed)
        energy_buf = wp.zeros(1, dtype=float, device=self.device)

        state_0, state_1 = self.state_0, self.state_1

        for step in range(self.horizon):
            # Set joint targets
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

            # Collide once per macro-step (matching training pattern)
            state_0.clear_forces()
            contacts = self.model.collide(state_0)

            # Substeps at 240 Hz (ping-pong states)
            for sub in range(self.num_substeps):
                state_0.clear_forces()
                self.solver.step(state_0, state_1, self.control, contacts, self.sim_dt)
                state_0, state_1 = state_1, state_0

                # Accumulate mechanical power: Σ|τ·qd| (sensitive to morphology)
                wp.launch(
                    accumulate_mechanical_power_kernel,
                    dim=1,
                    inputs=[
                        state_0.joint_q,
                        state_0.joint_qd,
                        self.control.joint_target_pos,
                        self.model.joint_target_ke,
                        self.model.joint_target_kd,
                        energy_buf,
                        self.qd_dof_start,
                        self.q_dof_start,
                        self.num_act_dofs,
                        self.total_qd_per_world,
                        self.total_q_per_world,
                        self.num_worlds,
                        self.sim_dt,
                    ],
                )

        # Read results from GPU
        wp.synchronize()

        mean_energy = float(energy_buf.numpy()[0])

        # Forward distance
        jq_final = state_0.joint_q.numpy()
        total_dist = 0.0
        for w in range(self.num_worlds):
            idx = w * self.joint_q_size
            total_dist += jq_final[idx] - jq_init[idx]
        mean_dist = total_dist / self.num_worlds

        # CoT = energy / (m * g * d)
        safe_dist = max(mean_dist, 0.1)
        cot = mean_energy / (self.total_mass * 9.81 * safe_dist)

        # Restore state references (ping-pong may have swapped them)
        self.state_0, self.state_1 = state_0, state_1

        return float(mean_dist), float(cot)

    def compute_fd_gradient(self, theta_np, actions_list, eps=0.01):
        """Compute gradient via central finite differences.

        Returns:
            (grad, mean_fwd_dist, cot) where grad points in the ascent
            direction for reward (= descent direction for CoT).
        """
        # Center evaluation for baseline metrics
        fwd_dist_center, cot_center = self._run_rollout(theta_np, actions_list)
        print(f"    [FD] Center: fwd_dist={fwd_dist_center:.4f} m, CoT={cot_center:.4f}")

        if np.isnan(cot_center) or np.isinf(cot_center):
            print(f"    [FD] Center CoT is NaN/Inf — returning zero gradient")
            return np.zeros(NUM_DESIGN_PARAMS), fwd_dist_center, cot_center

        grad = np.zeros(NUM_DESIGN_PARAMS)
        for i in range(NUM_DESIGN_PARAMS):
            theta_plus = theta_np.copy()
            theta_plus[i] += eps
            _, cot_plus = self._run_rollout(theta_plus, actions_list)

            theta_minus = theta_np.copy()
            theta_minus[i] -= eps
            _, cot_minus = self._run_rollout(theta_minus, actions_list)

            # Negative because we want to MINIMIZE CoT (= maximize reward)
            if np.isnan(cot_plus) or np.isnan(cot_minus):
                print(f"    [FD] param {i}: NaN in perturbation, setting grad=0")
                grad[i] = 0.0
            else:
                grad[i] = -(cot_plus - cot_minus) / (2 * eps)

            print(f"    [FD] param {i}: CoT+={cot_plus:.6f}, CoT-={cot_minus:.6f}, "
                  f"grad={grad[i]:+.6f}")

        # Clip gradients
        grad = np.clip(grad, -10.0, 10.0)
        print(f"    [FD] Gradient (clipped): {grad.tolist()}")

        return grad.astype(np.float64), fwd_dist_center, cot_center

    def cleanup(self):
        """Free GPU resources."""
        for attr in ('state_0', 'state_1', 'model', 'solver', 'control',
                     'pre_actions', 'base_quats_wp',
                     'joint_local_indices_wp', 'param_for_joint_wp',
                     'init_qpos_wp'):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass
        gc.collect()
        wp.synchronize()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="G1 Eval Worker: collect actions + BPTT gradient (GPU subprocess)"
    )
    parser.add_argument("--theta-file", type=str, required=True,
                        help="Path to theta.npy (design parameters)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained MimicKit checkpoint (model.pt)")
    parser.add_argument("--env-config", type=str, required=True,
                        help="Path to modified env config YAML")
    parser.add_argument("--engine-config", type=str, required=True,
                        help="Path to Newton engine config YAML")
    parser.add_argument("--agent-config", type=str, required=True,
                        help="Path to MimicKit agent config YAML")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Path to write eval_result.npz")
    parser.add_argument("--num-eval-worlds", type=int, default=32)
    parser.add_argument("--eval-horizon", type=int, default=100)
    args = parser.parse_args()

    theta = np.load(args.theta_file)
    print(f"  [EvalWorker] theta = {theta}")
    print(f"  [EvalWorker] checkpoint = {args.checkpoint}")
    print(f"  [EvalWorker] num_eval_worlds = {args.num_eval_worlds}, "
          f"eval_horizon = {args.eval_horizon}")

    # ----- Phase 1: Collect actions via MimicKit -----
    print(f"\n  [Phase 1] Collecting actions ({args.num_eval_worlds} worlds, "
          f"{args.eval_horizon} steps)...")

    actions_list = collect_actions_mimickit(
        mimickit_src_dir=str(MIMICKIT_SRC_DIR),
        env_config_path=args.env_config,
        engine_config_path=args.engine_config,
        agent_config_path=args.agent_config,
        checkpoint_path=args.checkpoint,
        num_worlds=args.num_eval_worlds,
        horizon=args.eval_horizon,
    )
    print(f"    Collected {len(actions_list)} steps x {actions_list[0].shape} actions")

    # ----- Phase 2+3: BPTT gradient -----
    print(f"\n  [Phase 2] Computing BPTT gradient...")

    mjcf_modifier = G1MJCFModifier(str(BASE_MJCF_PATH))
    diff_eval = DiffG1Eval(
        mjcf_modifier=mjcf_modifier,
        theta_np=theta,
        num_worlds=args.num_eval_worlds,
        horizon=args.eval_horizon,
        dt=1.0/30.0,
        num_substeps=16,
    )

    grad_theta, fwd_dist, cot_val = diff_eval.compute_gradient(actions_list)

    print(f"    BPTT gradient: {grad_theta}")
    print(f"    Forward distance: {fwd_dist:.4f} m")
    print(f"    Cost of Transport: {cot_val:.4f}")

    # ----- Save results -----
    np.savez(
        args.output_file,
        grad_theta=grad_theta,
        fwd_dist=np.array(fwd_dist),
        cot=np.array(cot_val),
    )
    print(f"  [EvalWorker] Results saved to {args.output_file}")

    # ----- Cleanup -----
    diff_eval.cleanup()
    del diff_eval, actions_list
    gc.collect()


if __name__ == "__main__":
    main()
