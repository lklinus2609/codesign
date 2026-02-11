#!/usr/bin/env python3
"""
G1 Eval Worker — GPU subprocess for PGHC co-design gradient computation.

Runs Phase 1 (action collection via MimicKit) and Phase 2 (BPTT gradient via
Warp/SolverSemiImplicit) in an isolated process so the parent orchestrator
never touches the GPU.

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
    base_quats stores the 4 quaternion components extracted from
    joint_X_p.numpy()[:, 3:7] in the same memory layout order.
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

    # Base quat components (same order as joint_X_p numpy layout)
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
    # quatf(x, y, z, w) maps to byte positions [3,4,5,6] — same layout as
    # the old direct writes to joint_X_p[idx, 3..6]
    joint_X_p[global_idx] = wp.transformf(pos, wp.quatf(nw, nx, ny, nz))


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

        # loss = -mean(fwd_dist), reward = -loss
        # ∂reward/∂theta = -∂loss/∂theta = -grad_np
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
        num_substeps=8,
    )

    grad_theta, fwd_dist = diff_eval.compute_gradient(actions_list)

    print(f"    BPTT gradient: {grad_theta}")
    print(f"    Forward distance: {fwd_dist:.4f} m")

    # ----- Save results -----
    np.savez(
        args.output_file,
        grad_theta=grad_theta,
        fwd_dist=np.array(fwd_dist),
    )
    print(f"  [EvalWorker] Results saved to {args.output_file}")

    # ----- Cleanup -----
    diff_eval.cleanup()
    del diff_eval, actions_list
    gc.collect()


if __name__ == "__main__":
    main()
