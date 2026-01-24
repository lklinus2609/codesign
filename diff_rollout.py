"""
Differentiable Rollout for Outer Loop Gradient Computation

This module provides differentiable simulation rollouts through Newton
for computing gradients of the design parameters.

Reference: Algorithm 1, Lines 20-23
    Line 20: Sample initial state s0
    Line 21: tau_val <- DifferentiableRollout(pi_theta*, phi_k, H)
    Line 22: L(phi_k) <- -sum_{t=0}^{H} r(s_t, a_t)
    Line 23: grad_phi L <- BPTT(L(phi_k))
"""

import numpy as np
import torch
import warp as wp

import newton


@wp.kernel
def compute_rollout_reward_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    root_body_idx: int,
    target_velocity: wp.vec3,
    upright_weight: float,
    velocity_weight: float,
    reward: wp.array(dtype=float),
):
    """
    Compute reward for a single timestep.

    Rewards:
    1. Upright bonus: penalize deviation from upright orientation
    2. Forward velocity: reward moving at target velocity

    Args:
        body_q: Body transforms [N] (wp.transform = pos + quat)
        body_qd: Body spatial velocities [N] (wp.spatial_vector = linear + angular)
        root_body_idx: Index of the root body (pelvis)
        target_velocity: Target forward velocity
        upright_weight: Weight for upright reward
        velocity_weight: Weight for velocity reward
        reward: Output reward array [1]
    """
    tid = wp.tid()
    if tid == 0:
        # Get root body state
        root_tf = body_q[root_body_idx]
        root_vel = body_qd[root_body_idx]

        # Extract position and orientation from transform
        root_pos = wp.transform_get_translation(root_tf)
        root_quat = wp.transform_get_rotation(root_tf)

        # Extract linear velocity from spatial vector (first 3 components)
        root_linear_vel = wp.spatial_top(root_vel)

        # Upright reward: z-component of up vector in world frame
        # For upright humanoid, local up (0,0,1) should align with world up
        local_up = wp.vec3(0.0, 0.0, 1.0)
        world_up = wp.quat_rotate(root_quat, local_up)
        upright_reward = world_up[2]  # Should be close to 1 when upright

        # Velocity reward: match target velocity
        vel_diff = root_linear_vel - target_velocity
        vel_error = wp.length(vel_diff)
        velocity_reward = wp.exp(-vel_error)

        # Combined reward
        total_reward = upright_weight * upright_reward + velocity_weight * velocity_reward

        reward[0] = total_reward


class DifferentiableRollout:
    """
    Performs differentiable simulation rollouts for outer loop optimization.

    This class wraps Newton's differentiable simulation capabilities to
    compute gradients of the cumulative reward with respect to design
    parameters (joint transforms).

    The rollout uses a frozen policy (no gradients through the policy network)
    and propagates gradients only through the physics simulation.
    """

    def __init__(
        self,
        model,
        horizon: int = 50,
        device: str = "cuda:0",
        use_semi_implicit: bool = True,
    ):
        """
        Initialize the differentiable rollout.

        Args:
            model: Newton Model object (must be finalized with requires_grad=True)
            horizon: Number of timesteps for the rollout (H in Algorithm 1)
            device: Computation device
            use_semi_implicit: If True, use SolverSemiImplicit for BPTT compatibility
        """
        self.model = model
        self.horizon = horizon
        self.device = device

        # Verify model supports gradients
        if not model.requires_grad:
            raise ValueError(
                "Model must be finalized with requires_grad=True for differentiable rollout"
            )

        # Create solver for differentiable simulation
        # SolverSemiImplicit is recommended for BPTT
        if use_semi_implicit:
            self.solver = newton.solvers.SolverSemiImplicit(model)
        else:
            # MuJoCo solver (may have limited BPTT support)
            self.solver = newton.solvers.SolverMuJoCo(model)

        # Simulation parameters
        self.dt = 1.0 / 60.0  # 60 Hz simulation

        # Reward weights
        self.upright_weight = 1.0
        self.velocity_weight = 0.5
        self.target_velocity = wp.vec3(0.5, 0.0, 0.0)  # Forward walking

        # Allocate state buffers
        self._init_state_buffers()

        # Reward accumulator (differentiable)
        self.reward_buffer = wp.zeros(1, dtype=float, requires_grad=True)

    def _init_state_buffers(self):
        """Initialize state buffers for rollout."""
        # We need H+1 states for H timesteps
        self.states = []
        for _ in range(self.horizon + 1):
            state = self.model.state(requires_grad=True)
            self.states.append(state)

        # Control buffer
        self.control = self.model.control()

        # Contacts buffer (recomputed each step)
        self.contacts = None

    def reset_to_state(self, initial_state):
        """
        Reset the rollout to a given initial state.

        Args:
            initial_state: Newton State object to copy from
        """
        # Copy initial state to first state buffer
        wp.copy(self.states[0].joint_q, initial_state.joint_q)
        wp.copy(self.states[0].joint_qd, initial_state.joint_qd)
        wp.copy(self.states[0].body_q, initial_state.body_q)
        wp.copy(self.states[0].body_qd, initial_state.body_qd)

    def set_actions_from_policy(self, policy_fn, obs_norm_fn):
        """
        Compute actions for all timesteps using the frozen policy.

        Since the policy is frozen, we precompute all actions before
        the differentiable forward pass.

        Args:
            policy_fn: Function that takes normalized obs and returns actions
            obs_norm_fn: Function to normalize observations

        Returns:
            List of action tensors for each timestep
        """
        actions = []

        # Get initial observation
        state = self.states[0]
        obs = self._state_to_obs(state)

        with torch.no_grad():
            for t in range(self.horizon):
                # Normalize observation
                norm_obs = obs_norm_fn(obs)

                # Get action from policy (deterministic/mode)
                action_dist = policy_fn(norm_obs)
                action = action_dist.mode

                actions.append(action)

                # Note: For precomputation, we can't get next obs without simulation
                # So we'll recompute obs during the actual rollout
                # This is just for initialization

        return actions

    def _state_to_obs(self, state) -> torch.Tensor:
        """
        Convert Newton state to observation tensor.

        This should match the observation space used by the policy.

        Args:
            state: Newton State object

        Returns:
            Observation tensor
        """
        # Extract relevant state information
        joint_q = wp.to_torch(state.joint_q)
        joint_qd = wp.to_torch(state.joint_qd)
        body_q = wp.to_torch(state.body_q)
        body_qd = wp.to_torch(state.body_qd)

        # Flatten and concatenate
        # Note: This is a simplified observation - actual implementation
        # should match the environment's observation space
        obs = torch.cat([
            joint_q.flatten(),
            joint_qd.flatten(),
        ])

        return obs

    def _action_to_control(self, action: torch.Tensor):
        """
        Convert action tensor to Newton control.

        Args:
            action: Action tensor from policy
        """
        # Convert torch tensor to numpy, then to warp
        action_np = action.detach().cpu().numpy()

        # Set joint target positions (assuming position control)
        target_pos = self.control.joint_target_pos.numpy()
        target_pos[:len(action_np)] = action_np
        self.control.joint_target_pos.assign(target_pos)

    def forward(self, policy_fn=None, obs_norm_fn=None) -> wp.array:
        """
        Execute differentiable forward rollout.

        Implements Algorithm 1, Lines 21-22:
            tau_val <- DifferentiableRollout(pi_theta*, phi_k, H)
            L(phi_k) <- -sum_{t=0}^{H} r(s_t, a_t)

        Args:
            policy_fn: Optional policy function (if None, uses zero actions)
            obs_norm_fn: Optional observation normalizer

        Returns:
            loss: Negative cumulative reward (for minimization)
        """
        # Reset reward accumulator
        self.reward_buffer.zero_()

        # Forward simulation loop
        for t in range(self.horizon):
            state_t = self.states[t]
            state_tp1 = self.states[t + 1]

            # Clear forces
            state_t.clear_forces()

            # Get action from policy (with no_grad since policy is frozen)
            if policy_fn is not None:
                with torch.no_grad():
                    obs = self._state_to_obs(state_t)
                    norm_obs = obs_norm_fn(obs) if obs_norm_fn else obs
                    action_dist = policy_fn(norm_obs)
                    action = action_dist.mode
                    self._action_to_control(action)

            # Compute contacts
            self.contacts = self.model.collide(state_t)

            # Step physics (differentiable)
            self.solver.step(
                state_t,
                state_tp1,
                self.control,
                self.contacts,
                self.dt
            )

            # Compute reward for this timestep
            self._compute_step_reward(state_tp1)

        # Loss is negative reward (for minimization)
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        wp.launch(
            negate_kernel,
            dim=1,
            inputs=[self.reward_buffer],
            outputs=[loss]
        )

        return loss

    def _compute_step_reward(self, state):
        """
        Compute and accumulate reward for a single timestep.

        Args:
            state: Current state after physics step
        """
        # Find root body index (assuming it's the first body)
        root_body_idx = 0

        step_reward = wp.zeros(1, dtype=float, requires_grad=True)

        wp.launch(
            compute_rollout_reward_kernel,
            dim=1,
            inputs=[
                state.body_q,
                state.body_qd,
                root_body_idx,
                self.target_velocity,
                self.upright_weight,
                self.velocity_weight,
            ],
            outputs=[step_reward]
        )

        # Accumulate reward
        wp.launch(
            add_kernel,
            dim=1,
            inputs=[step_reward],
            outputs=[self.reward_buffer]
        )

    def compute_design_gradient(self, policy_fn=None, obs_norm_fn=None):
        """
        Compute gradient of loss with respect to design parameters.

        Implements Algorithm 1, Line 23:
            grad_phi L <- BPTT(L(phi_k))

        Args:
            policy_fn: Policy function (frozen)
            obs_norm_fn: Observation normalizer

        Returns:
            Dictionary with loss and gradient information
        """
        # Create tape for recording operations
        tape = wp.Tape()

        with tape:
            # Forward pass (differentiable)
            loss = self.forward(policy_fn, obs_norm_fn)

        # Backward pass (BPTT through physics)
        tape.backward(loss)

        # Extract loss value
        loss_val = loss.numpy()[0]

        # Gradient is now available in model.joint_X_p.grad
        joint_X_p_grad = None
        if self.model.joint_X_p.grad is not None:
            joint_X_p_grad = self.model.joint_X_p.grad.numpy().copy()

        # Zero gradients for next call
        tape.zero()

        return {
            "loss": loss_val,
            "joint_X_p_grad": joint_X_p_grad,
            "cumulative_reward": -loss_val,
        }


@wp.kernel
def negate_kernel(
    input: wp.array(dtype=float),
    output: wp.array(dtype=float),
):
    """Negate input value."""
    tid = wp.tid()
    output[tid] = -input[tid]


@wp.kernel
def add_kernel(
    input: wp.array(dtype=float),
    output: wp.array(dtype=float),
):
    """Add input to output (accumulate)."""
    tid = wp.tid()
    output[tid] = output[tid] + input[tid]


class SimplifiedDiffRollout:
    """
    Simplified differentiable rollout that doesn't require a policy.

    Uses the current joint positions as targets, essentially simulating
    the robot trying to maintain its current pose. This is useful for
    testing the gradient flow without involving the full policy.
    """

    def __init__(self, model, horizon: int = 3):
        """
        Initialize simplified rollout.

        Args:
            model: Newton Model (must have requires_grad=True)
            horizon: Rollout horizon (default=3 for stability before contact)
        """
        self.model = model
        self.horizon = horizon
        # Use smaller timestep for stability
        self.dt = 1.0 / 120.0  # 120 Hz instead of 60 Hz

        # Use SolverSemiImplicit for reliable gradients
        self.solver = newton.solvers.SolverSemiImplicit(model)

        # State buffers
        self.state_0 = model.state(requires_grad=True)
        self.state_1 = model.state(requires_grad=True)
        self.control = model.control()

        # G1 robot standing height (pelvis z-coordinate)
        # This should be high enough to clear the ground
        self.standing_height = 0.75

    def _initialize_standing_pose(self, debug=False):
        """
        Initialize the robot in a stable standing pose.

        This sets the root body position to standing height and
        initializes all velocities to zero.
        """
        # Get joint_q as numpy for modification
        joint_q = self.model.joint_q.numpy().copy()

        # For floating base robot, joint_q starts with:
        # [root_x, root_y, root_z, root_quat_w, root_quat_x, root_quat_y, root_quat_z, ...]
        # Note: Newton/MuJoCo uses (w, x, y, z) quaternion order
        # Set root position to standing height
        joint_q[0] = 0.0  # x
        joint_q[1] = 0.0  # y
        joint_q[2] = self.standing_height  # z - elevated

        # Set root orientation to upright (identity quaternion)
        # MuJoCo/Newton quaternion order: (w, x, y, z)
        joint_q[3] = 1.0  # qw
        joint_q[4] = 0.0  # qx
        joint_q[5] = 0.0  # qy
        joint_q[6] = 0.0  # qz

        if debug:
            print(f"[DEBUG] Initial joint_q[0:7]: {joint_q[0:7]}")

        # Copy to state
        self.state_0.joint_q.assign(joint_q)

        # Zero all velocities
        joint_qd = self.model.joint_qd.numpy().copy()
        joint_qd[:] = 0.0
        self.state_0.joint_qd.assign(joint_qd)

    def forward_and_backward(self, initial_state=None, debug=False):
        """
        Run forward simulation and compute gradients.

        Args:
            initial_state: Optional initial state to copy from
            debug: If True, print debug information

        Returns:
            Dictionary with loss and gradients
        """
        # Initialize state
        if initial_state is not None:
            wp.copy(self.state_0.joint_q, initial_state.joint_q)
            wp.copy(self.state_0.joint_qd, initial_state.joint_qd)
        else:
            # Initialize to stable standing pose
            self._initialize_standing_pose(debug=debug)

        # Evaluate forward kinematics
        newton.eval_fk(
            self.model,
            self.state_0.joint_q,
            self.state_0.joint_qd,
            self.state_0
        )

        if debug:
            # Check initial body position
            body_q = self.state_0.body_q.numpy()
            root_pos = body_q[0, :3]  # First body position
            print(f"[DEBUG] Initial root body position: {root_pos}")

        tape = wp.Tape()
        loss = wp.zeros(1, dtype=float, requires_grad=True)

        with tape:
            state_a = self.state_0
            state_b = self.state_1

            for t in range(self.horizon):
                state_a.clear_forces()

                # Compute contacts
                contacts = self.model.collide(state_a)

                # Step simulation
                self.solver.step(state_a, state_b, self.control, contacts, self.dt)

                if debug:
                    # Check for NaN after each step
                    body_q = state_b.body_q.numpy()
                    root_pos = body_q[0, :3]
                    if np.any(np.isnan(root_pos)):
                        print(f"[DEBUG] NaN detected at step {t}, root_pos: {root_pos}")
                        break
                    elif t < 3 or t == self.horizon - 1:
                        print(f"[DEBUG] Step {t}: root_pos = {root_pos}")

                # Swap states
                state_a, state_b = state_b, state_a

            # Compute final loss (height of root body)
            # Negative height because we minimize loss
            final_state = state_a
            wp.launch(
                compute_height_loss_kernel,
                dim=1,
                inputs=[final_state.body_q],
                outputs=[loss]
            )

        # Get loss value before backward (for debugging)
        loss_val = loss.numpy()[0]

        # Check for NaN in loss
        if np.isnan(loss_val):
            print("[SimplifiedDiffRollout] Warning: Loss is NaN, skipping backward pass")
            tape.zero()
            return {
                "loss": 0.0,
                "joint_X_p_grad": None,
            }

        # Backward pass
        tape.backward(loss)

        # Get gradients
        joint_X_p_grad = None
        if self.model.joint_X_p.grad is not None:
            joint_X_p_grad = self.model.joint_X_p.grad.numpy().copy()

            # Check for NaN in gradients and clip
            if np.any(np.isnan(joint_X_p_grad)):
                print("[SimplifiedDiffRollout] Warning: NaN in gradients, zeroing")
                joint_X_p_grad = np.zeros_like(joint_X_p_grad)
            else:
                # Clip gradients to prevent explosion
                grad_clip = 10.0
                joint_X_p_grad = np.clip(joint_X_p_grad, -grad_clip, grad_clip)

        tape.zero()

        return {
            "loss": loss_val,
            "joint_X_p_grad": joint_X_p_grad,
        }


@wp.kernel
def compute_height_loss_kernel(
    body_q: wp.array(dtype=wp.transform),
    loss: wp.array(dtype=float),
):
    """
    Compute loss based on root body height.

    Loss = -height (we want to maximize height, i.e., stay upright)
    """
    tid = wp.tid()
    if tid == 0:
        # Root body is at index 0
        root_tf = body_q[0]
        root_pos = wp.transform_get_translation(root_tf)
        height = root_pos[2]  # z-component
        loss[0] = -height  # Negative because we minimize


if __name__ == "__main__":
    print("DifferentiableRollout module loaded successfully")
    print("This module requires a Newton model to test.")
