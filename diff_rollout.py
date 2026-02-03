"""
Differentiable Rollout for Outer Loop Gradient Computation (PGHC)

This module provides differentiable simulation rollouts through Newton
for computing gradients of the design parameters.

Implements Algorithm 1, Lines 20-23:
    Line 20: Sample initial state s0
    Line 21: tau_val <- DifferentiableRollout(pi_theta*, phi_k, H)
    Line 22: L(phi_k) <- Objective(trajectory)
    Line 23: grad_phi L <- BPTT(L(phi_k))

Objectives (Thesis Section III.C):
    - Cost of Transport (CoT): E_total / (m * g * d)
    - Height maintenance (simplified)
    - Velocity tracking

Reference: Masters_Thesis.pdf Section III.C, Equation 9
"""

import numpy as np
import torch
import warp as wp

import newton


# =============================================================================
# Warp Kernels for Differentiable Objectives
# =============================================================================

@wp.kernel
def compute_cot_step_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    joint_qd: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    root_body_idx: int,
    dt: float,
    energy_accumulator: wp.array(dtype=float),
    distance_accumulator: wp.array(dtype=float),
):
    """
    Compute energy and distance for Cost of Transport calculation.

    CoT = E_total / (m * g * d)

    For each timestep:
    - Energy: sum of |torque * angular_velocity| for all joints
    - Distance: displacement of root body in forward direction
    """
    tid = wp.tid()
    if tid == 0:
        # Energy: approximate as sum of |joint_stiffness * position_error * velocity|
        # Simplified: use joint velocities as proxy for energy consumption
        energy = float(0.0)
        for i in range(joint_qd.shape[0]):
            # Approximate mechanical power = stiffness * velocity^2
            ke = joint_target_ke[i] if i < joint_target_ke.shape[0] else 1000.0
            energy = energy + ke * joint_qd[i] * joint_qd[i] * dt

        # Distance: x-component of root body velocity * dt
        root_vel = body_qd[root_body_idx]
        root_linear_vel = wp.spatial_top(root_vel)
        distance = root_linear_vel[0] * dt  # Forward (x) direction

        # Accumulate (atomic add)
        wp.atomic_add(energy_accumulator, 0, energy)
        wp.atomic_add(distance_accumulator, 0, distance)


@wp.kernel
def compute_cot_loss_kernel(
    energy_total: wp.array(dtype=float),
    distance_total: wp.array(dtype=float),
    robot_mass: float,
    gravity: float,
    loss: wp.array(dtype=float),
):
    """
    Compute Cost of Transport loss.

    CoT = E_total / (m * g * d)

    We minimize CoT (lower = more efficient).
    """
    tid = wp.tid()
    if tid == 0:
        energy = energy_total[0]
        distance = distance_total[0]

        # Prevent division by zero
        if distance < 0.01:
            distance = 0.01

        cot = energy / (robot_mass * gravity * distance)
        loss[0] = cot


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


@wp.kernel
def compute_velocity_loss_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    root_body_idx: int,
    target_velocity: wp.vec3,
    loss: wp.array(dtype=float),
):
    """
    Compute loss for velocity tracking.

    Loss = ||v_current - v_target||^2
    """
    tid = wp.tid()
    if tid == 0:
        root_vel = body_qd[root_body_idx]
        root_linear_vel = wp.spatial_top(root_vel)

        vel_diff = root_linear_vel - target_velocity
        loss[0] = wp.dot(vel_diff, vel_diff)


@wp.kernel
def compute_upright_loss_kernel(
    body_q: wp.array(dtype=wp.transform),
    root_body_idx: int,
    loss: wp.array(dtype=float),
):
    """
    Compute loss for staying upright.

    Loss = 1 - (z-component of up vector in world frame)
    """
    tid = wp.tid()
    if tid == 0:
        root_tf = body_q[root_body_idx]
        root_quat = wp.transform_get_rotation(root_tf)

        # Local up vector
        local_up = wp.vec3(0.0, 0.0, 1.0)
        world_up = wp.quat_rotate(root_quat, local_up)

        # Loss is 1 - z_component (0 when perfectly upright)
        loss[0] = 1.0 - world_up[2]


@wp.kernel
def add_weighted_kernel(
    input_val: wp.array(dtype=float),
    weight: float,
    output: wp.array(dtype=float),
):
    """Add weighted input to output."""
    tid = wp.tid()
    if tid == 0:
        output[0] = output[0] + weight * input_val[0]


# =============================================================================
# Differentiable Rollout Classes
# =============================================================================

class DifferentiableRollout:
    """
    Performs differentiable simulation rollouts for outer loop optimization.

    This class wraps Newton's differentiable simulation capabilities to
    compute gradients of the cumulative reward/cost with respect to design
    parameters (joint transforms).

    The rollout uses a frozen policy (no gradients through the policy network)
    and propagates gradients only through the physics simulation.

    Supports multiple objective functions per thesis:
    - Cost of Transport (CoT)
    - Velocity tracking
    - Upright bonus
    """

    def __init__(
        self,
        model,
        horizon: int = 50,
        device: str = "cuda:0",
        use_semi_implicit: bool = True,
        objective_type: str = "cost_of_transport",
        robot_mass: float = 40.0,
        target_velocity: tuple = (0.5, 0.0, 0.0),
    ):
        """
        Initialize the differentiable rollout.

        Args:
            model: Newton Model object (must be finalized with requires_grad=True)
            horizon: Number of timesteps for the rollout (H in Algorithm 1)
            device: Computation device
            use_semi_implicit: If True, use SolverSemiImplicit for BPTT compatibility
            objective_type: "cost_of_transport", "height", or "combined"
            robot_mass: Robot mass in kg (for CoT normalization)
            target_velocity: Target forward velocity (m/s)
        """
        self.model = model
        self.horizon = horizon
        self.device = device
        self.objective_type = objective_type
        self.robot_mass = robot_mass
        self.gravity = 9.81
        self.target_velocity = wp.vec3(target_velocity[0], target_velocity[1], target_velocity[2])

        # Verify model supports gradients
        if not model.requires_grad:
            raise ValueError(
                "Model must be finalized with requires_grad=True for differentiable rollout"
            )

        # Create solver for differentiable simulation
        if use_semi_implicit:
            self.solver = newton.solvers.SolverSemiImplicit(model)
        else:
            self.solver = newton.solvers.SolverMuJoCo(model)

        # Simulation parameters
        self.dt = 1.0 / 60.0  # 60 Hz simulation

        # Allocate state buffers
        self._init_state_buffers()

        # Objective accumulators
        self.energy_accumulator = wp.zeros(1, dtype=float, requires_grad=True)
        self.distance_accumulator = wp.zeros(1, dtype=float, requires_grad=True)

        # Objective weights (for combined objective)
        self.cot_weight = 1.0
        self.upright_weight = 0.1
        self.velocity_weight = 0.2

        # Policy reference (set externally)
        self._policy_fn = None
        self._obs_normalizer = None

    def _init_state_buffers(self):
        """Initialize state buffers for rollout."""
        # We need H+1 states for H timesteps
        self.states = []
        for _ in range(self.horizon + 1):
            state = self.model.state(requires_grad=True)
            self.states.append(state)

        # Control buffer
        self.control = self.model.control()

    def set_policy(self, policy_fn, obs_normalizer=None):
        """
        Set the frozen policy for outer loop rollouts.

        Args:
            policy_fn: Policy network forward function
            obs_normalizer: Optional observation normalizer
        """
        self._policy_fn = policy_fn
        self._obs_normalizer = obs_normalizer

    def _state_to_obs(self, state) -> torch.Tensor:
        """
        Convert Newton state to observation tensor.

        This matches the observation space used by typical humanoid policies:
        - Root height, orientation, linear/angular velocity
        - Joint positions and velocities
        - Optionally: key body positions

        Args:
            state: Newton State object

        Returns:
            Observation tensor (flattened)
        """
        # Extract state components
        joint_q = state.joint_q.numpy()
        joint_qd = state.joint_qd.numpy()
        body_q = state.body_q.numpy()
        body_qd = state.body_qd.numpy()

        obs_parts = []

        # Root body position (x, y, z) - only z (height) is typically included
        root_pos = body_q[0, :3]
        obs_parts.append([root_pos[2]])  # Height only

        # Root body orientation (quaternion w, x, y, z)
        root_quat = body_q[0, 3:7]
        obs_parts.append(root_quat)

        # Root body linear velocity
        root_linear_vel = body_qd[0, :3]
        obs_parts.append(root_linear_vel)

        # Root body angular velocity
        root_angular_vel = body_qd[0, 3:6]
        obs_parts.append(root_angular_vel)

        # Joint positions (skip first 7 which are root position/orientation)
        if len(joint_q) > 7:
            obs_parts.append(joint_q[7:])

        # Joint velocities (skip first 6 which are root velocities)
        if len(joint_qd) > 6:
            obs_parts.append(joint_qd[6:])

        # Flatten and convert to torch
        obs_flat = np.concatenate([np.asarray(p).flatten() for p in obs_parts])
        return torch.tensor(obs_flat, dtype=torch.float32, device=self.device)

    def _action_to_control(self, action: torch.Tensor):
        """
        Convert action tensor to Newton control.

        Actions are interpreted as position targets for joint PD control.

        Args:
            action: Action tensor from policy
        """
        # Convert torch tensor to numpy
        action_np = action.detach().cpu().numpy().flatten()

        # Get current joint target positions
        target_pos = self.control.joint_target_pos.numpy()

        # Get default positions from model
        default_pos = self.model.joint_q.numpy()
        if len(default_pos) > 7:
            default_pos = default_pos[7:]  # Skip root DOFs

        # Apply action as offset from default pose
        # Action scaling typically [-1, 1] mapped to joint limits
        action_scale = 0.5  # Radians
        n_action = min(len(action_np), len(target_pos))

        for i in range(n_action):
            target_pos[i] = default_pos[i] if i < len(default_pos) else 0.0
            target_pos[i] += action_np[i] * action_scale

        self.control.joint_target_pos.assign(target_pos)

    def forward(self, policy_fn=None, obs_norm_fn=None) -> dict:
        """
        Execute differentiable forward rollout.

        Implements Algorithm 1, Lines 21-22:
            tau_val <- DifferentiableRollout(pi_theta*, phi_k, H)
            L(phi_k) <- Objective(trajectory)

        Args:
            policy_fn: Policy function (if None, uses zero actions)
            obs_norm_fn: Observation normalizer

        Returns:
            Dictionary with loss and objective values
        """
        # Use provided policy or stored policy
        policy = policy_fn or self._policy_fn
        obs_norm = obs_norm_fn or self._obs_normalizer

        # Reset accumulators
        self.energy_accumulator.zero_()
        self.distance_accumulator.zero_()

        # Forward simulation loop
        for t in range(self.horizon):
            state_t = self.states[t]
            state_tp1 = self.states[t + 1]

            # Clear forces
            state_t.clear_forces()

            # Get action from policy (frozen - no gradients)
            if policy is not None:
                with torch.no_grad():
                    obs = self._state_to_obs(state_t)
                    if obs_norm is not None:
                        obs = obs_norm(obs)
                    action = policy(obs)
                    if hasattr(action, 'mode'):
                        action = action.mode
                    self._action_to_control(action)

            # Compute contacts
            contacts = self.model.collide(state_t)

            # Step physics (differentiable)
            self.solver.step(
                state_t,
                state_tp1,
                self.control,
                contacts,
                self.dt
            )

            # Accumulate objective components
            self._accumulate_objective_step(state_tp1)

        # Compute final loss
        return self._compute_final_objective()

    def _accumulate_objective_step(self, state):
        """
        Accumulate objective components for a single timestep.

        Args:
            state: State after physics step
        """
        if self.objective_type in ["cost_of_transport", "combined"]:
            # Accumulate energy and distance for CoT
            wp.launch(
                compute_cot_step_kernel,
                dim=1,
                inputs=[
                    state.body_q,
                    state.body_qd,
                    state.joint_qd,
                    self.control.joint_target_ke,
                    0,  # root_body_idx
                    self.dt,
                ],
                outputs=[
                    self.energy_accumulator,
                    self.distance_accumulator,
                ]
            )

    def _compute_final_objective(self) -> dict:
        """
        Compute final objective value from accumulated components.

        Returns:
            Dictionary with loss and component values
        """
        loss = wp.zeros(1, dtype=float, requires_grad=True)

        if self.objective_type == "cost_of_transport":
            wp.launch(
                compute_cot_loss_kernel,
                dim=1,
                inputs=[
                    self.energy_accumulator,
                    self.distance_accumulator,
                    self.robot_mass,
                    self.gravity,
                ],
                outputs=[loss]
            )

        elif self.objective_type == "height":
            final_state = self.states[self.horizon]
            wp.launch(
                compute_height_loss_kernel,
                dim=1,
                inputs=[final_state.body_q],
                outputs=[loss]
            )

        elif self.objective_type == "combined":
            # Combine multiple objectives
            final_state = self.states[self.horizon]

            # CoT component
            cot_loss = wp.zeros(1, dtype=float, requires_grad=True)
            wp.launch(
                compute_cot_loss_kernel,
                dim=1,
                inputs=[
                    self.energy_accumulator,
                    self.distance_accumulator,
                    self.robot_mass,
                    self.gravity,
                ],
                outputs=[cot_loss]
            )
            wp.launch(
                add_weighted_kernel,
                dim=1,
                inputs=[cot_loss, self.cot_weight],
                outputs=[loss]
            )

            # Upright component
            upright_loss = wp.zeros(1, dtype=float, requires_grad=True)
            wp.launch(
                compute_upright_loss_kernel,
                dim=1,
                inputs=[final_state.body_q, 0],
                outputs=[upright_loss]
            )
            wp.launch(
                add_weighted_kernel,
                dim=1,
                inputs=[upright_loss, self.upright_weight],
                outputs=[loss]
            )

            # Velocity component
            vel_loss = wp.zeros(1, dtype=float, requires_grad=True)
            wp.launch(
                compute_velocity_loss_kernel,
                dim=1,
                inputs=[
                    final_state.body_q,
                    final_state.body_qd,
                    0,
                    self.target_velocity,
                ],
                outputs=[vel_loss]
            )
            wp.launch(
                add_weighted_kernel,
                dim=1,
                inputs=[vel_loss, self.velocity_weight],
                outputs=[loss]
            )

        loss_val = loss.numpy()[0]
        energy_val = self.energy_accumulator.numpy()[0]
        distance_val = self.distance_accumulator.numpy()[0]

        return {
            "loss": loss_val,
            "energy": energy_val,
            "distance": distance_val,
            "cot": energy_val / (self.robot_mass * self.gravity * max(distance_val, 0.01)),
        }

    def compute_design_gradient(self, policy_fn=None, obs_norm_fn=None) -> dict:
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
            result = self.forward(policy_fn, obs_norm_fn)

        # Create loss tensor for backward
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        loss.assign(np.array([result["loss"]]))

        # Backward pass (BPTT through physics)
        tape.backward(loss)

        # Extract gradient for joint_X_p
        joint_X_p_grad = None
        if self.model.joint_X_p.grad is not None:
            joint_X_p_grad = self.model.joint_X_p.grad.numpy().copy()

        # Zero gradients for next call
        tape.zero()

        return {
            "loss": result["loss"],
            "joint_X_p_grad": joint_X_p_grad,
            "cot": result["cot"],
            "energy": result["energy"],
            "distance": result["distance"],
        }


class SimplifiedDiffRollout:
    """
    Simplified differentiable rollout that doesn't require a policy.

    Uses the current joint positions as targets, essentially simulating
    the robot trying to maintain its current pose. This is useful for
    testing the gradient flow without involving the full policy.

    Supports Cost of Transport objective.
    """

    def __init__(
        self,
        model,
        horizon: int = 3,
        robot_mass: float = 40.0,
        objective_type: str = "height",
    ):
        """
        Initialize simplified rollout.

        Args:
            model: Newton Model (must have requires_grad=True)
            horizon: Rollout horizon (default=3 for stability before contact)
            robot_mass: Robot mass in kg
            objective_type: "height", "cot", or "combined"
        """
        self.model = model
        self.horizon = horizon
        self.robot_mass = robot_mass
        self.gravity = 9.81
        self.objective_type = objective_type

        # Use smaller timestep for stability
        self.dt = 1.0 / 120.0  # 120 Hz instead of 60 Hz

        # Use SolverSemiImplicit for reliable gradients
        self.solver = newton.solvers.SolverSemiImplicit(model)

        # State buffers
        self.state_0 = model.state(requires_grad=True)
        self.state_1 = model.state(requires_grad=True)
        self.control = model.control()

        # G1 robot standing height (pelvis z-coordinate)
        self.standing_height = 0.75

        # Accumulators for CoT
        self.energy_accumulator = wp.zeros(1, dtype=float, requires_grad=True)
        self.distance_accumulator = wp.zeros(1, dtype=float, requires_grad=True)

    def _initialize_standing_pose(self, debug=False):
        """
        Initialize the robot in a stable standing pose.
        """
        joint_q = self.model.joint_q.numpy().copy()

        # Set root position to standing height
        joint_q[0] = 0.0  # x
        joint_q[1] = 0.0  # y
        joint_q[2] = self.standing_height  # z - elevated

        # Set root orientation to upright (identity quaternion)
        joint_q[3] = 1.0  # qw
        joint_q[4] = 0.0  # qx
        joint_q[5] = 0.0  # qy
        joint_q[6] = 0.0  # qz

        if debug:
            print(f"[DEBUG] Initial joint_q[0:7]: {joint_q[0:7]}")

        self.state_0.joint_q.assign(joint_q)

        # Zero all velocities
        joint_qd = self.model.joint_qd.numpy().copy()
        joint_qd[:] = 0.0
        self.state_0.joint_qd.assign(joint_qd)

    def forward_and_backward(self, initial_state=None, debug=False) -> dict:
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
            self._initialize_standing_pose(debug=debug)

        # Evaluate forward kinematics
        newton.eval_fk(
            self.model,
            self.state_0.joint_q,
            self.state_0.joint_qd,
            self.state_0
        )

        if debug:
            body_q = self.state_0.body_q.numpy()
            root_pos = body_q[0, :3]
            print(f"[DEBUG] Initial root body position: {root_pos}")

        # Reset accumulators
        self.energy_accumulator.zero_()
        self.distance_accumulator.zero_()

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

                # Accumulate for CoT if needed
                if self.objective_type in ["cot", "combined"]:
                    wp.launch(
                        compute_cot_step_kernel,
                        dim=1,
                        inputs=[
                            state_b.body_q,
                            state_b.body_qd,
                            state_b.joint_qd,
                            self.control.joint_target_ke,
                            0,
                            self.dt,
                        ],
                        outputs=[
                            self.energy_accumulator,
                            self.distance_accumulator,
                        ]
                    )

                if debug:
                    body_q = state_b.body_q.numpy()
                    root_pos = body_q[0, :3]
                    if np.any(np.isnan(root_pos)):
                        print(f"[DEBUG] NaN detected at step {t}, root_pos: {root_pos}")
                        break
                    elif t < 3 or t == self.horizon - 1:
                        print(f"[DEBUG] Step {t}: root_pos = {root_pos}")

                # Swap states
                state_a, state_b = state_b, state_a

            # Compute final loss
            final_state = state_a

            if self.objective_type == "height":
                wp.launch(
                    compute_height_loss_kernel,
                    dim=1,
                    inputs=[final_state.body_q],
                    outputs=[loss]
                )
            elif self.objective_type == "cot":
                wp.launch(
                    compute_cot_loss_kernel,
                    dim=1,
                    inputs=[
                        self.energy_accumulator,
                        self.distance_accumulator,
                        self.robot_mass,
                        self.gravity,
                    ],
                    outputs=[loss]
                )
            elif self.objective_type == "combined":
                # Height component
                height_loss = wp.zeros(1, dtype=float, requires_grad=True)
                wp.launch(
                    compute_height_loss_kernel,
                    dim=1,
                    inputs=[final_state.body_q],
                    outputs=[height_loss]
                )
                wp.launch(
                    add_weighted_kernel,
                    dim=1,
                    inputs=[height_loss, 1.0],
                    outputs=[loss]
                )

                # CoT component (weighted less)
                cot_loss = wp.zeros(1, dtype=float, requires_grad=True)
                wp.launch(
                    compute_cot_loss_kernel,
                    dim=1,
                    inputs=[
                        self.energy_accumulator,
                        self.distance_accumulator,
                        self.robot_mass,
                        self.gravity,
                    ],
                    outputs=[cot_loss]
                )
                wp.launch(
                    add_weighted_kernel,
                    dim=1,
                    inputs=[cot_loss, 0.1],
                    outputs=[loss]
                )

        # Get loss value before backward
        loss_val = loss.numpy()[0]

        # Check for NaN in loss
        if np.isnan(loss_val):
            print("[SimplifiedDiffRollout] Warning: Loss is NaN, skipping backward pass")
            tape.zero()
            return {
                "loss": 0.0,
                "joint_X_p_grad": None,
                "cot": 0.0,
                "energy": 0.0,
                "distance": 0.0,
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

        # Compute CoT value
        energy_val = self.energy_accumulator.numpy()[0]
        distance_val = self.distance_accumulator.numpy()[0]
        cot_val = energy_val / (self.robot_mass * self.gravity * max(distance_val, 0.01))

        return {
            "loss": loss_val,
            "joint_X_p_grad": joint_X_p_grad,
            "cot": cot_val,
            "energy": energy_val,
            "distance": distance_val,
        }


if __name__ == "__main__":
    print("DifferentiableRollout module loaded successfully")
    print("Supports objectives: height, cost_of_transport, combined")
    print("This module requires a Newton model to test.")
