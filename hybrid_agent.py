"""
Hybrid AMP Agent for Co-Design Optimization

This agent implements Algorithm 1: Hybrid PPO Control + Differentiable Physics
Morphology optimization. It extends the AMPAgent with an outer loop that
optimizes design parameters (hip roll angle) using differentiable physics.

Architecture (Option B - Separate Models):
    - Inner Loop Model: requires_grad=False, uses SolverMuJoCo (stable for RL)
    - Outer Loop Model: requires_grad=True, uses SolverSemiImplicit (BPTT-friendly)
    - After outer loop update, sync joint_X_p from outer to inner model

Algorithm Overview:
    - Inner Loop (Lines 9-17): Standard AMP/PPO training for locomotion
    - Outer Loop (Lines 19-24): Differentiable physics for morphology optimization

Key Parameters:
    - warmup_iters: Initial iterations before activating outer loop (~10000)
    - outer_loop_freq: Outer loop runs every N inner iterations (200)
    - design_lr: Learning rate for design parameter (beta in Algorithm 1)
    - diff_horizon: Horizon for differentiable rollout (H in Algorithm 1)

Reference: algorithm1.pdf, CODESIGN_RULES.md
"""

import os
import numpy as np
import torch
import warp as wp

# Import newton for creating the differentiable model
import newton

# MimicKit imports
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIMICKIT_PATH = os.path.join(SCRIPT_DIR, '..', 'MimicKit', 'mimickit')
if MIMICKIT_PATH not in sys.path:
    sys.path.insert(0, MIMICKIT_PATH)

try:
    import learning.amp_agent as amp_agent
    from util.logger import Logger
    MIMICKIT_AVAILABLE = True
except ImportError:
    amp_agent = None
    Logger = None
    MIMICKIT_AVAILABLE = False

# Local imports - handle both package and script execution
if __name__ == "__main__":
    from parametric_g1 import ParametricG1Model
    from diff_rollout import DifferentiableRollout, SimplifiedDiffRollout
else:
    from .parametric_g1 import ParametricG1Model
    from .diff_rollout import DifferentiableRollout, SimplifiedDiffRollout


def create_diff_model_from_mjcf(mjcf_path: str, device: str):
    """
    Create a Newton model with requires_grad=True for differentiable simulation.

    This creates a SEPARATE model from the inner loop model, used only for
    gradient computation in the outer loop.

    Reference: newton/newton/examples/diffsim/example_diffsim_spring_cage.py

    Args:
        mjcf_path: Path to MuJoCo XML file
        device: Computation device

    Returns:
        Newton Model with requires_grad=True
    """
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    builder.add_mjcf(
        mjcf_path,
        floating=True,
        ignore_inertial_definitions=False,
        collapse_fixed_joints=False,
        enable_self_collisions=False,
    )

    builder.add_ground_plane()

    # Finalize with gradients enabled for BPTT
    model = builder.finalize(device=device, requires_grad=True)

    return model


class HybridAMPAgentBase:
    """
    Base mixin class for hybrid co-design functionality.

    This class contains the outer loop logic that can be mixed into
    either the standalone agent or the integrated AMPAgent.

    Attributes:
        _warmup_iters: Number of inner iterations before outer loop activates
        _outer_loop_freq: Frequency of outer loop updates
        _design_lr: Learning rate for design parameter (beta in Algorithm 1)
        _diff_horizon: Horizon for differentiable rollout (H)
        _parametric_model: ParametricG1Model for design parameter management
        _diff_model: Separate Newton model for differentiable rollout
        _inner_model: Reference to inner loop model (from environment)
    """

    def _load_hybrid_params(self, config):
        """
        Load hybrid co-design specific parameters.

        Args:
            config: Configuration dictionary
        """
        # Warmup period - inner loop trains before outer loop activates
        # Default: 10000 iterations for sufficient AMP convergence
        self._warmup_iters = config.get("warmup_iters", 10000)

        # Outer loop frequency (every N inner iterations)
        # Default: 200 iterations between outer loop updates
        self._outer_loop_freq = config.get("outer_loop_freq", 200)

        # Design parameter learning rate (beta in Algorithm 1)
        self._design_lr = config.get("design_learning_rate", 0.01)

        # Differentiable rollout horizon (H in Algorithm 1)
        self._diff_horizon = config.get("diff_horizon", 50)

        # Path to character MJCF file for creating diff model
        self._char_file = config.get("char_file", None)

        # Design parameter bounds
        self._design_param_min = config.get("design_param_min", -0.1745)
        self._design_param_max = config.get("design_param_max", 0.1745)
        self._design_param_init = config.get("design_param_init", 0.0)

        print(f"[HybridAgent] Warmup iterations: {self._warmup_iters}")
        print(f"[HybridAgent] Outer loop frequency: {self._outer_loop_freq}")
        print(f"[HybridAgent] Design learning rate: {self._design_lr}")
        print(f"[HybridAgent] Diff rollout horizon: {self._diff_horizon}")

    def _init_hybrid_components(self, device):
        """
        Initialize hybrid co-design components.

        Args:
            device: Computation device
        """
        # Initialize parametric model for design parameter management
        self._parametric_model = ParametricG1Model(
            device=device,
            theta_init=self._design_param_init,
            theta_min=self._design_param_min,
            theta_max=self._design_param_max,
        )

        # Outer loop model (created when char_file is provided)
        self._diff_model = None
        self._diff_rollout = None

        # Reference to inner loop model (set when environment is available)
        self._inner_model = None

        # Tracking
        self._outer_loop_count = 0
        self._design_history = []

    def setup_diff_model(self, char_file: str, device: str):
        """
        Set up the separate differentiable model for outer loop.

        This creates a Newton model with requires_grad=True using
        SolverSemiImplicit for BPTT compatibility.

        Args:
            char_file: Path to character MJCF file
            device: Computation device
        """
        if not os.path.exists(char_file):
            print(f"[HybridAgent] Warning: Character file not found: {char_file}")
            return

        print(f"[HybridAgent] Creating differentiable model from {char_file}")

        # Create separate model for outer loop with gradients enabled
        self._diff_model = create_diff_model_from_mjcf(char_file, device)

        # Attach parametric model to the diff model
        self._parametric_model.attach_model(self._diff_model)

        # Initialize differentiable rollout with the diff model
        self._diff_rollout = SimplifiedDiffRollout(
            self._diff_model,
            horizon=self._diff_horizon
        )

        print(f"[HybridAgent] Diff model created with {self._diff_model.joint_count} joints")
        print(f"[HybridAgent] Using SolverSemiImplicit for outer loop")

    def set_inner_model_reference(self, inner_model):
        """
        Set reference to the inner loop's Newton model.

        This is used to sync joint_X_p after outer loop updates.

        Args:
            inner_model: Inner loop Newton model (from environment)
        """
        self._inner_model = inner_model
        print(f"[HybridAgent] Inner model reference set")

    def _should_run_outer_loop(self) -> bool:
        """
        Determine if outer loop should execute this iteration.

        Outer loop runs when:
        1. Warmup period is complete (inner loop has learned basic locomotion)
        2. Current iteration aligns with outer loop frequency

        Returns:
            True if outer loop should run
        """
        if self._iter < self._warmup_iters:
            return False

        iters_since_warmup = self._iter - self._warmup_iters
        return iters_since_warmup % self._outer_loop_freq == 0

    def _outer_loop_update(self):
        """
        Execute outer loop morphology optimization.

        Implements Algorithm 1, Lines 19-24:
            Line 18: theta* <- theta_k (policy optimal for current design)
            Line 20: Sample initial state s0
            Line 21: tau_val <- DifferentiableRollout(pi_theta*, phi_k, H)
            Line 22: L(phi_k) <- -sum r(s_t, a_t)
            Line 23: grad_phi L <- BPTT(L(phi_k))
            Line 24: phi_{k+1} <- proj_C(phi_k - beta * grad_phi L)

        Returns:
            Dictionary with outer loop info
        """
        print(f"\n[HybridAgent] === Outer Loop Update {self._outer_loop_count + 1} ===")
        print(f"[HybridAgent] Iteration: {self._iter}")
        print(f"[HybridAgent] Current theta: {self._parametric_model.get_theta():.4f} rad "
              f"({self._parametric_model.get_theta_degrees():.2f} deg)")

        if self._diff_rollout is None:
            print("[HybridAgent] Warning: No differentiable rollout available")
            return {"outer_loop_status": "skipped_no_rollout"}

        # Lines 21-23: Differentiable rollout and BPTT
        # Using SimplifiedDiffRollout (no policy, just physics test)
        grad_info = self._diff_rollout.forward_and_backward()

        print(f"[HybridAgent] Outer loop loss: {grad_info['loss']:.4f}")

        # Extract gradient for design parameter
        design_grad = self._extract_design_gradient(grad_info)

        if design_grad is not None and not np.isnan(design_grad) and design_grad != 0.0:
            print(f"[HybridAgent] Design gradient: {design_grad:.6f}")

            # Line 24: Gradient descent with projection
            old_theta = self._parametric_model.get_theta()
            new_theta = old_theta - self._design_lr * design_grad
            new_theta = float(np.clip(new_theta, self._design_param_min, self._design_param_max))

            # Update parametric model (this updates the diff_model's joint_X_p)
            self._parametric_model.set_theta(new_theta, update_model=True)

            print(f"[HybridAgent] Updated theta: {old_theta:.4f} -> {new_theta:.4f} rad")
            print(f"[HybridAgent] New theta: {self._parametric_model.get_theta_degrees():.2f} deg")

            # CRITICAL: Sync joint_X_p to inner loop model
            self._sync_design_to_inner_model()

            # Record history
            self._design_history.append({
                "iter": self._iter,
                "theta": new_theta,
                "grad": design_grad,
                "loss": grad_info["loss"],
            })
        else:
            if design_grad is None:
                print("[HybridAgent] Warning: Could not extract design gradient")
            elif np.isnan(design_grad):
                print("[HybridAgent] Warning: Design gradient is NaN, skipping update")
            else:
                print("[HybridAgent] Warning: Design gradient is zero, skipping update")
            design_grad = 0.0

        self._outer_loop_count += 1

        return {
            "outer_loop_loss": grad_info["loss"],
            "outer_loop_grad": design_grad,
            "outer_loop_count": self._outer_loop_count,
        }

    def _sync_design_to_inner_model(self):
        """
        Synchronize design parameters from outer to inner model.

        After the outer loop updates joint_X_p in the diff model,
        we copy these values to the inner model used by the environment.

        This is the key step that makes Option B work - the two models
        stay synchronized on the design parameters.
        """
        if self._inner_model is None:
            print("[HybridAgent] Warning: No inner model reference, cannot sync")
            return

        if self._diff_model is None:
            print("[HybridAgent] Warning: No diff model, cannot sync")
            return

        # Get the updated joint_X_p from diff model
        diff_joint_X_p = self._diff_model.joint_X_p.numpy()

        # Get hip roll joint indices
        left_idx = self._parametric_model._left_hip_roll_idx
        right_idx = self._parametric_model._right_hip_roll_idx

        if left_idx is None or right_idx is None:
            print("[HybridAgent] Warning: Hip roll indices not set, cannot sync")
            return

        # Update only the hip roll joint transforms in the inner model
        inner_joint_X_p = self._inner_model.joint_X_p.numpy()
        inner_joint_X_p[left_idx] = diff_joint_X_p[left_idx]
        inner_joint_X_p[right_idx] = diff_joint_X_p[right_idx]
        self._inner_model.joint_X_p.assign(inner_joint_X_p)

        print(f"[HybridAgent] Synced joint_X_p to inner model (indices {left_idx}, {right_idx})")

    def _extract_design_gradient(self, grad_info) -> float:
        """
        Extract the design parameter gradient from full joint gradients.

        The gradient for theta is computed from the gradients of joint_X_p
        for the hip roll joints using the chain rule.

        Args:
            grad_info: Dictionary with joint_X_p_grad

        Returns:
            Scalar gradient for theta, or None if unavailable
        """
        joint_X_p_grad = grad_info.get("joint_X_p_grad")

        if joint_X_p_grad is None:
            return None

        left_idx = self._parametric_model._left_hip_roll_idx
        right_idx = self._parametric_model._right_hip_roll_idx

        if left_idx is None or right_idx is None:
            return 0.0

        # Chain rule: dL/dtheta = dL/dq * dq/dtheta
        # where q is the quaternion (w, x, y, z) for X-rotation
        # w = cos(theta/2), x = sin(theta/2), y = 0, z = 0

        theta = self._parametric_model.get_theta()
        half_theta = theta / 2.0

        # Derivatives of quaternion components w.r.t. theta
        dw_dtheta = -np.sin(half_theta) / 2.0
        dx_dtheta = np.cos(half_theta) / 2.0

        # Get quaternion gradients from joint_X_p gradient
        # joint_X_p format: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
        # Based on Warp transform storage order

        left_quat_grad = joint_X_p_grad[left_idx, 3:7]
        right_quat_grad = joint_X_p_grad[right_idx, 3:7]

        # Assuming (w, x, y, z) ordering for quaternion
        grad_w_left = left_quat_grad[0]
        grad_x_left = left_quat_grad[1]
        grad_w_right = right_quat_grad[0]
        grad_x_right = right_quat_grad[1]

        # Apply chain rule
        grad_theta_left = grad_w_left * dw_dtheta + grad_x_left * dx_dtheta
        grad_theta_right = grad_w_right * dw_dtheta + grad_x_right * dx_dtheta

        # Average gradients from both hips (symmetric design)
        grad_theta = (grad_theta_left + grad_theta_right) / 2.0

        return float(grad_theta)

    def get_design_history(self):
        """Get history of design parameter updates."""
        return self._design_history

    def _get_hybrid_state_dict(self):
        """Get hybrid-specific state for saving."""
        return {
            "outer_loop_count": self._outer_loop_count,
            "design_history": self._design_history,
            "parametric_model": self._parametric_model.get_state_dict(),
        }

    def _load_hybrid_state_dict(self, state):
        """Load hybrid-specific state."""
        self._outer_loop_count = state.get("outer_loop_count", 0)
        self._design_history = state.get("design_history", [])
        if "parametric_model" in state:
            self._parametric_model.load_state_dict(state["parametric_model"])


class HybridAMPAgent(HybridAMPAgentBase):
    """
    Standalone Hybrid Co-Design Agent for testing without MimicKit.

    This class can be used for standalone testing and debugging.
    For production training with MimicKit, use HybridAMPAgentIntegrated.
    """

    def __init__(self, config, env, device):
        """
        Initialize the standalone hybrid agent.

        Args:
            config: Configuration dictionary
            env: Environment instance (can be None for testing)
            device: Computation device
        """
        self._env = env
        self._device = device
        self._config = config
        self._iter = 0

        # Load hybrid parameters
        self._load_hybrid_params(config)

        # Initialize hybrid components
        self._init_hybrid_components(device)

    def train_iter(self):
        """
        Execute one training iteration.

        In standalone mode, this runs a dummy inner loop and
        conditionally runs the outer loop.

        Returns:
            Dictionary with training info
        """
        info = {}

        # Phase 1: Inner Loop (dummy for standalone)
        inner_info = self._dummy_inner_loop()
        info.update(inner_info)

        # Phase 2: Outer Loop (conditional)
        if self._should_run_outer_loop():
            outer_info = self._outer_loop_update()
            info.update(outer_info)

        # Update iteration counter
        self._iter += 1

        # Log design parameter
        info["design_param_theta"] = self._parametric_model.get_theta()
        info["design_param_degrees"] = self._parametric_model.get_theta_degrees()

        return info

    def _dummy_inner_loop(self):
        """Placeholder inner loop for standalone testing."""
        return {"inner_loss": 0.0, "inner_reward": 0.0}

    def save(self, filepath):
        """Save agent state."""
        state = {
            "iter": self._iter,
            "config": self._config,
            **self._get_hybrid_state_dict(),
        }
        torch.save(state, filepath)
        print(f"[HybridAgent] Saved to {filepath}")

    def load(self, filepath):
        """Load agent state."""
        state = torch.load(filepath, map_location=self._device, weights_only=False)
        self._iter = state.get("iter", 0)
        self._load_hybrid_state_dict(state)
        print(f"[HybridAgent] Loaded from {filepath}, iteration {self._iter}")


# Only define integrated agent if MimicKit is available
if MIMICKIT_AVAILABLE:
    class HybridAMPAgentIntegrated(amp_agent.AMPAgent, HybridAMPAgentBase):
        """
        Integrated Hybrid Agent that properly inherits from AMPAgent.

        This class should be used for production training with MimicKit.
        It overrides _train_iter() to inject the outer loop, and
        _log_train_info() to log design parameters.

        Usage:
            config = load_config("hybrid_g1_agent.yaml")
            env = build_env(...)
            agent = HybridAMPAgentIntegrated(config, env, device)
            agent.setup_diff_model(char_file, device)
            agent.set_inner_model_reference(env._engine._sim_model)
            agent.train_model(max_samples=50000000, ...)
        """

        def __init__(self, config, env, device):
            """
            Initialize the integrated hybrid agent.

            Args:
                config: Configuration dictionary (should include AMP and hybrid params)
                env: MimicKit environment instance
                device: Computation device
            """
            # Load hybrid parameters first (before super().__init__ calls _load_params)
            self._load_hybrid_params(config)

            # Initialize base AMP agent
            # This calls _load_params, _build_model, etc.
            super().__init__(config, env, device)

            # Initialize hybrid components
            self._init_hybrid_components(device)

            print("[HybridAgentIntegrated] Initialized with AMP + outer loop")

        def _train_iter(self):
            """
            Override training iteration to add outer loop.

            This calls the parent _train_iter() for AMP training,
            then conditionally runs the outer loop.

            Returns:
                Dictionary with training info
            """
            # Phase 1: Inner Loop - standard AMP training
            # This calls BaseAgent._train_iter() which does:
            # - _init_iter()
            # - _rollout_train()
            # - _build_train_data()
            # - _update_model()
            info = super()._train_iter()

            # Phase 2: Outer Loop - morphology optimization (conditional)
            if self._should_run_outer_loop():
                outer_info = self._outer_loop_update()
                info.update(outer_info)

            # Add design parameter to info
            info["design_param_theta"] = self._parametric_model.get_theta()
            info["design_param_degrees"] = self._parametric_model.get_theta_degrees()

            return info

        def _log_train_info(self, train_info, test_info, env_diag_info, start_time):
            """
            Override to add design parameter logging.

            Args:
                train_info: Training info dictionary
                test_info: Test info dictionary
                env_diag_info: Environment diagnostics
                start_time: Training start time
            """
            # Call parent logging
            super()._log_train_info(train_info, test_info, env_diag_info, start_time)

            # Log design parameters
            theta = self._parametric_model.get_theta()
            theta_deg = self._parametric_model.get_theta_degrees()

            self._logger.log("Design_Theta_Rad", theta, collection="3_Design")
            self._logger.log("Design_Theta_Deg", theta_deg, collection="3_Design")
            self._logger.log("Outer_Loop_Count", self._outer_loop_count, collection="3_Design")

            if "outer_loop_loss" in train_info:
                self._logger.log("Outer_Loop_Loss", train_info["outer_loop_loss"], collection="3_Design")
            if "outer_loop_grad" in train_info:
                self._logger.log("Outer_Loop_Grad", train_info["outer_loop_grad"], collection="3_Design")

        def save(self, out_file):
            """
            Override to include hybrid state in checkpoint.

            Args:
                out_file: Output file path
            """
            # Save base state
            super().save(out_file)

            # Save hybrid state separately
            hybrid_file = out_file.replace(".pt", "_hybrid.pt")
            hybrid_state = self._get_hybrid_state_dict()
            torch.save(hybrid_state, hybrid_file)

            if Logger:
                Logger.print(f"[HybridAgent] Saved hybrid state to {hybrid_file}")

        def load(self, in_file):
            """
            Override to load hybrid state from checkpoint.

            Args:
                in_file: Input file path
            """
            # Load base state
            super().load(in_file)

            # Load hybrid state
            hybrid_file = in_file.replace(".pt", "_hybrid.pt")
            if os.path.exists(hybrid_file):
                hybrid_state = torch.load(hybrid_file, map_location=self._device, weights_only=False)
                self._load_hybrid_state_dict(hybrid_state)
                if Logger:
                    Logger.print(f"[HybridAgent] Loaded hybrid state from {hybrid_file}")
            else:
                if Logger:
                    Logger.print(f"[HybridAgent] Warning: Hybrid state file not found: {hybrid_file}")

else:
    # Placeholder when MimicKit is not available
    class HybridAMPAgentIntegrated:
        """Placeholder - MimicKit not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("MimicKit is required for HybridAMPAgentIntegrated. "
                            "Use HybridAMPAgent for standalone testing.")


if __name__ == "__main__":
    print("Testing HybridAMPAgent...")

    # Create dummy config
    config = {
        "warmup_iters": 100,
        "outer_loop_freq": 20,
        "design_learning_rate": 0.01,
        "diff_horizon": 50,
        "design_param_init": 0.0,
    }

    # Create agent (standalone mode)
    agent = HybridAMPAgent(config, env=None, device="cpu")

    print(f"\nInitial design param: {agent._parametric_model.get_theta():.4f} rad")

    # Test should_run_outer_loop logic
    for i in range(150):
        agent._iter = i
        if agent._should_run_outer_loop():
            print(f"Outer loop would run at iteration {i}")

    print("\nHybridAMPAgent standalone test completed!")
