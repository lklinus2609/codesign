"""
Hybrid AMP Agent for Co-Design Optimization (PGHC Algorithm)

This agent implements Algorithm 1: Performance-Gated Hybrid Co-Design (PGHC)
from the thesis. It extends the AMPAgent with an outer loop that optimizes
design parameters using differentiable physics.

Architecture (Option B - Separate Models):
    - Inner Loop Model: requires_grad=False, uses SolverMuJoCo (stable for RL)
    - Outer Loop Model: requires_grad=True, uses SolverSemiImplicit (BPTT-friendly)
    - After outer loop update, sync joint_X_p from outer to inner model

PGHC Algorithm Overview:
    - Phase 1 (Lines 12-18): Performance-Gated Inner Loop
        - Train policy until stability metric delta_rel < delta_conv
    - Phase 2 (Lines 20-35): Trust-Region Outer Loop
        - Compute design gradient via BPTT
        - Adaptive trust region with performance validation

Key Parameters (Thesis Section II.D):
    - stability_window (W): Window size for moving average returns (default: 100)
    - stability_threshold (delta_conv): Convergence threshold (default: 0.05)
    - trust_region_threshold (xi): Performance degradation limit (default: 0.1)
    - design_lr_init (beta_init): Initial design learning rate

Reference: Masters_Thesis.pdf Algorithm 1, CODESIGN_RULES.md
"""

import os
from collections import deque
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
    Base mixin class for PGHC hybrid co-design functionality.

    This class contains the outer loop logic that can be mixed into
    either the standalone agent or the integrated AMPAgent.

    Implements Algorithm 1 from the thesis:
    - Stability gating (Lines 12-18): Policy convergence check
    - Adaptive trust region (Lines 24-35): Safe design updates

    Attributes:
        _gating_mode: "stability" (thesis algorithm) or "fixed" (legacy)
        _stability_window: Window W for moving average (default: 100)
        _stability_threshold: delta_conv threshold (default: 0.05)
        _trust_region_threshold: xi threshold (default: 0.1)
        _design_lr: Current adaptive learning rate beta
        _design_lr_init: Initial learning rate
        _design_lr_min: Minimum learning rate (prevents collapse)
        _design_lr_max: Maximum learning rate (prevents instability)
    """

    def _load_hybrid_params(self, config):
        """
        Load PGHC hybrid co-design parameters.

        Args:
            config: Configuration dictionary
        """
        # =================================================================
        # Gating Mode Selection
        # =================================================================
        # "stability": Use performance-gated triggering (thesis Algorithm 1)
        # "fixed": Use legacy warmup + frequency schedule
        self._gating_mode = config.get("gating_mode", "stability")

        # =================================================================
        # Stability Gating Parameters (Algorithm 1, Lines 12-18)
        # =================================================================
        # Window W for moving average returns
        self._stability_window = config.get("stability_window", 100)

        # Convergence threshold delta_conv (5% relative change)
        self._stability_threshold = config.get("stability_threshold", 0.05)

        # Small epsilon to prevent division by zero
        self._stability_epsilon = config.get("stability_epsilon", 1e-8)

        # Minimum iterations before stability check can pass
        # (ensures enough data for reliable metric)
        self._min_iters_for_stability = config.get(
            "min_iters_for_stability",
            2 * self._stability_window
        )

        # =================================================================
        # Legacy Fixed Schedule Parameters (fallback mode)
        # =================================================================
        self._warmup_iters = config.get("warmup_iters", 10000)
        self._outer_loop_freq = config.get("outer_loop_freq", 200)

        # =================================================================
        # Adaptive Trust Region Parameters (Algorithm 1, Lines 24-35)
        # =================================================================
        # Initial design learning rate (beta_init)
        self._design_lr_init = config.get("design_learning_rate", 0.01)
        self._design_lr = self._design_lr_init

        # Learning rate bounds
        self._design_lr_min = config.get("design_lr_min", 1e-5)
        self._design_lr_max = config.get("design_lr_max", 0.1)

        # Trust region threshold xi (10% performance degradation allowed)
        self._trust_region_threshold = config.get("trust_region_threshold", 0.1)

        # Learning rate adaptation factors
        self._lr_decay_factor = config.get("lr_decay_factor", 0.5)
        self._lr_growth_factor = config.get("lr_growth_factor", 1.5)

        # Maximum trust region iterations before giving up
        self._max_trust_region_iters = config.get("max_trust_region_iters", 10)

        # Small improvement threshold for LR growth
        self._small_improvement_threshold = config.get(
            "small_improvement_threshold", 0.01
        )

        # =================================================================
        # Differentiable Rollout Parameters
        # =================================================================
        self._diff_horizon = config.get("diff_horizon", 50)
        self._char_file = config.get("char_file", None)

        # =================================================================
        # Design Parameter Configuration
        # =================================================================
        self._design_param_min = config.get("design_param_min", -0.5236)  # -30 deg
        self._design_param_max = config.get("design_param_max", 0.5236)   # +30 deg
        self._design_param_init = config.get("design_param_init", 0.0)

        # =================================================================
        # Print Configuration Summary
        # =================================================================
        print(f"[PGHC] Gating mode: {self._gating_mode}")
        if self._gating_mode == "stability":
            print(f"[PGHC] Stability window (W): {self._stability_window}")
            print(f"[PGHC] Stability threshold (delta_conv): {self._stability_threshold}")
        else:
            print(f"[PGHC] Warmup iterations: {self._warmup_iters}")
            print(f"[PGHC] Outer loop frequency: {self._outer_loop_freq}")
        print(f"[PGHC] Initial design LR (beta): {self._design_lr_init}")
        print(f"[PGHC] Trust region threshold (xi): {self._trust_region_threshold}")
        print(f"[PGHC] Diff rollout horizon: {self._diff_horizon}")

    def _init_hybrid_components(self, device):
        """
        Initialize PGHC hybrid co-design components.

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

        # =================================================================
        # Stability Tracking (Algorithm 1, Lines 16-17)
        # =================================================================
        # Episode return history for moving average computation
        self._return_history = deque(maxlen=2 * self._stability_window)

        # Current stability metric value
        self._current_delta_rel = float('inf')

        # Flag indicating if stability was achieved at least once
        self._stability_achieved = False

        # Number of times stability was checked
        self._stability_checks = 0

        # =================================================================
        # Outer Loop Tracking
        # =================================================================
        self._outer_loop_count = 0
        self._design_history = []

        # Last objective value for trust region comparison
        self._last_objective_value = None

        # Trust region statistics
        self._trust_region_accepts = 0
        self._trust_region_rejects = 0

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
            print(f"[PGHC] Warning: Character file not found: {char_file}")
            return

        print(f"[PGHC] Creating differentiable model from {char_file}")

        # Create separate model for outer loop with gradients enabled
        self._diff_model = create_diff_model_from_mjcf(char_file, device)

        # Attach parametric model to the diff model
        self._parametric_model.attach_model(self._diff_model)

        # Initialize differentiable rollout with the diff model
        self._diff_rollout = SimplifiedDiffRollout(
            self._diff_model,
            horizon=self._diff_horizon
        )

        print(f"[PGHC] Diff model created with {self._diff_model.joint_count} joints")
        print(f"[PGHC] Using SolverSemiImplicit for outer loop")

    def set_inner_model_reference(self, inner_model):
        """
        Set reference to the inner loop's Newton model.

        This is used to sync joint_X_p after outer loop updates.

        Args:
            inner_model: Inner loop Newton model (from environment)
        """
        self._inner_model = inner_model
        print(f"[PGHC] Inner model reference set")

    def record_episode_return(self, episode_return: float):
        """
        Record an episode return for stability metric computation.

        Called by the training loop after each episode completes.

        Args:
            episode_return: Total reward from completed episode
        """
        self._return_history.append(episode_return)

    def _compute_stability_metric(self) -> float:
        """
        Compute the stability metric delta_rel.

        Implements Algorithm 1, Line 17:
            delta_rel = |R_t - R_{t-W}| / (|R_{t-W}| + epsilon)

        Returns:
            delta_rel value, or inf if not enough data
        """
        if len(self._return_history) < 2 * self._stability_window:
            return float('inf')

        # Convert to list for slicing
        history = list(self._return_history)

        # R_t: average of most recent W episodes
        R_t = np.mean(history[-self._stability_window:])

        # R_{t-W}: average of previous W episodes
        R_t_W = np.mean(history[-2*self._stability_window:-self._stability_window])

        # Compute relative change
        delta_rel = abs(R_t - R_t_W) / (abs(R_t_W) + self._stability_epsilon)

        self._current_delta_rel = delta_rel
        return delta_rel

    def _should_run_outer_loop(self) -> bool:
        """
        Determine if outer loop should execute this iteration.

        Implements Algorithm 1, Line 18:
            until delta_rel < delta_conv (Stability Gate Trigger)

        In "stability" mode: triggers when policy has converged
        In "fixed" mode: uses legacy warmup + frequency schedule

        Returns:
            True if outer loop should run
        """
        if self._gating_mode == "stability":
            return self._should_run_outer_loop_stability()
        else:
            return self._should_run_outer_loop_fixed()

    def _should_run_outer_loop_stability(self) -> bool:
        """
        Stability-gated outer loop trigger (thesis algorithm).

        Returns:
            True if stability gate is triggered
        """
        # Need minimum iterations for reliable stability metric
        if self._iter < self._min_iters_for_stability:
            return False

        # Compute stability metric
        delta_rel = self._compute_stability_metric()
        self._stability_checks += 1

        # Check if stability threshold is met
        if delta_rel < self._stability_threshold:
            self._stability_achieved = True
            print(f"[PGHC] Stability gate triggered: delta_rel={delta_rel:.4f} < {self._stability_threshold}")
            return True

        return False

    def _should_run_outer_loop_fixed(self) -> bool:
        """
        Fixed schedule outer loop trigger (legacy mode).

        Returns:
            True if iteration matches schedule
        """
        if self._iter < self._warmup_iters:
            return False

        iters_since_warmup = self._iter - self._warmup_iters
        return iters_since_warmup % self._outer_loop_freq == 0

    def _evaluate_objective(self, theta_val: float = None) -> float:
        """
        Evaluate the objective function J(phi, theta*) for trust region.

        Implements evaluation needed for Algorithm 1, Line 26:
            D = Objective(phi_k, theta*) - Objective(phi', theta*)

        Args:
            theta_val: Design parameter value (None = current)

        Returns:
            Objective value (lower is better)
        """
        if self._diff_rollout is None:
            return 0.0

        # Temporarily set theta if specified
        original_theta = None
        if theta_val is not None:
            original_theta = self._parametric_model.get_theta()
            self._parametric_model.set_theta(theta_val, update_model=True)

        # Run forward pass to get objective
        grad_info = self._diff_rollout.forward_and_backward()
        objective = grad_info.get("loss", 0.0)

        # Restore original theta
        if original_theta is not None:
            self._parametric_model.set_theta(original_theta, update_model=True)

        return objective

    def _outer_loop_update(self):
        """
        Execute outer loop morphology optimization with adaptive trust region.

        Implements Algorithm 1, Lines 20-35:
            - Line 21: Differentiable rollout
            - Line 22: Compute gradient via BPTT
            - Lines 24-35: Adaptive trust region update

        Returns:
            Dictionary with outer loop info
        """
        print(f"\n[PGHC] === Outer Loop Update {self._outer_loop_count + 1} ===")
        print(f"[PGHC] Iteration: {self._iter}")
        print(f"[PGHC] Current design LR (beta): {self._design_lr:.6f}")

        current_theta = self._parametric_model.get_theta()
        print(f"[PGHC] Current theta: {current_theta:.4f} rad "
              f"({self._parametric_model.get_theta_degrees():.2f} deg)")

        if self._diff_rollout is None:
            print("[PGHC] Warning: No differentiable rollout available")
            return {"outer_loop_status": "skipped_no_rollout"}

        # =================================================================
        # Line 21-22: Differentiable rollout and BPTT
        # =================================================================
        grad_info = self._diff_rollout.forward_and_backward()
        current_objective = grad_info.get("loss", 0.0)

        print(f"[PGHC] Current objective J(phi_k): {current_objective:.4f}")

        # Extract gradient for design parameter
        design_grad = self._extract_design_gradient(grad_info)

        if design_grad is None or np.isnan(design_grad):
            print("[PGHC] Warning: Invalid gradient, skipping update")
            return {"outer_loop_status": "skipped_invalid_grad"}

        print(f"[PGHC] Design gradient: {design_grad:.6f}")

        # =================================================================
        # Lines 24-35: Adaptive Trust Region
        # =================================================================
        accepted = False
        trust_region_iters = 0

        while not accepted and trust_region_iters < self._max_trust_region_iters:
            trust_region_iters += 1

            # Line 25: Compute candidate design
            candidate_theta = current_theta - self._design_lr * design_grad

            # Apply bounds projection
            candidate_theta = float(np.clip(
                candidate_theta,
                self._design_param_min,
                self._design_param_max
            ))

            # Line 26: Evaluate candidate objective
            candidate_objective = self._evaluate_objective(candidate_theta)

            # Compute performance change D
            D = current_objective - candidate_objective  # Improvement (positive = better)

            print(f"[PGHC]   TR iter {trust_region_iters}: "
                  f"candidate={candidate_theta:.4f}, "
                  f"J'={candidate_objective:.4f}, D={D:.4f}")

            # Line 27: Check trust region violation
            if D < -self._trust_region_threshold * abs(current_objective):
                # Violation: performance degraded too much
                # Line 28: Shrink step size
                self._design_lr *= self._lr_decay_factor
                self._design_lr = max(self._design_lr, self._design_lr_min)
                self._trust_region_rejects += 1
                print(f"[PGHC]   Trust region violated, shrinking LR to {self._design_lr:.6f}")
            else:
                # Lines 30-32: Accept update
                accepted = True
                self._trust_region_accepts += 1

                # Update theta
                old_theta = current_theta
                self._parametric_model.set_theta(candidate_theta, update_model=True)

                print(f"[PGHC] Update accepted: {old_theta:.4f} -> {candidate_theta:.4f} rad")
                print(f"[PGHC]   ({self._parametric_model.get_theta_degrees():.2f} deg)")

                # Line 31: Optionally grow LR if improvement is small
                if abs(D) < self._small_improvement_threshold * abs(current_objective):
                    self._design_lr *= self._lr_growth_factor
                    self._design_lr = min(self._design_lr, self._design_lr_max)
                    print(f"[PGHC]   Small D, growing LR to {self._design_lr:.6f}")

                # Sync to inner model
                self._sync_design_to_inner_model()

                # Record history
                self._design_history.append({
                    "iter": self._iter,
                    "theta": candidate_theta,
                    "grad": design_grad,
                    "loss": candidate_objective,
                    "improvement": D,
                    "design_lr": self._design_lr,
                    "trust_region_iters": trust_region_iters,
                })

        if not accepted:
            print(f"[PGHC] Warning: Trust region failed after {trust_region_iters} iterations")

        self._outer_loop_count += 1
        self._last_objective_value = current_objective

        return {
            "outer_loop_loss": current_objective,
            "outer_loop_grad": design_grad if design_grad is not None else 0.0,
            "outer_loop_count": self._outer_loop_count,
            "outer_loop_accepted": accepted,
            "outer_loop_lr": self._design_lr,
            "stability_delta_rel": self._current_delta_rel,
            "trust_region_iters": trust_region_iters,
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
            print("[PGHC] Warning: No inner model reference, cannot sync")
            return

        if self._diff_model is None:
            print("[PGHC] Warning: No diff model, cannot sync")
            return

        # Get the updated joint_X_p from diff model
        diff_joint_X_p = self._diff_model.joint_X_p.numpy()

        # Get hip roll joint indices
        left_idx = self._parametric_model._left_hip_roll_idx
        right_idx = self._parametric_model._right_hip_roll_idx

        if left_idx is None or right_idx is None:
            print("[PGHC] Warning: Hip roll indices not set, cannot sync")
            return

        # Update only the hip roll joint transforms in the inner model
        inner_joint_X_p = self._inner_model.joint_X_p.numpy()
        inner_joint_X_p[left_idx] = diff_joint_X_p[left_idx]
        inner_joint_X_p[right_idx] = diff_joint_X_p[right_idx]
        self._inner_model.joint_X_p.assign(inner_joint_X_p)

        print(f"[PGHC] Synced joint_X_p to inner model (indices {left_idx}, {right_idx})")

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

    def get_stability_metrics(self):
        """Get current stability tracking metrics."""
        return {
            "delta_rel": self._current_delta_rel,
            "stability_achieved": self._stability_achieved,
            "stability_checks": self._stability_checks,
            "return_history_len": len(self._return_history),
        }

    def get_trust_region_stats(self):
        """Get trust region acceptance statistics."""
        total = self._trust_region_accepts + self._trust_region_rejects
        accept_rate = self._trust_region_accepts / total if total > 0 else 0.0
        return {
            "accepts": self._trust_region_accepts,
            "rejects": self._trust_region_rejects,
            "accept_rate": accept_rate,
            "current_lr": self._design_lr,
        }

    def _get_hybrid_state_dict(self):
        """Get hybrid-specific state for saving."""
        return {
            "outer_loop_count": self._outer_loop_count,
            "design_history": self._design_history,
            "parametric_model": self._parametric_model.get_state_dict(),
            "design_lr": self._design_lr,
            "return_history": list(self._return_history),
            "stability_achieved": self._stability_achieved,
            "trust_region_accepts": self._trust_region_accepts,
            "trust_region_rejects": self._trust_region_rejects,
        }

    def _load_hybrid_state_dict(self, state):
        """Load hybrid-specific state."""
        self._outer_loop_count = state.get("outer_loop_count", 0)
        self._design_history = state.get("design_history", [])
        if "parametric_model" in state:
            self._parametric_model.load_state_dict(state["parametric_model"])
        self._design_lr = state.get("design_lr", self._design_lr_init)
        if "return_history" in state:
            self._return_history = deque(
                state["return_history"],
                maxlen=2 * self._stability_window
            )
        self._stability_achieved = state.get("stability_achieved", False)
        self._trust_region_accepts = state.get("trust_region_accepts", 0)
        self._trust_region_rejects = state.get("trust_region_rejects", 0)


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

        # Simulate episode returns for stability tracking
        # In real training, this comes from actual rollouts
        self._simulate_episode_return()

        # Phase 2: Outer Loop (conditional)
        if self._should_run_outer_loop():
            outer_info = self._outer_loop_update()
            info.update(outer_info)

        # Update iteration counter
        self._iter += 1

        # Log design parameter
        info["design_param_theta"] = self._parametric_model.get_theta()
        info["design_param_degrees"] = self._parametric_model.get_theta_degrees()
        info["stability_delta_rel"] = self._current_delta_rel

        return info

    def _dummy_inner_loop(self):
        """Placeholder inner loop for standalone testing."""
        return {"inner_loss": 0.0, "inner_reward": 0.0}

    def _simulate_episode_return(self):
        """Simulate episode return for testing stability gating."""
        # Simulate converging returns
        base_return = 100.0
        noise = np.random.randn() * (5.0 / (1 + self._iter * 0.001))
        self.record_episode_return(base_return + noise)

    def save(self, filepath):
        """Save agent state."""
        state = {
            "iter": self._iter,
            "config": self._config,
            **self._get_hybrid_state_dict(),
        }
        torch.save(state, filepath)
        print(f"[PGHC] Saved to {filepath}")

    def load(self, filepath):
        """Load agent state."""
        state = torch.load(filepath, map_location=self._device, weights_only=False)
        self._iter = state.get("iter", 0)
        self._load_hybrid_state_dict(state)
        print(f"[PGHC] Loaded from {filepath}, iteration {self._iter}")


# Only define integrated agent if MimicKit is available
if MIMICKIT_AVAILABLE:
    class HybridAMPAgentIntegrated(amp_agent.AMPAgent, HybridAMPAgentBase):
        """
        Integrated PGHC Agent that properly inherits from AMPAgent.

        This class should be used for production training with MimicKit.
        It overrides _train_iter() to inject the outer loop, and
        _log_train_info() to log design parameters and stability metrics.

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

            print("[PGHC] Initialized with AMP + PGHC outer loop")

        def _train_iter(self):
            """
            Override training iteration to add PGHC outer loop.

            This calls the parent _train_iter() for AMP training,
            records episode returns for stability tracking,
            then conditionally runs the outer loop.

            Returns:
                Dictionary with training info
            """
            # Phase 1: Inner Loop - standard AMP training
            info = super()._train_iter()

            # Record episode returns for stability metric
            # Note: In MimicKit, episode returns come from test_info
            if hasattr(self, '_mean_test_return') and self._mean_test_return is not None:
                self.record_episode_return(self._mean_test_return)

            # Phase 2: Outer Loop - morphology optimization (conditional)
            if self._should_run_outer_loop():
                outer_info = self._outer_loop_update()
                info.update(outer_info)

            # Add design parameter and stability metrics to info
            info["design_param_theta"] = self._parametric_model.get_theta()
            info["design_param_degrees"] = self._parametric_model.get_theta_degrees()
            info["stability_delta_rel"] = self._current_delta_rel

            return info

        def setup_video_recording(self, video_interval=500, fps=30):
            """
            Set up video recording for wandb.

            Args:
                video_interval: Record video every N iterations (0 to disable)
                fps: Frames per second for video
            """
            self._video_interval = video_interval
            self._video_fps = fps
            self._video_recorder = None

            if video_interval > 0:
                try:
                    from hybrid_codesign.video_recorder import HeadlessVideoRecorder
                    self._video_recorder = HeadlessVideoRecorder(
                        self._env, self, self._device, fps=fps
                    )
                    print(f"[PGHC] Headless video recording enabled (every {video_interval} iters)")
                except ImportError as e:
                    print(f"[PGHC] Video recording not available: {e}")

        def _log_train_info(self, train_info, test_info, env_diag_info, start_time):
            """
            Override to add PGHC-specific logging.

            Args:
                train_info: Training info dictionary
                test_info: Test info dictionary
                env_diag_info: Environment diagnostics
                start_time: Training start time
            """
            # Call parent logging
            super()._log_train_info(train_info, test_info, env_diag_info, start_time)

            # Store mean test return for stability tracking
            if test_info is not None and "mean_return" in test_info:
                self._mean_test_return = test_info["mean_return"]

            # Log design parameters
            theta = self._parametric_model.get_theta()
            theta_deg = self._parametric_model.get_theta_degrees()

            self._logger.log("Design_Theta_Rad", theta, collection="3_Design")
            self._logger.log("Design_Theta_Deg", theta_deg, collection="3_Design")
            self._logger.log("Outer_Loop_Count", self._outer_loop_count, collection="3_Design")
            self._logger.log("Design_LR", self._design_lr, collection="3_Design")

            # Log stability metrics
            self._logger.log("Stability_Delta_Rel", self._current_delta_rel, collection="3_Design")
            self._logger.log("Stability_Achieved", int(self._stability_achieved), collection="3_Design")

            # Log trust region stats
            tr_stats = self.get_trust_region_stats()
            self._logger.log("TR_Accept_Rate", tr_stats["accept_rate"], collection="3_Design")

            if "outer_loop_loss" in train_info:
                self._logger.log("Outer_Loop_Loss", train_info["outer_loop_loss"], collection="3_Design")
            if "outer_loop_grad" in train_info:
                self._logger.log("Outer_Loop_Grad", train_info["outer_loop_grad"], collection="3_Design")

            # Video recording (wandb only)
            if (hasattr(self, '_video_recorder') and
                self._video_recorder is not None and
                hasattr(self, '_video_interval') and
                self._video_interval > 0 and
                self._iter > 0 and
                self._iter % self._video_interval == 0):
                try:
                    self._video_recorder.record_and_log(
                        iteration=self._iter,
                        num_episodes=1,
                        max_steps=300,
                        prefix="policy"
                    )
                except Exception as e:
                    print(f"[PGHC] Video recording failed: {e}")

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
                Logger.print(f"[PGHC] Saved hybrid state to {hybrid_file}")

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
                    Logger.print(f"[PGHC] Loaded hybrid state from {hybrid_file}")
            else:
                if Logger:
                    Logger.print(f"[PGHC] Warning: Hybrid state file not found: {hybrid_file}")

else:
    # Placeholder when MimicKit is not available
    class HybridAMPAgentIntegrated:
        """Placeholder - MimicKit not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("MimicKit is required for HybridAMPAgentIntegrated. "
                            "Use HybridAMPAgent for standalone testing.")


if __name__ == "__main__":
    print("Testing PGHC HybridAMPAgent...")

    # Create config with PGHC parameters
    config = {
        "gating_mode": "stability",  # or "fixed" for legacy
        "stability_window": 50,
        "stability_threshold": 0.05,
        "design_learning_rate": 0.01,
        "trust_region_threshold": 0.1,
        "diff_horizon": 50,
        "design_param_init": 0.0,
    }

    # Create agent (standalone mode)
    agent = HybridAMPAgent(config, env=None, device="cpu")

    print(f"\nInitial design param: {agent._parametric_model.get_theta():.4f} rad")
    print(f"Gating mode: {agent._gating_mode}")

    # Test stability gating
    print("\nSimulating training iterations...")
    for i in range(300):
        agent._iter = i
        info = agent.train_iter()

        if i % 50 == 0:
            metrics = agent.get_stability_metrics()
            print(f"Iter {i}: delta_rel={metrics['delta_rel']:.4f}, "
                  f"history_len={metrics['return_history_len']}")

        if "outer_loop_count" in info and info.get("outer_loop_accepted", False):
            print(f"  -> Outer loop ran at iter {i}")

    print("\nPGHC HybridAMPAgent test completed!")
    print(f"Final stability metrics: {agent.get_stability_metrics()}")
    print(f"Trust region stats: {agent.get_trust_region_stats()}")
