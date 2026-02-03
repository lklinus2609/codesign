"""
Parametric G1 Model for Differentiable Morphology Optimization

This module provides a parameterized G1 humanoid model where joint oblique
angles can be optimized via differentiable physics.

Supports both single-parameter (legacy) and multi-parameter (PGHC) modes:
- Single parameter: Hip roll angle offset only (original implementation)
- Multi parameter: N joint oblique angles (thesis Algorithm 1)

Design Parameters:
    theta: Vector of joint oblique angle offsets (radians)
    - Each theta_i represents rotation about X-axis for joint i
    - Symmetric pairs share magnitude (opposite signs for left/right)
    - Bounds: [-0.5236, +0.5236] (±30 degrees per thesis)

Reference: Masters_Thesis.pdf Section III.A.1
"""

import math
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import warp as wp

# Default bounds for design parameters (±30 degrees in radians)
DEFAULT_THETA_MIN = -0.5236
DEFAULT_THETA_MAX = 0.5236

# Lower body joints for 15 DOF optimization
LOWER_BODY_JOINTS = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

# Symmetric pairs for lower body (6 independent parameters)
LOWER_BODY_SYMMETRIC_PAIRS = [
    ("left_hip_pitch_joint", "right_hip_pitch_joint"),
    ("left_hip_roll_joint", "right_hip_roll_joint"),
    ("left_hip_yaw_joint", "right_hip_yaw_joint"),
    ("left_knee_joint", "right_knee_joint"),
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
    ("left_ankle_roll_joint", "right_ankle_roll_joint"),
]

# Legacy single parameter (for backward compatibility)
LEFT_HIP_ROLL_JOINT = "left_hip_roll_joint"
RIGHT_HIP_ROLL_JOINT = "right_hip_roll_joint"


class ParametricG1Model:
    """
    Manages G1 humanoid model with differentiable joint oblique angle parameters.

    This class wraps a Newton model and provides methods to:
    1. Get/set design parameters (joint oblique angle offsets)
    2. Update the model's joint transforms based on parameters
    3. Compute gradients through the parameters

    Supports two modes:
    - Single parameter mode (legacy): Only hip roll angle
    - Multi parameter mode (PGHC): N joint oblique angles with symmetric pairs

    Attributes:
        theta: Current design parameter vector (torch.Tensor with grad)
        theta_min: Lower bound for theta
        theta_max: Upper bound for theta
        n_params: Number of independent design parameters
        joint_names: List of joint names being parameterized
        device: Computation device
    """

    def __init__(
        self,
        device: str = "cuda:0",
        theta_init: float = 0.0,
        theta_min: float = DEFAULT_THETA_MIN,
        theta_max: float = DEFAULT_THETA_MAX,
        joint_names: Optional[List[str]] = None,
        symmetric_pairs: Optional[List[Tuple[str, str]]] = None,
        multi_param_mode: bool = False,
    ):
        """
        Initialize the parametric G1 model.

        Args:
            device: Computation device (e.g., "cuda:0", "cpu")
            theta_init: Initial value for all oblique angle offsets (radians)
            theta_min: Minimum allowed value for theta
            theta_max: Maximum allowed value for theta
            joint_names: List of joints to parameterize (None = hip roll only)
            symmetric_pairs: Pairs of joints sharing parameters (left/right)
            multi_param_mode: If True, use multi-parameter mode
        """
        self.device = device
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.multi_param_mode = multi_param_mode

        # Set up joint configuration
        if joint_names is None or not multi_param_mode:
            # Legacy single parameter mode
            self.joint_names = [LEFT_HIP_ROLL_JOINT, RIGHT_HIP_ROLL_JOINT]
            self.symmetric_pairs = [(LEFT_HIP_ROLL_JOINT, RIGHT_HIP_ROLL_JOINT)]
            self.n_params = 1
            self.multi_param_mode = False
        else:
            # Multi parameter mode
            self.joint_names = joint_names
            self.symmetric_pairs = symmetric_pairs or []
            # Count independent parameters (symmetric pairs count as 1)
            paired_joints = set()
            for left, right in self.symmetric_pairs:
                paired_joints.add(left)
                paired_joints.add(right)
            unpaired = [j for j in joint_names if j not in paired_joints]
            self.n_params = len(self.symmetric_pairs) + len(unpaired)

        # Design parameters as differentiable tensor
        self.theta = torch.zeros(
            self.n_params,
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )
        if theta_init != 0.0:
            with torch.no_grad():
                self.theta.fill_(theta_init)

        # Joint index cache (populated when model is attached)
        self._joint_indices: Dict[str, int] = {}
        self._model = None

        # Store base quaternions for each joint (before parameterization)
        self._base_quaternions: Dict[str, np.ndarray] = {}

        # Map from parameter index to joint names
        self._param_to_joints: List[List[str]] = []
        self._build_param_mapping()

        # Legacy compatibility
        self._left_hip_roll_idx = None
        self._right_hip_roll_idx = None

    def _build_param_mapping(self):
        """Build mapping from parameter indices to joint names."""
        self._param_to_joints = []
        used_joints = set()

        # First, add symmetric pairs
        for left, right in self.symmetric_pairs:
            if left in self.joint_names and right in self.joint_names:
                self._param_to_joints.append([left, right])
                used_joints.add(left)
                used_joints.add(right)

        # Then, add unpaired joints
        for joint in self.joint_names:
            if joint not in used_joints:
                self._param_to_joints.append([joint])

    def attach_model(self, newton_model):
        """
        Attach a Newton model and find joint indices.

        This must be called after the Newton model is finalized but before
        any parameter updates.

        Args:
            newton_model: A finalized Newton Model object

        Raises:
            ValueError: If required joints are not found in the model
        """
        self._model = newton_model
        joint_keys = newton_model.joint_key

        # Find indices for all parameterized joints
        for joint_name in self.joint_names:
            try:
                idx = joint_keys.index(joint_name)
                self._joint_indices[joint_name] = idx
            except ValueError:
                raise ValueError(
                    f"Joint '{joint_name}' not found in model. "
                    f"Available joints: {joint_keys}"
                )

        # Store base quaternions from the model
        joint_X_p_np = newton_model.joint_X_p.numpy()
        for joint_name, idx in self._joint_indices.items():
            # joint_X_p is [N, 7] where [0:3] is position, [3:7] is quaternion
            self._base_quaternions[joint_name] = joint_X_p_np[idx, 3:7].copy()

        # Legacy compatibility: store hip roll indices
        if LEFT_HIP_ROLL_JOINT in self._joint_indices:
            self._left_hip_roll_idx = self._joint_indices[LEFT_HIP_ROLL_JOINT]
        if RIGHT_HIP_ROLL_JOINT in self._joint_indices:
            self._right_hip_roll_idx = self._joint_indices[RIGHT_HIP_ROLL_JOINT]

        print(f"[ParametricG1] Attached model with {len(joint_keys)} joints")
        print(f"[ParametricG1] Parameterizing {len(self.joint_names)} joints "
              f"({self.n_params} independent parameters)")
        if self.multi_param_mode:
            print(f"[ParametricG1] Mode: Multi-parameter (PGHC)")
        else:
            print(f"[ParametricG1] Mode: Single-parameter (legacy)")

    @staticmethod
    def angle_to_x_rotation_quat(theta: float) -> np.ndarray:
        """
        Convert an angle to a quaternion representing rotation about X-axis.

        The quaternion format is (w, x, y, z) matching Warp's convention.

        Args:
            theta: Rotation angle in radians

        Returns:
            Quaternion as numpy array [w, x, y, z]
        """
        half_theta = theta / 2.0
        w = math.cos(half_theta)
        x = math.sin(half_theta)
        y = 0.0
        z = 0.0
        return np.array([w, x, y, z], dtype=np.float32)

    @staticmethod
    def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions (q1 * q2).

        Quaternion format: (w, x, y, z)

        Args:
            q1: First quaternion
            q2: Second quaternion

        Returns:
            Product quaternion (normalized)
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        result = np.array([w, x, y, z], dtype=np.float32)
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        return result

    def compute_joint_quaternion(self, joint_name: str, theta_val: float) -> np.ndarray:
        """
        Compute the quaternion for a joint given a theta value.

        Combines the delta rotation with the base quaternion.

        Args:
            joint_name: Name of the joint
            theta_val: Oblique angle offset in radians

        Returns:
            Combined quaternion as numpy array
        """
        base_quat = self._base_quaternions.get(joint_name)
        if base_quat is None:
            raise ValueError(f"Base quaternion not found for joint: {joint_name}")

        delta_quat = self.angle_to_x_rotation_quat(theta_val)
        return self.quat_multiply(delta_quat, base_quat)

    def update_model_morphology(self, theta_vals: np.ndarray = None):
        """
        Update the Newton model's joint transforms based on current theta.

        This modifies the joint_X_p array in-place.

        Args:
            theta_vals: Optional explicit theta values. If None, uses self.theta

        Raises:
            RuntimeError: If no model is attached
        """
        if self._model is None:
            raise RuntimeError("No model attached. Call attach_model() first.")

        if theta_vals is None:
            theta_vals = self.theta.detach().cpu().numpy()

        # Get current joint_X_p as numpy for modification
        joint_X_p_np = self._model.joint_X_p.numpy()

        # Update each parameterized joint
        for param_idx, joint_list in enumerate(self._param_to_joints):
            if param_idx >= len(theta_vals):
                continue

            theta_val = float(theta_vals[param_idx])

            for i, joint_name in enumerate(joint_list):
                if joint_name not in self._joint_indices:
                    continue

                joint_idx = self._joint_indices[joint_name]

                # For symmetric pairs, second joint may need opposite sign
                # (depending on kinematic convention)
                effective_theta = theta_val
                # Note: For G1, both hips use same sign for symmetric stance
                # Uncomment below if opposite signs needed:
                # if i == 1 and len(joint_list) == 2:
                #     effective_theta = -theta_val

                quat = self.compute_joint_quaternion(joint_name, effective_theta)
                joint_X_p_np[joint_idx, 3:7] = quat

        # Write back to warp array
        self._model.joint_X_p.assign(joint_X_p_np)

    def project_bounds(self, theta_tensor: torch.Tensor) -> torch.Tensor:
        """
        Project theta to valid range [theta_min, theta_max].

        Implements Algorithm 1, Line 24: proj_C(phi - beta * grad)

        Args:
            theta_tensor: Input theta value(s)

        Returns:
            Clamped theta value(s)
        """
        return torch.clamp(theta_tensor, self.theta_min, self.theta_max)

    def get_theta(self, param_idx: int = 0) -> float:
        """Get theta value for a specific parameter index."""
        if param_idx >= self.n_params:
            return 0.0
        return self.theta[param_idx].item()

    def get_theta_all(self) -> np.ndarray:
        """Get all theta values as numpy array."""
        return self.theta.detach().cpu().numpy()

    def get_theta_degrees(self, param_idx: int = 0) -> float:
        """Get theta value in degrees for a specific parameter index."""
        return math.degrees(self.get_theta(param_idx))

    def get_theta_degrees_all(self) -> np.ndarray:
        """Get all theta values in degrees."""
        return np.degrees(self.get_theta_all())

    def set_theta(self, value: float, param_idx: int = 0, update_model: bool = True):
        """
        Set theta to a new value for a specific parameter.

        Args:
            value: New theta value in radians
            param_idx: Parameter index (0 for single-param mode)
            update_model: If True, also update the attached model's morphology
        """
        if param_idx >= self.n_params:
            return

        with torch.no_grad():
            self.theta[param_idx] = value

        if update_model and self._model is not None:
            self.update_model_morphology()

    def set_theta_all(self, values: np.ndarray, update_model: bool = True):
        """
        Set all theta values.

        Args:
            values: Array of theta values (length must match n_params)
            update_model: If True, also update the attached model's morphology
        """
        if len(values) != self.n_params:
            raise ValueError(f"Expected {self.n_params} values, got {len(values)}")

        with torch.no_grad():
            for i, v in enumerate(values):
                self.theta[i] = v

        if update_model and self._model is not None:
            self.update_model_morphology()

    def get_gradient(self, param_idx: int = 0) -> Optional[float]:
        """Get gradient for a specific parameter."""
        if self.theta.grad is None:
            return None
        if param_idx >= self.n_params:
            return None
        return self.theta.grad[param_idx].item()

    def get_gradient_all(self) -> Optional[np.ndarray]:
        """Get all gradients as numpy array."""
        if self.theta.grad is None:
            return None
        return self.theta.grad.detach().cpu().numpy()

    def zero_grad(self):
        """Zero out gradients."""
        if self.theta.grad is not None:
            self.theta.grad.zero_()

    def apply_gradient_step(self, learning_rate: float, param_idx: int = None):
        """
        Apply a gradient descent step to theta.

        Implements Algorithm 1, Line 24:
            phi_{k+1} = proj_C(phi_k - beta * grad_phi L)

        Args:
            learning_rate: Learning rate (beta in Algorithm 1)
            param_idx: Specific parameter to update (None = all)

        Returns:
            Dictionary with update info
        """
        if self.theta.grad is None:
            raise RuntimeError("No gradient computed for theta. Run backward pass first.")

        old_theta = self.get_theta_all().copy()

        with torch.no_grad():
            if param_idx is not None:
                # Update single parameter
                if param_idx < self.n_params:
                    new_val = self.theta[param_idx] - learning_rate * self.theta.grad[param_idx]
                    new_val = torch.clamp(new_val, self.theta_min, self.theta_max)
                    self.theta[param_idx] = new_val
            else:
                # Update all parameters
                new_theta = self.theta - learning_rate * self.theta.grad
                new_theta = self.project_bounds(new_theta)
                self.theta.copy_(new_theta)

        # Update model morphology
        if self._model is not None:
            self.update_model_morphology()

        # Zero gradients for next iteration
        self.zero_grad()

        return {
            "theta_old": old_theta,
            "theta_new": self.get_theta_all(),
            "theta_grad": self.theta.grad.detach().cpu().numpy() if self.theta.grad is not None else None,
            "theta_degrees": self.get_theta_degrees_all(),
        }

    def get_param_joint_names(self, param_idx: int) -> List[str]:
        """Get joint names associated with a parameter index."""
        if param_idx >= len(self._param_to_joints):
            return []
        return self._param_to_joints[param_idx]

    def get_state_dict(self) -> dict:
        """Get state dictionary for saving."""
        return {
            "theta": self.get_theta_all().tolist(),
            "theta_min": self.theta_min,
            "theta_max": self.theta_max,
            "n_params": self.n_params,
            "multi_param_mode": self.multi_param_mode,
            "joint_names": self.joint_names,
            "joint_indices": dict(self._joint_indices),
            # Legacy compatibility
            "left_hip_roll_idx": self._left_hip_roll_idx,
            "right_hip_roll_idx": self._right_hip_roll_idx,
        }

    def load_state_dict(self, state_dict: dict):
        """Load state from dictionary."""
        theta_vals = state_dict.get("theta", [0.0])
        if isinstance(theta_vals, (int, float)):
            theta_vals = [theta_vals]

        # Resize theta if needed
        if len(theta_vals) != self.n_params:
            print(f"[ParametricG1] Warning: theta size mismatch "
                  f"({len(theta_vals)} vs {self.n_params}), resizing")
            if len(theta_vals) < self.n_params:
                theta_vals = list(theta_vals) + [0.0] * (self.n_params - len(theta_vals))
            else:
                theta_vals = theta_vals[:self.n_params]

        self.set_theta_all(np.array(theta_vals), update_model=False)
        self.theta_min = state_dict.get("theta_min", DEFAULT_THETA_MIN)
        self.theta_max = state_dict.get("theta_max", DEFAULT_THETA_MAX)


def create_parametric_model(
    config: dict,
    device: str = "cuda:0"
) -> ParametricG1Model:
    """
    Factory function to create a ParametricG1Model from config.

    Args:
        config: Configuration dictionary (from env config)
        device: Computation device

    Returns:
        Configured ParametricG1Model
    """
    joint_names = config.get("design_joints", None)
    symmetric_pairs = config.get("symmetric_pairs", None)

    # Convert symmetric_pairs from list of lists to list of tuples
    if symmetric_pairs is not None:
        symmetric_pairs = [tuple(pair) for pair in symmetric_pairs]

    theta_min = config.get("design_param_min", DEFAULT_THETA_MIN)
    theta_max = config.get("design_param_max", DEFAULT_THETA_MAX)
    theta_init = config.get("design_param_init", 0.0)

    multi_param = joint_names is not None and len(joint_names) > 2

    return ParametricG1Model(
        device=device,
        theta_init=theta_init,
        theta_min=theta_min,
        theta_max=theta_max,
        joint_names=joint_names,
        symmetric_pairs=symmetric_pairs,
        multi_param_mode=multi_param,
    )


def test_quaternion_math():
    """Unit test for quaternion operations."""
    print("Testing quaternion math...")

    # Test identity
    q_identity = ParametricG1Model.angle_to_x_rotation_quat(0.0)
    assert np.allclose(q_identity, [1, 0, 0, 0]), f"Identity failed: {q_identity}"

    # Test 90 degree rotation
    q_90 = ParametricG1Model.angle_to_x_rotation_quat(math.pi / 2)
    expected_90 = [math.cos(math.pi/4), math.sin(math.pi/4), 0, 0]
    assert np.allclose(q_90, expected_90, atol=1e-6), f"90deg failed: {q_90}"

    # Test quaternion multiplication (two 45deg rotations = 90deg)
    q_45 = ParametricG1Model.angle_to_x_rotation_quat(math.pi / 4)
    q_combined = ParametricG1Model.quat_multiply(q_45, q_45)
    assert np.allclose(q_combined, expected_90, atol=1e-6), f"Multiply failed: {q_combined}"

    print("All quaternion tests passed!")


if __name__ == "__main__":
    test_quaternion_math()

    # Test single parameter mode (legacy)
    print("\n=== Testing Single Parameter Mode (Legacy) ===")
    model_single = ParametricG1Model(device="cpu", theta_init=0.05)
    print(f"N params: {model_single.n_params}")
    print(f"Theta: {model_single.get_theta():.4f} rad ({model_single.get_theta_degrees():.2f} deg)")

    # Test multi parameter mode
    print("\n=== Testing Multi Parameter Mode ===")
    model_multi = ParametricG1Model(
        device="cpu",
        theta_init=0.0,
        joint_names=LOWER_BODY_JOINTS,
        symmetric_pairs=LOWER_BODY_SYMMETRIC_PAIRS,
        multi_param_mode=True,
    )
    print(f"N params: {model_multi.n_params}")
    print(f"Joint names: {model_multi.joint_names}")
    print(f"Param mapping: {model_multi._param_to_joints}")

    # Test setting values
    model_multi.set_theta(0.1, param_idx=0)
    model_multi.set_theta(-0.05, param_idx=1)
    print(f"Theta values: {model_multi.get_theta_all()}")
    print(f"Theta degrees: {model_multi.get_theta_degrees_all()}")

    # Test bounds projection
    large_vals = np.array([1.0, -1.0, 0.5, -0.5, 0.3, -0.3])
    model_multi.set_theta_all(large_vals, update_model=False)
    projected = model_multi.project_bounds(model_multi.theta)
    print(f"Projected: {projected.detach().numpy()}")

    # Test factory function
    print("\n=== Testing Factory Function ===")
    config = {
        "design_joints": LOWER_BODY_JOINTS,
        "symmetric_pairs": [list(p) for p in LOWER_BODY_SYMMETRIC_PAIRS],
        "design_param_init": 0.0,
        "design_param_min": -0.5236,
        "design_param_max": 0.5236,
    }
    model_from_config = create_parametric_model(config, device="cpu")
    print(f"N params from config: {model_from_config.n_params}")

    print("\nParametricG1Model tests completed!")
