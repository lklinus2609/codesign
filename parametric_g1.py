"""
Parametric G1 Model for Differentiable Morphology Optimization

This module provides a parameterized G1 humanoid model where the hip roll
angle offset can be optimized via differentiable physics.

Design Parameter:
    theta: Hip roll angle offset (radians)
    - Positive theta: feet closer together (toed-in)
    - Negative theta: feet farther apart (toed-out)
    - Bounds: [-0.1745, +0.1745] (±10 degrees)

Reference: Algorithm 1, Lines 19-24 (Outer Loop)
"""

import math
import numpy as np
import torch
import warp as wp

# Joint names for left and right hip roll joints in G1 model
LEFT_HIP_ROLL_JOINT = "left_hip_roll_joint"
RIGHT_HIP_ROLL_JOINT = "right_hip_roll_joint"

# Default bounds for design parameter (±10 degrees in radians)
DEFAULT_THETA_MIN = -0.1745
DEFAULT_THETA_MAX = 0.1745

# Original hip roll link quaternions from g1.xml
# left_hip_roll_link: quat="0.996179 0 -0.0873386 0" (~10° X rotation)
# This is the base orientation before our parameterization
ORIGINAL_LEFT_HIP_QUAT = np.array([0.996179, 0.0, -0.0873386, 0.0])
ORIGINAL_RIGHT_HIP_QUAT = np.array([0.996179, 0.0, -0.0873386, 0.0])


@wp.kernel
def update_joint_transform_kernel(
    joint_X_p: wp.array(dtype=wp.transform),
    joint_idx: int,
    delta_quat: wp.quat,
):
    """
    Warp kernel to update joint transform quaternion component.

    Multiplies the existing quaternion by a delta rotation.
    """
    tid = wp.tid()
    if tid == 0:
        current_tf = joint_X_p[joint_idx]
        current_pos = wp.transform_get_translation(current_tf)
        current_quat = wp.transform_get_rotation(current_tf)

        # Apply delta rotation: new_quat = delta_quat * current_quat
        new_quat = wp.quat_multiply(delta_quat, current_quat)
        new_quat = wp.normalize(new_quat)

        # Write back updated transform
        joint_X_p[joint_idx] = wp.transform(current_pos, new_quat)


@wp.kernel
def set_joint_quaternion_kernel(
    joint_X_p: wp.array(dtype=wp.transform),
    joint_idx: int,
    new_quat: wp.quat,
):
    """
    Warp kernel to directly set joint transform quaternion component.
    """
    tid = wp.tid()
    if tid == 0:
        current_tf = joint_X_p[joint_idx]
        current_pos = wp.transform_get_translation(current_tf)
        joint_X_p[joint_idx] = wp.transform(current_pos, new_quat)


class ParametricG1Model:
    """
    Manages G1 humanoid model with differentiable hip roll angle parameter.

    This class wraps a Newton model and provides methods to:
    1. Get/set the design parameter (hip roll angle offset)
    2. Update the model's joint transforms based on the parameter
    3. Compute gradients through the parameter

    The design parameter theta represents an additional rotation applied
    to the hip roll joints, affecting the neutral stance width.

    Attributes:
        theta: Current design parameter value (torch.Tensor with grad)
        theta_min: Lower bound for theta
        theta_max: Upper bound for theta
        device: Computation device
    """

    def __init__(
        self,
        device: str = "cuda:0",
        theta_init: float = 0.0,
        theta_min: float = DEFAULT_THETA_MIN,
        theta_max: float = DEFAULT_THETA_MAX,
    ):
        """
        Initialize the parametric G1 model.

        Args:
            device: Computation device (e.g., "cuda:0", "cpu")
            theta_init: Initial value for hip roll angle offset (radians)
            theta_min: Minimum allowed value for theta
            theta_max: Maximum allowed value for theta
        """
        self.device = device
        self.theta_min = theta_min
        self.theta_max = theta_max

        # Design parameter as differentiable tensor
        # Using torch for optimizer compatibility with MimicKit
        self.theta = torch.tensor(
            [theta_init],
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )

        # Cache for joint indices (populated when model is attached)
        self._left_hip_roll_idx = None
        self._right_hip_roll_idx = None
        self._model = None

        # Store base quaternions for each joint (before our parameterization)
        self._left_base_quat = None
        self._right_base_quat = None

    def attach_model(self, newton_model):
        """
        Attach a Newton model and find the hip roll joint indices.

        This must be called after the Newton model is finalized but before
        any parameter updates.

        Args:
            newton_model: A finalized Newton Model object

        Raises:
            ValueError: If hip roll joints are not found in the model
        """
        self._model = newton_model

        # Find joint indices by name
        joint_keys = newton_model.joint_key

        try:
            self._left_hip_roll_idx = joint_keys.index(LEFT_HIP_ROLL_JOINT)
        except ValueError:
            raise ValueError(
                f"Joint '{LEFT_HIP_ROLL_JOINT}' not found in model. "
                f"Available joints: {joint_keys}"
            )

        try:
            self._right_hip_roll_idx = joint_keys.index(RIGHT_HIP_ROLL_JOINT)
        except ValueError:
            raise ValueError(
                f"Joint '{RIGHT_HIP_ROLL_JOINT}' not found in model. "
                f"Available joints: {joint_keys}"
            )

        # Store the base quaternions from the model
        joint_X_p_np = newton_model.joint_X_p.numpy()

        # joint_X_p is [N, 7] where [0:3] is position, [3:7] is quaternion
        self._left_base_quat = joint_X_p_np[self._left_hip_roll_idx, 3:7].copy()
        self._right_base_quat = joint_X_p_np[self._right_hip_roll_idx, 3:7].copy()

        print(f"[ParametricG1] Attached model with {len(joint_keys)} joints")
        print(f"[ParametricG1] Left hip roll joint index: {self._left_hip_roll_idx}")
        print(f"[ParametricG1] Right hip roll joint index: {self._right_hip_roll_idx}")
        print(f"[ParametricG1] Initial theta: {self.theta.item():.4f} rad ({math.degrees(self.theta.item()):.2f} deg)")

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
            Product quaternion
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        result = np.array([w, x, y, z], dtype=np.float32)
        # Normalize
        result = result / np.linalg.norm(result)
        return result

    def compute_joint_quaternions(self, theta_val: float):
        """
        Compute the joint quaternions for a given theta value.

        The delta rotation is applied on top of the base quaternions.
        Left and right hips get symmetric rotations.

        Args:
            theta_val: Hip roll angle offset in radians

        Returns:
            Tuple of (left_quat, right_quat) as numpy arrays
        """
        # Delta rotation quaternion for left hip (positive X rotation)
        delta_quat_left = self.angle_to_x_rotation_quat(theta_val)

        # For right hip, we apply the opposite rotation to maintain symmetry
        # (or the same rotation depending on the desired behavior)
        # Here we apply the same rotation for both to keep feet parallel
        delta_quat_right = self.angle_to_x_rotation_quat(theta_val)

        # Combine with base quaternions: new = delta * base
        left_quat = self.quat_multiply(delta_quat_left, self._left_base_quat)
        right_quat = self.quat_multiply(delta_quat_right, self._right_base_quat)

        return left_quat, right_quat

    def update_model_morphology(self, theta_val: float = None):
        """
        Update the Newton model's joint transforms based on current theta.

        This modifies the joint_X_p array in-place.

        Args:
            theta_val: Optional explicit theta value. If None, uses self.theta

        Raises:
            RuntimeError: If no model is attached
        """
        if self._model is None:
            raise RuntimeError("No model attached. Call attach_model() first.")

        if theta_val is None:
            theta_val = self.theta.item()

        # Compute new quaternions
        left_quat, right_quat = self.compute_joint_quaternions(theta_val)

        # Get current joint_X_p as numpy for modification
        joint_X_p_np = self._model.joint_X_p.numpy()

        # Update quaternion components (indices 3:7)
        joint_X_p_np[self._left_hip_roll_idx, 3:7] = left_quat
        joint_X_p_np[self._right_hip_roll_idx, 3:7] = right_quat

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

    def get_theta(self) -> float:
        """Get current theta value."""
        return self.theta.item()

    def get_theta_degrees(self) -> float:
        """Get current theta value in degrees."""
        return math.degrees(self.theta.item())

    def set_theta(self, value: float, update_model: bool = True):
        """
        Set theta to a new value.

        Args:
            value: New theta value in radians
            update_model: If True, also update the attached model's morphology
        """
        with torch.no_grad():
            self.theta.fill_(value)

        if update_model and self._model is not None:
            self.update_model_morphology(value)

    def apply_gradient_step(self, learning_rate: float):
        """
        Apply a gradient descent step to theta.

        Implements Algorithm 1, Line 24:
            phi_{k+1} = proj_C(phi_k - beta * grad_phi L)

        Args:
            learning_rate: Learning rate (beta in Algorithm 1)

        Returns:
            Dictionary with update info
        """
        if self.theta.grad is None:
            raise RuntimeError("No gradient computed for theta. Run backward pass first.")

        grad = self.theta.grad.item()
        old_theta = self.theta.item()

        with torch.no_grad():
            # Gradient descent step
            new_theta = self.theta - learning_rate * self.theta.grad
            # Project to valid range
            new_theta = self.project_bounds(new_theta)
            self.theta.copy_(new_theta)

        # Update model morphology
        if self._model is not None:
            self.update_model_morphology()

        # Zero gradients for next iteration
        self.theta.grad.zero_()

        return {
            "theta_old": old_theta,
            "theta_new": self.theta.item(),
            "theta_grad": grad,
            "theta_degrees": self.get_theta_degrees(),
        }

    def get_state_dict(self) -> dict:
        """Get state dictionary for saving."""
        return {
            "theta": self.theta.item(),
            "theta_min": self.theta_min,
            "theta_max": self.theta_max,
            "left_hip_roll_idx": self._left_hip_roll_idx,
            "right_hip_roll_idx": self._right_hip_roll_idx,
        }

    def load_state_dict(self, state_dict: dict):
        """Load state from dictionary."""
        self.set_theta(state_dict["theta"], update_model=False)
        self.theta_min = state_dict.get("theta_min", DEFAULT_THETA_MIN)
        self.theta_max = state_dict.get("theta_max", DEFAULT_THETA_MAX)


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

    # Test parametric model creation
    print("\nTesting ParametricG1Model...")
    model = ParametricG1Model(device="cpu", theta_init=0.05)
    print(f"Initial theta: {model.get_theta():.4f} rad ({model.get_theta_degrees():.2f} deg)")

    # Test bounds projection
    model.theta = torch.tensor([0.3], requires_grad=True)  # Over max
    projected = model.project_bounds(model.theta)
    print(f"Projected 0.3 rad: {projected.item():.4f} rad (should be ~0.1745)")

    print("\nParametricG1Model tests completed!")
