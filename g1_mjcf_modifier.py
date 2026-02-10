"""
G1 MJCF Modifier for PGHC Co-Design

Parses the base G1 MJCF and modifies body frame quaternions for 6 symmetric
lower-body joint pairs based on design parameter theta (oblique angles).

Each theta[i] adds an X-axis rotation to the body frame of one symmetric pair:
  new_quat = quat_from_x_rotation(theta[i]) * original_quat

MuJoCo quaternion convention: (w, x, y, z)
"""

import copy
import math
import os
import xml.etree.ElementTree as ET
import numpy as np
import yaml


# 6 symmetric pairs of body names whose frame quaternions we modify
SYMMETRIC_PAIRS = [
    ("left_hip_pitch_link", "right_hip_pitch_link"),
    ("left_hip_roll_link", "right_hip_roll_link"),
    ("left_hip_yaw_link", "right_hip_yaw_link"),
    ("left_knee_link", "right_knee_link"),
    ("left_ankle_pitch_link", "right_ankle_pitch_link"),
    ("left_ankle_roll_link", "right_ankle_roll_link"),
]

NUM_DESIGN_PARAMS = len(SYMMETRIC_PAIRS)  # 6


def quat_from_x_rotation(angle):
    """Quaternion for rotation around X axis. Returns (w, x, y, z)."""
    half = angle * 0.5
    return (math.cos(half), math.sin(half), 0.0, 0.0)


def quat_multiply(q1, q2):
    """Hamilton product q1 * q2. Both in (w, x, y, z) format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )


def quat_normalize(q):
    """Normalize quaternion (w, x, y, z)."""
    w, x, y, z = q
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (w/norm, x/norm, y/norm, z/norm)


class G1MJCFModifier:
    """Modifies G1 MJCF body frame quaternions for co-design."""

    def __init__(self, base_mjcf_path):
        self.base_mjcf_path = os.path.abspath(base_mjcf_path)
        self.tree = ET.parse(self.base_mjcf_path)
        self.root = self.tree.getroot()

        # Build name→element map and extract original quaternions
        self._body_map = {}
        for body in self.root.iter("body"):
            name = body.get("name")
            if name:
                self._body_map[name] = body

        # Store original quaternions for all parameterized bodies
        self._original_quats = {}  # body_name → (w, x, y, z)
        for left, right in SYMMETRIC_PAIRS:
            for name in (left, right):
                elem = self._body_map.get(name)
                if elem is None:
                    raise ValueError(f"Body '{name}' not found in MJCF")
                quat_str = elem.get("quat")
                if quat_str:
                    vals = [float(v) for v in quat_str.split()]
                    self._original_quats[name] = tuple(vals)  # (w, x, y, z)
                else:
                    # No quat means identity
                    self._original_quats[name] = (1.0, 0.0, 0.0, 0.0)

    def generate(self, theta_np, output_path):
        """Write modified MJCF to output_path.

        Args:
            theta_np: numpy array of shape (6,) — oblique angles in radians
            output_path: where to write the modified XML
        """
        assert len(theta_np) == NUM_DESIGN_PARAMS

        # Deep copy the tree
        tree = copy.deepcopy(self.tree)
        root = tree.getroot()

        body_map = {}
        for body in root.iter("body"):
            name = body.get("name")
            if name:
                body_map[name] = body

        for i, (left, right) in enumerate(SYMMETRIC_PAIRS):
            angle = float(theta_np[i])
            delta_q = quat_from_x_rotation(angle)

            for name in (left, right):
                base_q = self._original_quats[name]
                new_q = quat_normalize(quat_multiply(delta_q, base_q))
                quat_str = f"{new_q[0]:.8f} {new_q[1]:.8f} {new_q[2]:.8f} {new_q[3]:.8f}"
                body_map[name].set("quat", quat_str)

        # Write — use same directory as base for relative mesh paths
        tree.write(output_path, xml_declaration=False)

    def generate_env_config(self, mjcf_path, base_env_config_path, output_path):
        """Copy env config YAML, replacing char_file with the modified MJCF path.

        Args:
            mjcf_path: path to the modified MJCF
            base_env_config_path: path to the base amp_g1_env.yaml
            output_path: where to write the modified config
        """
        with open(base_env_config_path, "r") as f:
            config = yaml.safe_load(f)

        config["char_file"] = mjcf_path
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
