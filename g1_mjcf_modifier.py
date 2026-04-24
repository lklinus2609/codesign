"""
G1 MJCF Modifier for GBC (Gradient-Based Co-Design)

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


# Design groups: each group is a list of body names sharing a single scalar design
# parameter. Pairs (2 bodies) enforce L/R symmetry; singletons (1 body) are for
# midline joints (waist).
#
# Lower-body scope: 6 leg pairs = 6 params covering 12 joints.
LOWER_SYMMETRIC_GROUPS = [
    ["left_hip_pitch_link", "right_hip_pitch_link"],
    ["left_hip_roll_link", "right_hip_roll_link"],
    ["left_hip_yaw_link", "right_hip_yaw_link"],
    ["left_knee_link", "right_knee_link"],
    ["left_ankle_pitch_link", "right_ankle_pitch_link"],
    ["left_ankle_roll_link", "right_ankle_roll_link"],
]

# Full-body symmetric scope: 6 leg pairs + 3 waist singletons + 7 arm pairs
# = 16 params covering all 29 G1 joints.
FULL_SYMMETRIC_GROUPS = LOWER_SYMMETRIC_GROUPS + [
    ["waist_yaw_link"],
    ["waist_roll_link"],
    ["torso_link"],  # body hosts the waist_pitch_joint (name mismatch — see BODY_TO_JOINT)
    ["left_shoulder_pitch_link", "right_shoulder_pitch_link"],
    ["left_shoulder_roll_link", "right_shoulder_roll_link"],
    ["left_shoulder_yaw_link", "right_shoulder_yaw_link"],
    ["left_elbow_link", "right_elbow_link"],
    ["left_wrist_roll_link", "right_wrist_roll_link"],
    ["left_wrist_pitch_link", "right_wrist_pitch_link"],
    ["left_wrist_yaw_link", "right_wrist_yaw_link"],
]

DESIGN_GROUPS_BY_SCOPE = {
    "lower": LOWER_SYMMETRIC_GROUPS,
    "full": FULL_SYMMETRIC_GROUPS,
}

# Some G1 bodies host a joint whose name doesn't match the
# body_name.replace("_link","_joint") convention. Map exceptions here.
BODY_TO_JOINT = {
    "torso_link": "waist_pitch_joint",
}


def body_to_joint_name(body_name):
    """Resolve the joint name hosted by a given body, handling G1 exceptions."""
    if body_name in BODY_TO_JOINT:
        return BODY_TO_JOINT[body_name]
    return body_name.replace("_link", "_joint")


def group_param_name(group):
    """Short label for a design group.
    Pairs: uses the 'left_*' body name stripped of '_link'
           (e.g. 'left_hip_pitch') — back-compat with existing wandb keys.
    Singletons: the body name stripped of '_link' (e.g. 'waist_yaw').
    """
    first = group[0]
    return first[:-len("_link")] if first.endswith("_link") else first


# Back-compat aliases (point at lower-body scope, the pre-refactor default).
SYMMETRIC_PAIRS = [tuple(g) for g in LOWER_SYMMETRIC_GROUPS]
NUM_DESIGN_PARAMS = len(LOWER_SYMMETRIC_GROUPS)  # 6


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

    def __init__(self, base_mjcf_path, scope="lower"):
        self.base_mjcf_path = os.path.abspath(base_mjcf_path)
        self.tree = ET.parse(self.base_mjcf_path)
        self.root = self.tree.getroot()
        self.scope = scope
        self.groups = DESIGN_GROUPS_BY_SCOPE[scope]

        # Build name→element map and extract original quaternions
        self._body_map = {}
        for body in self.root.iter("body"):
            name = body.get("name")
            if name:
                self._body_map[name] = body

        # Store original quaternions for all parameterized bodies
        self._original_quats = {}  # body_name → (w, x, y, z)
        for group in self.groups:
            for name in group:
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
            theta_np: numpy array — one oblique angle in radians per design
                group (len == len(self.groups)).
            output_path: where to write the modified XML
        """
        assert len(theta_np) == len(self.groups)

        # Deep copy the tree
        tree = copy.deepcopy(self.tree)
        root = tree.getroot()

        body_map = {}
        for body in root.iter("body"):
            name = body.get("name")
            if name:
                body_map[name] = body

        for i, group in enumerate(self.groups):
            angle = float(theta_np[i])
            delta_q = quat_from_x_rotation(angle)

            for name in group:
                base_q = self._original_quats[name]
                new_q = quat_normalize(quat_multiply(delta_q, base_q))
                quat_str = f"{new_q[0]:.8f} {new_q[1]:.8f} {new_q[2]:.8f} {new_q[3]:.8f}"
                body_map[name].set("quat", quat_str)

        # Write — use same directory as base for relative mesh paths
        tree.write(output_path, xml_declaration=False)

    def generate_env_config(self, mjcf_path, base_env_config_path, output_path):
        """Copy env config YAML, replacing char_file with the modified MJCF path.

        All file paths in the config are resolved to absolute so the config
        works regardless of the consumer's cwd (subprocess or in-process import).

        Args:
            mjcf_path: path to the modified MJCF (should be absolute)
            base_env_config_path: path to the base amp_g1_env.yaml
            output_path: where to write the modified config
        """
        with open(base_env_config_path, "r") as f:
            config = yaml.safe_load(f)

        # Resolve file paths relative to MimicKit root (grandparent of env config dir)
        base_dir = os.path.dirname(os.path.abspath(base_env_config_path))
        mimickit_root = os.path.abspath(os.path.join(base_dir, "..", ".."))

        config["char_file"] = os.path.abspath(mjcf_path)

        # Resolve motion_file if present and relative
        if "motion_file" in config and not os.path.isabs(config["motion_file"]):
            config["motion_file"] = os.path.join(mimickit_root, config["motion_file"])

        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
