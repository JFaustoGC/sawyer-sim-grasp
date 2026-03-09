"""
sawyer.py — Sawyer-specific constants and enumerations.

Defines all named entities (links, joints, cameras) for the Sawyer robot 
so callers never pass raw strings or magic numbers.
"""

from __future__ import annotations
from enum import Enum


class Joint(str, Enum):
    """The seven joints of the Sawyer right arm, in proximal→distal order."""
    J0 = 'right_j0'
    J1 = 'right_j1'
    J2 = 'right_j2'
    J3 = 'right_j3'
    J4 = 'right_j4'
    J5 = 'right_j5'
    J6 = 'right_j6'

    @classmethod
    def all(cls) -> list[str]:
        """Return list of all 7 joint name strings."""
        return [j.value for j in cls]


class Link(str, Enum):
    """Primary structural links of interest."""
    BASE     = 'right_arm_base_link'
    PEDESTAL = 'base' # Often the world/pedestal connection
    HAND     = 'right_hand'
    WRIST    = 'right_wrist'
    GRIPPER  = 'right_gripper_tip'


class ArmLink(str, Enum):
    """Sequential links of the 7-DOF arm."""
    L0 = 'right_l0'
    L1 = 'right_l1'
    L2 = 'right_l2'
    L3 = 'right_l3'
    L4 = 'right_l4'
    L5 = 'right_l5'
    L6 = 'right_l6'


class HeadLink(str, Enum):
    """Links associated with the head and screen."""
    HEAD   = 'head'
    SCREEN = 'screen'


class GripperLink(str, Enum):
    """Links for end-effector components."""
    BASE = 'right_gripper_base'
    TIP  = 'right_gripper_tip'
    CONNECTOR_PLATE = 'right_connector_plate_base'
    FT_SENSOR = 'right_ft_sensor_link'
    ADAPTER_TOP = 'right_adapter_top'
    ADAPTER_BOTTOM = 'right_adapter_bottom'
    PNEUMATIC_BASE = 'right_pneumatic_gripper_base'
    L_FINGER_MOUNT = 'right_l_finger_mount'
    R_FINGER_MOUNT = 'right_r_finger_mount'
    L_FINGER = 'right_gripper_l_finger'
    R_FINGER = 'right_gripper_r_finger'
    L_FINGER_BASE = 'right_gripper_l_finger_base'
    R_FINGER_BASE = 'right_gripper_r_finger_base'
    CROSSBAR = 'right_gripper_crossbar'
    L_FINGER_TIP = 'right_gripper_l_finger_tip'
    R_FINGER_TIP = 'right_gripper_r_finger_tip'


class Camera(str, Enum):
    """Cameras available on the Sawyer."""
    HEAD = 'head'
    HAND = 'hand'


class Limb(str, Enum):
    """Robot limbs (Sawyer only has a right arm)."""
    RIGHT = 'right'


class GripperState(str, Enum):
    """High-level gripper state."""
    OPEN    = 'open'
    CLOSED  = 'closed'
    UNKNOWN = 'unknown'


class RobotMode(str, Enum):
    """Operating mode of the robot."""
    REAL = 'real'
    SIM  = 'sim'


class RobotStatus(str, Enum):
    """High-level robot lifecycle status."""
    ENABLED  = 'enabled'
    DISABLED = 'disabled'
    STOPPED  = 'stopped'
    ERROR    = 'error'
    ESTOP    = 'estop'


class LightName(str, Enum):
    """Named LEDs on the Sawyer."""
    HEAD_RED    = 'head_red_light'
    HEAD_GREEN  = 'head_green_light'
    HEAD_BLUE   = 'head_blue_light'
    HAND_RED    = 'right_hand_red_light'
    HAND_GREEN  = 'right_hand_green_light'
    HAND_BLUE   = 'right_hand_blue_light'
