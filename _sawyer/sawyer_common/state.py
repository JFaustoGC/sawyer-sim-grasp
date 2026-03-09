"""
state.py — Typed, timestamped robot state tree.

Every sub-state carries its own timestamp so callers can detect staleness.
All state objects are frozen (immutable snapshots).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .geometry import JointAngles, Pose
from .sawyer import GripperState as GripperStateEnum, RobotMode, RobotStatus


# ── Base ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Timestamped:
    """Every sub-state knows when it was sampled."""
    timestamp: float  # time.time() seconds


# ── Sub-states ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ArmState(Timestamped):
    joint_angles: JointAngles
    joint_velocities: JointAngles
    joint_efforts: JointAngles
    endpoint_pose: Optional[Pose]
    endpoint_velocity: Dict[str, Any]
    endpoint_effort: Dict[str, Any]


@dataclass(frozen=True)
class GripperState(Timestamped):
    position: float
    is_grasping: bool
    state: GripperStateEnum
    mode: RobotMode


@dataclass(frozen=True)
class NavigatorState(Timestamped):
    buttons: Dict[str, Any]


@dataclass(frozen=True)
class CuffState(Timestamped):
    lower: bool
    upper: bool
    squeeze: bool


@dataclass(frozen=True)
class RobotStatusState(Timestamped):
    enabled: bool
    stopped: bool
    error: bool
    estop_button: bool


# ── Top-level snapshot ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RobotState(Timestamped):
    """Complete robot state snapshot — immutable, timestamped."""
    arm: ArmState
    gripper: GripperState
    navigator: Optional[NavigatorState]
    cuff: Optional[CuffState]
    robot_status: Optional[RobotStatusState]

    def age(self) -> float:
        """Seconds since this state was captured."""
        return time.time() - self.timestamp

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> RobotState:
        """Parse the raw dict from ZMQ get_full_state / SUB stream."""
        now = d.get('timestamp', time.time())
        robot = d.get('robot', {})
        grip = d.get('gripper', {})

        # Arm
        arm_ts = robot.get('timestamp', now)
        
        # Parse endpoint pose which might be a list [x,y,z,qx,qy,qz,qw] or a dict
        raw_pose = robot.get('endpoint_pose')
        ep_pose = None
        if isinstance(raw_pose, dict):
            ep_pose = Pose.from_dict(raw_pose)
        elif isinstance(raw_pose, (list, tuple)) and len(raw_pose) == 7:
            ep_pose = Pose(
                position=Position.from_list(raw_pose[:3]),
                orientation=Quaternion.from_list(raw_pose[3:])
            )

        arm = ArmState(
            timestamp=arm_ts,
            joint_angles=JointAngles.from_list(robot.get('joint_angles', [0.0] * 7)),
            joint_velocities=JointAngles.from_list(robot.get('joint_velocities', [0.0] * 7)),
            joint_efforts=JointAngles.from_list(robot.get('joint_efforts', [0.0] * 7)),
            endpoint_pose=ep_pose,
            endpoint_velocity=robot.get('endpoint_velocity', {}),
            endpoint_effort=robot.get('endpoint_effort', {}),
        )

        # Gripper
        grip_ts = grip.get('timestamp', now)
        grip_state_str = grip.get('state', 'unknown')
        try:
            grip_enum = GripperStateEnum(grip_state_str)
        except ValueError:
            grip_enum = GripperStateEnum.UNKNOWN
        mode_str = grip.get('mode', 'sim')
        try:
            grip_mode = RobotMode(mode_str)
        except ValueError:
            grip_mode = RobotMode.SIM

        gripper = GripperState(
            timestamp=grip_ts,
            position=float(grip.get('position', 0.0)),
            is_grasping=bool(grip.get('is_grasping', False)),
            state=grip_enum,
            mode=grip_mode,
        )

        # Navigator (optional)
        nav_raw = d.get('navigator')
        navigator = None
        if nav_raw is not None:
            navigator = NavigatorState(
                timestamp=nav_raw.get('timestamp', now),
                buttons=nav_raw,
            )

        # Cuff (optional)
        cuff_raw = d.get('cuff')
        cuff = None
        if cuff_raw is not None:
            cuff = CuffState(
                timestamp=cuff_raw.get('timestamp', now),
                lower=bool(cuff_raw.get('lower', False)),
                upper=bool(cuff_raw.get('upper', False)),
                squeeze=bool(cuff_raw.get('squeeze', False)),
            )

        # Robot status (optional)
        status_raw = d.get('robot_status')
        robot_status = None
        if status_raw is not None:
            robot_status = RobotStatusState(
                timestamp=status_raw.get('timestamp', now),
                enabled=bool(status_raw.get('enabled', False)),
                stopped=bool(status_raw.get('stopped', False)),
                error=bool(status_raw.get('error', False)),
                estop_button=bool(status_raw.get('estop_button', False)),
            )

        return RobotState(
            timestamp=now,
            arm=arm,
            gripper=gripper,
            navigator=navigator,
            cuff=cuff,
            robot_status=robot_status,
        )
