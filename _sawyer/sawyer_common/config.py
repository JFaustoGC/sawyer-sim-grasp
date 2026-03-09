"""
config.py — Typed configuration for the Sawyer robot client.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from .sawyer import RobotMode


@dataclass
class ConnectionConfig:
    host: str = 'localhost'
    command_port: int = 5555
    state_port: int = 5556


@dataclass
class GripperConfig:
    device_id: str = 'stp_021709TP00448'
    max_position: float = 0.041667
    min_position: float = 0.0


@dataclass
class RobotConfig:
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    mode: RobotMode = RobotMode.SIM
    gripper: GripperConfig = field(default_factory=GripperConfig)
    urdf_path: Optional[str] = None
    state_rate_hz: float = 100.0

    @staticmethod
    def from_env() -> RobotConfig:
        """Build config from environment variables (backwards compatible)."""
        mode_str = os.environ.get('ROBOT_MODE', 'sim').lower()
        try:
            mode = RobotMode(mode_str)
        except ValueError:
            mode = RobotMode.SIM

        conn = ConnectionConfig(
            host=os.environ.get('ROBOT_HOST', 'localhost'),
            command_port=int(os.environ.get('ROBOT_COMMAND_PORT', '5555')),
            state_port=int(os.environ.get('ROBOT_STATE_PORT', '5556')),
        )

        gripper = GripperConfig(
            device_id=os.environ.get('GRIPPER_DEVICE_ID', 'stp_021709TP00448'),
        )

        return RobotConfig(
            connection=conn,
            mode=mode,
            gripper=gripper,
            urdf_path=os.environ.get('ROBOT_URDF_PATH'),
        )
