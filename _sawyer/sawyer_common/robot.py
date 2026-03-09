from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

from .geometry import JointAngles, JointTrajectory, Pose, JointsLike

class ArmProvider(ABC):
    """Protocol for arm joint control and cartesian information."""
    
    @abstractmethod
    def get_joints(self) -> JointAngles:
        """Return the current joint angles of the robot arm."""
        pass

    @abstractmethod
    def move_to_joints(self, joints: JointsLike, timeout: float = 30.0) -> bool:
        """Move the arm to the specified joint angles, blocking until complete."""
        pass

    @abstractmethod
    def execute_trajectory(self, trajectory: JointTrajectory, rate_hz: float = 100.0) -> bool:
        """Execute a full joint trajectory."""
        pass

    @abstractmethod
    def stream_trajectory(self, trajectory: JointTrajectory, release_time: Optional[float] = None) -> bool:
        """
        Stream a joint trajectory directly to the controllers.
        
        Args:
            trajectory: The path to execute.
            release_time: Optional time (seconds from start) at which to fire the gripper release.
        """
        pass

    @abstractmethod
    def get_pose(self) -> Optional[Pose]:
        """Return the current Cartesian pose of the end-effector."""
        pass


class GripperProvider(ABC):
    """Protocol for end-effector manipulation."""
    
    @abstractmethod
    def open_gripper(self) -> bool:
        """Open the end-effector."""
        pass

    @abstractmethod
    def close_gripper(self) -> bool:
        """Close the end-effector."""
        pass


class DisplayProvider(ABC):
    """Protocol for screen manipulation."""
    
    @abstractmethod
    def display_image(self, image: np.ndarray) -> bool:
        """Display an OpenCV/NumPy image array on the robot's screen."""
        pass
        
    @abstractmethod
    def clear_display(self) -> bool:
        """Clear the robot's screen."""
        pass


class LightsProvider(ABC):
    """Protocol for robot LEDs."""
    
    @abstractmethod
    def set_lights(self, color: str) -> bool:
        """Set the robot's LEDs (e.g. 'red', 'green', 'blue', 'off')."""
        pass


class ButtonProvider(ABC):
    """Protocol for reading physical robot buttons and dials."""

    @abstractmethod
    def get_navigator_state(self) -> Dict[str, Any]:
        """Get the state of the navigator wheel and buttons."""
        pass

    @abstractmethod
    def get_cuff_state(self) -> Dict[str, Any]:
        """Get the state of the arm cuff buttons."""
        pass


class RobotLifecycle(ABC):
    """Protocol for robot state and connection management."""
    
    @abstractmethod
    def enable(self) -> bool:
        """Enable all joints and motors."""
        pass

    @abstractmethod
    def disable(self) -> bool:
        """Disable all joints and motors."""
        pass

    @abstractmethod
    def reset(self) -> bool:
        """Reset faults and clear errors."""
        pass

    @abstractmethod
    def stop(self) -> bool:
        """Trigger an emergency stop."""
        pass

    @abstractmethod
    def sleep(self, duration: float) -> None:
        """Sleep for the specified duration, maintaining current state."""
        pass

    @abstractmethod
    def get_robot_status(self) -> Dict[str, Any]:
        """Get high-level robot health/status dictionary."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Cleanly close the connection to the robot."""
        pass

    def __enter__(self) -> "RobotLifecycle":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class Teleportable(ABC):
    """Protocol for robots that can be instantly 'teleported' to a state (Simulation)."""
    
    @abstractmethod
    def teleport_joints(self, joints: JointsLike) -> None:
        """Instantly set joint positions, bypassing physics/controllers."""
        pass

    @abstractmethod
    def teleport_gripper(self, position: float) -> None:
        """Instantly set gripper position."""
        pass


class Robot(ArmProvider, GripperProvider, DisplayProvider, LightsProvider, ButtonProvider, RobotLifecycle, ABC):
    """
    Unified abstract Robot Interface implementing all physical capabilities.
    """
    pass


class SimRobot(ArmProvider, GripperProvider, Teleportable, RobotLifecycle, ABC):
    """
    Unified abstract SimRobot Interface implementing all simulated physical capabilities.
    """
    pass
