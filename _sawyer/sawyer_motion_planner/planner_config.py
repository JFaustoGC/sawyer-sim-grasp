from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class MotionProfile(str, Enum):
    SLOW    = "slow"
    MEDIUM  = "medium"
    FAST    = "fast"
    EXPRESS = "express"


# Specific tuned limits for Sawyer joints
# Each array is [j0, j1, j2, j3, j4, j5, j6]
LIMIT_PROFILES = {
    MotionProfile.SLOW: {
        "vel": np.array([0.37, 0.2825, 0.415, 0.415, 0.74, 0.74, 0.965]),
        "acc": np.array([1.5, 1.5, 3.0, 3.0, 3.0, 3.0, 3.0])
    },
    MotionProfile.MEDIUM: {
        "vel": np.array([0.88, 0.678, 0.996, 0.996, 1.776, 1.776, 2.316]),
        "acc": np.array([3.5, 2.5, 5.0, 5.0, 5.0, 5.0, 5.0])
    },
    MotionProfile.FAST: {
        "vel": np.array([1.48, 1.13, 1.66, 1.66, 2.96, 2.96, 3.86]),
        "acc": np.array([7.0, 5.0, 8.0, 8.0, 8.0, 8.0, 8.0])
    },
    MotionProfile.EXPRESS: {
        "vel": np.array([1.48, 1.13, 1.66, 1.66, 2.96, 2.96, 3.86]),
        "acc": np.array([10.0, 8.0, 10.0, 10.0, 12.0, 12.0, 12.0])
    },
}

@dataclass
class PlannerConfig:
    dt: float = 0.01  # 100Hz
    
    # Default profile
    default_profile: MotionProfile = MotionProfile.SLOW
    
    # Weights for CasADi (used for complex tasks like tossing)
    w_smooth: float = 1.0      
    w_slack: float  = 10000.0  
    w_track: float  = 10.0     
    w_goal: float   = 5000.0   
    w_pos: float    = 5000.0   
    w_ori: float    = 5000.0   
    
    # Posture regularization (keeps joints near a natural config)
    w_reg: float = 0.1
    q_natural: np.ndarray = field(default_factory=lambda: np.array([0.0, -0.5, 0.0, 1.0, 0.0, -0.5, 0.0]))

    # Cartesian solver settings
    solver_steps: int = 30
    solver_max_iter: int = 500
    solver_max_cpu_time: float = 3.0

    # Global safety limits (capped at express)
    max_vel: np.ndarray = field(default_factory=lambda: LIMIT_PROFILES[MotionProfile.EXPRESS]["vel"])
    max_acc: np.ndarray = field(default_factory=lambda: LIMIT_PROFILES[MotionProfile.EXPRESS]["acc"])
