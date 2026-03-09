"""
constants.py — Physical and operational parameters for the Sawyer robot.
"""

# ── Robot Physical Parameters ────────────────────────────────────────────────

# Sawyer link lengths (meters)
L1 = 0.4000
L2 = 0.4000
L3_WRIST = 0.13375
GRIPPER_LENGTH = 0.2088
L3_TOTAL = L3_WRIST + GRIPPER_LENGTH

# Offsets from origin
BASE_OFFSET_XYZ = [0.081, 0.0, 0.317]

# Gripper parameters
GRIPPER_NOMINAL_WIDTH = 0.01  # For collision checks
GRIPPER_MAX_WIDTH = 0.02

# Standard 7-DOF joint limits (Sawyer)
SAWYER_JOINT_LIMITS_MIN = [-3.0503, -3.8095, -3.0426, -3.0439, -2.9761, -2.9761, -4.7124]
SAWYER_JOINT_LIMITS_MAX = [3.0503, 2.2736, 3.0426, 3.0439, 2.9761, 2.9761, 4.7124]

# Motor Torque Limits (Nm) - Approximate for Sawyer
SAWYER_TORQUE_LIMITS = [80.0, 80.0, 40.0, 40.0, 9.0, 9.0, 9.0]


# ── Trajectory & Planning ───────────────────────────────────────────────────

PLANNING_DT = 0.01  # Default timestep (seconds)

# Tossing-specific 3-DOF limits (from tossing_config.py)
TOSSING_POS_LIMITS_MIN = [-3.8095, -3.0439, -2.9761]
TOSSING_POS_LIMITS_MAX = [ 2.2736,  3.0439,  2.9761]

TOSSING_PROFILES = {
    "slow": {
        "vel": [0.2825, 0.415, 0.74],
        "accel": [1.5, 3.0, 3.0],
    },
    "medium": {
        "vel": [0.678, 0.996, 1.776],
        "accel": [2.5, 5.0, 5.0],
    },
    "fast": {
        "vel": [1.13, 1.66, 2.96],
        "accel": [5.0, 8.0, 8.0],
    },
    "express": {
        "vel": [1.13, 1.66, 2.96],
        "accel": [8.0, 10.0, 12.0],
    },
}
