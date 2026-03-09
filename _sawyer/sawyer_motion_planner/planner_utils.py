"""
planner_utils.py — Shared helpers for SawyerPlanner and TossPlanner.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from sawyer_common.geometry import JointAngles
from sawyer_common.sawyer import Joint
from .kinematics import CasadiKinematics
from .planner_config import LIMIT_PROFILES, MotionProfile


def _ensure_joint_array(q: Union[np.ndarray, JointAngles]) -> np.ndarray:
    """Convert JointAngles or array-like q to a numpy array."""
    if hasattr(q, "to_array"):
        return np.asarray(q.to_array(), dtype=float)
    return np.asarray(q, dtype=float)


def min_jerk_poly(
    tau: np.ndarray, duration: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """5th-order minimum-jerk: s, sd, sdd at normalized tau in [0,1]."""
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
    sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / duration**2
    return s, sd, sdd


def get_profile_limits(
    profile: MotionProfile, joint_indices: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice LIMIT_PROFILES by active joint indices."""
    limits = LIMIT_PROFILES[profile]
    v_lim = np.array([limits["vel"][i] for i in joint_indices])
    a_lim = np.array([limits["acc"][i] for i in joint_indices])
    return v_lim, a_lim


def expand_to_7dof(
    q_reduced: np.ndarray,
    qd_reduced: np.ndarray,
    qdd_reduced: np.ndarray,
    active_joints: List[int],
    locked_values: Dict[int, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand reduced-DOF arrays (M, n_active) to 7-DOF arrays (M, 7)."""
    M = len(q_reduced)
    Q7 = np.zeros((M, 7))
    Qd7 = np.zeros((M, 7))
    Qdd7 = np.zeros((M, 7))
    for idx, val in locked_values.items():
        Q7[:, idx] = val
    for local_i, global_i in enumerate(active_joints):
        Q7[:, global_i] = q_reduced[:, local_i]
        Qd7[:, global_i] = qd_reduced[:, local_i]
        Qdd7[:, global_i] = qdd_reduced[:, local_i]
    return Q7, Qd7, Qdd7


def build_reduced_kinematics(
    full_kin: CasadiKinematics,
    active_joints: List[int],
    q_full: Union[np.ndarray, JointAngles],
    locked_overrides: Optional[Dict[str, float]] = None,
) -> Tuple[CasadiKinematics, Dict[int, float]]:
    """
    Build a reduced-DOF kinematics model with locked joints.

    Returns (reduced_kin, locked_values) where locked_values maps
    global joint index -> locked value.
    """
    q_full_arr = _ensure_joint_array(q_full)
    all_joint_names = Joint.all()
    active_names = [all_joint_names[i] for i in active_joints]
    locked_joints = {
        all_joint_names[i]: float(q_full_arr[i])
        for i in range(7)
        if i not in active_joints
    }
    if locked_overrides:
        locked_joints.update(locked_overrides)

    locked_by_index = {
        all_joint_names.index(jname): val
        for jname, val in locked_joints.items()
    }

    kin = CasadiKinematics(
        full_kin.urdf_path,
        base_link=full_kin.base_link,
        end_link=full_kin.end_link,
        active_joints=active_names,
        locked_joints=locked_joints,
    )
    return kin, locked_by_index


def trapezoidal_time(
    dq: np.ndarray, v_lim: np.ndarray, a_lim: np.ndarray
) -> float:
    """Minimum time for trapezoidal velocity profile (per joint, take max)."""
    t_candidates = []
    for i in range(len(dq)):
        d = abs(dq[i])
        if d < 1e-8:
            t_candidates.append(0.0)
            continue
        t_acc = v_lim[i] / a_lim[i]
        d_acc = 0.5 * a_lim[i] * t_acc**2
        if 2 * d_acc >= d:
            t_candidates.append(2 * np.sqrt(d / a_lim[i]))
        else:
            d_coast = d - 2 * d_acc
            t_candidates.append(2 * t_acc + d_coast / v_lim[i])
    return max(t_candidates) if t_candidates else 0.0
