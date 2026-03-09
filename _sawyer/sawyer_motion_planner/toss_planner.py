"""
toss_planner.py — Simplified, fixed-endpoint tossing trajectory planner.

Assumptions
-----------
* Robot is already at q_start when planning begins (no approach motion here).
* J0 is pre-aligned to face the target; J2, J4 = 0; J6 = 1.766.
* Active joints default to [J1, J3, J5] (indices 1, 3, 5).
  Pass active_joints=[5] for single-joint (J5-only) bench tests.
* Release position AND velocity are hard constraints.
* A braking phase is appended analytically after the throw.
* The planner returns a TossResult containing the full JointTrajectory
  (throw + brake, sampled at 100Hz) and the release_index into that
  trajectory.  The release_index is where the gripper should fire.

Primary API — takes joint configs directly:

    result = planner.plan_toss(
        q_start_full   = q_now,          # 7-DOF start config (hard constraint)
        q_release_full = q_rel,          # 7-DOF release config (hard constraint)
        v_release      = 3.2,            # required EE speed at release (m/s)
        theta_release  = np.pi / 4,      # release angle above horizontal
    )

Convenience API — takes Cartesian positions, does IK internally:

    result = planner.plan_toss_from_positions(
        target_xyz      = np.array([1.2, 0.0, 0.1]),
        release_pos_xyz = np.array([0.5, 0.0, 0.4]),
        q_start_full    = q_now,
        theta_release   = np.pi / 4,
    )
"""

from __future__ import annotations

import numpy as np
import casadi as ca
from dataclasses import dataclass
from scipy.interpolate import interp1d
from typing import Optional, List, Union

from sawyer_common.geometry import JointAngles, JointTrajectory
from sawyer_common.sawyer import Joint
from .kinematics import CasadiKinematics
from .planner_config import LIMIT_PROFILES, MotionProfile, PlannerConfig
from .planner_utils import get_profile_limits, build_reduced_kinematics, expand_to_7dof


def _ensure_joint_array(q: Union[np.ndarray, JointAngles]) -> np.ndarray:
    """Convert a JointAngles or array-like object to a numpy array."""
    if hasattr(q, "to_array"):
        return np.asarray(q.to_array(), dtype=float)
    return np.asarray(q, dtype=float)


# ── Physics helper ────────────────────────────────────────────────────────────

def compute_release_speed(
    target_xyz: np.ndarray,
    release_pos_world: np.ndarray,
    theta_rad: float = np.pi / 4,
    g: float = 9.81,
) -> float:
    """
    Return the required EE speed (m/s) to hit target_xyz from release_pos_world
    at angle theta_rad above horizontal, using projectile motion.

    Operates in the sagittal plane — only horizontal distance and height delta
    matter; lateral offset is assumed to be handled by J0 alignment.

    Raises ValueError if the target is geometrically unreachable at this angle.
    """
    dx = target_xyz[0] - release_pos_world[0]
    dy = target_xyz[1] - release_pos_world[1]
    R  = np.sqrt(dx**2 + dy**2)          # horizontal range in the sagittal plane
    H  = target_xyz[2] - release_pos_world[2]   # signed height delta

    denom = 2 * np.cos(theta_rad)**2 * (R * np.tan(theta_rad) - H)
    if denom <= 0:
        raise ValueError(
            f"Target unreachable at theta={np.degrees(theta_rad):.1f}°  "
            f"R={R:.3f}m  H={H:.3f}m — increase angle or reduce target height."
        )
    return float(np.sqrt(g * R**2 / denom))


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class TossResult:
    """Output of TossPlanner.plan_toss."""
    trajectory: JointTrajectory
    release_index: int          # index into trajectory at which to fire the gripper
    v_achieved: float           # EE speed at release (m/s) — for verification
    T_throw: float              # optimised throw duration (s)
    T_brake: float              # appended brake duration (s)


# ── Planner ───────────────────────────────────────────────────────────────────

class TossPlanner:
    """
    Fixed-endpoint tossing trajectory optimiser.

    Parameters
    ----------
    kinematics : CasadiKinematics
        Full 7-DOF model — the planner will build its own reduced sub-model
        from the active_joints argument at plan time.
    config : PlannerConfig, optional
    """

    # Locked non-active joints (world frame values for a standard toss pose)
    LOCKED_DEFAULTS = {
        Joint.J0: 0.0,
        Joint.J2: 0.0,
        Joint.J4: 0.0,
        Joint.J6: 1.766,
    }

    def __init__(self, kinematics: CasadiKinematics, config: Optional[PlannerConfig] = None):
        self._full_kin = kinematics
        self.cfg = config or PlannerConfig()

    # ── Public API ────────────────────────────────────────────────────────────

    def align_j0(self, target: np.ndarray) -> float:
        """
        Return the J0 angle needed to face target (world XY).
        Always returns 0.0 for now — real impl: arctan2(y, x).
        Move the robot to this angle before calling plan_toss.
        """
        return 0.0

    def plan_toss_from_positions(
        self,
        target_xyz: np.ndarray,
        release_pos_xyz: np.ndarray,
        q_start_full: np.ndarray,
        theta_release: float = np.pi / 4,
        active_joints: List[int] = [1, 3, 5],
        **kwargs,
    ) -> Optional[TossResult]:
        """
        Convenience wrapper: compute v_release from projectile physics and the
        release config via IK, then call plan_toss.

        Args:
            target_xyz:       Cup position in world frame.
            release_pos_xyz:  Desired EE Cartesian position at release.
            q_start_full:     (7,) start joint config.
            theta_release:    Release angle above horizontal (rad).
            active_joints:    Indices of joints to optimise.
            **kwargs:         Forwarded to plan_toss (N, T_max, T_brake, ...).
        """
        v_release = compute_release_speed(target_xyz, release_pos_xyz, theta_release)

        q_start_arr = _ensure_joint_array(q_start_full)

        # IK for release config
        from .planner import SawyerPlanner
        full_planner = SawyerPlanner(self._full_kin)
        q_release_arr = full_planner.compute_ik(q_start_arr, release_pos_xyz)
        if q_release_arr is None:
            print("[TossPlanner] IK failed for release position.")
            return None

        return self.plan_toss(
            q_start_full=q_start_arr,
            q_release_full=q_release_arr,
            v_release=v_release,
            theta_release=theta_release,
            active_joints=active_joints,
            **kwargs,
        )

    def plan_toss(
        self,
        q_start_full: Union[np.ndarray, JointAngles],    # (7,) full joint config at start — HARD constraint
        q_release_full: Union[np.ndarray, JointAngles],  # (7,) full joint config at release — HARD constraint
        v_release: float,            # required EE speed at release (m/s)
        theta_release: float = np.pi / 4,
        active_joints: List[int] = [1, 3, 5],   # indices into the 7-joint array
        N: int = 60,                 # optimisation knot points for the throw phase
        T_max: float = 3.0,         # upper bound for free throw duration
        T_brake: float = 1.2,        # fixed analytic braking duration
        ipopt_verbose: bool = False,
        j1_max_deg: float = -32.0,   # J1 hard upper bound (deg); must be ≤ this throughout — prevents curling into own links
    ) -> Optional[TossResult]:
        """
        Plan a tossing trajectory with hard start/release constraints.

        The trajectory has two phases:
          [0 .. release_index]              — optimised throw
          [release_index .. release_index+brake_steps]  — analytic braking back to q_start

        Returns TossResult or None if the solver fails.
        """
        q_start_arr = q_start_full.values
        q_release_arr = q_release_full.values

        # Build reduced kinematics using shared utility
        locked_overrides = {
            jname: val for jname, val in self.LOCKED_DEFAULTS.items()
        }
        kin, locked_by_index = build_reduced_kinematics(
            self._full_kin, active_joints, q_start_arr,
            locked_overrides=locked_overrides,
        )
        n = len(active_joints)

        q0  = q_start_arr[active_joints]
        qR  = q_release_arr[active_joints]

        v_lim, a_lim = get_profile_limits(MotionProfile.EXPRESS, active_joints)

        # ── Optimisation ──────────────────────────────────────────────────────
        opti = ca.Opti()

        T  = opti.variable()          # free throw duration
        opti.subject_to(opti.bounded(0.15, T, T_max))
        dt = T / N

        Q   = opti.variable(n, N + 1)
        Qd  = opti.variable(n, N + 1)
        Qdd = opti.variable(n, N + 1)

        # Trapezoidal integration
        for k in range(N):
            opti.subject_to(
                Q[:, k+1] == Q[:, k] + dt / 2.0 * (Qd[:, k] + Qd[:, k+1])
            )
            opti.subject_to(
                Qd[:, k+1] == Qd[:, k] + dt / 2.0 * (Qdd[:, k] + Qdd[:, k+1])
            )

        # Hard: start position and zero velocity
        opti.subject_to(Q[:, 0]  == q0)
        opti.subject_to(Qd[:, 0] == np.zeros(n))

        # Hard: release position
        opti.subject_to(Q[:, -1] == qR)

        # Hard: release velocity — EE must match v_release in direction theta_release
        # Constrains both magnitude AND direction (vx, vz components individually)
        J = kin.jacobian(Q[:, -1])    # (3, n) at release config
        opti.subject_to(J[0, :] @ Qd[:, -1] == v_release * np.cos(theta_release))
        opti.subject_to(J[2, :] @ Qd[:, -1] == v_release * np.sin(theta_release))

        # Joint / velocity / acceleration limits
        for i in range(n):
            opti.subject_to(opti.bounded(kin.q_min[i], Q[i, :], kin.q_max[i]))
            opti.subject_to(opti.bounded(-v_lim[i],    Qd[i, :],  v_lim[i]))
            opti.subject_to(opti.bounded(-a_lim[i],    Qdd[i, :], a_lim[i]))

        # Hard J1 upper bound — J1 must stay ≤ j1_max_deg throughout (no positive or near-zero values)
        if 1 in active_joints:
            j1_local = active_joints.index(1)
            opti.subject_to(Q[j1_local, :] <= np.deg2rad(j1_max_deg))

        # Objective: minimise jerk-like (smooth Qdd) + small time penalty
        opti.minimize(ca.sumsqr(Qdd) * 1e-3 + T * 0.1)

        solver_opts = {
            'ipopt.print_level': 5 if ipopt_verbose else 0,
            'print_time':        1 if ipopt_verbose else 0,
            'ipopt.sb':          'yes',
            'ipopt.max_iter':    400,
        }
        opti.solver('ipopt', solver_opts)

        try:
            sol = opti.solve()
        except Exception as e:
            print(f"[TossPlanner] Solver failed: {e}")
            return None

        T_opt  = float(sol.value(T))
        q_arr  = np.atleast_2d(sol.value(Q)).T    # (N+1, n)
        qd_arr = np.atleast_2d(sol.value(Qd)).T
        qdd_arr = np.atleast_2d(sol.value(Qdd)).T

        # Verify achieved EE speed at release
        J_num      = np.array(kin.jacobian(q_arr[-1]).full() if hasattr(kin.jacobian(q_arr[-1]), 'full')
                               else kin.jacobian(q_arr[-1])).reshape(3, n)
        qd_final   = qd_arr[-1].reshape(n, 1)
        v_ee_achieved = float(np.linalg.norm((J_num[[0, 2], :] @ qd_final).flatten()))

        print(f"[TossPlanner] T_opt={T_opt:.3f}s  "
              f"v_achieved={v_ee_achieved:.3f} m/s  (target={v_release:.3f})")

        # ── Resample throw to 100Hz ───────────────────────────────────────────
        orig_t = np.linspace(0.0, T_opt, N + 1)
        std_t = np.arange(0.0, T_opt, self.cfg.dt)

        def _resample(arr):
            out = interp1d(orig_t, arr, axis=0)(std_t)
            return out.reshape(len(std_t), -1)

        q_throw = _resample(q_arr)
        qd_throw = _resample(qd_arr)
        qdd_throw = _resample(qdd_arr)
        release_index = len(std_t) - 1

        # ── Analytic braking phase ────────────────────────────────────────────
        q_brake, qd_brake, qdd_brake = self._brake_phase(
            q_start=q_arr[-1],
            qd_start=qd_arr[-1],
            q_rest=q_arr[-1],
            T_brake=T_brake,
        )

        # ── Concatenate (skip first point of brake to avoid duplicate) ────────
        q_3dof   = np.vstack([q_throw,   q_brake[1:]])
        qd_3dof  = np.vstack([qd_throw,  qd_brake[1:]])
        qdd_3dof = np.vstack([qdd_throw, qdd_brake[1:]])

        # ── Expand back to 7-DOF ──────────────────────────────────────────────
        Q7, Qd7, Qdd7 = expand_to_7dof(
            q_3dof, qd_3dof, qdd_3dof,
            active_joints, locked_by_index,
        )

        traj = JointTrajectory.from_arrays(Q7, Qd7, Qdd7, self.cfg.dt)

        return TossResult(
            trajectory=traj,
            release_index=release_index,
            v_achieved=v_ee_achieved,
            T_throw=T_opt,
            T_brake=T_brake,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _brake_phase(
        self,
        q_start: np.ndarray,
        qd_start: np.ndarray,
        q_rest: np.ndarray,
        T_brake: float,
    ):
        """
        Free deceleration: coast from (q_start, qd_start) to zero velocity.

        No position constraint — the arm just smoothly decelerates wherever
        it happens to be.  Uses a linear velocity ramp (constant deceleration).
        Sampled at self.cfg.dt.
        Returns Q (M, n), Qd (M, n), Qdd (M, n).
        """
        steps = int(T_brake / self.cfg.dt) + 1
        t_arr = np.linspace(0.0, T_brake, steps)

        # Linear ramp: qd goes from qd_start to 0 over T_brake
        # qd(t)  = qd_start * (1 - t/T_brake)
        # q(t)   = q_start + qd_start * (t - t²/(2*T_brake))
        # qdd(t) = -qd_start / T_brake  (constant)
        qdd_const = -qd_start / T_brake

        Q_b, Qd_b, Qdd_b = [], [], []
        for t in t_arr:
            s = t / T_brake
            q   = q_start + qd_start * (t - t**2 / (2.0 * T_brake))
            qd  = qd_start * (1.0 - s)
            qdd = qdd_const

            Q_b.append(q)
            Qd_b.append(qd)
            Qdd_b.append(qdd)

        return np.array(Q_b), np.array(Qd_b), np.array(Qdd_b)
