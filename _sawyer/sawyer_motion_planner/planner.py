"""
planner.py — Unified Motion Planner for the Sawyer Robot.

Combines profile-based Point-to-Point (Joint/Cartesian) planning with
specialized Tossing trajectory optimization.
"""

from __future__ import annotations
import casadi as ca
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
from scipy.spatial.transform import Rotation, Slerp

from sawyer_common.geometry import Pose, Pose2D, Point2D, JointAngles, JointTrajectory, JointState
from sawyer_common.sawyer import Joint, Link
from .kinematics import CasadiKinematics
from scipy.interpolate import interp1d
from .planner_config import PlannerConfig, LIMIT_PROFILES
from .planner_utils import min_jerk_poly, get_profile_limits


class SawyerPlanner:
    def __init__(self, kinematics: CasadiKinematics, config: Optional[PlannerConfig] = None):
        self.model = kinematics
        self.cfg = config if config else PlannerConfig()

    # ── Inverse Kinematics ───────────────────────────────────────────────────

    def compute_ik(self, q_guess: np.ndarray,
                   target_pos: np.ndarray,
                   target_quat: np.ndarray = np.array([0, 1, 0, 0]),
                   soft_limits: bool = False) -> Optional[np.ndarray]:
        """Numerical IK using CasADi optimization.

        Args:
            soft_limits: When True, joint limits are penalised rather than
                         enforced as hard constraints. Useful for intermediate
                         Cartesian waypoints where strict feasibility is not
                         required.
        """
        opti = ca.Opti()
        q = opti.variable(self.model.n_dof)

        pos_err = self.model.fk_pos(q) - ca.DM(target_pos)
        rot_k = self.model.fk_rot(q)
        dot_prod = ca.dot(rot_k, ca.DM(target_quat))
        ori_err = 1.0 - (dot_prod * dot_prod)

        cost = self.cfg.w_pos * ca.dot(pos_err, pos_err) + self.cfg.w_ori * ori_err

        if soft_limits:
            # Soft quadratic penalty for exceeding joint limits
            q_mid = ca.DM((self.model.q_min + self.model.q_max) / 2.0)
            q_range = ca.DM((self.model.q_max - self.model.q_min) / 2.0)
            normalised = (q - q_mid) / q_range  # in [-1, 1] when within limits
            # Penalise being near or beyond limits
            cost += 10.0 * ca.dot(normalised**2, normalised**2)
        else:
            opti.subject_to(opti.bounded(self.model.q_min, q, self.model.q_max))

        opti.minimize(cost)
        opti.set_initial(q, q_guess)
        opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'})

        try:
            sol = opti.solve()
            return sol.value(q)
        except:
            return None

    # ── Trajectory Generation ────────────────────────────────────────────────

    def plan_joint(
        self,
        q_start: Union[np.ndarray, JointAngles],
        q_goal: Union[np.ndarray, JointAngles, np.ndarray],
        profile: 'MotionProfile' = None,
        target_pos: Optional[np.ndarray] = None,
        target_quat: Optional[np.ndarray] = None,
    ) -> Optional[JointTrajectory]:
        """
        Joint-space P2P movement.

        When ``target_pos`` (and optionally ``target_quat``) are given instead
        of ``q_goal``, IK is solved first to obtain the goal joint config.
        This is a convenience overload — the trajectory is still interpolated
        in joint space with a minimum-jerk profile.
        """
        from .planner_config import MotionProfile
        if profile is None:
            profile = MotionProfile.SLOW

        q_start_arr = np.asarray(q_start.to_array() if hasattr(q_start, 'to_array') else q_start)

        # Resolve goal: either direct joint angles or via IK
        if target_pos is not None:
            if target_quat is None:
                target_quat = np.array([0, 1, 0, 0])
            q_goal_arr = self.compute_ik(q_start_arr, target_pos, target_quat)
            if q_goal_arr is None:
                return None
        else:
            q_goal_arr = np.asarray(q_goal.to_array() if hasattr(q_goal, 'to_array') else q_goal)

        dq = q_goal_arr - q_start_arr
        v_lim, a_lim = get_profile_limits(profile, list(range(7)))

        # Calculate time-optimal duration based on profile limits
        t_v = 1.875 * np.abs(dq) / v_lim
        t_a = np.sqrt(5.77 * np.abs(dq) / a_lim)
        duration = float(np.max([np.max(t_v), np.max(t_a), 0.1]))

        steps = int(duration / self.cfg.dt)
        tau = np.linspace(0, 1, steps + 1)

        s, sd, sdd = min_jerk_poly(tau, duration)

        Q   = q_start_arr + np.outer(s, dq)
        Qd  = np.outer(sd, dq)
        Qdd = np.outer(sdd, dq)

        return JointTrajectory.from_arrays(Q, Qd, Qdd, self.cfg.dt)

    def plan_cartesian(
        self,
        q_start: Union[np.ndarray, JointAngles],
        target_pos: np.ndarray,
        target_quat: np.ndarray = np.array([0, 1, 0, 0]),
        profile: 'MotionProfile' = None,
    ) -> Optional[JointTrajectory]:
        """
        Cartesian-space planning via CasADi trajectory optimization.

        Optimizes the full trajectory at once with:
        - Linear position tracking in Cartesian space
        - Orientation tracking via quaternion dot-product cost
        - Min-jerk reference for smooth time allocation
        - Posture regularization toward a natural configuration
        - Profile-based velocity/acceleration limits

        The coarse solver solution is upsampled to the configured dt (100 Hz).
        """
        from .planner_config import MotionProfile
        if profile is None:
            profile = MotionProfile.SLOW

        q_start_arr = np.asarray(q_start.to_array() if hasattr(q_start, 'to_array') else q_start)
        target_pos = np.asarray(target_pos)
        target_quat = np.asarray(target_quat)

        # Current EE pose
        p0 = np.array(self.model.fk_pos(q_start_arr)).flatten()
        q0_quat = np.array(self.model.fk_rot(q_start_arr)).flatten()

        # Hemisphere consistency for shortest-path rotation
        if np.dot(q0_quat, target_quat) < 0:
            target_quat = -target_quat

        # Duration from Cartesian distance + profile limits
        dist = np.linalg.norm(target_pos - p0)
        v_lim, a_lim = get_profile_limits(profile, list(range(7)))
        # Use the slowest joint's velocity as a rough linear speed bound
        linear_speed = float(np.min(v_lim)) * 0.15  # conservative
        duration = max(dist / linear_speed, 0.5) if linear_speed > 0 else 2.0

        # SLERP interpolator for orientation reference
        key_rots = Rotation.from_quat(np.vstack([q0_quat, target_quat]))
        slerp = Slerp([0.0, 1.0], key_rots)

        # ── CasADi trajectory optimization ─────────────────────────────────
        N = self.cfg.solver_steps
        dt_coarse = duration / N

        opti = ca.Opti()
        Q = opti.variable(self.model.n_dof, N)
        V = opti.variable(self.model.n_dof, N)
        A = opti.variable(self.model.n_dof, N)

        q_nat = ca.DM(self.cfg.q_natural)
        q_tgt = ca.DM(target_quat)
        total_cost = 0.0

        for k in range(N):
            q_k, v_k, a_k = Q[:, k], V[:, k], A[:, k]

            # ── Dynamics (Euler integration) ──
            if k == 0:
                opti.subject_to(q_k == ca.DM(q_start_arr) + v_k * dt_coarse)
                opti.subject_to(v_k == 0)  # start from rest
            else:
                opti.subject_to(q_k == Q[:, k-1] + v_k * dt_coarse)
                opti.subject_to(v_k == V[:, k-1] + a_k * dt_coarse)

            # ── Joint limits & velocity/acceleration limits ──
            opti.subject_to(opti.bounded(self.model.q_min, q_k, self.model.q_max))
            opti.subject_to(opti.bounded(-ca.DM(a_lim), a_k, ca.DM(a_lim)))
            opti.subject_to(opti.bounded(-ca.DM(v_lim), v_k, ca.DM(v_lim)))

            # ── Cartesian tracking costs ──
            alpha = (k + 1) / N
            # Min-jerk blending factor instead of linear
            s = 10 * alpha**3 - 15 * alpha**4 + 6 * alpha**5
            pos_ref = p0 * (1 - s) + target_pos * s
            pos_k = self.model.fk_pos(q_k)
            err_pos = pos_k - ca.DM(pos_ref)
            total_cost += self.cfg.w_pos * ca.dot(err_pos, err_pos)

            # Orientation: SLERP reference at this step
            quat_ref = slerp(s).as_quat()  # [x,y,z,w]
            rot_k = self.model.fk_rot(q_k)
            dot_prod = ca.dot(rot_k, ca.DM(quat_ref))
            total_cost += self.cfg.w_ori * (1.0 - dot_prod * dot_prod)

            # Smoothness
            total_cost += self.cfg.w_smooth * ca.dot(a_k, a_k)

            # Posture regularization
            total_cost += self.cfg.w_reg * ca.dot(q_k - q_nat, q_k - q_nat)

        # ── Terminal constraints ──
        opti.subject_to(V[:, -1] == 0)  # stop at end

        # Hard constraint on final Cartesian pose
        pos_final = self.model.fk_pos(Q[:, -1])
        opti.subject_to(pos_final == ca.DM(target_pos))
        rot_final = self.model.fk_rot(Q[:, -1])
        dot_final = ca.dot(rot_final, q_tgt)
        opti.subject_to(dot_final * dot_final >= 0.999)

        # Hard constraint on initial config
        opti.subject_to(Q[:, 0] == ca.DM(q_start_arr))

        opti.minimize(total_cost)

        # Warm start: linear interpolation in joint space
        for k in range(N):
            opti.set_initial(Q[:, k], q_start_arr)

        opti.solver('ipopt', {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': self.cfg.solver_max_iter,
            'ipopt.max_cpu_time': self.cfg.solver_max_cpu_time,
            'print_time': 0,
        })

        try:
            sol = opti.solve()
        except RuntimeError:
            print("[CARTESIAN] CasADi solver failed to converge")
            return None

        # ── Extract & upsample to 100 Hz ──────────────────────────────────
        res_q = sol.value(Q).T   # (N, n_dof)
        res_v = sol.value(V).T
        res_a = sol.value(A).T

        t_coarse = np.linspace(0, duration, N)
        n_fine = int(duration / self.cfg.dt)
        t_fine = np.linspace(0, duration, n_fine)

        q_fine = interp1d(t_coarse, res_q, axis=0, kind='cubic')(t_fine)
        v_fine = interp1d(t_coarse, res_v, axis=0, kind='linear')(t_fine)
        a_fine = interp1d(t_coarse, res_a, axis=0, kind='linear')(t_fine)

        return JointTrajectory.from_arrays(q_fine, v_fine, a_fine, self.cfg.dt)
