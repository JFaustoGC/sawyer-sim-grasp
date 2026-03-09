"""
sawyer_mujoco.py — MuJoCo backend for the Sawyer robot client.

Implements the same sawyer_common interfaces as PyBulletRobot, using the
`mujoco` Python bindings against the MJCF model produced by
sawyer_description.compiler (SimTarget.MUJOCO).
"""

import time
from typing import Optional, Dict, Any, List

import mujoco
import numpy as np

from sawyer_common.robot import SimRobot
from sawyer_common.geometry import JointAngles, JointTrajectory, Pose, JointsLike
from sawyer_common.sawyer import Joint, Link


# MuJoCo actuator / joint name constants for the Sawyer MJCF (robosuite model)
_ARM_JOINT_NAMES = [
    "right_j0", "right_j1", "right_j2", "right_j3",
    "right_j4", "right_j5", "right_j6",
]

# Robosuite MJCF uses torque actuators named torq_right_jX
_ARM_ACTUATOR_NAMES = [
    "torq_right_j0", "torq_right_j1", "torq_right_j2", "torq_right_j3",
    "torq_right_j4", "torq_right_j5", "torq_right_j6",
]

# Gripper finger actuators (motor type: ctrl = force in N, negative = close/squeeze)
_GRIPPER_ACTUATOR_NAMES: list[str] = [
    "mot_right_gripper_l_finger",
    "mot_right_gripper_r_finger",
]
_GRIPPER_JOINT_NAMES: list[str] = [
    "right_gripper_l_finger_joint",
    "right_gripper_r_finger_joint",
]

# Gripper joint range (metres) — matches ctrlrange in the MJCF
_GRIPPER_OPEN_FORCE  =  100.0  # N — push fingers open
_GRIPPER_CLOSE_FORCE = -100.0  # N — squeeze fingers closed (pneumatic constant force)

# End-effector site name in the MJCF (added by compiler on right_hand body)
_EEF_SITE_NAME = "right_eef"

# PD gains for the torque controller (applied each physics step).
# Robosuite default: kp=150, damping_ratio=1 (critically damped).
# KD = 2 * damping_ratio * sqrt(KP * rotor_inertia).
# With armature [0.5, 0.5, 0.25, 0.25, 0.1, 0.1, 0.1]:
#   KD_i = 2 * 1.0 * sqrt(150 * armature_i)
_KP = np.array([150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0])
_KD = np.array([ 17.3,  17.3,  12.2,  12.2,   7.7,   7.7,   7.7])


class MujocoRobot(SimRobot):
    """
    Facade for the Sawyer robot in a MuJoCo simulation environment.

    Mirrors the PyBulletRobot API so both backends are drop-in replacements
    for one another against the sawyer_common abstract interfaces.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        gui: bool = True,
        start_pos: Optional[List[float]] = None,
        show_contacts: bool = False,
        show_collision_geoms: bool = False,
        show_ee_frame: bool = False,
    ) -> None:
        """
        Initialize the MuJoCo model and (optionally) a passive viewer.

        Args:
            model_path:           Absolute path to a MuJoCo MJCF (.xml) file.
                                  If None, the sawyer_description compiler generates one.
            gui:                  Launch a passive MuJoCo viewer window when True.
            start_pos:            [x, y, z] base position override.
            show_contacts:        Show contact force arrows and contact points.
            show_collision_geoms: Show group-3 collision geometry as cyan boxes.
            show_ee_frame:        Show site frames (end-effector axes).
        """
        self._gui = gui

        # ── Build / locate MJCF ──────────────────────────────────────────────
        if model_path is None:
            from sawyer_description.compiler import RobotCompiler, SimTarget
            compiler = RobotCompiler()
            model_path = compiler.compile(SimTarget.MUJOCO)

        # ── Load model ───────────────────────────────────────────────────────
        self._model: mujoco.MjModel = mujoco.MjModel.from_xml_path(model_path)
        self._data: mujoco.MjData = mujoco.MjData(self._model)

        # Add realistic rotor inertia (armature) to stabilise PD control.
        # The robosuite MJCF ships with zero armature; without it the wrist
        # joints are nearly massless and even small torques create huge
        # accelerations, making a simple PD controller oscillate.
        _ARMATURE = [0.5, 0.5, 0.25, 0.25, 0.1, 0.1, 0.1]
        for i, name in enumerate(_ARM_JOINT_NAMES):
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid != -1:
                self._model.dof_armature[self._model.jnt_dofadr[jid]] = _ARMATURE[i]

        # Keep the original robosuite actuator torque limits (±80, ±40, ±9)
        # which match the robot.xml defaults for Sawyer with custom gripper.

        # Extend finger joint range to allow slight negative (inward) travel.
        # At joint=0.0 (URDF minimum) the finger gap is 6.0 cm — wider than a
        # 5.7 cm cube. Allowing -0.004 m lets the fingers squeeze 3 mm past
        # the natural rest position to generate real contact (normal) force.
        # Joint range is fixed: 0.0 (closed) to 0.006 (open). No runtime extension needed.

        # Run one forward pass to populate xpos / xquat etc.
        mujoco.mj_forward(self._model, self._data)

        # Eliminate slow slippage
        self._model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC   # required for noslip + impratio
        self._model.opt.noslip_iterations = 3                    # post-process solver: fully prevents slow slip
        self._model.opt.impratio = 10                            # further reduces slippage inside elliptic cone

        # ── Resolve joint / actuator indices ─────────────────────────────────
        self._arm_qpos_indices: List[int] = []
        for name in _ARM_JOINT_NAMES:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                print(f"WARNING: arm joint '{name}' not found in model.")
                self._arm_qpos_indices.append(-1)
            else:
                self._arm_qpos_indices.append(self._model.jnt_qposadr[jid])

        self._arm_actuator_ids: List[int] = []
        for name in _ARM_ACTUATOR_NAMES:
            aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self._arm_actuator_ids.append(aid)  # -1 if missing → silently ignored

        # Cache dof addresses for the arm joints (used every PD step)
        self._arm_dof_indices: List[int] = []
        for name in _ARM_JOINT_NAMES:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._arm_dof_indices.append(
                self._model.jnt_dofadr[jid] if jid != -1 else -1
            )

        self._gripper_actuator_ids: List[int] = []
        for name in _GRIPPER_ACTUATOR_NAMES:
            aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid != -1:
                self._gripper_actuator_ids.append(aid)

        # Gripper joint qpos addresses (for teleportation)
        self._gripper_qpos_indices: List[int] = []
        self._gripper_dof_indices: List[int] = []
        for name in _GRIPPER_JOINT_NAMES:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid != -1:
                self._gripper_qpos_indices.append(self._model.jnt_qposadr[jid])
                dof = self._model.jnt_dofadr[jid]
                self._gripper_dof_indices.append(dof)
                # High damping matches Gazebo electric gripper (damping=100)
                self._model.dof_damping[dof] = 100.0

        # Hold gripper open on startup (will be explicitly opened/closed by script)
        for aid in self._gripper_actuator_ids:
            self._data.ctrl[aid] = _GRIPPER_OPEN_FORCE

        # End-effector site id
        self._eef_site_id: int = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, _EEF_SITE_NAME
        )

        # ── Optional base position override ──────────────────────────────────
        if start_pos is not None:
            self._apply_base_pos(start_pos)

        # ── Visualization settings ────────────────────────────────────────
        self._model.vis.map.force = 0.02   # contact force arrow scale
        self._model.vis.scale.framelength = 0.04
        self._model.vis.scale.framewidth = 0.005

        # Make group-3 collision geoms visible as semi-transparent cyan (only if requested)
        if gui and show_collision_geoms:
            for i in range(self._model.ngeom):
                if self._model.geom_group[i] == 3:
                    self._model.geom_rgba[i] = [0.0, 0.8, 0.8, 0.25]

        # ── Launch viewer ────────────────────────────────────────────────────
        self._viewer = None
        if gui:
            try:
                from mujoco.viewer import launch_passive
                self._viewer = launch_passive(self._model, self._data)
                self._viewer.opt.geomgroup[3] = show_collision_geoms
                self._viewer.opt.sitegroup[3] = show_ee_frame
                self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = show_contacts
                self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = show_contacts
                self._viewer.opt.frame = (
                    mujoco.mjtFrame.mjFRAME_SITE if show_ee_frame else mujoco.mjtFrame.mjFRAME_NONE
                )
            except Exception as exc:
                print(f"WARNING: Could not launch MuJoCo viewer: {exc}")

        self._enabled = True

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_base_pos(self, pos: List[float]) -> None:
        """Override the free-joint root position if one exists."""
        for name in ("root", "base", "right_arm_base_link"):
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid != -1 and self._model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                adr = self._model.jnt_qposadr[jid]
                self._data.qpos[adr:adr + 3] = pos
                mujoco.mj_forward(self._model, self._data)
                return

    def _sync_viewer(self) -> None:
        """Push current state to the passive viewer (no-op if headless)."""
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def _pd_step(self, q_des: np.ndarray, qd_des: np.ndarray) -> None:
        """
        Apply one physics step with a PD + gravity-compensation controller.

        Uses priority-based torque allocation to prevent limit-cycle
        oscillation when actuators saturate:
            1. Gravity/Coriolis compensation  (highest priority)
            2. Velocity damping               (prevents oscillation)
            3. Position correction             (remaining budget)

        Args:
            q_des:  Desired joint positions (7,) rad
            qd_des: Desired joint velocities (7,) rad/s
        """
        q  = np.array([self._data.qpos[adr] if adr != -1 else 0.0
                        for adr in self._arm_qpos_indices], dtype=float)
        qd = np.array([self._data.qvel[adr] if adr != -1 else 0.0
                        for adr in self._arm_dof_indices], dtype=float)
        grav = np.array([self._data.qfrc_bias[adr] if adr != -1 else 0.0
                          for adr in self._arm_dof_indices], dtype=float)

        tau_pos = _KP * (q_des - q)
        tau_vel = _KD * (qd_des - qd)

        for i, aid in enumerate(self._arm_actuator_ids):
            if aid != -1:
                lo = self._model.actuator_ctrlrange[aid, 0]
                hi = self._model.actuator_ctrlrange[aid, 1]

                # 1) Gravity compensation (highest priority)
                g = np.clip(grav[i], lo, hi)
                # 2) Damping — allocate within remaining budget
                d = np.clip(tau_vel[i], lo - g, hi - g)
                # 3) Position — whatever is left
                p = np.clip(tau_pos[i], lo - g - d, hi - g - d)

                self._data.ctrl[aid] = float(g + d + p)

        mujoco.mj_step(self._model, self._data)
        
        # If no gripper actuators, hold fingers open via teleport
        if not self._gripper_actuator_ids:
            self.teleport_gripper(0.006)

        self._sync_viewer()

    def _step(self) -> None:
        """Advance physics by one timestep with zero torque and sync viewer."""
        mujoco.mj_step(self._model, self._data)
        self._sync_viewer()

    # ── Teleportable Interface ────────────────────────────────────────────────

    def teleport_joints(self, joints: JointsLike) -> None:
        """Instantly set arm joint positions, bypassing the physics integrator."""
        angles = JointAngles(joints).values.tolist()
        for i, (qpos_adr, dof_adr) in enumerate(
            zip(self._arm_qpos_indices, self._arm_dof_indices)
        ):
            if qpos_adr != -1 and i < len(angles):
                self._data.qpos[qpos_adr] = angles[i]
            if dof_adr != -1:
                self._data.qvel[dof_adr] = 0.0
        mujoco.mj_forward(self._model, self._data)
        self._sync_viewer()

    def teleport_gripper(self, position: float) -> None:
        """Instantly set gripper finger positions."""
        for adr in self._gripper_qpos_indices:
            self._data.qpos[adr] = position
        mujoco.mj_forward(self._model, self._data)
        self._sync_viewer()

    # ── ArmProvider Interface ─────────────────────────────────────────────────

    def get_joints(self) -> JointAngles:
        """Return the current arm joint angles."""
        positions = [
            float(self._data.qpos[adr]) if adr != -1 else 0.0
            for adr in self._arm_qpos_indices
        ]
        return JointAngles(positions)

    def move_to_joints(self, joints: JointsLike, timeout: float = 30.0) -> bool:
        """
        Drive the arm to target joint angles via PD velocity control.

        Runs the physics engine, sending τ = kp*(q_des−q) + kd*(0−qd)
        each step until all joints converge within 0.01 rad or timeout.
        """
        q_des  = np.array(JointAngles(joints).values.tolist())
        qd_des = np.zeros(7)

        start = time.time()
        while time.time() - start < timeout:
            self._pd_step(q_des, qd_des)
            q = np.array([self._data.qpos[adr] for adr in self._arm_qpos_indices])
            if np.allclose(q, q_des, atol=0.01):
                return True

        return False

    def execute_trajectory(self, trajectory: JointTrajectory, rate_hz: float = 100.0) -> bool:
        """
        Execute a joint trajectory using PD control at each waypoint.

        Uses position and velocity from each JointState point.  When
        velocity data is absent the desired velocity defaults to zero.
        """
        dt = 1.0 / rate_hz
        steps_per_point = max(1, int(dt / self._model.opt.timestep))

        for point in trajectory.points:
            q_des  = np.array(point.position.values.tolist())
            qd_des = np.array(point.velocity.values.tolist()) if point.velocity else np.zeros(7)
            for _ in range(steps_per_point):
                self._pd_step(q_des, qd_des)

        return True

    def stream_trajectory(
        self, trajectory: JointTrajectory, release_time: Optional[float] = None
    ) -> bool:
        """
        Stream a joint trajectory respecting per-point timestamps via PD control.

        Uses both position and velocity targets from each JointState point.
        Optionally opens the gripper at `release_time` seconds from start.
        """
        if not trajectory.points:
            return True

        dt_sim = self._model.opt.timestep
        start_wall = time.time()

        for idx, point in enumerate(trajectory.points):
            # Gripper release check
            elapsed = time.time() - start_wall
            if release_time is not None and elapsed >= release_time:
                self.open_gripper()
                release_time = None

            q_des  = np.array(point.position.values.tolist())
            qd_des = np.array(point.velocity.values.tolist()) if point.velocity else np.zeros(7)

            # How long to hold this waypoint
            if idx + 1 < len(trajectory.points):
                segment_duration = (trajectory.points[idx + 1].time_from_start
                                    - point.time_from_start)
            else:
                segment_duration = dt_sim  # last point: one step

            steps = max(1, int(segment_duration / dt_sim))
            for _ in range(steps):
                self._pd_step(q_des, qd_des)

        return True

    def get_pose(self) -> Optional[Pose]:
        """Return the Cartesian pose of the end-effector site."""
        if self._eef_site_id == -1:
            return None

        pos = self._data.site_xpos[self._eef_site_id].copy()

        # Convert 3×3 rotation matrix → quaternion (w, x, y, z) → (x, y, z, w)
        mat = self._data.site_xmat[self._eef_site_id].reshape(3, 3)
        quat_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(quat_wxyz, mat.flatten())
        # mujoco convention: [w, x, y, z] → Pose convention: {x, y, z, w}
        qx, qy, qz, qw = quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]

        return Pose.from_dict({
            "position":    {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
        })

    # ── GripperProvider Interface ─────────────────────────────────────────────

    def open_gripper(self) -> bool:
        """Command the gripper to open."""
        if not self._gripper_actuator_ids:
            self.teleport_gripper(0.006)
        else:
            for aid in self._gripper_actuator_ids:
                self._data.ctrl[aid] = _GRIPPER_OPEN_FORCE
        return True

    def close_gripper(self) -> bool:
        """Command the gripper to close."""
        if not self._gripper_actuator_ids:
            self.teleport_gripper(0.0)
        else:
            for aid in self._gripper_actuator_ids:
                self._data.ctrl[aid] = _GRIPPER_CLOSE_FORCE
        return True

    # ── RobotLifecycle Interface ──────────────────────────────────────────────

    def enable(self) -> bool:
        self._enabled = True
        return True

    def disable(self) -> bool:
        """Disable by zeroing all actuator control signals."""
        self._enabled = False
        self._data.ctrl[:] = 0.0
        return True

    def reset(self) -> bool:
        """Reset the simulation to its initial state (keyframe 0 if available)."""
        mujoco.mj_resetData(self._model, self._data)
        if self._model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self._model, self._data, 0)
        mujoco.mj_forward(self._model, self._data)
        self._sync_viewer()
        self._enabled = True
        return True

    def stop(self) -> bool:
        return self.disable()

    def sleep(self, duration: float) -> None:
        """Sleep for the specified duration, maintaining current state."""
        q_des = np.array([self._data.qpos[adr] if adr != -1 else 0.0 for adr in self._arm_qpos_indices])
        qd_des = np.zeros(7)
        steps = max(1, int(duration / self._model.opt.timestep))
        for _ in range(steps):
            self._pd_step(q_des, qd_des)

    def get_robot_status(self) -> Dict[str, Any]:
        return {
            "enabled":   self._enabled,
            "stopped":   not self._enabled,
            "error":     False,
            "estop":     False,
            "simulated": True,
        }

    def close(self) -> None:
        """Close the MuJoCo viewer (if open)."""
        try:
            if self._viewer is not None:
                self._viewer.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"MujocoRobot(gui={self._gui})"
