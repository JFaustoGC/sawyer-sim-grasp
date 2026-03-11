"""
sim.py — SawyerSim: student-friendly Sawyer MuJoCo simulation.

This file automatically adds the bundled _sawyer/ packages to Python's
search path so no installation is needed.  Just run your script directly.
"""

from __future__ import annotations

# ── Bootstrap: add bundled packages to sys.path ───────────────────────────────
import sys
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))          # .../sawyer_student/
_ROOT = _os.path.dirname(_HERE)                                # .../sawyer-student-lab/
_VENDOR = _os.path.join(_ROOT, "_sawyer")

if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
# ─────────────────────────────────────────────────────────────────────────────

import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mujoco
import numpy as np

from sawyer_common.sawyer import Link
from sawyer_description.compiler import RobotCompiler, SimTarget
from sawyer_motion_planner import CasadiKinematics, SawyerPlanner
from sawyer_motion_planner.planner_config import MotionProfile
from sawyer_mujoco import MujocoRobot

# ── Speed presets ─────────────────────────────────────────────────────────────

_SPEED_MAP = {
    "slow":   MotionProfile.SLOW,
    "medium": MotionProfile.MEDIUM,
    "fast":   MotionProfile.FAST,
}

# Initial "ready" joint configuration (safe pose, avoids collisions at start)
_READY_JOINTS = [0.0, -0.9, 0.0, 1.8, 0.0, 0.6, 1.776047]

# How high the robot base sits above the MuJoCo world floor
_ROBOT_BASE_Z = 0.15


# ── Internal object descriptor ────────────────────────────────────────────────

@dataclass
class _PendingGeom:
    name: str
    shape: str           # "box" | "sphere" | "cylinder" | "mesh"
    position: List[float]
    size: Optional[List[float]] = None   # box: [x, y, z] full dims
    radius: Optional[float] = None
    height: Optional[float] = None       # cylinder only
    mass: float = 0.1
    color: Optional[List[float]] = None  # [r, g, b, a]
    mesh_asset_name: Optional[str] = None  # for mesh geoms


# ── Main class ────────────────────────────────────────────────────────────────

class SawyerSim:
    """
    Beginner-friendly Sawyer robot simulation.

    Typical usage::

        from sawyer_student import SawyerSim

        sim = SawyerSim()
        sim.add_box("cube", position=[0.45, 0.0, 0.028], size=[0.057, 0.057, 0.057])
        sim.start()

        sim.move_to(0.45, 0.0, 0.20)
        sim.close_gripper()
        sim.wait(1.0)
        sim.open_gripper()
        sim.close()
    """

    def __init__(self, gui: bool = True) -> None:
        self._gui = gui
        self._pending_geoms: List[_PendingGeom] = []
        self._mesh_assets: List[Tuple[str, str]] = []  # (asset_name, abs_path)

        self._compiler: Optional[RobotCompiler] = None
        self._robot: Optional[MujocoRobot] = None
        self._planner: Optional[SawyerPlanner] = None
        self._kin: Optional[CasadiKinematics] = None
        self._started = False

    # ── Object loading ────────────────────────────────────────────────────────

    def add_box(
        self,
        name: str,
        position: List[float],
        size: List[float],
        mass: float = 0.1,
        color: Optional[List[float]] = None,
    ) -> None:
        """
        Add a box to the scene.

        Args:
            name:     Unique label (e.g. ``"cube"``).
            position: [x, y, z] centre position in the world frame (metres).
                      z=0 is the floor; the robot base is at z=0.
            size:     [width, depth, height] full extents in metres.
                      The default cube used in demos is [0.057, 0.057, 0.057].
            mass:     Mass in kg.
            color:    [r, g, b, a] each between 0 and 1.
        """
        self._check_not_started()
        self._pending_geoms.append(_PendingGeom(
            name=name, shape="box", position=list(position),
            size=list(size), mass=mass,
            color=color or [0.9, 0.2, 0.2, 1.0],
        ))

    def add_sphere(
        self,
        name: str,
        position: List[float],
        radius: float,
        mass: float = 0.1,
        color: Optional[List[float]] = None,
    ) -> None:
        """
        Add a sphere to the scene.

        Args:
            name:     Unique label.
            position: [x, y, z] centre in metres.
            radius:   Radius in metres.
        """
        self._check_not_started()
        self._pending_geoms.append(_PendingGeom(
            name=name, shape="sphere", position=list(position),
            radius=radius, mass=mass,
            color=color or [0.2, 0.6, 0.9, 1.0],
        ))

    def add_cylinder(
        self,
        name: str,
        position: List[float],
        radius: float,
        height: float,
        mass: float = 0.1,
        color: Optional[List[float]] = None,
    ) -> None:
        """
        Add a cylinder to the scene.

        Args:
            name:     Unique label.
            position: [x, y, z] centre in metres.
            radius:   Radius in metres.
            height:   Full height in metres.
        """
        self._check_not_started()
        self._pending_geoms.append(_PendingGeom(
            name=name, shape="cylinder", position=list(position),
            radius=radius, height=height, mass=mass,
            color=color or [0.4, 0.8, 0.4, 1.0],
        ))

    def add_object_from_urdf(
        self,
        urdf_path: str,
        position: List[float],
        name: str = "object",
    ) -> None:
        """
        Load an object from a URDF file.

        Only the first link that has a ``<collision><geometry>`` is used.
        Supported shapes: ``box``, ``sphere``, ``cylinder``, ``mesh``.

        Args:
            urdf_path: Path to the ``.urdf`` file.
            position:  [x, y, z] centre position in the world frame.
            name:      Unique label for the object.
        """
        self._check_not_started()
        geom = self._parse_urdf(os.path.abspath(urdf_path), list(position), name)
        self._pending_geoms.append(geom)

    # ── Scene launch ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Compile the robot, build the scene, and open the viewer.

        Call this after all ``add_*`` calls and before any robot control.
        """
        if self._started:
            raise RuntimeError("sim.start() has already been called.")

        print("[SawyerSim] Compiling robot description…")
        self._compiler = RobotCompiler()

        # Kinematics use the PyBullet URDF
        pb_urdf = self._compiler.compile(SimTarget.PYBULLET)
        self._kin = CasadiKinematics(pb_urdf, Link.BASE.value, Link.GRIPPER.value)
        self._planner = SawyerPlanner(self._kin)

        print("[SawyerSim] Building MuJoCo scene…")
        mjcf_path = self._compiler.compile(SimTarget.MUJOCO)
        scene_path = self._build_scene_xml(mjcf_path)

        print("[SawyerSim] Launching simulation…")
        try:
            self._robot = MujocoRobot(
                model_path=scene_path,
                gui=self._gui,
                start_pos=[0, 0, _ROBOT_BASE_Z],
            )
        finally:
            os.unlink(scene_path)

        self._apply_collision_masks()
        self._robot.enable()

        print("[SawyerSim] Teleporting to ready pose…")
        self._robot.teleport_joints(_READY_JOINTS)
        self._robot.sleep(0.5)

        self._started = True
        print("[SawyerSim] Ready!")

    # ── Robot control ─────────────────────────────────────────────────────────

    def move_to(self, x: float, y: float, z: float, speed: str = "medium") -> bool:
        """
        Move the gripper to a position in the world frame.

        Args:
            x, y, z: Target position in metres (z=0 is the floor).
            speed:   ``"slow"``, ``"medium"`` (default), or ``"fast"``.

        Returns:
            ``True`` on success, ``False`` if the position is unreachable.
        """
        self._check_started()
        profile = _SPEED_MAP.get(speed, MotionProfile.MEDIUM)
        target = np.array([x, y, z - _ROBOT_BASE_Z])  # world → robot base frame
        q_rot = np.array(self._kin.fk_rot(
            np.asarray(self._robot.get_joints().values)
        )).flatten()

        traj = self._planner.plan_cartesian(
            self._robot.get_joints(), target, q_rot, profile=profile
        )
        if traj is None:
            print(f"[SawyerSim] Warning: could not reach ({x:.3f}, {y:.3f}, {z:.3f})")
            return False
        self._robot.execute_trajectory(traj, rate_hz=100)
        return True

    def open_gripper(self) -> None:
        """Open the gripper fingers."""
        self._check_started()
        self._robot.open_gripper()

    def close_gripper(self) -> None:
        """Close the gripper fingers."""
        self._check_started()
        self._robot.close_gripper()

    def get_joints(self) -> List[float]:
        """
        Return the current joint angles (7 values, radians).

        Joint order: j0, j1, j2, j3, j4, j5, j6 (base to wrist).
        The last joint (j6) controls the wrist rotation / gripper orientation.
        """
        self._check_started()
        return list(self._robot.get_joints().values)

    def set_joints(self, angles: List[float]) -> None:
        """
        Instantly teleport the robot to a joint configuration (no animation).

        Args:
            angles: 7 joint angles in radians [j0 … j6].
                    Use ``get_joints()`` to read the current values first.

        Example::

            # Rotate the wrist to change gripper orientation
            joints = sim.get_joints()
            joints[6] += 0.5   # rotate wrist ~30°
            sim.set_joints(joints)
        """
        self._check_started()
        if len(angles) != 7:
            raise ValueError(f"Expected 7 joint angles, got {len(angles)}")
        self._robot.teleport_joints(list(angles))
        self._robot.sleep(0.1)

    def move_joints(self, angles: List[float], speed: str = "medium") -> bool:
        """
        Move to a joint configuration using a smooth planned trajectory.

        Args:
            angles: 7 target joint angles in radians [j0 … j6].
            speed:  ``"slow"``, ``"medium"`` (default), or ``"fast"``.

        Returns:
            ``True`` on success, ``False`` if planning failed.

        Example::

            # Move to the ready pose
            sim.move_joints([0.0, -0.9, 0.0, 1.8, 0.0, 0.6, 1.776047])
        """
        self._check_started()
        if len(angles) != 7:
            raise ValueError(f"Expected 7 joint angles, got {len(angles)}")
        profile = _SPEED_MAP.get(speed, MotionProfile.MEDIUM)
        traj = self._planner.plan_joint(
            self._robot.get_joints(), list(angles), profile=profile
        )
        if traj is None:
            print("[SawyerSim] Warning: joint planning failed.")
            return False
        self._robot.execute_trajectory(traj, rate_hz=100)
        return True

    # ── Sensing ───────────────────────────────────────────────────────────────

    def get_position(self) -> Tuple[float, float, float]:
        """Return end-effector (x, y, z) in the world frame (metres)."""
        self._check_started()
        pose = self._robot.get_pose()
        if pose is None:
            return (0.0, 0.0, 0.0)
        return (pose.position.x, pose.position.y, pose.position.z + _ROBOT_BASE_Z)

    def get_pose(self) -> dict:
        """
        Return the full end-effector pose.

        Keys: ``x``, ``y``, ``z``, ``qx``, ``qy``, ``qz``, ``qw``.
        """
        self._check_started()
        pose = self._robot.get_pose()
        if pose is None:
            return {"x": 0.0, "y": 0.0, "z": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
        p, o = pose.position, pose.orientation
        return {"x": p.x, "y": p.y, "z": p.z + _ROBOT_BASE_Z,
                "qx": o.x, "qy": o.y, "qz": o.z, "qw": o.w}

    # ── Utilities ─────────────────────────────────────────────────────────────

    def wait(self, seconds: float) -> None:
        """Run the simulation for ``seconds`` seconds."""
        self._check_started()
        self._robot.sleep(seconds)

    def reset(self) -> None:
        """Teleport the robot back to its ready pose."""
        self._check_started()
        self._robot.teleport_joints(_READY_JOINTS)
        self._robot.sleep(0.3)

    def close(self) -> None:
        """Shut down the simulation and close the viewer."""
        if self._robot is not None:
            self._robot.close()
            self._robot = None
        self._started = False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_not_started(self):
        if self._started:
            raise RuntimeError("Cannot add objects after sim.start().")

    def _check_started(self):
        if not self._started:
            raise RuntimeError("Call sim.start() first.")

    def _build_scene_xml(self, robot_mjcf_path: str) -> str:
        tree = ET.parse(robot_mjcf_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise RuntimeError("No <worldbody> in robot MJCF")

        # Raise robot base above floor
        base_body = worldbody.find("body[@name='base']")
        if base_body is not None:
            base_body.set("pos", f"0 0 {_ROBOT_BASE_Z:.4f}")

        # Lighting
        light = ET.SubElement(worldbody, "light")
        light.set("pos", "0 0 3"); light.set("dir", "0 0 -1")
        light.set("directional", "true"); light.set("castshadow", "true")

        # Floor
        floor = ET.SubElement(worldbody, "geom")
        floor.set("name", "floor"); floor.set("type", "plane")
        floor.set("size", "2 2 0.1"); floor.set("rgba", "0.8 0.9 0.8 1")
        floor.set("contype", "1"); floor.set("conaffinity", "1")

        # Mesh assets (from add_object_from_urdf)
        if self._mesh_assets:
            asset_el = root.find("asset")
            if asset_el is None:
                asset_el = ET.SubElement(root, "asset")
            for asset_name, abs_path in self._mesh_assets:
                m = ET.SubElement(asset_el, "mesh")
                m.set("name", asset_name); m.set("file", abs_path)

        # Inject objects
        for pg in self._pending_geoms:
            self._inject_geom(worldbody, pg)

        # Write temp file beside the MJCF so relative asset paths resolve
        mjcf_dir = os.path.dirname(os.path.abspath(robot_mjcf_path))
        fd, tmp = tempfile.mkstemp(suffix=".xml", prefix="sawyer_scene_", dir=mjcf_dir)
        os.close(fd)
        tree.write(tmp, encoding="unicode")
        return tmp

    def _inject_geom(self, worldbody: ET.Element, pg: _PendingGeom) -> None:
        px, py, pz = pg.position
        pz_robot = pz - _ROBOT_BASE_Z  # convert to robot-base frame

        body = ET.SubElement(worldbody, "body")
        body.set("name", pg.name)
        body.set("pos", f"{px:.4f} {py:.4f} {pz_robot:.4f}")
        ET.SubElement(body, "freejoint")

        r, g, b, a = pg.color or [0.8, 0.5, 0.2, 1.0]
        geom = ET.SubElement(body, "geom")
        geom.set("name", f"{pg.name}_geom")
        geom.set("rgba", f"{r} {g} {b} {a}")
        geom.set("mass", str(pg.mass))
        geom.set("contype", "1"); geom.set("conaffinity", "1")
        geom.set("friction", "2.0 0.005 0.0001")

        if pg.shape == "box":
            sx, sy, sz = pg.size
            geom.set("type", "box")
            geom.set("size", f"{sx/2:.4f} {sy/2:.4f} {sz/2:.4f}")
        elif pg.shape == "sphere":
            geom.set("type", "sphere")
            geom.set("size", f"{pg.radius:.4f}")
        elif pg.shape == "cylinder":
            geom.set("type", "cylinder")
            geom.set("size", f"{pg.radius:.4f} {pg.height/2:.4f}")
        elif pg.shape == "mesh":
            geom.set("type", "mesh")
            geom.set("mesh", pg.mesh_asset_name)

    def _apply_collision_masks(self) -> None:
        """Robot geoms → layer 2; environment → layer 1 (anti-ghosting)."""
        model = self._robot._model
        env_names = {"floor"} | {pg.name for pg in self._pending_geoms}
        count = 0
        for i in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            bid = model.geom_bodyid[i]
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            combined = (gname + " " + bname).lower()
            if any(k in combined for k in env_names):
                model.geom_contype[i] = 1
                model.geom_conaffinity[i] = 1
            else:
                model.geom_contype[i] = 2
                model.geom_conaffinity[i] = 1
                count += 1
        print(f"[SawyerSim] Anti-ghosting applied to {count} robot geoms.")

    def _parse_urdf(self, urdf_path: str, position: List[float], name: str) -> _PendingGeom:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        urdf_dir = os.path.dirname(urdf_path)

        shape = size = radius = height = mesh_asset = color = None

        for link in root.iter("link"):
            col = link.find("collision")
            if col is None:
                continue
            geo = col.find("geometry")
            if geo is None:
                continue

            box_el      = geo.find("box")
            sphere_el   = geo.find("sphere")
            cyl_el      = geo.find("cylinder")
            mesh_el     = geo.find("mesh")

            if box_el is not None:
                shape = "box"
                size = [float(v) for v in box_el.get("size", "0.1 0.1 0.1").split()]
            elif sphere_el is not None:
                shape = "sphere"
                radius = float(sphere_el.get("radius", "0.05"))
            elif cyl_el is not None:
                shape = "cylinder"
                radius = float(cyl_el.get("radius", "0.05"))
                height = float(cyl_el.get("length", "0.1"))
            elif mesh_el is not None:
                shape = "mesh"
                fn = mesh_el.get("filename", "")
                abs_mesh = os.path.abspath(os.path.join(urdf_dir, fn))
                if not os.path.isfile(abs_mesh):
                    raise FileNotFoundError(f"Mesh not found: {abs_mesh}")
                mesh_asset = f"{name}_mesh"
                self._mesh_assets.append((mesh_asset, abs_mesh))

            # Apply collision origin offset to position
            origin = col.find("origin")
            if origin is not None:
                xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
                position = [position[i] + xyz[i] for i in range(3)]
            break  # first link only

        if shape is None:
            raise ValueError(f"No supported collision geometry in: {urdf_path}")

        # Visual color
        for link in root.iter("link"):
            vis = link.find("visual")
            if vis is None:
                continue
            mat = vis.find("material")
            if mat is not None:
                cel = mat.find("color")
                if cel is not None:
                    color = [float(v) for v in cel.get("rgba", "0.8 0.5 0.2 1").split()]
                    break

        return _PendingGeom(
            name=name, shape=shape, position=position,
            size=size, radius=radius, height=height,
            mass=0.1, color=color or [0.8, 0.5, 0.2, 1.0],
            mesh_asset_name=mesh_asset,
        )
