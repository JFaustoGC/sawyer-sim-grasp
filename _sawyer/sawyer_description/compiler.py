"""
compiler.py — Pure-Python robot URDF compiler / post-processor.

Takes a canonical URDF and applies simulator-specific modifications:
  - PYBULLET:  Per-link RGBA colors, original DAE/STL meshes, absolute paths
  - MUJOCO:    Robosuite OBJ meshes (if available), stripped Gazebo tags
  - DRAKE:     Absolute mesh paths, stripped Gazebo tags
  - GAZEBO:    Keeps Gazebo plugin/sensor tags
  - GENERIC:   Clean URDF with relative ../meshes/ paths

Usage:
    from sawyer_description.compiler import RobotCompiler, SimTarget

    compiler = RobotCompiler()
    path = compiler.compile(SimTarget.PYBULLET, output_path="/tmp/sawyer_pb.urdf")
"""

import math
import os
import re
import copy
import xml.etree.ElementTree as ET
from enum import Enum
from typing import Optional, Dict, List, Tuple


class SimTarget(Enum):
    """Target simulator for URDF post-processing."""
    GENERIC  = "generic"
    PYBULLET = "pybullet"
    MUJOCO   = "mujoco"
    DRAKE    = "drake"
    GAZEBO   = "gazebo"


# ── Per-link color palette for PyBullet ──────────────────────────────────────

LINK_COLORS: Dict[str, Tuple[float, float, float, float]] = {
    # Base / pedestal
    "base":                     (0.20, 0.20, 0.22, 1.0),
    "right_arm_base_link":      (0.20, 0.20, 0.22, 1.0),
    "pedestal":                 (0.15, 0.15, 0.17, 1.0),

    # Arm links — rainbow progression
    "right_l0":                 (0.85, 0.12, 0.12, 1.0),  # red
    "right_l1":                 (0.95, 0.55, 0.10, 1.0),  # orange
    "right_l2":                 (0.95, 0.85, 0.10, 1.0),  # yellow
    "right_l3":                 (0.20, 0.75, 0.20, 1.0),  # green
    "right_l4":                 (0.20, 0.50, 0.90, 1.0),  # blue
    "right_l5":                 (0.55, 0.25, 0.85, 1.0),  # purple
    "right_l6":                 (0.85, 0.20, 0.60, 1.0),  # magenta

    # Hand / end-effector
    "right_hand":               (0.70, 0.70, 0.70, 1.0),
    "head":                     (0.40, 0.40, 0.42, 1.0),

    # Adapter / sensor chain
    "right_gripper_base":           (0.30, 0.30, 0.30, 1.0),
    "right_connector_plate_base":   (0.15, 0.15, 0.15, 1.0),  # black adapter
    "right_connector_plate_mount":  (0.15, 0.15, 0.15, 1.0),
    "right_ft_sensor_link":         (0.60, 0.60, 0.62, 1.0),  # silver sensor
    "right_adapter_top":            (0.55, 0.55, 0.57, 1.0),  # light gray adapter
    "right_adapter_bottom":         (0.55, 0.55, 0.57, 1.0),

    # Gripper — dark accents
    "right_pneumatic_gripper_base": (0.40, 0.15, 0.55, 1.0),  # purple (mocha base)
    "right_gripper":                (0.30, 0.30, 0.30, 1.0),
    "right_gripper_l_finger":       (0.25, 0.25, 0.25, 1.0),
    "right_gripper_r_finger":       (0.25, 0.25, 0.25, 1.0),
    "right_gripper_l_finger_tip":   (0.35, 0.35, 0.35, 1.0),
    "right_gripper_r_finger_tip":   (0.35, 0.35, 0.35, 1.0),
    "right_gripper_l_finger_base":  (0.30, 0.30, 0.30, 1.0),
    "right_gripper_r_finger_base":  (0.30, 0.30, 0.30, 1.0),
    "right_gripper_l_finger_body":  (0.20, 0.20, 0.20, 1.0),
    "right_gripper_r_finger_body":  (0.20, 0.20, 0.20, 1.0),
    "right_gripper_tip":            (0.90, 0.90, 0.90, 1.0),
    "right_l_finger_mount":           (0.30, 0.30, 0.30, 1.0),
    "right_r_finger_mount":           (0.30, 0.30, 0.30, 1.0),
    "right_gripper_crossbar":                   (0.35, 0.35, 0.35, 1.0),
}

# Default color for any link not in the palette
_DEFAULT_COLOR = (0.50, 0.50, 0.50, 1.0)


class RobotCompiler:
    """
    Pure-Python URDF post-processor for multi-simulator support.

    Starts from a pre-compiled canonical URDF (from xacro or hand-edited)
    and applies simulator-specific processing.
    """

    def __init__(self, package_root: Optional[str] = None):
        """
        Args:
            package_root: Root of the sawyer_description package.
                          Defaults to the directory containing this file.
        """
        if package_root is None:
            package_root = os.path.dirname(os.path.abspath(__file__))
        self.package_root = package_root
        self.urdf_dir = os.path.join(package_root, "urdf")
        self.meshes_dir = os.path.join(package_root, "meshes")

    def compile(
        self,
        target: SimTarget = SimTarget.GENERIC,
        source_urdf: Optional[str] = None,
        output_path: Optional[str] = None,
        with_gripper: bool = True,
    ) -> str:
        """
        Compile a URDF for a specific simulator target.

        Args:
            target: Which simulator to post-process for.
            source_urdf: Path to the source URDF. Defaults to the canonical
                         assembled/sawyer_pneumatic.urdf.
            output_path: Where to write the result. Defaults to
                         urdf/sim/sawyer_{target.value}.urdf.

        Returns:
            Absolute path to the compiled URDF.
        """
        # ── MuJoCo special path: generate MJCF (not URDF) ────────────────────
        if target == SimTarget.MUJOCO:
            return self._compile_mujoco(output_path, with_gripper=with_gripper)

        # ── Standard URDF path ───────────────────────────────────────────────
        if source_urdf is None:
            if with_gripper:
                source_urdf = os.path.join(
                    self.urdf_dir, "assembled", "sawyer_pneumatic.urdf"
                )
            else:
                # Arm-only URDF (no gripper chain)
                source_urdf = os.path.join(
                    self.urdf_dir, "assembled", "sawyer_electric.urdf"
                )

        if output_path is None:
            os.makedirs(os.path.join(self.urdf_dir, "sim"), exist_ok=True)
            output_path = os.path.join(
                self.urdf_dir, "sim", f"sawyer_{target.value}.urdf"
            )

        if not os.path.isfile(source_urdf):
            raise FileNotFoundError(f"Source URDF not found: {source_urdf}")

        tree = ET.parse(source_urdf)
        root = tree.getroot()

        # Apply simulator-specific post-processing
        if target == SimTarget.PYBULLET:
            self._apply_pybullet(root, source_urdf)
        elif target == SimTarget.DRAKE:
            self._apply_drake(root, source_urdf)
        elif target == SimTarget.GAZEBO:
            pass  # Keep everything as-is
        else:
            # GENERIC: strip Gazebo, keep relative paths
            self._strip_gazebo(root)

        # Write output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tree.write(output_path, xml_declaration=True, encoding="UTF-8")

        return output_path

    # ── PyBullet Post-Processor ──────────────────────────────────────────────

    def _apply_pybullet(self, root: ET.Element, source_urdf: str) -> None:
        """Add per-link RGBA materials, resolve mesh paths to absolute."""
        self._strip_gazebo(root)
        self._resolve_mesh_paths_absolute(root, source_urdf)
        self._add_per_link_colors(root)

    def _add_per_link_colors(self, root: ET.Element) -> None:
        """
        Add <material> with RGBA color to each link's visual elements.

        If a link already has a <material>, it is replaced. Links not in
        LINK_COLORS get the default gray.
        """
        for link in root.findall(".//link"):
            link_name = link.get("name", "")
            rgba = LINK_COLORS.get(link_name, _DEFAULT_COLOR)
            rgba_str = f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"

            for visual in link.findall("visual"):
                # Remove existing material
                for mat in visual.findall("material"):
                    visual.remove(mat)

                # Add new material with the link color
                mat_name = f"color_{link_name}"
                mat_el = ET.SubElement(visual, "material", {"name": mat_name})
                ET.SubElement(mat_el, "color", {"rgba": rgba_str})

    # ── MuJoCo: Native MJCF Generation ────────────────────────────────────────

    def _compile_mujoco(self, output_path: Optional[str] = None, with_gripper: bool = True) -> str:
        """
        Generate native MJCF for MuJoCo by copying the robosuite robot.xml,
        resolving mesh paths to absolute, and injecting the gripper/adapter
        chain from the pneumatic gripper URDF.

        MuJoCo's URDF loader is limited — it cannot render multi-visual
        OBJ meshes. The robosuite project maintains a proper MJCF model
        with per-part materials and mesh declarations, which we use as the
        canonical MuJoCo description for the arm. The gripper/adapter chain
        is converted from URDF to MJCF and injected at the right_hand body.
        """
        # Locate the robosuite MJCF source
        robosuite_xml = os.path.join(
            self.meshes_dir, "sawyer_description", "robosuite", "robot.xml"
        )

        # Fallback: try to find it via the robosuite package
        if not os.path.isfile(robosuite_xml):
            try:
                import robosuite as rs
                robosuite_xml = os.path.join(
                    os.path.dirname(rs.__file__),
                    "models", "assets", "robots", "sawyer", "robot.xml",
                )
            except ImportError:
                pass

        if not os.path.isfile(robosuite_xml):
            raise FileNotFoundError(
                f"Cannot find robosuite robot.xml. "
                f"Expected at: {robosuite_xml}"
            )

        if output_path is None:
            os.makedirs(os.path.join(self.urdf_dir, "sim"), exist_ok=True)
            output_path = os.path.join(
                self.urdf_dir, "sim", "sawyer_mujoco.xml"
            )

        # Parse and resolve mesh paths to absolute
        tree = ET.parse(robosuite_xml)
        root = tree.getroot()

        source_dir = os.path.dirname(os.path.abspath(robosuite_xml))
        for mesh in root.findall(".//mesh[@file]"):
            rel_path = mesh.get("file", "")
            if rel_path and not os.path.isabs(rel_path):
                abs_path = os.path.normpath(os.path.join(source_dir, rel_path))
                # Fallback: strip obj_meshes/ prefix (robosuite uses obj_meshes/
                # but our local copy stores meshes directly in link subdirs)
                if not os.path.isfile(abs_path):
                    stripped = rel_path.replace("obj_meshes/", "", 1)
                    alt_path = os.path.normpath(os.path.join(source_dir, stripped))
                    if os.path.isfile(alt_path):
                        abs_path = alt_path
                mesh.set("file", abs_path)

        # Ensure collision primitive geoms (cylinders/spheres without mesh refs)
        # are kept for physics and marked as collision-only (invisible).
        # Visual geoms have mesh="..." attribute; collision geoms use primitive shapes.
        for body in root.findall(".//body"):
            for geom in body.findall("geom"):
                if geom.get("mesh") is None:
                    geom.set("group", "3")        # invisible in default rendering
                    geom.set("contype", "1")
                    geom.set("conaffinity", "1")

        # Inject the gripper/adapter chain into the right_hand body
        if with_gripper:
            self._inject_gripper_into_mjcf(root)
            # Add collision primitives for the gripper finger tips.
            # The gripper visual meshes are contype=0 (no collision) because
            # the STLs have degenerate geometry. Simple boxes approximate the
            # finger pads well enough for grasping.
            # Local axes of the finger_tip bodies:
            #   X = across finger width
            #   Y = along finger (forward/downward toward tip)
            #   Z = between fingers (sideways) — NOT the forward direction
            # Old code used Z=0.035 which displaced boxes 3.4cm sideways. Fixed to Y.
            _finger_col = {
                "right_gripper_l_finger_tip": {"pos": "0 0.000 0.003", "euler": "0 1.5708 0", "size": "0.007 0.020 0.010"},
                "right_gripper_r_finger_tip": {"pos": "0 0.000 -0.003", "euler": "0 1.5708 0", "size": "0.007 0.020 0.010"},
            }
            for fname, attrs in _finger_col.items():
                fbody = root.find(f".//body[@name='{fname}']")
                if fbody is not None:
                    ET.SubElement(fbody, "geom", {
                        "name": f"{fname}_col",
                        "type": "box",
                        "pos": attrs["pos"],
                        "euler": attrs["euler"],
                        "size": attrs["size"],
                        "contype": "1",
                        "conaffinity": "1",
                        "group": "3",
                        "rgba": "0.3 0.3 0.3 0",
                        "friction": "5 0.005 0.0001",
                        "solimp": "0.99 0.999 0.001",
                        "solref": "0.004 1",
                        "condim": "6",
                    })

        # Ensure an end-effector site exists on right_hand (the robosuite
        # model places right_center at the base, not the EE).
        # Remove the misplaced right_center from wherever it is
        for body in root.findall(".//body"):
            for site in body.findall("site"):
                if site.get("name") == "right_center":
                    body.remove(site)

        # Add EE site on right_gripper_tip (the actual tool tip)
        tip_body = root.find(".//body[@name='right_gripper_tip']")
        if tip_body is None:
            # Fallback to right_hand if tip body doesn't exist
            tip_body = root.find(".//body[@name='right_hand']")
        if tip_body is not None:
            ET.SubElement(tip_body, "site", {
                "name": "right_eef",
                "pos": "-0.020 0 0",  # offset to pinch center (midpoint of finger_tip bodies in local frame)
                "size": "0.005",
                "rgba": "1 0.3 0.3 0.3",
                "group": "3",
            })

        # Enable multiccd for better contact point coverage on flat surfaces
        option_el = root.find("option")
        if option_el is None:
            option_el = ET.SubElement(root, "option")
        flag_el = option_el.find("flag")
        if flag_el is None:
            flag_el = ET.SubElement(option_el, "flag")
        flag_el.set("multiccd", "enable")

        # Write output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tree.write(output_path, xml_declaration=True, encoding="UTF-8")

        return output_path

    # ── MuJoCo Gripper Injection ──────────────────────────────────────────────

    @staticmethod
    def _rpy_to_quat(r: float, p: float, y: float) -> Tuple[float, float, float, float]:
        """Convert roll-pitch-yaw (radians) to quaternion (w, x, y, z)."""
        cr, sr = math.cos(r / 2), math.sin(r / 2)
        cp, sp = math.cos(p / 2), math.sin(p / 2)
        cy, sy = math.cos(y / 2), math.sin(y / 2)
        return (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )

    def _inject_gripper_into_mjcf(self, mjcf_root: ET.Element) -> None:
        """
        Parse the pneumatic gripper URDF, convert to MJCF body/geom
        elements, and inject them into the right_hand body of the MJCF tree.

        Skips injection gracefully if the gripper URDF or its mesh files
        are not yet available (e.g. during gripper design iteration).
        """
        gripper_urdf_path = os.path.join(
            self.urdf_dir, "base", "pneumatic_gripper.urdf"
        )
        if not os.path.isfile(gripper_urdf_path):
            print(f"WARNING: Gripper URDF not found: {gripper_urdf_path}")
            return

        # Pre-flight: verify that all referenced mesh files exist before
        # injecting.  If any are missing, skip injection entirely so the
        # arm-only MJCF still loads in MuJoCo.
        #
        # Mesh paths in the gripper URDF use ../meshes/... which is relative
        # to the urdf/ directory (not urdf/base/ where the file lives).
        gripper_tree = ET.parse(gripper_urdf_path)
        gripper_root = gripper_tree.getroot()
        for mesh_el in gripper_root.iter("mesh"):
            filename = mesh_el.get("filename", "")
            if not filename:
                continue
            if os.path.isabs(filename):
                abs_path = filename
            else:
                abs_path = os.path.normpath(os.path.join(self.urdf_dir, filename))
            if not os.path.isfile(abs_path):
                print(f"WARNING: Gripper mesh missing ({os.path.basename(abs_path)}), "
                      f"skipping gripper injection.")
                return

        # Find the right_hand body in the MJCF
        right_hand = mjcf_root.find(".//body[@name='right_hand']")
        if right_hand is None:
            print("WARNING: 'right_hand' body not found in MJCF")
            return

        # Find or create the <asset> section
        asset = mjcf_root.find("asset")
        if asset is None:
            asset = ET.SubElement(mjcf_root, "asset")

        # Build link and joint maps (gripper_root already parsed above)
        links: Dict[str, ET.Element] = {}
        for link in gripper_root.findall("link"):
            links[link.get("name")] = link

        children: Dict[str, List[Tuple[ET.Element, str]]] = {}
        for joint in gripper_root.findall("joint"):
            parent_el = joint.find("parent")
            child_el = joint.find("child")
            if parent_el is None or child_el is None:
                continue
            parent_name = parent_el.get("link")
            child_name = child_el.get("link")
            children.setdefault(parent_name, []).append((joint, child_name))

        # The gripper URDF has a joint from right_hand → right_gripper_base.
        # Start building from right_gripper_base under the MJCF right_hand body.
        gripper_start_joints = children.get("right_hand", [])
        if not gripper_start_joints:
            print("WARNING: No joints from right_hand in gripper URDF")
            return

        # Resolve mesh paths relative to the urdf/ directory (not urdf/base/)
        # because the gripper URDF uses ../meshes/... paths.
        for joint_el, child_name in gripper_start_joints:
            self._convert_urdf_link_to_mjcf(
                child_name, joint_el, links, children,
                right_hand, asset, self.urdf_dir,
            )

        # Add position actuators for the finger slide joints so the gripper
        # generates real contact (squeeze) force instead of just teleporting.
        actuator = mjcf_root.find("actuator")
        if actuator is None:
            actuator = ET.SubElement(mjcf_root, "actuator")

        finger_joints = {
            "right_gripper_l_finger_joint": "mot_right_gripper_l_finger",
            "right_gripper_r_finger_joint": "mot_right_gripper_r_finger",
        }
        for joint_name, act_name in finger_joints.items():
            # Motor actuator: applies constant force = ctrl (N).
            # Positive ctrl = open (joint increases), negative ctrl = close (squeeze).
            # This gives steady grip force independent of finger position,
            # which is essential for maintaining contact during dynamic motions.
            jel = mjcf_root.find(f".//joint[@name='{joint_name}']")
            if jel is not None:
                ET.SubElement(actuator, "motor", {
                    "name":       act_name,
                    "joint":      joint_name,
                    "gear":       "1",
                    "ctrlrange":  "-200 200",
                    "forcerange": "-200 200",
                })

    def _convert_urdf_link_to_mjcf(
        self,
        link_name: str,
        joint_el: ET.Element,
        links: Dict[str, ET.Element],
        children: Dict[str, List[Tuple[ET.Element, str]]],
        parent_body: ET.Element,
        asset: ET.Element,
        urdf_dir: str,
    ) -> None:
        """Recursively convert a URDF link to an MJCF body element."""
        link = links.get(link_name)
        if link is None:
            return

        # Extract joint transform → MJCF body pos/quat
        body_attribs: Dict[str, str] = {"name": link_name}
        origin = joint_el.find("origin") if joint_el is not None else None
        if origin is not None:
            body_attribs["pos"] = origin.get("xyz", "0 0 0")
            rpy_str = origin.get("rpy", "0 0 0")
            r, p, y = (float(v) for v in rpy_str.split())
            if abs(r) > 1e-6 or abs(p) > 1e-6 or abs(y) > 1e-6:
                w, qx, qy, qz = self._rpy_to_quat(r, p, y)
                body_attribs["quat"] = f"{w:.6f} {qx:.6f} {qy:.6f} {qz:.6f}"
        else:
            body_attribs["pos"] = "0 0 0"

        body = ET.SubElement(parent_body, "body", body_attribs)

        # Inertial
        inertial = link.find("inertial")
        if inertial is not None:
            mass_el = inertial.find("mass")
            inertia_el = inertial.find("inertia")
            inertial_origin = inertial.find("origin")

            mass_val = mass_el.get("value", "0.001") if mass_el is not None else "0.001"
            # Clamp tiny masses
            if float(mass_val) < 1e-4:
                mass_val = "1e-04"

            inertial_pos = "0 0 0"
            inertial_quat = None
            if inertial_origin is not None:
                inertial_pos = inertial_origin.get("xyz", "0 0 0")
                rpy_str = inertial_origin.get("rpy", "0 0 0")
                r, p, y = (float(v) for v in rpy_str.split())
                if abs(r) > 1e-6 or abs(p) > 1e-6 or abs(y) > 1e-6:
                    w, qx, qy, qz = self._rpy_to_quat(r, p, y)
                    inertial_quat = f"{w:.6f} {qx:.6f} {qy:.6f} {qz:.6f}"

            diag = "0.0001 0.0001 0.0001"
            if inertia_el is not None:
                ixx = max(float(inertia_el.get("ixx", "0")), 1e-6)
                iyy = max(float(inertia_el.get("iyy", "0")), 1e-6)
                izz = max(float(inertia_el.get("izz", "0")), 1e-6)
                diag = f"{ixx} {iyy} {izz}"

            inertial_attribs = {
                "mass": mass_val,
                "pos": inertial_pos,
                "diaginertia": diag,
            }
            if inertial_quat:
                inertial_attribs["quat"] = inertial_quat
            ET.SubElement(body, "inertial", inertial_attribs)

        # Joint (only for non-fixed types)
        if joint_el is not None:
            jtype = joint_el.get("type", "fixed")
            if jtype in ("prismatic", "revolute"):
                axis_el = joint_el.find("axis")
                axis = axis_el.get("xyz", "0 0 1") if axis_el is not None else "0 0 1"
                limit = joint_el.find("limit")

                jnt_attribs: Dict[str, str] = {
                    "name": joint_el.get("name"),
                    "type": "slide" if jtype == "prismatic" else "hinge",
                    "axis": axis,
                }
                if limit is not None:
                    lo = limit.get("lower", "0")
                    hi = limit.get("upper", "0.01")
                    jnt_attribs["range"] = f"{lo} {hi}"
                    jnt_attribs["limited"] = "true"
                ET.SubElement(body, "joint", jnt_attribs)

        # Visual geometry → MJCF geoms
        for visual in link.findall("visual"):
            geom_elem = visual.find("geometry")
            if geom_elem is None:
                continue

            mesh_el = geom_elem.find("mesh")
            if mesh_el is None:
                continue

            mesh_filename = mesh_el.get("filename", "")
            if not mesh_filename:
                continue

            # Resolve mesh path
            if not os.path.isabs(mesh_filename):
                mesh_abs = os.path.normpath(
                    os.path.join(urdf_dir, mesh_filename)
                )
            else:
                mesh_abs = mesh_filename

            # Unique mesh asset name
            base_name = os.path.splitext(os.path.basename(mesh_filename))[0]
            mesh_asset_name = f"gripper_{base_name}"

            # Register mesh in <asset>
            mesh_asset_attribs: Dict[str, str] = {
                "name": mesh_asset_name,
                "file": mesh_abs,
            }
            mesh_scale = mesh_el.get("scale")
            if mesh_scale:
                mesh_asset_attribs["scale"] = mesh_scale
            ET.SubElement(asset, "mesh", mesh_asset_attribs)

            # Build geom attributes
            geom_attribs: Dict[str, str] = {
                "type": "mesh",
                "mesh": mesh_asset_name,
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            }

            # Visual origin transform
            vis_origin = visual.find("origin")
            if vis_origin is not None:
                xyz = vis_origin.get("xyz", "0 0 0")
                if xyz.strip() != "0 0 0":
                    geom_attribs["pos"] = xyz
                rpy_str = vis_origin.get("rpy", "0 0 0")
                r, p, y = (float(v) for v in rpy_str.split())
                if abs(r) > 1e-6 or abs(p) > 1e-6 or abs(y) > 1e-6:
                    w, qx, qy, qz = self._rpy_to_quat(r, p, y)
                    geom_attribs["quat"] = f"{w:.6f} {qx:.6f} {qy:.6f} {qz:.6f}"

            # Material color
            mat = visual.find("material")
            if mat is not None:
                color = mat.find("color")
                if color is not None:
                    geom_attribs["rgba"] = color.get("rgba", "0.5 0.5 0.5 1")
                else:
                    geom_attribs["rgba"] = "0.5 0.5 0.5 1"
            else:
                geom_attribs["rgba"] = "0.5 0.5 0.5 1"

            ET.SubElement(body, "geom", geom_attribs)

        # Skip collision meshes — the gripper collision STLs contain
        # degenerate geometry that fails MuJoCo's convex hull (qhull).
        # Visual-only geoms are sufficient for the gripper chain.

        # Recurse for child links
        for child_joint, child_name in children.get(link_name, []):
            self._convert_urdf_link_to_mjcf(
                child_name, child_joint, links, children,
                body, asset, urdf_dir,
            )

    # ── Drake Post-Processor ─────────────────────────────────────────────────

    def _apply_drake(self, root: ET.Element, source_urdf: str) -> None:
        """Resolve mesh paths, convert STL→OBJ, strip collisions, complete materials, color links."""
        self._strip_gazebo(root)
        self._strip_collisions(root)
        self._resolve_mesh_paths_absolute(root, source_urdf)
        self._convert_stl_to_obj(root)
        self._complete_materials(root)
        self._add_per_link_colors(root)

    # ── Common Utilities ─────────────────────────────────────────────────────

    def _convert_stl_to_obj(self, root: ET.Element) -> None:
        """
        Convert any visual STL meshes to OBJ (baking the scale into vertices).

        Drake's Meshcat only renders .obj / .gltf / .dae — not .stl.
        For each STL visual mesh we load it via trimesh, apply the URDF scale,
        export as OBJ alongside the original, and update the URDF element.
        """
        try:
            import trimesh
        except ImportError:
            print("WARNING: trimesh not installed — cannot convert STL→OBJ for Drake")
            return

        obj_cache_dir = os.path.join(self.meshes_dir, "_drake_obj_cache")
        os.makedirs(obj_cache_dir, exist_ok=True)

        for mesh_el in root.iter("mesh"):
            filename = mesh_el.get("filename", "")
            if not filename:
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext != ".stl":
                continue

            # Parse scale (URDF allows "sx sy sz" or omitted = 1 1 1)
            scale_str = mesh_el.get("scale", "1 1 1")
            scale = [float(s) for s in scale_str.split()]
            if len(scale) == 1:
                scale = scale * 3

            # Derive a stable OBJ filename from the source path
            rel_key = os.path.relpath(filename, self.meshes_dir) if os.path.isabs(filename) else filename
            obj_name = rel_key.replace(os.sep, "_").replace("/", "_")
            obj_name = os.path.splitext(obj_name)[0] + ".obj"
            obj_path = os.path.join(obj_cache_dir, obj_name)

            # Convert only if OBJ doesn't exist or STL is newer
            if not os.path.isfile(obj_path) or (
                os.path.isfile(filename) and os.path.getmtime(filename) > os.path.getmtime(obj_path)
            ):
                if not os.path.isfile(filename):
                    continue
                mesh_data = trimesh.load(filename, force="mesh")
                mesh_data.apply_scale(scale)
                mesh_data.export(obj_path, file_type="obj")

            # Update URDF element: point to OBJ, remove scale
            mesh_el.set("filename", obj_path)
            if "scale" in mesh_el.attrib:
                del mesh_el.attrib["scale"]

    def _strip_gazebo(self, root: ET.Element) -> None:
        """Remove all <gazebo> tags from the URDF."""
        for gazebo in root.findall("gazebo"):
            root.remove(gazebo)

    def _strip_collisions(self, root: ET.Element) -> None:
        """Remove all <collision> elements from links (MuJoCo ignores them)."""
        for link in root.findall(".//link"):
            for collision in link.findall("collision"):
                link.remove(collision)

    def _strip_namonly_materials(self, root: ET.Element) -> None:
        """
        Remove <material> elements that reference a name without inline
        color data. MuJoCo's URDF loader can choke on undefined material refs.
        """
        for visual in root.findall(".//visual"):
            for mat in visual.findall("material"):
                # Keep materials that have a <color> child
                if mat.find("color") is not None:
                    continue
                # Remove name-only material references
                visual.remove(mat)

    def _complete_materials(self, root: ET.Element) -> None:
        """
        Ensure every <material> has an inline <color> element.
        Drake requires materials to be fully defined on first use.

        Builds a name→rgba map from top-level <material> definitions,
        then fills in any reference that lacks a <color> child.
        """
        # Build a lookup from top-level material definitions
        mat_map: Dict[str, str] = {}
        for mat in root.findall("material"):
            name = mat.get("name", "")
            color = mat.find("color")
            if color is not None and name:
                mat_map[name] = color.get("rgba", "0.5 0.5 0.5 1")

        # Fill in missing color on visual material refs
        for visual in root.findall(".//visual"):
            for mat in visual.findall("material"):
                if mat.find("color") is not None:
                    continue
                name = mat.get("name", "")
                rgba = mat_map.get(name, "0.5 0.5 0.5 1")
                ET.SubElement(mat, "color", {"rgba": rgba})

    def _resolve_mesh_paths_absolute(self, root: ET.Element, source_urdf: str) -> None:
        """
        Convert relative mesh paths to absolute paths.

        The source URDFs use paths like ``../meshes/...`` which were
        originally relative to the flat ``urdf/`` directory. Since we
        moved URDFs into subfolders (``urdf/assembled/``, ``urdf/base/``),
        we resolve ``../meshes/`` relative to ``urdf/`` (not the subfolder)
        so that ``../meshes/`` correctly reaches the package-level ``meshes/``.

        Also handles a legacy quirk where some URDFs contain a redundant
        ``/meshes/`` segment (e.g. ``sawyer_description/meshes/robosuite/``
        when the real path is ``sawyer_description/robosuite/``).
        """
        for mesh in root.iter("mesh"):
            filename = mesh.get("filename", "")
            if not filename or os.path.isabs(filename):
                continue

            # Resolve relative to urdf/ directory (not the subfolder)
            # so ../meshes/ correctly → package_root/meshes/
            abs_path = os.path.normpath(os.path.join(self.urdf_dir, filename))

            # Fallback: strip redundant /meshes/ segment from legacy URDFs
            # The source URDFs have paths like .../meshes/sawyer_description/meshes/robosuite/
            # but the actual layout is .../meshes/sawyer_description/robosuite/
            if not os.path.isfile(abs_path):
                # Find first /meshes/ and check if remainder has another /meshes/
                marker = "/meshes/"
                idx = abs_path.find(marker)
                if idx >= 0:
                    after_first = abs_path[idx + len(marker):]
                    if marker.lstrip("/") in after_first:
                        fixed = abs_path[:idx] + marker + after_first.replace("meshes/", "", 1)
                        if os.path.isfile(fixed):
                            abs_path = fixed

            mesh.set("filename", abs_path)

