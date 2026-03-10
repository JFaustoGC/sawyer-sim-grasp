"""
geometry.py — strongly-typed spatial primitives for the robot client.

Types:
    Position        — 3D cartesian point (x, y, z) in metres
    Quaternion      — unit quaternion orientation (x, y, z, w)
    Euler           — roll/pitch/yaw in radians (XYZ convention)
    Pose            — position + orientation
    Pose2D          — planar position + orientation (x, z, theta)
    JointAngles     — ordered joint angles (j0 … j6) in radians
    JointState      — joint angles + velocities + efforts
    JointTrajectory — sequence of joint states over time

All types are dataclasses, JSON-serialisable (via to_dict/from_dict),
and provide .to_array() for high-performance library (NumPy/CasADi) compatibility.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Sequence, Optional, Union, Dict, Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation


# ── Position ─────────────────────────────────────────────────────────────────

@dataclass
class Position:
    """3D cartesian position in metres."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @classmethod
    def from_list(cls, v: Sequence[float]) -> Position:
        """Create from a [x, y, z] list."""
        return cls(float(v[0]), float(v[1]), float(v[2]))

    @classmethod
    def from_dict(cls, d: Union[Dict[str, float], Sequence[float]]) -> Position:
        """Create from a dictionary with 'x', 'y', 'z' keys or a list [x, y, z]."""
        if isinstance(d, (list, tuple)):
            return cls.from_list(d)
        return cls(float(d['x']), float(d['y']), float(d['z']))

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {'x': self.x, 'y': self.y, 'z': self.z}

    def to_array(self) -> np.ndarray:
        """Returns [x, y, z] as a numpy array."""
        return np.array([self.x, self.y, self.z])

    def __add__(self, other: Position) -> Position:
        """Element-wise addition."""
        return Position(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Position) -> Position:
        """Element-wise subtraction."""
        return Position(self.x - other.x, self.y - other.y, self.z - other.z)

    def distance_to(self, other: Position) -> float:
        """Euclidean distance to another position."""
        return float(np.linalg.norm(self.to_array() - other.to_array()))

    def __repr__(self) -> str:
        return f"Position(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"


@dataclass
class Point2D:
    """2D cartesian point (x, y) in metres."""
    x: float = 0.0
    y: float = 0.0

    @classmethod
    def from_list(cls, v: Sequence[float]) -> Point2D:
        return cls(float(v[0]), float(v[1]))

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def __repr__(self) -> str:
        return f"Point2D(x={self.x:.3f}, y={self.y:.3f})"


# ── Quaternion ────────────────────────────────────────────────────────────────

@dataclass
class Quaternion:
    """
    Unit quaternion orientation.
    Convention: (x, y, z, w) — same as ROS / Eigen / Pinocchio / SciPy.
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    @classmethod
    def identity(cls) -> Quaternion:
        """Return the identity quaternion [0, 0, 0, 1]."""
        return cls(0.0, 0.0, 0.0, 1.0)

    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> Quaternion:
        """Create from roll/pitch/yaw in radians (XYZ extrinsic convention)."""
        r = Rotation.from_euler('xyz', [roll, pitch, yaw])
        # cast to any to avoid scipy typing issues if necessary
        q: Any = r.as_quat()
        return cls(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> Quaternion:
        """Create from a 3×3 rotation matrix."""
        r = Rotation.from_matrix(matrix)
        q: Any = r.as_quat()
        return cls(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    @classmethod
    def from_list(cls, v: Sequence[float]) -> Quaternion:
        """Create from a [x, y, z, w] list."""
        return cls(float(v[0]), float(v[1]), float(v[2]), float(v[3]))

    @classmethod
    def from_dict(cls, d: Union[Dict[str, float], Sequence[float]]) -> Quaternion:
        """Create from a dictionary with 'x', 'y', 'z', 'w' keys or a list [x, y, z, w]."""
        if isinstance(d, (list, tuple)):
            return cls.from_list(d)
        return cls(float(d['x']), float(d['y']), float(d['z']), float(d['w']))

    def _rot(self) -> Rotation:
        """Internal scipy.spatial.transform.Rotation instance."""
        return Rotation.from_quat([self.x, self.y, self.z, self.w])

    def to_euler(self) -> Euler:
        """Return roll/pitch/yaw in radians (XYZ extrinsic)."""
        rpy: Any = self._rot().as_euler('xyz')
        return Euler(float(rpy[0]), float(rpy[1]), float(rpy[2]))

    def to_matrix(self) -> np.ndarray:
        """Return 3×3 rotation matrix."""
        return self._rot().as_matrix()

    def to_array(self) -> np.ndarray:
        """Return [x, y, z, w] as a numpy array."""
        return np.array([self.x, self.y, self.z, self.w])

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {'x': self.x, 'y': self.y, 'z': self.z, 'w': self.w}

    def normalized(self) -> Quaternion:
        """Return unit quaternion."""
        arr = self.to_array()
        norm = np.linalg.norm(arr)
        if norm < 1e-9:
            return Quaternion.identity()
        arr /= norm
        return Quaternion(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))

    def __mul__(self, other: Quaternion) -> Quaternion:
        """Composition (self * other)."""
        r = self._rot() * other._rot()
        q: Any = r.as_quat()
        return Quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def __repr__(self) -> str:
        rpy = self.to_euler()
        return (f"Quaternion(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f}, w={self.w:.4f}) "
                f"[rpy={rpy.roll:.3f}, {rpy.pitch:.3f}, {rpy.yaw:.3f} rad]")


# ── Euler ─────────────────────────────────────────────────────────────────────

@dataclass
class Euler:
    """Roll / pitch / yaw in radians (XYZ extrinsic convention)."""
    roll:  float = 0.0
    pitch: float = 0.0
    yaw:   float = 0.0

    @classmethod
    def from_degrees(cls, roll: float, pitch: float, yaw: float) -> Euler:
        """Create from roll/pitch/yaw in degrees."""
        return cls(float(np.radians(roll)), float(np.radians(pitch)), float(np.radians(yaw)))

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> Euler:
        """Create from a dictionary with 'roll', 'pitch', 'yaw' keys."""
        return cls(float(d['roll']), float(d['pitch']), float(d['yaw']))

    def to_quaternion(self) -> Quaternion:
        """Convert to unit quaternion."""
        return Quaternion.from_euler(self.roll, self.pitch, self.yaw)

    def to_array(self) -> np.ndarray:
        """Return [roll, pitch, yaw] as a numpy array."""
        return np.array([self.roll, self.pitch, self.yaw])

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {'roll': self.roll, 'pitch': self.pitch, 'yaw': self.yaw}

    def __repr__(self) -> str:
        r, p, y = np.degrees([self.roll, self.pitch, self.yaw])
        return f"Euler(r={self.roll:.3f}, p={self.pitch:.3f}, y={self.yaw:.3f} rad) [{r:.1f}°, {p:.1f}°, {y:.1f}°]"


# ── Pose ──────────────────────────────────────────────────────────────────────

@dataclass
class Pose:
    """6-DOF end-effector pose: position (m) + orientation (quaternion)."""
    position:    Position   = field(default_factory=Position)
    orientation: Quaternion = field(default_factory=Quaternion.identity)

    @classmethod
    def from_position_euler(cls, x: float, y: float, z: float,
                            roll: float = 0.0, pitch: float = 0.0,
                            yaw: float = 0.0) -> Pose:
        """Convenience constructor from xyz + rpy (radians)."""
        return cls(Position(x, y, z), Quaternion.from_euler(roll, pitch, yaw))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Pose:
        """Create from a dictionary representation."""
        return cls(
            position=Position.from_dict(d['position']),
            orientation=Quaternion.from_dict(d['orientation'])
        )

    def to_matrix(self) -> np.ndarray:
        """Return 4×4 homogeneous transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self.orientation.to_matrix()
        T[:3,  3] = self.position.to_array()
        return T

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> Pose:
        """Create from a 4×4 homogeneous transformation matrix."""
        return cls(Position.from_list(T[:3, 3]), Quaternion.from_matrix(T[:3, :3]))

    def to_array(self) -> np.ndarray:
        """Returns [x, y, z, qx, qy, qz, qw] as a numpy array."""
        return np.concatenate([self.position.to_array(), self.orientation.to_array()])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'position': self.position.to_dict(),
            'orientation': self.orientation.to_dict()
        }

    def __repr__(self) -> str:
        rpy = self.orientation.to_euler()
        return (f"Pose(pos=({self.position.x:.3f}, {self.position.y:.3f}, {self.position.z:.3f}) "
                f"rpy=({rpy.roll:.3f}, {rpy.pitch:.3f}, {rpy.yaw:.3f}))")


# ── Pose2D ────────────────────────────────────────────────────────────────────

@dataclass
class Pose2D:
    """Planar 3-DOF pose (x, z, theta) where theta is pitch."""
    x: float = 0.0
    z: float = 0.0
    theta: float = 0.0  # Pitch in radians

    @classmethod
    def from_list(cls, v: Sequence[float]) -> Pose2D:
        """Create from a [x, z, theta] list."""
        return cls(float(v[0]), float(v[1]), float(v[2]))

    def to_array(self) -> np.ndarray:
        """Returns [x, z, theta] as a numpy array."""
        return np.array([self.x, self.z, self.theta])

    def to_pose(self, y: float = 0.0) -> Pose:
        """Convert to 6-DOF Pose (assumes rotation is purely pitch)."""
        return Pose.from_position_euler(self.x, y, self.z, pitch=self.theta)

    @classmethod
    def from_pose(cls, pose: Pose) -> Pose2D:
        """Extract (x, z, pitch) from 6-DOF Pose."""
        rpy = pose.orientation.to_euler()
        return cls(pose.position.x, pose.position.z, float(rpy.pitch))

    def __repr__(self) -> str:
        deg = float(np.degrees(self.theta))
        return f"Pose2D(x={self.x:.3f}, z={self.z:.3f} m, theta={deg:.1f}°)"


# ── JointAngles ───────────────────────────────────────────────────────────────

class JointAngles:
    """
    Joint angles for a 7-DOF arm (Sawyer) in radians.
    Behaves as a 7-element float64 array.
    """
    __slots__ = ['_values']

    def __init__(self, values: Union['JointAngles', Sequence[float], np.ndarray]):
        if isinstance(values, JointAngles):
            arr = np.copy(values._values)
        else:
            arr = np.array(values, dtype=np.float64, copy=None)
            arr = np.atleast_1d(arr.squeeze())
        if arr.shape != (7,):
            raise ValueError(f"Expected exactly 7 joint values for Sawyer, got {len(arr) if arr.ndim == 1 else arr.shape}")
        self._values = arr

    @property
    def values(self) -> np.ndarray:
        return self._values

    def __array__(self) -> np.ndarray:
        return self._values

    def __getitem__(self, idx: int) -> float:
        return self._values[idx]

    def __setitem__(self, idx: int, value: float):
        self._values[idx] = value

    def __len__(self) -> int:
        return 7

    def __iter__(self):
        return iter(self._values)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, (JointAngles, Sequence, np.ndarray)):
            return False
        try:
            return bool(np.allclose(self._values, np.asarray(other, dtype=np.float64)))
        except (ValueError, TypeError):
            return False

    def __repr__(self) -> str:
        vals = ", ".join(f"j{i}:{x:.3f}" for i, x in enumerate(self._values))
        return f"JointAngles([{vals}])"


JointsLike = Union[JointAngles, Sequence[float], np.ndarray]


# ── Trajectories ─────────────────────────────────────────────────────────────

@dataclass
class JointState:
    """Full state of the joints at a single point in time."""
    position: JointAngles
    velocity: Optional[JointAngles] = None
    acceleration: Optional[JointAngles] = None
    effort: Optional[JointAngles] = None
    time_from_start: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary for JSON serialisation."""
        d: Dict[str, Any] = {
            'position': self.position.values.tolist(),
            'time_from_start': self.time_from_start
        }
        if self.velocity is not None: d['velocity'] = self.velocity.values.tolist()
        if self.acceleration is not None: d['acceleration'] = self.acceleration.values.tolist()
        if self.effort is not None: d['effort'] = self.effort.values.tolist()
        return d


@dataclass
class JointTrajectory:
    """A sequence of joint states (e.g., from a planner)."""
    points: List[JointState] = field(default_factory=list)

    @classmethod
    def from_arrays(cls, Q: np.ndarray, 
                    Qd: Optional[np.ndarray] = None, 
                    Qdd: Optional[np.ndarray] = None, 
                    dt: float = 0.01) -> 'JointTrajectory':
        """
        Build from raw NumPy matrices (rows=time_steps, cols=joints).
        """
        traj = cls()
        for i in range(Q.shape[0]):
            pos = JointAngles(Q[i])
            vel = JointAngles(Qd[i]) if Qd is not None else None
            acc = JointAngles(Qdd[i]) if Qdd is not None else None
            traj.points.append(JointState(
                position=pos, 
                velocity=vel, 
                acceleration=acc, 
                time_from_start=float(i * dt)
            ))
        return traj

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Convert to nested dictionary for JSON serialisation."""
        return {'points': [p.to_dict() for p in self.points]}

    def to_position_array(self) -> np.ndarray:
        """Extract all positions into a (N, 7) array."""
        return np.stack([p.position.values for p in self.points])

    def to_velocity_array(self) -> np.ndarray:
        """Extract all velocities into a (N, 7) array."""
        vels = []
        for p in self.points:
            if p.velocity is not None:
                vels.append(p.velocity.values)
            else:
                vels.append(np.zeros(7))
        return np.stack(vels)

    def to_acceleration_array(self) -> np.ndarray:
        """Extract all accelerations into a (N, 7) array."""
        accs = []
        for p in self.points:
            if p.acceleration is not None:
                accs.append(p.acceleration.values)
            else:
                accs.append(np.zeros(7))
        return np.stack(accs)

    def resample(self, dt: float = 0.01) -> 'JointTrajectory':
        """Resample trajectory to uniform dt (default 100Hz) using linear interpolation."""
        orig_times = np.array([p.time_from_start for p in self.points])
        T_total = orig_times[-1] if len(orig_times) > 0 else 0
        if T_total <= 0:
            return self
        new_times = np.arange(0.0, T_total, dt)
        if len(new_times) == 0:
            return self

        Q = self.to_position_array()
        Qd = self.to_velocity_array()
        Qdd = self.to_acceleration_array()

        Q_new = interp1d(orig_times, Q, axis=0)(new_times)
        Qd_new = interp1d(orig_times, Qd, axis=0)(new_times)
        Qdd_new = interp1d(orig_times, Qdd, axis=0)(new_times)

        return JointTrajectory.from_arrays(Q_new, Qd_new, Qdd_new, dt)
