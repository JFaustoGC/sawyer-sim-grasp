import numpy as np
from sawyer_common.geometry import Quaternion
from scipy.spatial.transform import Rotation as R

class RotationPrimitive:
    def __init__(self, num_rotations=4, total_deg = 180):
        self.num_rotations = num_rotations
        self.total_deg = total_deg
        self.angle_step = total_deg / num_rotations

    def get_angle(self, idx):
        return (idx % self.num_rotations) * self.angle_step

    def get_quaternion(self, idx):
        angle_deg = self.get_angle(idx)
        yaw = np.deg2rad(angle_deg)
        
        # Sawyer Gripper: Roll=pi (facing down), Pitch=0, Yaw=target
        # Using 'zyx' intrinsic rotation
        rot = R.from_euler('zyx', [yaw, 0.0, np.pi])
        quat = rot.as_quat() # Returns [x, y, z, w]

        q = Quaternion()
        q.x, q.y, q.z, q.w = quat
        return q

    @staticmethod
    def get_random_flat_quaternion():
        """Random Yaw for object reset."""
        yaw = np.random.uniform(0, 2 * np.pi)
        return [0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)]