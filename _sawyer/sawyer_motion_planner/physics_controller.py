"""
Physics Controller for Ballistic Tossing Trajectory Planning

Implements the analytical ballistic equations from the TossingBot paper to calculate
the required release position and velocity for a given target landing location.

Based on the assumptions:
1. Aerial trajectory is linear in XY plane (orthogonal drag negligible)
2. Release position at fixed radius cd from robot base origin
3. Release height rz at fixed constant ch
4. Release velocity angled 45° upwards in direction of target

Reference: TossingBot: Learning to Throw Arbitrary Objects with Residual Physics
"""
import numpy as np
from sawyer_common.geometry import Position, Pose2D
from sawyer_common.constants import L3_TOTAL, BASE_OFFSET_XYZ


class PhysicsController:
    """
    Ballistic physics controller for tossing trajectory planning.

    Given a target landing position, calculates the required release position
    and velocity magnitude using projectile motion equations.
    """

    def __init__(self, release_radius=0.7, release_height=0.04, gravity=9.8,
                 max_speed=2.0, min_speed=0.5):
        """
        Initialize the physics controller.

        Args:
            release_radius: Fixed distance cd from robot base origin (meters)
            release_height: Fixed release height ch above base (meters)
            gravity: Gravitational acceleration constant (m/s²), default 9.8
            max_speed: Maximum physically achievable release speed (m/s)
            min_speed: Minimum release speed (m/s)
        """
        self.cd = release_radius  # Release radius
        self.ch = release_height  # Release height
        self.g = gravity          # Gravity
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.release_angle_deg = 45.0  # Fixed 45° release angle

    def calculate_release_params(self, target_pos):
        """
        Calculate release position and velocity for a target landing location.

        Uses the TossingBot ballistic equations:
        - θ = arctan(py / px)
        - rx = cd * sin(θ)
        - ry = cd * cos(θ)
        - rz = ch
        - ||v|| = sqrt(g * (px² + py²) / (rz - pz - sqrt(px² + py²)))

        Args:
            target_pos: Target landing position [x, y, z] in robot frame (meters)

        Returns:
            dict with keys:
                'release_pos': Position object in robot frame
                'release_speed': ||v|| magnitude of release velocity (m/s)
                'release_angle': angle in degrees (always 45°)
                'theta': base angle alignment (radians)
                'valid': bool indicating if solution is physically achievable
                'error_msg': error message if not valid
        """
        # Ensure target_pos is handled as a numpy array internally
        if hasattr(target_pos, 'to_array'):
            p_vec = target_pos.to_array()
        else:
            p_vec = np.array(target_pos)
            
        px, py, pz = p_vec[0], p_vec[1], p_vec[2]

        # Calculate base angle alignment (equation 1)
        theta = np.arctan2(py, px)

        # Calculate release position at fixed radius and height
        rx = self.cd * np.sin(theta)
        ry = self.cd * np.cos(theta)
        rz = self.ch

        release_pos = Position(rx, ry, rz)

        # Calculate horizontal distance to target
        d_horizontal = np.sqrt(px**2 + py**2)

        # Calculate vertical drop
        delta_z = rz - pz

        # Check if target is behind release position (impossible)
        if d_horizontal <= self.cd:
            return {
                'release_pos': release_pos,
                'release_speed': None,
                'release_angle': self.release_angle_deg,
                'theta': theta,
                'valid': False,
                'error_msg': f'Target too close: horizontal distance {d_horizontal:.3f}m <= release radius {self.cd:.3f}m'
            }

        # Calculate required velocity magnitude (equation 2)
        # ||v|| = sqrt(g * d² / (rz - pz - d))
        denominator = delta_z - d_horizontal

        if denominator <= 0:
            return {
                'release_pos': release_pos,
                'release_speed': None,
                'release_angle': self.release_angle_deg,
                'theta': theta,
                'valid': False,
                'error_msg': f'Invalid trajectory: rz - pz - d = {denominator:.3f} <= 0 (target too high or too far)'
            }

        v_squared = self.g * d_horizontal**2 / denominator
        v_magnitude = np.sqrt(v_squared)

        # Check physical limits
        if v_magnitude > self.max_speed:
            return {
                'release_pos': release_pos,
                'release_speed': v_magnitude,
                'release_angle': self.release_angle_deg,
                'theta': theta,
                'valid': False,
                'error_msg': f'Required speed {v_magnitude:.3f} m/s exceeds max {self.max_speed:.3f} m/s'
            }

        if v_magnitude < self.min_speed:
            return {
                'release_pos': release_pos,
                'release_speed': v_magnitude,
                'release_angle': self.release_angle_deg,
                'theta': theta,
                'valid': False,
                'error_msg': f'Required speed {v_magnitude:.3f} m/s below min {self.min_speed:.3f} m/s'
            }

        # Valid solution
        return {
            'release_pos': release_pos,
            'release_speed': v_magnitude,
            'release_angle': self.release_angle_deg,
            'theta': theta,
            'valid': True,
            'error_msg': None
        }

    def get_release_velocity_components(self, release_speed, theta):
        """
        Calculate velocity components from magnitude and direction.

        At 45° release angle:
        - vx = ||v|| * cos(45°) * sin(θ) = ||v|| / sqrt(2) * sin(θ)
        - vy = ||v|| * cos(45°) * cos(θ) = ||v|| / sqrt(2) * cos(θ)
        - vz = ||v|| * sin(45°) = ||v|| / sqrt(2)

        Args:
            release_speed: Magnitude of release velocity ||v|| (m/s)
            theta: Base angle alignment (radians)

        Returns:
            dict with keys:
                'vx': X component of velocity (m/s)
                'vy': Y component of velocity (m/s)
                'vz': Z component of velocity (m/s)
                'v_horizontal': Horizontal velocity magnitude (m/s)
        """
        angle_rad = np.deg2rad(self.release_angle_deg)

        # At 45°: horizontal and vertical components are equal
        v_horizontal = release_speed * np.cos(angle_rad)  # ||v|| / sqrt(2)
        v_vertical = release_speed * np.sin(angle_rad)    # ||v|| / sqrt(2)

        # Decompose horizontal velocity into X and Y
        vx = v_horizontal * np.sin(theta)
        vy = v_horizontal * np.cos(theta)
        vz = v_vertical

        return {
            'vx': vx,
            'vy': vy,
            'vz': vz,
            'v_horizontal': v_horizontal
        }

    def predict_landing_position(self, release_pos, release_speed, theta):
        """
        Predict landing position given release parameters (forward calculation).

        Useful for verification and debugging.

        Args:
            release_pos: [rx, ry, rz] release position
            release_speed: Magnitude of release velocity (m/s)
            theta: Base angle alignment (radians)

        Returns:
            [px, py, pz] predicted landing position
        """
        rx, ry, rz = release_pos

        # Get velocity components
        vel = self.get_release_velocity_components(release_speed, theta)
        vx, vy, vz = vel['vx'], vel['vy'], vel['vz']
        v_horizontal = vel['v_horizontal']

        # Time to reach ground (pz = 0 in most cases, or table height)
        # Solving: 0 = rz + vz*t - 0.5*g*t²
        # Using quadratic formula: t = (vz + sqrt(vz² + 2*g*rz)) / g
        discriminant = vz**2 + 2 * self.g * rz
        if discriminant < 0:
            return None  # Should not happen with valid parameters

        t_land = (vz + np.sqrt(discriminant)) / self.g

        # Landing position
        px = rx + vx * t_land
        py = ry + vy * t_land
        pz = 0.0  # Assuming landing on ground level

        return np.array([px, py, pz])

    def print_trajectory_info(self, target_pos, params):
        """
        Print detailed trajectory information for debugging.

        Args:
            target_pos: Target landing position
            params: Result from calculate_release_params()
        """
        print("\n" + "="*60)
        print("BALLISTIC TRAJECTORY CALCULATION")
        print("="*60)
        print(f"Target Position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] m")
        print(f"Release Position: [{params['release_pos'][0]:.3f}, {params['release_pos'][1]:.3f}, {params['release_pos'][2]:.3f}] m")
        print(f"Release Radius: {self.cd:.3f} m")
        print(f"Release Height: {self.ch:.3f} m")
        print(f"Base Alignment θ: {np.rad2deg(params['theta']):.2f}°")
        print(f"Release Angle: {params['release_angle']:.1f}°")

        if params['valid']:
            print(f"Required Speed: {params['release_speed']:.3f} m/s")
            print("Status: ✓ VALID")

            # Show velocity components
            vel = self.get_release_velocity_components(params['release_speed'], params['theta'])
            print(f"Velocity Components: vx={vel['vx']:.3f}, vy={vel['vy']:.3f}, vz={vel['vz']:.3f} m/s")

            # Verify by forward calculation
            predicted = self.predict_landing_position(
                params['release_pos'], params['release_speed'], params['theta']
            )
            if predicted is not None:
                error = np.linalg.norm(predicted[:2] - target_pos[:2])
                print(f"Predicted Landing: [{predicted[0]:.3f}, {predicted[1]:.3f}, {predicted[2]:.3f}] m")
                print(f"Prediction Error: {error*1000:.1f} mm")
        else:
            print(f"Required Speed: {params['release_speed']:.3f} m/s" if params['release_speed'] else "N/A")
            print(f"Status: ✗ INVALID - {params['error_msg']}")

        print("="*60 + "\n")
