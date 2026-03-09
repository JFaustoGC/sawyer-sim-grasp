"""Plotting and animation functions for 3R manipulator trajectories."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib import animation
import casadi as ca

# Import your new structure
from .robot_kinematics import RobotKinematics
from .tossing_config import ROBOT_PARAMS

def animate_3r_trajectory(q_traj, dt):
    """
    Animate a 3R manipulator given a trajectory of joint configurations.
    Args:
        q_traj: Trajectory of joint configurations (N x 3 array)
        dt: Time step between configurations
    """
    # Initialize Kinematics
    rk = RobotKinematics()
    
    # Extract Params for drawing links
    l1 = ROBOT_PARAMS["l1"]
    l2 = ROBOT_PARAMS["l2"]
    l3 = ROBOT_PARAMS["l3"]
    gripper_len = ROBOT_PARAMS["gripper_len"]
    # For visualization, we treat the gripper as a distinct block
    total_l3 = l3
    
    base_x = ROBOT_PARAMS["base_offset"][0]
    base_z = ROBOT_PARAMS["base_offset"][2]
    
    # Visual params
    gripper_w = ROBOT_PARAMS["gripper_width"]
    gripper_l = 0.05 # Visual length of the gripper box

    q_traj = np.asarray(q_traj)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid(True)

    # Axis limits
    total_length = l1 + l2 + l3 + gripper_len
    ax.set_xlim(-0.5, 1.3)
    ax.set_ylim(-0.1, 1.3)

    # Precompute Path using the Kinematics Class
    N = len(q_traj)
    ee_path = np.zeros((N, 2))
    
    # We also need intermediate link positions for the stick figure
    # Since the Class only gives EE, we do a quick local calculation for links
    joint_positions_all = []
    
    for i in range(N):
        q = q_traj[i]
        
        # 1. Get EE from Class (Source of Truth)
        pose = rk.forward_kinematics(q)
        ee_path[i] = pose[0:2]

        # 2. Calculate Joints for Stick Figure
        t1 = -q[0]
        t2 = -q[0] - q[1]
        t3 = -q[0] - q[1] - q[2]
        
        p0 = (base_x, base_z)
        p1 = (base_x + l1 * np.cos(t1), base_z + l1 * np.sin(t1))
        p2 = (p1[0] + l2 * np.cos(t2), p1[1] + l2 * np.sin(t2))
        p3 = (p2[0] + l3 * np.cos(t3), p2[1] + l3 * np.sin(t3))
        # p_ee is the end of the gripper
        p_ee = (pose[0], pose[1]) 

        joint_positions_all.append([p0, p1, p2, p3, p_ee])

    ax.plot(ee_path[:, 0], ee_path[:, 1], 'k--', label='Reference Path', zorder=5)
    ax.legend()

    # Plot elements
    line, = ax.plot([], [], '-', lw=4, color='blue')
    joint_circles = [Circle((0, 0), 0.015, color='black', zorder=20) for _ in range(4)]
    for circ in joint_circles:
        ax.add_patch(circ)
        
    gripper_patch = Rectangle((0, 0), gripper_l, gripper_w, angle=0.0, color='red', zorder=10)
    ax.add_patch(gripper_patch)

    # Containers for arrows (mutable lists to allow updates in closure)
    arrows = {"x": None, "z": None}

    def init():
        line.set_data([], [])
        for circ in joint_circles:
            circ.center = (0, 0)
        gripper_patch.set_xy((0, 0))
        return (line, gripper_patch) + tuple(joint_circles)

    def animate_frame(i):
        joints = joint_positions_all[i]
        
        # Update Link Lines (Base -> J1 -> J2 -> J3)
        # We don't draw line to EE, that's the gripper patch
        line_points = joints[:-1] 
        line.set_data([p[0] for p in line_points], [p[1] for p in line_points])
        
        # Update Joint Circles
        for circ, pos in zip(joint_circles, line_points):
            circ.center = pos

        # Update Gripper Patch
        # joints[-2] is the wrist (J3), joints[-1] is the tip
        wrist_x, wrist_y = joints[-2]
        
        # Global angle of the end effector
        theta = -q_traj[i, 0] - q_traj[i, 1] - q_traj[i, 2]

        gripper_patch.set_width(gripper_len)
        gripper_patch.set_height(gripper_w)
        gripper_patch.set_angle(np.rad2deg(theta))
        
        # Calculate bottom-left corner of rectangle for rotation around center-ish
        # We want the wrist to be the start of the gripper
        # Simple rotation logic:
        # The rectangle anchors at bottom-left. We need to shift it so (0, height/2) is at wrist
        
        corner_x = wrist_x + (gripper_w/2.0) * np.sin(theta)
        corner_y = wrist_y - (gripper_w/2.0) * np.cos(theta)
        
        gripper_patch.set_xy((corner_x, corner_y))

        # EE frame arrows (Force visualization)
        ee_x, ee_y = joints[-1]
        
        if arrows["x"]: arrows["x"].remove()
        if arrows["z"]: arrows["z"].remove()
        
        arrows["x"] = ax.arrow(ee_x, ee_y, 0.07 * np.cos(theta), 0.07 * np.sin(theta),
                              head_width=0.02, head_length=0.04, fc='r', ec='r', zorder=30)
        arrows["z"] = ax.arrow(ee_x, ee_y, -0.07 * np.sin(theta), 0.07 * np.cos(theta),
                              head_width=0.02, head_length=0.04, fc='g', ec='g', zorder=30)
        
        # Return list of artists to redraw
        return (line, gripper_patch) + tuple(joint_circles) + (arrows["x"], arrows["z"])

    interval_ms = int(dt * 1000)
    ani = animation.FuncAnimation(fig, animate_frame, frames=len(q_traj), init_func=init,
                                  blit=True, interval=interval_ms, repeat=False)
    plt.show()


def plot_joint_trajectories(solution):
    """
    Plot joint positions, velocities, and accelerations over time.
    Args:
        solution: Dictionary containing 'time', 'Q', 'Qd', 'Qdd' arrays
    """
    time = solution['time']
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(time, solution['Q'].T)
    plt.title('Joint Positions (q)')
    plt.ylabel('rad')
    plt.legend(['q0', 'q1', 'q2'])
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, solution['Qd'].T)
    plt.title('Joint Velocities (qd)')
    plt.ylabel('rad/s')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time, solution['Qdd'].T)
    plt.title('Joint Accelerations (qdd)')
    plt.ylabel('rad/s^2')
    plt.xlabel('Time [s]')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_ee_kinematics(solution):
    """
    Plot end-effector position, velocity magnitude, and acceleration magnitude.
    Args:
        solution: Dictionary containing 'time', 'Q', 'Qd', 'Qdd' arrays
    """
    # 1. Instantiate the Class
    rk = RobotKinematics()
    
    # 2. Get the CasADi functions to use for numeric evaluation
    # We use these because 'jdot_qdot' is readily available there
    funcs = rk.get_casadi_funcs()
    
    Q_opt = solution['Q']
    Qd_opt = solution['Qd']
    Qdd_opt = solution['Qdd']
    time = solution['time']
    N = Q_opt.shape[1]

    ee_pos = np.zeros((2, N))
    ee_vel = np.zeros((2, N))
    ee_acc = np.zeros((2, N))

    print("Computing kinematics for plotting...")

    for k in range(N):
        q_k = Q_opt[:, k]
        qd_k = Qd_opt[:, k]
        qdd_k = Qdd_opt[:, k]

        # Use CasADi functions with numeric input
        # CasADi automatically casts numpy inputs -> numpy outputs when calling the Function object
        
        # Position
        fk_val = np.array(funcs['fk'](q_k)).flatten()
        ee_pos[:, k] = fk_val[0:2]

        # Velocity = J * qd
        jac_val = np.array(funcs['jacobian'](q_k))
        v = np.dot(jac_val[0:2, :], qd_k)
        ee_vel[:, k] = v

        # Acceleration = J_dot*qd + J*qdd
        # funcs['jdot_qdot'] returns the J_dot*qd term
        jdqd_val = np.array(funcs['jdot_qdot'](q_k, qd_k)).flatten()
        
        a = jdqd_val[0:2] + np.dot(jac_val[0:2, :], qdd_k)
        ee_acc[:, k] = a

    # Magnitudes
    ee_vel_mag = np.linalg.norm(ee_vel, axis=0)
    ee_acc_mag = np.linalg.norm(ee_acc, axis=0)

    # Plot
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time, ee_pos[0, :], label='x')
    plt.plot(time, ee_pos[1, :], label='z')
    plt.title('End-Effector Position')
    plt.ylabel('Position [m]')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, ee_vel_mag, label='|v|', color='r')
    plt.axhline(y=np.max(ee_vel_mag), color='k', linestyle=':', label='Max: {:.2f}'.format(np.max(ee_vel_mag)))
    plt.title('End-Effector Velocity Magnitude')
    plt.ylabel('Velocity [m/s]')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time, ee_acc_mag, label='|a|', color='g')
    plt.title('End-Effector Acceleration Magnitude')
    plt.ylabel('Acceleration [m/s^2]')
    plt.xlabel('Time [s]')
    plt.grid(True)

    plt.tight_layout()
    plt.show()