"""
kinematics.py — CasADi-wrapped Pinocchio kinematics for the Sawyer robot.
"""

import os
import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin
from typing import List, Dict, Optional, Any, Union, cast

# We cast the modules to Any to silence Pylance attribute errors on C++ extensions
p: Any = pin
cp: Any = cpin

class CasadiKinematics:
    """
    Bridge between Pinocchio's efficient robotics algorithms and CasADi's 
    symbolic optimization framework.
    """
    
    def __init__(self, urdf_path: str, base_link: str = "base", end_link: str = "right_gripper_tip",
                 active_joints: Optional[List[str]] = None,
                 locked_joints: Optional[Dict[str, float]] = None):
        
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found at {urdf_path}")
            
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.end_link = end_link
        self.requested_active_joints = active_joints
        self.locked_joints = locked_joints if locked_joints else {}
        
        # 1. Load the model using standard Pinocchio
        self.model: Any = p.buildModelFromUrdf(self.urdf_path)
        self.data: Any = self.model.createData()
        
        # 2. Identify ACTIVE joints
        self.active_joint_names: List[str] = []
        self.active_joint_ids: List[int] = []
        
        n_joints = int(self.model.njoints)
        for i in range(1, n_joints):
            name = str(self.model.names[i])
            if self.requested_active_joints is None:
                if name.startswith("right_j"):
                    self.active_joint_names.append(name)
                    self.active_joint_ids.append(i)
            else:
                if name in self.requested_active_joints:
                    self.active_joint_names.append(name)
                    self.active_joint_ids.append(i)
        
        self.n_dof = len(self.active_joint_names)
        
        # Extract Limits for ACTIVE joints
        self.q_min: np.ndarray = np.zeros(self.n_dof)
        self.q_max: np.ndarray = np.zeros(self.n_dof)
        
        for i, jid in enumerate(self.active_joint_ids):
            idx = int(self.model.joints[jid].idx_q)
            self.q_min[i] = float(self.model.lowerPositionLimit[idx])
            self.q_max[i] = float(self.model.upperPositionLimit[idx])
        
        # 3. Build CasADi symbolic model
        self.cmodel: Any = cp.Model(self.model)
        self.cdata: Any = self.cmodel.createData()
        
        # 4. Define Symbolic Functions
        self.fk_pos: Any = None
        self.fk_rot: Any = None
        self.fk_full: Any = None
        self.fk_tau: Any = None # Added for torque calculation
        self.fk_M: Any = None   # Mass matrix
        self.fk_nle: Any = None # Non-linear effects (Coriolis + Gravity)
        self.jacobian: Any = None
        
        self._build_functions()

    def _build_functions(self) -> None:
        """Internal: Compile CasADi symbolic graphs into callable functions."""
        cq_active = ca.SX.sym('q', self.n_dof)
        cv_active = ca.SX.sym('v', self.n_dof)
        ca_active = ca.SX.sym('a', self.n_dof)
        
        cq_full = ca.SX.zeros(int(self.model.nq))
        cv_full = ca.SX.zeros(int(self.model.nv))
        ca_full = ca.SX.zeros(int(self.model.nv))
        
        for i in range(1, int(self.model.njoints)):
            name = str(self.model.names[i])
            idx_q = int(self.model.joints[i].idx_q)
            idx_v = int(self.model.joints[i].idx_v)
            
            if name in self.active_joint_names:
                active_idx = self.active_joint_names.index(name)
                cq_full[idx_q] = cq_active[active_idx]
                cv_full[idx_v] = cv_active[active_idx]
                ca_full[idx_v] = ca_active[active_idx]
            elif name in self.locked_joints:
                cq_full[idx_q] = self.locked_joints[name]
            
        cp.forwardKinematics(self.cmodel, self.cdata, cq_full)
        cp.updateFramePlacements(self.cmodel, self.cdata)
        
        frame_id = None
        for i in range(len(self.model.frames)):
            frame = self.model.frames[i]
            if str(frame.name) == self.end_link:
                if frame.type == p.FrameType.BODY:
                    frame_id = i
                    break
                if frame_id is None:
                    frame_id = i  # fallback: first match
        if frame_id is None or frame_id >= len(self.model.frames):
            raise ValueError(f"Frame {self.end_link} not found in URDF.")
            
        pos = self.cdata.oMf[frame_id].translation
        rot = self.cdata.oMf[frame_id].rotation
        quat = self._rot_to_quat_ca(rot)
        
        self.fk_pos = ca.Function('fk_pos', [cq_active], [pos])
        self.fk_rot = ca.Function('fk_rot', [cq_active], [quat])
        self.fk_full = ca.Function('fk_full', [cq_active], [ca.vertcat(pos, quat)])
        
        # Symbolic Inverse Dynamics (Torque)
        tau_full = cp.rnea(self.cmodel, self.cdata, cq_full, cv_full, ca_full)
        tau_active = []
        for jid in self.active_joint_ids:
            idx_v = int(self.model.joints[jid].idx_v)
            tau_active.append(tau_full[idx_v])
        
        self.fk_tau = ca.Function('fk_tau', [cq_active, cv_active, ca_active], [ca.vertcat(*tau_active)])
        
        # Symbolic Mass Matrix (M)
        M_full = cp.crba(self.cmodel, self.cdata, cq_full)
        # crba returns upper triangular part, so symmetrize it
        M_sym = ca.triu(M_full) + ca.triu(M_full).T - ca.diag(ca.diag(M_full))
        
        M_active = ca.SX(self.n_dof, self.n_dof)
        for i, jid_i in enumerate(self.active_joint_ids):
            for j, jid_j in enumerate(self.active_joint_ids):
                M_active[i, j] = M_sym[int(self.model.joints[jid_i].idx_v), int(self.model.joints[jid_j].idx_v)]
                
        self.fk_M = ca.Function('fk_M', [cq_active], [M_active])
        
        # Symbolic Non-Linear Effects (Coriolis + Gravity)
        nle_full = cp.nonLinearEffects(self.cmodel, self.cdata, cq_full, cv_full)
        nle_active = []
        for jid in self.active_joint_ids:
            nle_active.append(nle_full[int(self.model.joints[jid].idx_v)])
            
        self.fk_nle = ca.Function('fk_nle', [cq_active, cv_active], [ca.vertcat(*nle_active)])
        
        cp.computeJointJacobians(self.cmodel, self.cdata, cq_full)
        J_full = cp.getFrameJacobian(self.cmodel, self.cdata, frame_id, p.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        J_active_list = []
        for jid in self.active_joint_ids:
            idx_v = int(self.model.joints[jid].idx_v)
            J_active_list.append(J_full[:, idx_v])
        
        J_active = ca.horzcat(*J_active_list)
        self.jacobian = ca.Function('jacobian', [cq_active], [J_active[:3, :]])

    def _rot_to_quat_ca(self, R: Any) -> Any:
        """Helper to convert CasADi rotation matrix to quaternion."""
        t = R[0,0] + R[1,1] + R[2,2]
        w = ca.sqrt(ca.fmax(0, 1+t))/2.0
        x = ca.sqrt(ca.fmax(0, 1+R[0,0]-R[1,1]-R[2,2]))/2.0
        y = ca.sqrt(ca.fmax(0, 1-R[0,0]+R[1,1]-R[2,2]))/2.0
        z = ca.sqrt(ca.fmax(0, 1-R[0,0]-R[1,1]+R[2,2]))/2.0
        x = ca.copysign(x, R[2,1]-R[1,2])
        y = ca.copysign(y, R[0,2]-R[2,0])
        z = ca.copysign(z, R[1,0]-R[0,1])
        return ca.vertcat(x,y,z,w)

    # ── Dynamics (Numeric) ───────────────────────────────────────────────────

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        q_full = self._expand_q(q)
        M_full = p.crba(self.model, self.data, q_full)
        M_active = np.zeros((self.n_dof, self.n_dof))
        for i, jid_i in enumerate(self.active_joint_ids):
            idx_i = int(self.model.joints[jid_i].idx_v)
            for j, jid_j in enumerate(self.active_joint_ids):
                idx_j = int(self.model.joints[jid_j].idx_v)
                M_active[i, j] = M_full[idx_i, idx_j]
        return M_active

    def gravity_torques(self, q: np.ndarray) -> np.ndarray:
        g_full = p.computeGeneralizedGravity(self.model, self.data, self._expand_q(q))
        return self._extract_active(g_full)

    def inverse_dynamics(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        tau_full = p.rnea(self.model, self.data, self._expand_q(q), self._expand_v(v), self._expand_v(a))
        return self._extract_active(tau_full)

    # ── Internal Helpers ─────────────────────────────────────────────────────

    def _expand_q(self, q_active: np.ndarray) -> np.ndarray:
        q_full = np.zeros(int(self.model.nq))
        for i, _ in enumerate(self.active_joint_names):
            jid = self.active_joint_ids[i]
            q_full[int(self.model.joints[jid].idx_q)] = q_active[i]
        for name, val in self.locked_joints.items():
            if name in self.model.names:
                jid = int(self.model.getJointId(name))
                q_full[int(self.model.joints[jid].idx_q)] = val
        return q_full

    def _expand_v(self, v_active: np.ndarray) -> np.ndarray:
        v_full = np.zeros(int(self.model.nv))
        for i, jid in enumerate(self.active_joint_ids):
            v_full[int(self.model.joints[jid].idx_v)] = v_active[i]
        return v_full

    def _extract_active(self, full_vec: np.ndarray) -> np.ndarray:
        res = np.zeros(self.n_dof)
        for i, jid in enumerate(self.active_joint_ids):
            res[i] = full_vec[int(self.model.joints[jid].idx_v)]
        return res
