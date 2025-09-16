import numpy as np
import mujoco



class PegInsertionEnv:
    def __init__(self)->None:
        self.mj_model = mujoco.MjModel.from_xml_path('assets/scene.xml')
        self.data = mujoco.MjData(self.mj_model)
        self.mocap_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base_mocap") 
        self.finger_act_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
        self.ctrlrange = self.mj_model.actuator_ctrlrange[self.finger_act_id].copy()  
        self.peg_site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "peg_tip")
        self.hole_site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "hole_entry")

        self.ctrl_dt = 0.001
        self.reset()
    
    def step(self, action: np.ndarray) -> np.ndarray:
        #action: [px, py, pz, qw, qx, qy, qz, grip_force]

        #position (m) and orientation (quat wxyz) for the mocap base handle
        # - grip_force in [0,1]  (mapped to actuator ctrlrange)
        #  (if you pass 0..255 directly, set self._GRIP_IS_RAW=True below)
        
        action = np.asarray(action, dtype=float).flatten()
        assert action.size == 8
        pos = action[:3]
        quat = action[3:7]
        grip = action[7]

        self.data.mocap_pos[0] = pos
        self.data.mocap_quat[0] = quat

        ctrl_min, ctrl_max = self.ctrlrange
        grip = float(np.clip(grip,0.0, 1.0))
        grip_cmd = ctrl_min + grip *(ctrl_max-ctrl_min)
        self.data.ctrl[self.finger_act_id] = grip_cmd

        mujoco.mj_step(self.mj_model, self.data)

        obs = self._get_obs()
        reward = 0.0
        terminated = False
        info= {}
        return obs, reward, terminated, info
        
        
    def reset(self, qpos= None, qvel= None) ->np.ndarray:
        
        mujoco.mj_resetData(self.mj_model, self.data)
        if qpos is not None:
            self.data.qpos[:] = qpos
        if qvel is not None:
            self.data.qvel[:] = qvel
        mujoco.mj_forward(self.mj_model, self.data)
        return self._get_obs()
    
    def _get_obs(self):
        
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        return {"qpos": qpos, "qvel": qvel}
    # in PegInsertionEnv
    def get_mocap_pose(self):
        return (self.data.mocap_pos[0].copy(),
                self.data.mocap_quat[0].copy())

    def set_mocap(self, pos, quat):
        q = np.asarray(quat, float); q /= (np.linalg.norm(q) + 1e-12)
        self.data.mocap_pos[0]  = np.asarray(pos, float)
        self.data.mocap_quat[0] = q
        
