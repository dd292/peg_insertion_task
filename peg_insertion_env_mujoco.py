import numpy as np
import mujoco
from collections import defaultdict



class PegInsertionEnv:
    def __init__(self)->None:
        self.mj_model = mujoco.MjModel.from_xml_path('assets/scene.xml')
        self.data = mujoco.MjData(self.mj_model)
        self.geom_ids = {
            "table":  mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor"),          
            "peg":    mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "peg_collision"),  
            "cuboid": mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "cuboid_collision"),}
        self.body_ids = {
            "gripper": mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base"),
            "peg": mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "peg"),
        }
        self.gripper_pad_ids= [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_pad1"),
                            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_pad2"),          
                            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_pad1"),          
                            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_pad2")]
        self.mocap_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base_mocap") 
        self.finger_act_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
        self.ctrlrange = self.mj_model.actuator_ctrlrange[self.finger_act_id].copy()  
        self.peg_site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "peg_tip")
        self.hole_site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "hole_entry")
        self.sid_jaw_L   = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "jaw_left_tip")
        self.sid_jaw_R   = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "jaw_right_tip")

        self._mocap_home_pos  = self.data.mocap_pos[0].copy()
        self._mocap_home_quat = self.data.mocap_quat[0].copy()
        self._grip_home = 1.0
        self.ctrl_dt = 0.001
        self.reset()
    
    def _apply_grip_fraction(self, frac: float):
        frac = float(np.clip(frac, 0.0, 1.0))
        lo, hi = self.ctrlrange
        self.data.ctrl[self.finger_act_id] = lo + frac * (hi - lo)
    def step(self, action: np.ndarray) -> np.ndarray:

        action = np.asarray(action, dtype=float).flatten()
        assert action.size == 8
        pos = action[:3]
        quat = action[3:7]
        grip = action[7]

        self.data.mocap_pos[0] = pos
        self.data.mocap_quat[0] = quat

        self._apply_grip_fraction(grip)

        mujoco.mj_step(self.mj_model, self.data)

        obs = self._get_obs()
        
        reward = 0.0
        terminated = False
        #Update terminate when success
        if self.has_peg_inserted():
            terminated = True

        info= {}
        info["grip_pose"] = self.data.xpos[self.body_ids["gripper"]].copy()
        info["grip_quat"] = self.data.xquat[self.body_ids["gripper"]].copy()
        info["jaw_sep"] = self.get_jaw_separation()
        info["peg_pos"] = self.data.xpos[self.body_ids["peg"]].copy()
        info["peg_quat"] = self.data.xquat[self.body_ids["peg"]].copy()

        return obs, reward, terminated, info
        
        
    def reset(self, qpos= None, qvel= None) ->np.ndarray:
        
        mujoco.mj_resetData(self.mj_model, self.data)
        if qpos is not None:
            self.data.qpos[:] = qpos
        if qvel is not None:
            self.data.qvel[:] = qvel
        self.data.mocap_pos[0]  = self._mocap_home_pos
        self.data.mocap_quat[0] = self._mocap_home_quat
        self._apply_grip_fraction(self._grip_home)
        mujoco.mj_forward(self.mj_model, self.data)
        return self._get_obs()
    
    def _get_obs(self):
        obs = {}
        obs["qpos"] = self.data.qpos.copy()
        obs["qvel"] = self.data.qvel.copy()
        obs["contacts"] = self._contacts_summary()
        return obs
    
    # in PegInsertionEnv
    def get_mocap_pose(self):
        return (self.data.mocap_pos[0].copy(),
                self.data.mocap_quat[0].copy())

    def set_mocap(self, pos, quat):
        q = np.asarray(quat, float); q /= (np.linalg.norm(q) + 1e-12)
        self.data.mocap_pos[0]  = np.asarray(pos, float)
        self.data.mocap_quat[0] = q
    
    def has_peg_inserted(self, dist_thresh=0.005, depth_thresh=0.01):
        
        tip_pos = self.data.site_xpos[self.peg_site_id]
        hole_pos = self.data.site_xpos[self.hole_site_id]

        # XY distance
        lateral_dist = np.linalg.norm(tip_pos[:2] - hole_pos[:2])
        # Z depth (peg tip below hole)
        depth = hole_pos[2] - tip_pos[2]

        return lateral_dist < dist_thresh and depth > depth_thresh
            
    def _label_from_geom(self, geom_id: int) -> str:
        
        for name, gid in self.geom_ids.items():
            if geom_id == gid:
                return name
        
        body_id = self.mj_model.geom_bodyid[geom_id]
        for name, bid in self.body_ids.items():
            if body_id == bid:
                return name
        mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if geom_id in self.gripper_pad_ids:
            return "gripper"
        return "other"


    def _contacts_summary(self):

        out = defaultdict(lambda: {"fn":0.0, "ft1":0.0, "ft2":0.0, "ft":0.0, "count":0})
        ncon = self.data.ncon
        c6 = np.zeros(6, dtype=float)  # [fn, ft1, ft2, tn, tt1, tt2] in contact frame

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            a = self._label_from_geom(con.geom1)
            b = self._label_from_geom(con.geom2)
            pair = tuple(sorted((a, b)))
            if "other" in pair:
                continue  

            mujoco.mj_contactForce(self.mj_model, self.data, i, c6)
            
            fn, ft1, ft2 = float(c6[0]), float(c6[1]), float(c6[2])
            # if abs(fn)>0 or abs(ft1) or abs(ft2)>0:
            #     print(f"non zero force{(a, b)}") Debug contact
            ft = float((ft1*ft1 + ft2*ft2) ** 0.5)

            agg = out[pair]
            agg["fn"]   += fn
            agg["ft1"]  += ft1
            agg["ft2"]  += ft2
            agg["ft"]   += ft       
            agg["count"] += 1
        
        return out
    
    def get_jaw_separation(self):

        if self.sid_jaw_L >= 0 and self.sid_jaw_R >= 0:
            pL = self.data.site_xpos[self.sid_jaw_L]
            pR = self.data.site_xpos[self.sid_jaw_R]
            return float(np.linalg.norm(pL - pR))
        return float("nan")