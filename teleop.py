import time, math, numpy as np, glfw, mujoco as mj
from peg_insertion_env_mujoco import PegInsertionEnv
from keyboard_glfw import GlfwKeyboardDevice
from logger import Logger

VEL_SMOOTH      = 0.15     
ANG_SMOOTH      = 0.15     
def clamp(v, lo, hi):
    return np.minimum(np.maximum(v, lo), hi)

def axis_angle_quat(axis, angle):
    a = np.asarray(axis, float)
    n = np.linalg.norm(a) + 1e-12
    a = a / n
    s = math.sin(0.5*angle)
    return np.array([math.cos(0.5*angle), a[0]*s, a[1]*s, a[2]*s], dtype=float)
def quat_multiply(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)


def integrate_body_rates(quat, body_rates_xyz, dt):
    #apply roll pitch yaw to the body rate
    rx, ry, rz = body_rates_xyz
    if abs(rx)+abs(ry)+abs(rz) < 1e-12:
        return quat
    qx = axis_angle_quat([1,0,0], rx*dt)  
    qy = axis_angle_quat([0,1,0], ry*dt)  
    qz = axis_angle_quat([0,0,1], rz*dt)  
    dq = quat_multiply(qx, quat_multiply(qy, qz))  
    q  = quat_multiply(quat, dq)                   
    return q / (np.linalg.norm(q) + 1e-12)


class TeleopApp:
    def __init__(self, env, device_factory):
        self.env = env

        # workspace bounds
        self.ws_min = getattr(env, "workspace_min", np.array([-0.40, -0.40, 0.02], float))
        self.ws_max = getattr(env, "workspace_max", np.array([+0.40, +0.40, 0.45], float))

        # glfw / renderer
        if not glfw.init(): raise RuntimeError("GLFW init failed")
        self.win = glfw.create_window(1280, 800, "Teleop Modular", None, None)
        glfw.make_context_current(self.win)
        glfw.swap_interval(1)

        # camera
        self.cam = mj.MjvCamera(); mj.mjv_defaultCamera(self.cam)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.2, -60, -20
        self.cam.lookat = np.array([0.0, 0.0, 0.12])

        # scene
        self.opt = mj.MjvOption(); mj.mjv_defaultOption(self.opt)
        self.scn = mj.MjvScene(self.env.mj_model, maxgeom=10000)
        self.ctx = mj.MjrContext(self.env.mj_model, mj.mjtFontScale.mjFONTSCALE_150)

        # camera mouse
        self._orbit = False
        self._pan = False
        self._mouse_last = (0.0, 0.0)
        glfw.set_cursor_pos_callback(self.win, self._on_cursor)
        glfw.set_mouse_button_callback(self.win, self._on_mouse_button)
        glfw.set_scroll_callback(self.win, self._on_scroll)

        # device (inject whatever you want here)
        self.device = device_factory(self.win, self.env.ctrl_dt)
        # smoothed commands
        self.v_smooth = np.zeros(3)
        self.w_smooth = np.zeros(3)
        self.max_time_steps = 100000
        print(self.device.help())
    
    def create_log (self, cmd_pos, cmd_quat, cmd_gripper, cur_obs):
        self.log.append([cmd_pos,  cmd_quat, cmd_gripper, cur_obs])
    # camera callbacks
    def _on_cursor(self, win, x, y):
        dx = x - self._mouse_last[0]
        dy = y - self._mouse_last[1]
        self._mouse_last = (x, y)
        if self._orbit:
            self.cam.azimuth  -= 0.25 * dx
            self.cam.elevation -= 0.20 * dy
            self.cam.elevation = float(clamp(self.cam.elevation, -89.9, 89.9))
        elif self._pan:
            s = 0.0015 * self.cam.distance
            self.cam.lookat[0] -= s * dx
            self.cam.lookat[1] += s * dy

    def _on_mouse_button(self, win, button, action, mods):
        pressed = (action == glfw.PRESS)
        if button == glfw.MOUSE_BUTTON_LEFT:  self._orbit = pressed
        elif button == glfw.MOUSE_BUTTON_RIGHT: self._pan = pressed

    def _on_scroll(self, win, xoff, yoff):
        self.cam.distance *= (1.0 - 0.08*yoff)
        self.cam.distance = float(clamp(self.cam.distance, 0.2, 5.0))

    def run(self):
        sim_time= 0.0
        ctrl_dt = self.env.mj_model.opt.timestep
        wall0 = time.perf_counter()
        cur_timesteps = 0
        done = False
        logger = Logger()

        while True: 
        
            if done or cur_timesteps>self.max_time_steps or glfw.window_should_close(self.win):
                break
            glfw.poll_events()

            
            state = self.device.poll()
            v_cmd = state["v_world"]
            w_cmd = state["w_body"]    
            grip  = float(np.clip(state["grip"], 0.0, 1.0))
            buttons = state.get("buttons", {})

            
            if buttons.get("reset"): 
                self.env.reset()
                
            if buttons.get("home"):
                pos, quat = self.env.get_mocap_pose()
                pos[:] = [0.0, 0.0, 0.18]
                self.env.set_mocap(pos, quat)

            
            self.v_smooth = (1.0 - VEL_SMOOTH)*self.v_smooth + VEL_SMOOTH*v_cmd
            self.w_smooth = (1.0 - ANG_SMOOTH)*self.w_smooth + ANG_SMOOTH*w_cmd

            
        
            pos, quat = self.env.get_mocap_pose()
            pos = pos + self.v_smooth * ctrl_dt
            quat = integrate_body_rates(quat, self.w_smooth, ctrl_dt)

            pos = clamp(pos, self.ws_min, self.ws_max)
            self.env.set_mocap(pos, quat)

            
            action = np.array([pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3], grip], dtype=float)

            _, _, done, info = self.env.step(action)
            

            # contacts (aggregated)
            contacts = self.env._contacts_summary()
            peg_table   = contacts.get(("peg","table"),    {"fn":0,"ft1":0,"ft2":0,"ft":0, "count":0})
            peg_cuboid  = contacts.get(("cuboid","peg"),   {"fn":0,"ft1":0,"ft2":0,"ft":0, "count":0})
            grip_table  = contacts.get(("gripper","table"),{"fn":0,"ft1":0,"ft2":0,"ft":0, "count":0})
            grip_peg    = contacts.get(("gripper","peg"),  {"fn":0,"ft1":0,"ft2":0,"ft":0, "count":0})

            logger.log(
                # commanded
                cmd_pos=pos.copy(), cmd_quat=quat.copy(), cmd_grip=float(grip),
                # gripper actual
                grip_pos=info["grip_pose"], grip_quat=info["grip_quat"], jaw_sep_m=info["jaw_sep"],
                # peg actual
                peg_pos=info["peg_pos"], peg_quat=info["peg_quat"],
                # contacts (sum over all contact points for each pair)
                peg_table_fn=peg_table["fn"],   peg_table_ft=peg_table["ft"],   peg_table_cnt=peg_table["count"],
                peg_cuboid_fn=peg_cuboid["fn"], peg_cuboid_ft=peg_cuboid["ft"], peg_cuboid_cnt=peg_cuboid["count"],
                grip_table_fn=grip_table["fn"], grip_table_ft=grip_table["ft"], grip_table_cnt=grip_table["count"],
                grip_peg_fn=grip_peg["fn"],     grip_peg_ft=grip_peg["ft"],     grip_peg_cnt=grip_peg["count"],
            )
            

            sim_time += ctrl_dt
            wall_elapsed = time.perf_counter() - wall0
            if sim_time> wall_elapsed:
                time.sleep(sim_time - wall_elapsed)
            cur_timesteps+=1

            
            fbw, fbh = glfw.get_framebuffer_size(self.win)
            viewport = mj.MjrRect(0, 0, fbw, fbh)
            mj.mjv_updateScene(self.env.mj_model, self.env.data, self.opt, None, self.cam,
                               mj.mjtCatBit.mjCAT_ALL, self.scn)
            mj.mjr_render(viewport, self.scn, self.ctx)
            glfw.swap_buffers(self.win)

        glfw.terminate()
        if done:
            print("Sucess, Episode end")
            logger.to_csv("teleop_log.csv")
        elif cur_timesteps>self.max_time_steps:
            print("Reached max_time_steps, Episode end")
            logger.to_csv("teleop_log.csv")


def make_glfw_keyboard_device(win, ctrl_dt):
    return GlfwKeyboardDevice(win, ctrl_dt, speed_preset=2)

if __name__ == "__main__":
    env = PegInsertionEnv()
    app = TeleopApp(env, device_factory=make_glfw_keyboard_device)
    app.run()