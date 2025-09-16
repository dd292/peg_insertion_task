import time, math, numpy as np, glfw, mujoco as mj
from peg_insertion_env import PegInsertionEnv

XY_GAIN = 0.0015     # meters per pixel
Z_GAIN  = 0.0030     # meters per scroll tick
YAW_GAIN= 0.004      # rad per pixel (right-drag)

class MouseTeleop:
    def __init__(self, env):
        self.env = env
        # input state
        self._mouse_left  = False
        self._mouse_right = False
        self._last_x = 0.0
        self._last_y = 0.0
        self._z_scroll_pending = 0.0
        self._yaw = 0.0
        self._grip = 0.0

        # init window & viewer-independent context
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        self.win = glfw.create_window(1280, 800, "Mouse→Hand Teleop", None, None)
        glfw.make_context_current(self.win)
        glfw.swap_interval(1)

        # connect callbacks
        glfw.set_cursor_pos_callback(self.win, self._on_cursor)
        glfw.set_mouse_button_callback(self.win, self._on_mouse_button)
        glfw.set_scroll_callback(self.win, self._on_scroll)
        glfw.set_key_callback(self.win, self._on_key)

        # simple renderer (no UI)
        self.cam = mj.MjvCamera(); mj.mjv_defaultCamera(self.cam)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.2, -60, -20
        self.cam.lookat = np.array([0.0, 0.0, 0.15])
        self.opt = mj.MjvOption(); mj.mjv_defaultOption(self.opt)
        self.scn = mj.MjvScene(self.env.mj_model, maxgeom=10_000)
        self.ctx = mj.MjrContext(self.env.mj_model, mj.mjtFontScale.mjFONTSCALE_150)

    # ---------- callbacks ----------
    def _on_cursor(self, win, x, y):
        dx = x - self._last_x
        dy = y - self._last_y
        self._last_x, self._last_y = x, y

        if not (self._mouse_left or self._mouse_right):
            return

        pos, quat = self.env.get_mocap_pose()

        if self._mouse_left:
            pos[0] += XY_GAIN * dx
            pos[1] -= XY_GAIN * dy

        if self._mouse_right:
            self._yaw += YAW_GAIN * dx
            quat = np.array([math.cos(0.5*self._yaw), 0.0, 0.0, math.sin(0.5*self._yaw)])

        self.env.set_mocap(pos, quat)

    def _on_mouse_button(self, win, button, action, mods):
        pressed = (action == glfw.PRESS)
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._mouse_left = pressed
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._mouse_right = pressed

    def _on_scroll(self, win, xoff, yoff):
        self._z_scroll_pending += yoff

    def _on_key(self, win, key, sc, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(self.win, True)
        elif key == glfw.KEY_LEFT_BRACKET:   # '[' close
            self._grip = max(0.0, self._grip - 0.05)
        elif key == glfw.KEY_RIGHT_BRACKET:  # ']' open
            self._grip = min(1.0, self._grip + 0.05)
        elif key == glfw.KEY_R:              # reset sim
            self.env.reset()

    # ---------- main loop ----------
    def run(self):
        ctrl_dt = self.env.ctrl_dt
        next_ctrl = time.perf_counter()

        print("Controls: L-drag=XY, Scroll=Z, R-drag=Yaw, '[' close, ']' open, R reset, Esc quit")

        while not glfw.window_should_close(self.win):
            glfw.poll_events()

            # apply scroll to Z
            if abs(self._z_scroll_pending) > 1e-9:
                pos, quat = self.env.get_mocap_pose()
                pos[2] += Z_GAIN * self._z_scroll_pending
                self._z_scroll_pending = 0.0
                self.env.set_mocap(pos, quat)

            now = time.perf_counter()
            if now >= next_ctrl:
                # build action from current mocap + grip
                pos, quat = self.env.get_mocap_pose()
                action = np.array([pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3], self._grip])
                self.env.step(action)
                next_ctrl += ctrl_dt

            # draw
            fbw, fbh = glfw.get_framebuffer_size(self.win)
            viewport = mj.MjrRect(0, 0, fbw, fbh)
            mj.mjv_updateScene(self.env.mj_model, self.env.data, self.opt, None, self.cam,
                               mj.mjtCatBit.mjCAT_ALL, self.scn)
            mj.mjr_render(viewport, self.scn, self.ctx)
            glfw.swap_buffers(self.win)

        glfw.terminate()

env = PegInsertionEnv()
mouse_collector = MouseTeleop(env)
mouse_collector.run()