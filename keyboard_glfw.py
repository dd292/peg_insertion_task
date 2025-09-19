import time, math, numpy as np, glfw, mujoco as mj
from peg_insertion_env_mujoco import PegInsertionEnv


BASE_XY_SPEED   = 0.25     # m/s
BASE_Z_SPEED    = 0.20     # m/s
BASE_YAW_RATE   = 0.80     # rad/s
BASE_PITCH_RATE = 0.80     # rad/s
BASE_ROLL_RATE  = 0.80     # rad/s
GRIP_RATE       = 10.0     # per second




class InputDevice:
    def poll(self) -> dict:
        raise NotImplementedError

    def help(self) -> str:
        return ""


class GlfwKeyboardDevice(InputDevice):
    def __init__(self, win, ctrl_dt, speed_preset=2):
        self.win = win
        self.dt = ctrl_dt
        self.speed_preset = speed_preset
        self._mods = 0
        self._grip = 0.0

        
        self._v_cmd = np.zeros(3)
        self._w_cmd = np.zeros(3)

        
        self._pressed = set()

        # bind callbacks
        glfw.set_key_callback(self.win, self._on_key)

    def _on_key(self, win, key, sc, action, mods):
        self._mods = mods
        press_or_repeat = (action in (glfw.PRESS, glfw.REPEAT))
        release = (action == glfw.RELEASE)

        # base rates scaled by current preset
        s = 0.5
        ax = ay = az = 0.0
        rx = ry = rz = 0.0
        reset = home = False

        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(self.win, True); return
        if action == glfw.PRESS and key == glfw.KEY_R:
            reset = True
            self._grip = 0.0
        if press_or_repeat:
            # translation (world XY, Z)
            if key == glfw.KEY_W: ay += BASE_XY_SPEED * s
            if key == glfw.KEY_S: ay -= BASE_XY_SPEED * s
            if key == glfw.KEY_D: ax += BASE_XY_SPEED * s
            if key == glfw.KEY_A: ax -= BASE_XY_SPEED * s
            if key == glfw.KEY_T: az += BASE_Z_SPEED  * s
            if key == glfw.KEY_G: az -= BASE_Z_SPEED  * s
            # rotation (body-fixed)
            if key == glfw.KEY_Z: rx -= BASE_ROLL_RATE  * s  # roll -
            if key == glfw.KEY_X: rx += BASE_ROLL_RATE  * s  # roll +
            if key == glfw.KEY_C: ry -= BASE_PITCH_RATE * s  # pitch -
            if key == glfw.KEY_V: ry += BASE_PITCH_RATE * s  # pitch +
            if key == glfw.KEY_B: rz -= BASE_YAW_RATE   * s  # yaw -
            if key == glfw.KEY_N: rz += BASE_YAW_RATE   * s  # yaw +
            # gripper
            if key == glfw.KEY_LEFT_BRACKET:
                self._grip = max(0.0, self._grip - GRIP_RATE*self.dt)
            if key == glfw.KEY_RIGHT_BRACKET:
                self._grip = min(1.0, self._grip + GRIP_RATE*self.dt)

            self._pressed.add(key)

        if release:
            self._pressed.discard(key)
            # zero released axes
            if key in (glfw.KEY_W, glfw.KEY_S): ay = 0.0
            if key in (glfw.KEY_A, glfw.KEY_D): ax = 0.0
            if key in (glfw.KEY_T, glfw.KEY_G): az = 0.0
            if key in (glfw.KEY_Z, glfw.KEY_X): rx = 0.0
            if key in (glfw.KEY_C, glfw.KEY_V): ry = 0.0
            if key in (glfw.KEY_B, glfw.KEY_N): rz = 0.0

        # update per-axis desired rates (merge so multiple keys can be held)
        if press_or_repeat or release:
            # linear
            if key in (glfw.KEY_A, glfw.KEY_D, glfw.KEY_W, glfw.KEY_S):
                self._v_cmd[0], self._v_cmd[1] = (ax, ay) if (ax or ay) else (0.0, 0.0)
            if key in (glfw.KEY_T, glfw.KEY_G):
                self._v_cmd[2] = az if (az or release) else self._v_cmd[2]
            # angular (body)
            if key in (glfw.KEY_Z, glfw.KEY_X):
                self._w_cmd[0] = rx if (rx or release) else self._w_cmd[0]
            if key in (glfw.KEY_C, glfw.KEY_V):
                self._w_cmd[1] = ry if (ry or release) else self._w_cmd[1]
            if key in (glfw.KEY_B, glfw.KEY_N):
                self._w_cmd[2] = rz if (rz or release) else self._w_cmd[2]

        # store buttons for this tick
        self._buttons = {
            "reset": reset,
            "home": home,
        }

    def poll(self):
        # Return the last computed commands; TeleopApp will smooth and integrate
        return {
            "v_world": self._v_cmd.copy(),
            "w_body":  self._w_cmd.copy(),
            "grip":    float(self._grip),
            "buttons": getattr(self, "_buttons", {"reset":False,"home":False,"speed":None})
        }

    def help(self) -> str:
        return (
            "GLFW Keyboard:\n"
            "  Move: W/A/S/D (XY), T/G (Z)\n"
            "  Rotate (body): Z/X=Roll, C/V=Pitch, B/N=Yaw\n"
            "  Grip: [ / ]  Reset: R   Quit: Esc\n"
        )