import time
import numpy as np
import mujoco
import glfw
from peg_insertion_env_mujoco import PegInsertionEnv

RTF = 1.0   # 1.0 for realtime, <1 for slow-mo, >1 for faster-than-realtime

def run_agent():
    env = PegInsertionEnv()
    sim_dt = env.mj_model.opt.timestep

    if not glfw.init():
        raise RuntimeError("GLFW initialization failed")
    window = glfw.create_window(1280, 720, "Peg Insertion Agent", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 1.2
    cam.azimuth = -60
    cam.elevation = -20
    cam.lookat[:] = [0.0, 0.0, 0.12]

    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    scn = mujoco.MjvScene(env.mj_model, maxgeom=10_000)
    ctx = mujoco.MjrContext(env.mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)

    step_count = 0
    max_steps = 5000
    while step_count < max_steps and not glfw.window_should_close(window):
        glfw.poll_events()

        #read pos
        hole_pos = env.data.site_xpos[env.hole_site_id].copy()
        peg_pos  = env.data.site_xpos[env.peg_site_id].copy()
        grip_pos, grip_quat = env.get_mocap_pose()

        #initalize constant command (This could be sent by an agent)
        #eg action = agent.step(hole_pos, peg_pos, grip_pos)
        cmd_pos  = np.array([0.0, 0.0, 0.15])
        cmd_quat = np.array([1.0, 0.0, 0.0, 0.0])
        cmd_grip = 0.5
        action = np.concatenate([cmd_pos, cmd_quat, [cmd_grip]])
        env.step(action)

        #render
        viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
        mujoco.mjv_updateScene(env.mj_model, env.data, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(viewport, scn, ctx)
        glfw.swap_buffers(window)

        #pace it wuth real time factor
        time.sleep(sim_dt / RTF)
        step_count += 1

    print("Simulation finished after", step_count, "steps")
    glfw.terminate()

if __name__ == "__main__":
    run_agent()
