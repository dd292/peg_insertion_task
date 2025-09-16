import mujoco
import mujoco.viewer
import time
from peg_insertion_env import PegInsertionEnv
import numpy as np

env = PegInsertionEnv()
m = env.mj_model
d = env.data
action = np.random.randn(8)

with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()
        
        
        viewer.sync()
        # with viewer.lock():
        print(viewer.perturb.refpos)
        env.step(action)
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)