import numpy as np


def scripted_action(world, agent, obs=None):
    # if world.world_step in [0, 30, 70, 120, 150, 200, 250, 310]:  # random number
    if world.world_step in [0, 70, 150, 250, 310]:
        if agent.state.index == 0:
            vel = np.random.uniform(100, 500)
            omega = np.random.uniform(-np.pi/6, np.pi/6)
        else:
            vel = np.random.uniform(60, 300)
            omega = np.random.uniform(-np.pi/6, np.pi/6)
    else:
        if agent.state.index == 0:
            vel = np.random.uniform(100, 500)
        else:
            vel = np.random.uniform(60, 300)
        omega = 0

    return vel, omega