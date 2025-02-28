import numpy as np


def scripted_action(world, agent):
    # let the blue uav to fly towards the red uav, the following setting raises a question, the red uav
    # if hard to escape since the omega and vel is limited
    # omega = 0
    #
    # if agent.state.index == 0:
    #     vel = 500
    # else:
    #     vel = 300
    #
    # ene_agent = world.policy_agents[agent.state.index]
    # if ene_agent.state.is_alive:
    #     d_vec = ene_agent.state.p_pos - agent.state.p_pos
    #     course_angle = np.arctan2(d_vec[1], d_vec[0])
    #     agent.state.fly_ceta = course_angle
    #
    # else:
    #     for tmp_agent in world.policy_agents:
    #         if tmp_agent.state.is_alive:
    #             d_vec = tmp_agent.state.p_pos - agent.state.p_pos
    #             course_angle = np.arctan2(d_vec[1], d_vec[0])
    #             agent.state.fly_ceta = course_angle
    #
    #             break
    #
    # return vel, omega

    # A new setting with a relative lower speed
    # omega = 0
    #
    # if agent.state.index == 0:
    #     vel = 350
    # else:
    #     vel = 200
    #
    # ene_agent = world.policy_agents[agent.state.index]
    # if ene_agent.state.is_alive:
    #     d_vec = ene_agent.state.p_pos - agent.state.p_pos
    #     course_angle = np.arctan2(d_vec[1], d_vec[0])
    #     agent.state.fly_ceta = course_angle
    #
    # else:
    #     for tmp_agent in world.policy_agents:
    #         if tmp_agent.state.is_alive:
    #             d_vec = tmp_agent.state.p_pos - agent.state.p_pos
    #             course_angle = np.arctan2(d_vec[1], d_vec[0])
    #             agent.state.fly_ceta = course_angle
    #
    #             break
    if world.world_step == 0:
        if agent.state.index == 0:
            vel = np.random.uniform(100, 500)
            omega = np.random.uniform(-np.pi/6, np.pi/6)
        else:
            vel = np.random.uniform(60, 300)
            omega = np.random.uniform(-np.pi/6, np.pi/6)
    else:
        if agent.state.index == 0:
            vel = np.random.uniform(100, 500)
            # vel = 500
        else:
            vel = np.random.uniform(60, 300)
            # vel = 300
        omega = 0

    return vel, omega