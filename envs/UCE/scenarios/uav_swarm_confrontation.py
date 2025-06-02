import numpy as np
from light_mappo.envs.UCE.core import World, Agent
from light_mappo.envs.UCE.core import Action
from light_mappo.envs.UCE.scenario import BaseScenario
from light_mappo.envs.UCE.randompolicy import scripted_action
# from light_mappo.envs.UCE.randomwalkpolicy import scripted_action

from gymnasium import spaces


class Scenario(BaseScenario):

    def make_world(self, args):
        world = World()
        world.world_length = args.episode_length
        world.task_num = args.task_num

        # set any world properties first
        world.num_red_agents = args.num_red_agents
        world.num_blue_agents = args.num_blue_agents
        world.num_agents = world.num_red_agents + world.num_blue_agents

        # add agents with color blue and red
        world.agents = []
        for index in range(world.num_red_agents):
            agent = Agent(index=index, uav_color='Red')
            agent.name = 'agent %d' % index
            agent.size = 0.05
            world.agents.append(agent)

        if args.train_stage == 'curriculum':
            for index in range(world.num_blue_agents):
                agent = Agent(index=index, uav_color='Blue')
                agent.name = 'agent %d' % index
                # change the size of the agent
                agent.size = 0.05
                world.agents.append(agent)

                # set action for scripted agents
                agent.action_callback = scripted_action

        elif args.train_stage == "self-play":
            for index in range(world.num_blue_agents):
                agent = Agent(index=index, uav_color='Blue')
                agent.name = 'agent %d' % index
                agent.size = 0.05
                world.agents.append(agent)

                # set action for scripted agents, though it's controlled by neural network
                agent.action_callback = scripted_action

                if agent.state.index == 0:
                    tmp_action_space = spaces.Box(low=np.array([100, -np.pi / 6]), high=np.array([500, np.pi / 6]),
                                                      shape=(2,), dtype=np.float32)
                else:
                    tmp_action_space = spaces.Box(low=np.array([60, -np.pi / 6]), high=np.array([300, np.pi / 6]),
                                                      dtype=np.float32)

                world.action_space.append(tmp_action_space)

            world.selfplay = True

        # make initial conditions
        self.reset_world(args, world)
        return world

    def reset_world(self, args, world):
        world.world_step = 0
        
        # set different color for red and blue uav
        for agent in world.agents:
            agent.reward_index = 0
            if agent.color == 'Red':
                agent.rgb_color = np.array([0.85, 0.35, 0.35])
            elif agent.color == 'Blue':
                agent.rgb_color = np.array([0.35, 0.35, 0.85])

        # set random initial states of pos, omega and vel and determined states
        for agent in world.agents:
            if agent.color == 'Red':
                agent.state.fly_ceta = np.random.uniform(0, np.pi/2)
            else:
                agent.state.fly_ceta = np.random.uniform(-np.pi/2, -np.pi)
            agent.state.is_alive = 1
            agent.state.omega = np.random.uniform(-np.pi/6, np.pi/6)

            # for different tasks, set different initial positions and velocities adn so on
            if world.task_num == 1 or world.task_num == 2:
                if agent.color == 'Red':
                    agent.state.p_pos = np.random.uniform(0, 80000, world.dim_p)
                else:
                    agent.state.p_pos = np.random.uniform(120000, 200000, world.dim_p)

                if agent.state.index == 0:
                    agent.state.p_vel = np.random.uniform(100, 500)
                    agent.state.phi = np.pi / 3
                    agent.state.rho = 15000
                else:
                    agent.state.p_vel = np.random.uniform(60, 300)
                    agent.state.phi = np.pi / 4
                    agent.state.rho = 10000

        if args.train_stage == "self-play":
            world.selfplay = True
            obs_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(21,), dtype=np.float32)
            act_space = spaces.Box(low=np.array([100, -np.pi / 6]), high=np.array([500, np.pi / 6]), shape=(2,),
                                   dtype=np.float32)
            world.scripted_act_selfplay = Action(args, obs_space, act_space, model_path=args.selfplay_model)

    # TODO: complete this benchmark_data function
    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for l in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            return tuple(dists)

    # gather local observation for each agent
    def observation(self, agent, world):
        return world.get_observation(agent)

    def reward(self, agent, world):
        # Agents are rewarded based on the destruction of the adversary
        if agent.color == 'Red':
            return agent.reward_index

    # TODO: complete this info function
    # # get the info of the current world state
    # def info(self, agent, world):
    #     return None
    #
    # def agent_reward(self, agent, world):
    #     # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
    #     shaped_reward = True
    #     shaped_adv_reward = True
    #
    #     # Calculate negative reward for adversary
    #     adversary_agents = self.adversaries(world)
    #     if shaped_adv_reward:  # distance-based adversary reward
    #         adv_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
    #     else:  # proximity-based adversary reward (binary)
    #         adv_rew = 0
    #         for a in adversary_agents:
    #             if np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) < 2 * a.goal_a.size:
    #                 adv_rew -= 5
    #
    #     # Calculate positive reward for agents
    #     good_agents = self.good_agents(world)
    #     if shaped_reward:  # distance-based agent reward
    #         pos_rew = -min(
    #             [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
    #     else:  # proximity-based agent reward (binary)
    #         pos_rew = 0
    #         if min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents]) \
    #                 < 2 * agent.goal_a.size:
    #             pos_rew += 5
    #         pos_rew -= min(
    #             [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
    #     return pos_rew + adv_rew
    #
    # def adversary_reward(self, agent, world):
    #     # Rewarded based on proximity to the goal landmark
    #     shaped_reward = True
    #     if shaped_reward:  # distance-based reward
    #         return -np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
    #     else:  # proximity-based reward (binary)
    #         adv_rew = 0
    #         if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < 2 * agent.goal_a.size:
    #             adv_rew += 5
    #         return adv_rew
    #
    #
