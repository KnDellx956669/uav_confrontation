import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from numba import jit

from ...algorithms.algorithm.r_actor_critic import R_Actor


def _t2n(x):
    return x.detach().cpu().numpy()


# physical/external base state of all entities
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


# state of agents velocity, position, is_alive, ceta, index, omega, phi, rho
class AgentState(EntityState):
    def __init__(self, index):
        super(AgentState, self).__init__()
        self.is_alive = 1
        self.fly_ceta = None
        self.index = index
        self.phi = None
        self.rho = None
        self.omega = None

    # set the vector of velocity
    @property
    def vel_vec(self):
        if self.fly_ceta and self.p_vel is not None:
            vel_vec = self.p_vel * np.array([np.cos(self.fly_ceta), np.sin(self.fly_ceta)])
            return vel_vec


# action of the agent
class Action(object):
    def __init__(self, args, observation_space, act_space, model_path=None):
        # physical action
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.num_agents = args.num_scripted_agents
        self.actor = R_Actor(args, observation_space, act_space, model_path=model_path)
        self.recurrent_N = args.recurrent_N
        self.hidden_size = args.hidden_size
        self.rnn_states = np.zeros(
            (self.episode_length + 1, self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32)

    def act(self, obs, masks, step):
        step = step - 1
        actions, action_log_probs, rnn_states = self.actor(obs, self.rnn_states[step], masks)
        self.rnn_states[step + 1] = _t2n(rnn_states).copy()
        return actions


# properties of agent entities
class Agent:
    def __init__(self, index, uav_color):
        super(Agent, self).__init__()
        # name
        self.name = ''
        # color and size of the agent
        self.color = uav_color
        self.rgb_color = None
        self.size = 0.05
        # agents are movable by default
        self.movable = True
        # state
        self.state = AgentState(index=index)
        # action
        self.action_callback = None
        # index of reward
        self.reward_index = 0

        # define the action space
        if self.state.index == 0:
            action_low = np.array([100, -np.pi/6], dtype=np.float32)
            action_high = np.array([500, np.pi/6], dtype=np.float32)
            self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        else:
            action_low = np.array([60, -np.pi/6], dtype=np.float32)
            action_high = np.array([300, np.pi/6], dtype=np.float32)
            self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)


# multi-agent world
class World(object):
    # TODO:how to settle out the question about the relation between the coordinates and the velocity
    def __init__(self):
        # list of agents
        self.agents = []
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep, every timestep means 1 second
        self.dt = 4
        # number of agents
        self.num_agents = 6

        self.world_step = 0
        self.num_agents = 0
        self.num_red_agents = 0
        self.num_blue_agents = 0

        # the api for task, by default, the task number is 1
        self.task_num = 1

        # serves for self-play
        self.scripted_act_selfplay = None
        self.selfplay = None
        self.action_space = []

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def get_observation(self, agent):
        """need to get fri_dist_i, fri_ang_alpha_i, fri_ang_beta_i, ene_dist_i, ene_ang_alpha_i, ene_ang_beta_i
        fri_num_liv, fri_ene_liv"""
        if not agent.state.is_alive:
            return np.zeros((self.num_red_agents*7, ))

        fri_dist_i, ene_dist_i = self.get_dist(agent)
        fri_dist_i, ene_dist_i = fri_dist_i / 282800, ene_dist_i / 282800
        fri_ang_alpha_i, fri_ang_beta_i, ene_ang_alpha_i, ene_ang_beta_i = self.get_ang(agent)
        fri_ang_alpha_i, fri_ang_beta_i, ene_ang_alpha_i, ene_ang_beta_i = \
            fri_ang_alpha_i / np.pi, fri_ang_beta_i / np.pi, ene_ang_alpha_i / np.pi, ene_ang_beta_i / np.pi
        fri_liv_i, ene_liv_i = self.get_num_liv()

        # delete self element
        fri_dist_i = np.delete(fri_dist_i, agent.state.index, axis=0)
        fri_ang_alpha_i = np.delete(fri_ang_alpha_i, agent.state.index, axis=0)
        fri_ang_beta_i = np.delete(fri_ang_beta_i, agent.state.index, axis=0)

        # use np.concatenate to concatenate the arrays
        observation = np.concatenate([
            fri_dist_i.flatten(),
            fri_ang_alpha_i.flatten(),
            fri_ang_beta_i.flatten(),
            ene_dist_i.flatten(),
            ene_ang_alpha_i.flatten(),
            ene_ang_beta_i.flatten(),
            fri_liv_i.flatten(),
            ene_liv_i.flatten()
        ])

        return observation

    # get the tensor of live index with the shape of (n, 1)
    def get_num_liv(self):
        l_r = np.ones((self.num_red_agents, 1))
        l_b = np.ones((self.num_blue_agents, 1))

        for agent in self.agents:
            if agent.color == 'Red':
                l_r[agent.state.index, 0] = 1 if agent.state.is_alive else 0
            elif agent.color == 'Blue':
                l_b[agent.state.index, 0] = 1 if agent.state.is_alive else 0

        return l_r, l_b

    def get_dist(self, agent):
        fri_dist = np.zeros((self.num_red_agents if agent.color == 'Red' else self.num_blue_agents, 1))
        ene_dist = np.zeros((self.num_blue_agents if agent.color == 'Red' else self.num_red_agents, 1))

        for tmp_agent in self.agents:
            if tmp_agent.color == agent.color:
                fri_dist[tmp_agent.state.index, 0] = (np.linalg.norm(agent.state.p_pos - tmp_agent.state.p_pos)) if tmp_agent.state.is_alive else 0
            else:
                tmp_d = (np.linalg.norm(agent.state.p_pos - tmp_agent.state.p_pos)) if tmp_agent.state.is_alive else 0
                ene_dist[tmp_agent.state.index, 0] = tmp_d

        return fri_dist, ene_dist

    def get_ang(self, agent):
        num_agents = self.num_red_agents if agent.color == 'Red' else self.num_blue_agents
        fri_ang_alpha = torch.zeros(num_agents, 1)
        ene_ang_alpha = torch.zeros(self.num_blue_agents if agent.color == 'Red' else self.num_red_agents, 1)
        fri_ang_beta = torch.zeros(num_agents, 1)
        ene_ang_beta = torch.zeros(self.num_blue_agents if agent.color == 'Red' else self.num_red_agents, 1)

        v_self_vec = torch.tensor(agent.state.vel_vec)

        for tmp_agent in self.agents:
            if (tmp_agent.state.index == agent.state.index) and (tmp_agent.color == agent.color):
                continue

            v_other_vec = torch.tensor(tmp_agent.state.vel_vec)
            d_vec = torch.tensor(tmp_agent.state.p_pos - agent.state.p_pos)

            cos_alpha = torch.dot(d_vec, v_self_vec) / (torch.norm(d_vec) * torch.norm(v_self_vec))
            cos_beta = torch.dot(d_vec, v_other_vec) / (torch.norm(d_vec) * torch.norm(v_other_vec))
            cos_alpha = torch.clamp(cos_alpha, -1, 1)
            cos_beta = torch.clamp(cos_beta, -1, 1)

            if tmp_agent.color == agent.color:
                fri_ang_alpha[tmp_agent.state.index, 0] = torch.acos(cos_alpha) if tmp_agent.state.is_alive else 0
                fri_ang_beta[tmp_agent.state.index, 0] = torch.acos(cos_beta) if tmp_agent.state.is_alive else 0
            else:
                ene_ang_alpha[tmp_agent.state.index, 0] = torch.acos(cos_alpha) if tmp_agent.state.is_alive else 0
                ene_ang_beta[tmp_agent.state.index, 0] = torch.acos(cos_beta) if tmp_agent.state.is_alive else 0

        return fri_ang_alpha.numpy(), fri_ang_beta.numpy(), ene_ang_alpha.numpy(), ene_ang_beta.numpy()

    # update state of the world
    def step(self):
        self.world_step += 1

        # set actions for scripted agents
        # get obs for blue uavs and masks, step
        if self.selfplay:
            obs = []
            masks = np.ones((self.num_blue_agents, 1))
            for agent in self.scripted_agents:
                obs.append(self.get_observation(agent))
                masks[agent.state.index] = 0 if not agent.state.is_alive else 1
            obs = np.stack(obs, axis=0)
            obs = obs.reshape(self.num_blue_agents, -1)
            actions = self.scripted_act_selfplay.act(obs, masks, self.world_step)

            for i, agent in enumerate(self.scripted_agents):
                self._set_action(actions[i], agent, self.action_space[i])
        else:
            for agent in self.scripted_agents:
                agent.state.p_vel, agent.state.omega = agent.action_callback(self, agent)

        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

        # update agent's living index
        if self.task_num == 1:
            self.update_living_index_1()
        elif self.task_num == 2:
            self.update_living_index_2()

    def update_agent_state(self, agent):

        # update the state of the agents
        agent.state.p_pos += np.array(agent.state.vel_vec) * self.dt
        agent.state.fly_ceta += np.array(agent.state.omega) * self.dt

    def update_living_index_1(self):
        """Update the living index of the agents."""
        # Initialize matrices to track the circumstances
        circumstance_red_all_live = np.ones((self.num_red_agents, self.num_blue_agents))
        circumstance_blue_all_live = np.ones((self.num_blue_agents, self.num_red_agents))

        # Precompute the agent conditions: only need to iterate over "alive" agents
        live_red_agents = [agent for agent in self.agents if agent.state.is_alive and agent.color == 'Red']
        live_blue_agents = [agent for agent in self.agents if agent.state.is_alive and agent.color == 'Blue']

        # Iterate over all red agents
        for red_agent in live_red_agents:
            # Get relative values for this red agent
            tmp_fri_dist, tmp_ene_dist = self.get_dist(red_agent)
            tmp_fri_ang_alpha, tmp_fri_ang_beta, tmp_ene_ang_alpha, tmp_ene_ang_beta = self.get_ang(red_agent)

            # Use a list to punish the relative distance between blue and red
            tmp_blue_dist_list = []

            # Iterate over all blue agents
            for blue_agent in live_blue_agents:
                # Precompute necessary values for blue agent
                blue_index_value = tmp_ene_dist[blue_agent.state.index, 0]
                tmp_blue_dist_list.append(blue_index_value)
                blue_ang_alpha_value = tmp_ene_ang_alpha[blue_agent.state.index, 0]
                blue_ang_beta_value = tmp_ene_ang_beta[blue_agent.state.index, 0]

                # Precompute boolean flags for distances and angles (avoid re-calculation)
                dist_index = blue_index_value <= red_agent.state.rho
                dist_self_index = blue_index_value <= blue_agent.state.rho
                ang_adv_index = blue_ang_alpha_value <= red_agent.state.phi
                ang_self_index = (np.pi - blue_ang_beta_value) <= blue_agent.state.phi

                if dist_index and ang_adv_index:
                    circumstance_blue_all_live[blue_agent.state.index, red_agent.state.index] = 0
                    red_agent.reward_index += 1  # Update reward index

                if dist_self_index and ang_self_index:
                    circumstance_red_all_live[red_agent.state.index, blue_agent.state.index] = 0

            # Receive punish reward with respect to reward
            # if tmp_blue_dist_list:
            #     min_d = min(tmp_blue_dist_list)
            #     penalty_reward = min_d / (200000 * 1.414 * 400)
            #     red_agent.reward_index -= penalty_reward

        # Update the living status of all agents (after completing the inner loops)
        for tmp_agent in self.agents:
            if not tmp_agent.state.is_alive:
                continue

            if tmp_agent.color == 'Red':
                # Check if this red agent is "dead" in circumstance matrix
                if np.any(circumstance_red_all_live[tmp_agent.state.index] != 1):
                    tmp_agent.state.is_alive = 0
                    tmp_agent.reward_index = -1
                    # pass
            if tmp_agent.color == 'Blue':
                # Check if this blue agent is "dead" in circumstance matrix
                if np.any(circumstance_blue_all_live[tmp_agent.state.index] != 1):
                    tmp_agent.state.is_alive = 0

    def update_living_index_2(self):
        """Update the living index of the agents."""
        # Initialize matrices to track the circumstances
        circumstance_red_all_live = np.ones((self.num_red_agents, self.num_blue_agents))
        circumstance_blue_all_live = np.ones((self.num_blue_agents, self.num_red_agents))

        # Precompute the agent conditions: only need to iterate over "alive" agents
        live_red_agents = [agent for agent in self.agents if agent.state.is_alive and agent.color == 'Red']
        live_blue_agents = [agent for agent in self.agents if agent.state.is_alive and agent.color == 'Blue']

        # Iterate over all red agents
        for red_agent in live_red_agents:
            # Get relative values for this red agent
            tmp_fri_dist, tmp_ene_dist = self.get_dist(red_agent)
            tmp_fri_ang_alpha, tmp_fri_ang_beta, tmp_ene_ang_alpha, tmp_ene_ang_beta = self.get_ang(red_agent)

            # Judge whether agent goes over the area
            lower_bound = 0
            uppper_bound = 200000

            out_of_bounds = np.any((red_agent.state.p_pos < lower_bound) | (red_agent.state.p_pos > uppper_bound))

            if out_of_bounds:
                red_agent.reward_index -= 5

            # Iterate over all blue agents
            for blue_agent in live_blue_agents:
                # Precompute necessary values for blue agent
                blue_index_value = tmp_ene_dist[blue_agent.state.index, 0]
                blue_ang_alpha_value = tmp_ene_ang_alpha[blue_agent.state.index, 0]
                blue_ang_beta_value = tmp_ene_ang_beta[blue_agent.state.index, 0]

                # Precompute boolean flags for distances and angles (avoid re-calculation)
                dist_index = blue_index_value <= red_agent.state.rho
                dist_self_index = blue_index_value <= blue_agent.state.rho
                ang_adv_index = blue_ang_alpha_value <= red_agent.state.phi
                ang_self_index = (np.pi - blue_ang_beta_value) <= blue_agent.state.phi

                if dist_index and ang_adv_index:
                    if blue_agent.state.index == 0:
                        circumstance_blue_all_live[blue_agent.state.index, red_agent.state.index] = 0
                        red_agent.reward_index += 3  # Update reward index
                    else:
                        circumstance_blue_all_live[blue_agent.state.index, red_agent.state.index] = 0
                        red_agent.reward_index += 1  # Update reward index

                if dist_self_index and ang_self_index:
                    circumstance_red_all_live[red_agent.state.index, blue_agent.state.index] = 0

        # Update the living status of all agents (after completing the inner loops)
        for red_agent in live_red_agents:
            if np.any(circumstance_red_all_live[red_agent.state.index] != 1):
                red_agent.state.is_alive = 0
                if red_agent.state.index == 0:
                    red_agent.reward_index -= 3
                else:
                    red_agent.reward_index -= 1

            # Rewards of distance
            tmp_fri_dist, tmp_ene_dist = self.get_dist(red_agent)
            for other_red_agent in live_red_agents:
                if red_agent.state.index == other_red_agent.state.index:
                    continue

                if tmp_fri_dist[other_red_agent.state.index, 0] > 60000:
                    red_agent.reward_index -= 5

        for blue_agent in live_blue_agents:
            # Check if this blue agent is "dead" in circumstance matrix
            if np.any(circumstance_blue_all_live[blue_agent.state.index] != 1):
                blue_agent.state.is_alive = 0

        # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        # process action
        action = self.scale_action(action, agent, action_space)
        if agent.state.is_alive:
            agent.state.p_vel = action[0]
            agent.state.omega = action[1]
        else:
            agent.state.p_vel = 0
            agent.state.omega = 0

        action = action[2:]
        assert len(action) == 0, "action should not be empty"

    def scale_action(self, action, agent, action_space):
        low = action_space.low
        high = action_space.high
        action = _t2n(action)
        action = low + (action + 1.0) * 0.5 * (high - low)
        clipped_action = np.clip(action, low, high)

        return clipped_action

