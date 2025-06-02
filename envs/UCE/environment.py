import gymnasium as gym
from gym.envs.registration import EnvSpec
import numpy as np
from gymnasium import spaces
from numba import jit


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, args, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, post_step_callback=None,
                 shared_viewer=True):

        self.args = args
        self.world = world
        self.world_length = self.world.world_length
        self.current_step = 0
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.post_step_callback = post_step_callback

        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in self.agents:

            if agent.state.index == 0:
                tmp_action_space = spaces.Box(low=np.array([100, -np.pi/6]), high=np.array([500, np.pi/6]), shape=(2,), dtype=np.float32)
            else:
                tmp_action_space = spaces.Box(low=np.array([60, -np.pi/6]), high=np.array([300, np.pi/6]), dtype=np.float32)

            self.action_space.append(tmp_action_space)

            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))

        for all_agent in self.world.policy_agents:
            share_obs_dim += len(observation_callback(all_agent, self.world))
        self.share_observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def step(self, action_n):
        self.current_step += 1

        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        # set action for each agent
        for i, agent in enumerate(self.world.policy_agents):
            self._set_action(action_n[i], agent, self.action_space[i])

        # advance world state
        self.world.step()

        # the reward of the agent can be positive
        # record observation for each agent
        for i, agent in enumerate(self.world.policy_agents):
            obs_n.append(self._get_obs(agent))
            reward_n.append([self._get_reward(agent)])
            done_n.append(self._get_done(agent))
            info = {'individual_reward': self._get_reward(agent)}
            info_n.append(info)

        # record the winning result
        num_ene_liv = len([agent for agent in self.world.scripted_agents if agent.state.is_alive])
        num_self_liv = len([agent for agent in self.world.policy_agents if agent.state.is_alive])

        if self.current_step >= self.world_length:
            self_win = (num_self_liv > num_ene_liv)
            draw = (num_self_liv == num_ene_liv)
            if draw:
                for info in info_n:
                    info['WinningResult'] = {'SelfWin': False, 'EneWin': False}
                    info['bad_transition'] = True
            else:
                for info in info_n:
                    info['WinningResult'] = {'SelfWin': self_win, 'EneWin': not self_win}
                    info['bad_transition'] = True
        elif num_ene_liv == 0:
            for info in info_n:
                info['WinningResult'] = {'SelfWin': True, 'EneWin': False}
        elif num_self_liv == 0:
            for info in info_n:
                info['WinningResult'] = {'SelfWin': False, 'EneWin': True}

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        # reset all agents' rewards to be 0
        for agent in self.world.policy_agents:
            agent.reward_index = 0

        # so far, the reward can be positive
        return obs_n, reward_n, done_n, info_n


    def reset(self):
        self.current_step = 0
        # reset world
        self.reset_callback(self.args, self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        for agent in self.world.policy_agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # TODO:all the agents are dead or the time is over
    # TODO:unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        # if self.done_callback is None:
        #     if self.current_step >= self.world_length:
        #         return True
        #     else:
        #         exist_num_red = 0
        #         for agent in self.world.agents:
        #             if agent.state.is_alive and agent.color == 'Red':
        #                 exist_num_red += 1
        #         if exist_num_red == 0:
        #             return True
        #
        #         exist_num_blue = 0
        #         for agent in self.world.agents:
        #             if agent.state.is_alive and agent.color == 'Blue':
        #                 exist_num_blue += 1
        #         if exist_num_blue == 0:
        #             return True
        #
        #         return False
        num_ene_liv = len([agent for agent in self.world.scripted_agents if agent.state.is_alive])
        if num_ene_liv == 0:
            return True

        if self.done_callback is None:
            if self.current_step >= self.world_length:
                return True
            else:
                if not agent.state.is_alive:
                    return True
                else:
                    return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

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

    def _set_preset_action(self, action, agent, action_space, time=None):
        # process action
        action = self.scale_action(action, action_space)
        agent.state.p_vel = action[0]
        agent.state.omega = np.random.uniform(-np.pi/6000, np.pi/6000)

        action = action[2:]
        assert len(action) == 0, "action should be empty"

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def scale_action(self, action, agent, action_space):
        low = action_space.low
        high = action_space.high
        action = low + (action + 1.0) * 0.5 * (high - low)
        clipped_action = np.clip(action, low, high)
        # if not np.all((action <= high) & (action >= low)):
        #     omega = 0
        #
        #     if agent.state.index == 0:
        #         vel = 350
        #     else:
        #         vel = 200
        #
        #     ene_agent = self.world.scripted_agents[agent.state.index]
        #     if ene_agent.state.is_alive:
        #         d_vec = ene_agent.state.p_pos - agent.state.p_pos
        #         course_angle = np.arctan2(d_vec[1], d_vec[0])
        #         agent.state.fly_ceta = course_angle
        #
        #     else:
        #         for tmp_agent in self.world.scripted_agents:
        #             if tmp_agent.state.is_alive:
        #                 d_vec = tmp_agent.state.p_pos - agent.state.p_pos
        #                 course_angle = np.arctan2(d_vec[1], d_vec[0])
        #                 agent.state.fly_ceta = course_angle
        #
        #                 break
        #
        #     clipped_action = np.array((vel, omega))
        # else:
        #     clipped_action = action

        # omega = 0
        #
        # if agent.state.index == 0:
        #     vel = 350
        # else:
        #     vel = 200
        #
        # ene_agent = self.world.scripted_agents[agent.state.index]
        # if ene_agent.state.is_alive:
        #     d_vec = ene_agent.state.p_pos - agent.state.p_pos
        #     course_angle = np.arctan2(d_vec[1], d_vec[0])
        #     agent.state.fly_ceta = course_angle
        #
        # else:
        #     for tmp_agent in self.world.scripted_agents:
        #         if tmp_agent.state.is_alive:
        #             d_vec = tmp_agent.state.p_pos - agent.state.p_pos
        #             course_angle = np.arctan2(d_vec[1], d_vec[0])
        #             agent.state.fly_ceta = course_angle
        #
        #             break

        # clipped_action = np.array((vel, omega))
        return clipped_action

    # render environment
    def render(self, mode='human'):
        # create and update the current step text
        # print("self.current_step: ", self.current_step)

        if mode == 'human':
            pass

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                from light_mappo.envs.UCE import rendering
                self.viewers[i] = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from light_mappo.envs.UCE import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for agent in self.world.agents:
                # geom和xform都是对象，不是已经被渲染好的，是需要被渲染的
                agent.size = 1000
                geom = rendering.make_circle(agent.size)
                xform = rendering.Transform()
                if 'agent' in agent.name:
                    geom.set_color(*agent.rgb_color, alpha=0.5)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add a fan for every agent
            for agent in self.world.agents:
                color_tmp = (1.0, 0.2, 0.2) if agent.color == 'Red' else (0.2, 0.2, 1.0)
                geom_fan = rendering.make_fan(agent.state.rho, 2 * agent.state.phi, color=color_tmp)
                xform_fan = rendering.Transform()
                geom_fan.add_attr(xform_fan)
                self.render_geoms.append(geom_fan)
                self.render_geoms_xform.append(xform_fan)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)
                # create and update the current step text
                # from light_mappo.envs.UCE.rendering import TextRenderer
                # step_text = TextRenderer(text="Step: 0")
                # step_text.text = f"Step: {self.current_step}"
                # viewer.add_geom(step_text)

        results = []
        for i in range(len(self.viewers)):
            # update bounds to center around agent
            from light_mappo.envs.UCE import rendering
            self.viewers[i].set_bounds(0, 200000, 0, 200000)

            # update geometry positions
            for e, agent in enumerate(self.world.agents):
                if agent.state.is_alive:
                    self.render_geoms_xform[e].set_translation(*agent.state.p_pos)

                    # set the coordinate and the angle of the fan
                    self.render_geoms_xform[e + self.n].set_translation(*agent.state.p_pos)
                    self.render_geoms_xform[e + self.n].set_rotation(agent.state.fly_ceta)
                else:
                    for viewer in self.viewers:
                        viewer.geoms[e] = None
                        viewer.geoms[e + self.n] = None

            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array=(mode == 'rgb_array')))

        return results

        # gather global observation for each agent whose shape is (num_agents, obs)
    @property
    def share_observation(self):
        share_obs = []
        # for agent in self.world.policy_agents TODO:比较纠结是一半智能体还是全部的
        for agent in self.world.policy_agents:
            share_obs.append(self.world.get_observation(agent))

        share_obs = np.stack(share_obs, axis=0)
        return share_obs.reshape((-1,))
