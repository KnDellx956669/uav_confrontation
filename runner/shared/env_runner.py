"""
# @Time    : 2021/7/1 7:15 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_runner.py
"""

import time
import numpy as np
import torch
import imageio


from light_mappo.runner.shared.base_runner import Runner

from collections import deque


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)
        self.noise_vector = None
        self.sigma = 1
        self.reset_noise_interval = -1
        self.noise_dim = 5
        self.reset_noise()
        self.complete_episode_count = 0
        self.record_full_win_history = []

    def reset_noise(self):
        "init noise"
        if self.noise_vector is None:
            self.noise_vector = []
            for i in range(self.num_agents):
                self.noise_vector.append(np.random.randn(self.noise_dim) * self.sigma)
            self.noise_vector = np.array(self.noise_vector)
        else:
            # shuffle noise
            np.random.shuffle(self.noise_vector)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        # set whether to use pre-set
        if self.use_preset:
            for episode in range(episodes):
                if self.use_linear_lr_decay:
                    self.trainer.policy.lr_decay(episode, episodes)
                if episode <= 100:
                    for step in range(self.episode_length):
                        # Sample actions
                        (
                            values,
                            actions,
                            action_log_probs,
                            rnn_states,
                            rnn_states_critic,
                            actions_env,
                        ) = self.collect(step)

                        # Obser reward and next obs
                        obs, rewards, dones, infos = self.envs.step_preset(actions_env)

                        # render the first env of all the n_rollout_threads envs in every episode
                        self.envs.envs[0].render()
                        time.sleep(0.001)

                        data = (
                            obs,
                            rewards,
                            dones,
                            infos,
                            values,
                            actions,
                            action_log_probs,
                            rnn_states,
                            rnn_states_critic,
                        )

                        # insert data into buffer
                        self.insert(data)

                    # compute return and update network
                    self.compute()
                    train_infos = self.train()

                    # post process
                    total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

                    # save model
                    if episode % self.save_interval == 0 or episode == episodes - 1:
                        self.save()

                    # log information
                    if episode % self.log_interval == 0:
                        end = time.time()
                        print(
                            "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                                self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start)),
                            )
                        )
                        train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                        print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                        self.log_train(train_infos, total_num_steps)
                        # self.log_env(env_infos, total_num_steps)

                    # eval
                    if episode % self.eval_interval == 0 and self.use_eval:
                        self.eval(total_num_steps)

                else:
                    for step in range(self.episode_length):
                        # Sample actions
                        (
                            values,
                            actions,
                            action_log_probs,
                            rnn_states,
                            rnn_states_critic,
                            actions_env,
                        ) = self.collect(step)

                        # Obser reward and next obs
                        obs, rewards, dones, infos = self.envs.step(actions_env)

                        # render the first env of all the n_rollout_threads envs in every episode
                        # self.envs.envs[0].render()
                        # time.sleep(0.001)

                        data = (
                            obs,
                            rewards,
                            dones,
                            infos,
                            values,
                            actions,
                            action_log_probs,
                            rnn_states,
                            rnn_states_critic,
                        )

                        # insert data into buffer
                        self.insert(data)

                    # compute return and update network
                    self.compute()
                    train_infos = self.train()

                    # post process
                    total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

                    # save model
                    if episode % self.save_interval == 0 or episode == episodes - 1:
                        self.save()

                    # log information
                    if episode % self.log_interval == 0:
                        end = time.time()
                        print(
                            "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                                self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start)),
                            )
                        )
                        train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                        print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                        self.log_train(train_infos, total_num_steps)
                        # self.log_env(env_infos, total_num_steps)
                    # eval
                    if episode % self.eval_interval == 0 and self.use_eval:
                        self.eval(total_num_steps)

        else:
            self.complete_episode_count = 0
            self.record_full_win_history = []
            win_history = deque(maxlen=100)
            for episode in range(episodes):
                if self.use_linear_lr_decay:
                    self.trainer.policy.lr_decay(episode, episodes)

                if self.use_NVMAPPO and self.reset_noise_interval != -1:
                    if episode % self.reset_noise_interval == 0:
                        print('shuffle noise vector...')
                        self.reset_noise()

                for step in range(self.episode_length):
                    # Sample actions
                    (
                        values,
                        actions,
                        action_log_probs,
                        rnn_states,
                        rnn_states_critic,
                        actions_env,
                    ) = self.collect(step)

                    # Obser reward and next obs
                    obs, rewards, dones, infos = self.envs.step(actions_env)  # shape of rewards is [5, 3, 1]
                    # render the first env of all the n_rollout_threads envs in every episode
                    self.envs.envs[0].render()
                    # time.sleep(0.02)
                    data = (
                        obs,
                        rewards,
                        dones,
                        infos,
                        values,
                        actions,
                        action_log_probs,
                        rnn_states,
                        rnn_states_critic,
                    )

                    # insert data into buffer
                    self.insert(data)

                    if np.all(dones):
                        # record winning result
                        win_history.append(int(infos[0][0]['WinningResult']['SelfWin']))
                        self.complete_episode_count += 1

                # compute return and update network
                self.compute()
                train_infos = self.train()

                # post process
                total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

                # save model
                if episode % self.save_interval == 0 or episode == episodes - 1:
                    self.save()

                # log information
                if episode % self.log_interval == 0:
                    end = time.time()
                    print(
                        "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                            self.all_args.scenario_name,
                            self.algorithm_name,
                            self.experiment_name,
                            episode,
                            episodes,
                            total_num_steps,
                            self.num_env_steps,
                            int(total_num_steps / (end - start)),
                        )
                    )
                    # so far, the reward can be positive !!!!!!!!!!
                    # the output clearly shows the positive rewards
                    train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                    episode_rewards = np.sum(self.buffer.rewards, axis=(0, 2))
                    print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                    # for i in range(5):
                    #     print("the {}th episode rewards is {}".format(i+1, episode_rewards[i][0]))
                    self.log_train(train_infos, total_num_steps)

                if len(win_history) == 100 and self.n_rollout_threads == 1:
                    win_rate = sum(win_history) / len(win_history)
                    self.record_full_win_history.append(win_rate)
                    assert self.complete_episode_count % 100 == 0
                    self.log_env(self.record_full_win_history, self.complete_episode_count/100)
                    win_history.clear()

                # eval
                if episode % self.eval_interval == 0 and self.use_eval:
                    self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()  # shape = [env_num, agent_num, obs_dim]

        # replay buffer
        if self.use_centralized_V:  # use this, the final shape is (5, 3, 72)
            share_obs = self.envs.get_share_obs()
            share_obs = np.expand_dims(share_obs, 1).repeat(
                self.num_agents, axis=1
            )  # shape = shape = [env_num, agent_num， agent_num * obs_dim]
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))  # [env_num, agent_num, 1]
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))  # [env_num, agent_num, action_dim]
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )  # [env_num, agent_num, 1]
        rnn_states = np.array(
            np.split(_t2n(rnn_states), self.n_rollout_threads)
        )  # [env_num, agent_num, 1, hidden_size]
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )  # [env_num, agent_num, 1, hidden_size]
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == "MultiDiscrete":
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == "Discrete":
            # actions  --> actions_env : shape:[10, 1] --> [5, 2, 5]
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            # TODO: change the shape of actions_env to fit the environment
            actions_env = actions
            # actions_env = torch.tanh(torch.tensor(actions_env)).numpy()
            # raise NotImplementedError

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        dones_env = np.all(dones, axis=1)  # distinguish

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array(
            [[[0.0] if hasattr(info[agent_id], 'bad_transition') and info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in
             infos])

        if self.use_centralized_V:
            share_obs = self.envs.get_share_obs()
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            # share_obs = obs.reshape(self.n_rollout_threads, -1)
            # share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        # if share_obs only for red_agents, then share_obs.shape=(5, 3, 72)
        # but we need a global observation which is (5, 3, 144) the extra 72 in dim=2 means the state of the blue_agents
        # which is not included in the share_obs before
        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            active_masks=active_masks,
            bad_masks=bad_masks
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            if self.eval_envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i] + 1)[
                        eval_actions[:, :, i]
                    ]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == "Discrete":
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos["eval_average_episode_rewards"] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos["eval_average_episode_rewards"])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs

        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render("rgb_array")[0][0]
                all_frames.append(image)
            else:
                envs.render("human")

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == "Discrete":
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    actions_env = actions

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    # calc_end = time.time()
                    # elapsed = calc_end - calc_start
                    # if elapsed < self.all_args.ifi:
                    #     time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render("human")

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            # imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
            pass