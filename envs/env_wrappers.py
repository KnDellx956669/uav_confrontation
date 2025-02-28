"""
# @Time    : 2021/7/1 8:44 上午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_wrappers.py
Modified from OpenAI Baselines code to work with multi-agent envs
"""

import numpy as np
import torch.multiprocessing as mp
import cloudpickle
from torch.multiprocessing import Pipe


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


# single env
class DummyVecEnv():
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.num_envs = len(env_fns)
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space
        self.actions = None

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i] = self.envs[i].reset()

        # so far, the rewards can be positive
        self.actions = None
        return obs, rews, dones, infos

    def step_preset(self, actions):
        self.step_async(actions)
        return self.step_preset_wait()

    def step_preset_wait(self):
        results = [env.step_preset(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i] = self.envs[i].reset()

        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        obs = [env.reset() for env in self.envs] # [env_num, agent_num, obs_dim]
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError

    def get_share_obs(self):
        return np.array([env.share_observation for env in self.envs])


class SubprocVecEnv():
    def __init__(self, env_fns):
        """
        env_fns: List of callables that create gym environments
        """
        self.waiting = False
        self.closed = False

        self.envs = [fn() for fn in env_fns]
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.processes = [
            mp.Process(target=self.worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]

        for process in self.processes:
            process.daemon = True  # Set daemon to True for subprocess cleanup
            process.start()

        for remote in self.work_remotes:
            remote.close()

        # Get spaces from the first environment
        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    @staticmethod
    def worker(remote, parent_remote, env_fn_wrapper):
        parent_remote.close()
        env = env_fn_wrapper()
        while True:
            try:
                cmd, data = remote.recv()
                if cmd == 'step':
                    obs, reward, done, info = env.step(data)
                    if done:
                        obs = env.reset()
                    remote.send((obs, reward, done, info))
                elif cmd == 'reset':
                    obs = env.reset()
                    remote.send(obs)
                elif cmd == 'get_spaces':
                    remote.send((env.observation_space, env.action_space))
                elif cmd == 'close':
                    env.close()
                    remote.close()
                    break
                else:
                    raise NotImplementedError
            except EOFError:
                break

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rewards, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rewards), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_share_obs(self):
        return np.array([env.share_observation for env in self.envs])