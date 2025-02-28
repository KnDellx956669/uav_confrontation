import sys


from light_mappo.envs.env_wrappers import DummyVecEnv
from light_mappo.envs.UCE.UCE_env import UCEEnv
from light_mappo.config import get_config


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            from light_mappo.envs.env_continuous import ContinuousActionEnv

            # env = ContinuousActionEnv()
            # from envs.env_discrete import DiscreteActionEnv
            # env = DiscreteActionEnv()
            env = UCEEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    pass


def main(args):
    parser = get_config()


if __name__ == "__main__":
    main(sys.argv[1:])