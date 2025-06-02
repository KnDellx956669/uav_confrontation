# !/usr/bin/env python
import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from light_mappo.config import get_config
from light_mappo.envs.env_wrappers import DummyVecEnv, SubprocVecEnv
from light_mappo.envs.UCE.UCE_env import UCEEnv

"""Train script for MPEs."""


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # TODO 注意注意，这里选择连续还是离散可以选择注释上面两行，或者下面两行。
            # TODO Important, here you can choose continuous or discrete action space by uncommenting the above two lines or the below two lines.
            # from light_mappo.envs.env_continuous import ContinuousActionEnv
            #
            # env = ContinuousActionEnv()
            env = UCEEnv(all_args)

            # from envs.env_discrete import DiscreteActionEnv

            # env = DiscreteActionEnv()

            env.seed(all_args.seed)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


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
    parser.add_argument("--scenario_name", type=str, default="uav_swarm_confrontation", help="Which scenario to run on")
    # parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=6, help="number of players")
    parser.add_argument("--num_policy_agents", type=int, default=3, help="number of policy players")
    parser.add_argument("--num_scripted_agents", type=int, default=3, help="number of policy players")
    parser.add_argument("--num_red_agents", type=int, default=3, help="number of red players")
    parser.add_argument("--num_blue_agents", type=int, default=3, help="number of blue players")
    parser.add_argument("--task_num", type=int, default=1, help="the number of tasks")
    parser.add_argument("--train_stage", type=str, default="self-play", help="train stage (curriculum or self-play)")
    parser.add_argument("--use_preset", default=False, help="use preset physical parameters")
    parser.add_argument("--selfplay_model", default="D:/light_mappo/light_mappo/results/MyEnv/uav_swarm_confrontation/rmappo/check/run86/models/actor.pt", help="model for blue uavs in self-play stage")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert (
            all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False
        ), "check recurrent policy!"
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents
    num_policy_agents = all_args.num_policy_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "num_policy_agents": num_policy_agents,
        "device": device,
        "run_dir": run_dir,
        "use_preset": all_args.use_preset,
        "use_NVMAPPO": all_args.use_NVMAPPO
    }

    # run experiments
    if all_args.share_policy:
        from light_mappo.runner.shared.env_runner import EnvRunner as Runner
    else:
        from light_mappo.runner.separated.env_runner import EnvRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
