#!/usr/bin/env python
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import time
from light_mappo.envs.UCE.scenarios import load
from UCE.environment import MultiAgentEnv
from UCE.policy import InteractivePolicy
import UCE.scenarios as scenarios
from light_mappo.config import get_config


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="uav_swarm_confrontation", help="Which scenario to run on")
    # parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=6, help="number of players")
    parser.add_argument("--num_policy_agents", type=int, default=3, help="number of policy players")
    parser.add_argument("--num_scripted_agents", type=int, default=3, help="number of policy players")
    parser.add_argument("--num_red_agents", type=int, default=3, help="number of red players")
    parser.add_argument("--num_blue_agents", type=int, default=3, help="number of blue players")
    parser.add_argument("--task_num", type=int, default=1, help="the number of tasks")
    parser.add_argument("--train_stage", type=str, default="curriculum", help="train stage (curriculum or self-play)")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    # parse arguments
    parser = get_config()
    all_args = parse_args(args, parser)

    # load scenario from script
    scenario = load(all_args.scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(all_args)
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world,
                        scenario.reward, scenario.observation, scenario.info)    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    # policies = [InteractivePolicy(env,i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        # for i, policy in enumerate(policies):
        #     act_n.append(policy.action(obs_n[i]))
        # step environment
        _, reward_n, done_n, _ = env.step(act_n)
        # render all agent views
        env.render()
        time.sleep(0.1)
        # display rewards
        #for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))


if __name__ == "__main__":
    main(sys.argv[1:])
