from .env_wo_resp import Teastore
from ray.tune.registry import register_env
from ray.rllib.algorithms import ppo, sac, dqn
import ray
import gym
import shutil
import os
from ray.tune.logger import pretty_print
from ray import tune, air
import tensorboard
import string
import random
import datetime
from .metric_callbacks import MetricCallbacks


def train_agent():
    num_iterations = 10 #10000
    number_of_rollout_workers = 8
    evaluating_interval = 3000
    number_of_gpus = 0
    save_interval = 500

    ray.init(ignore_reinit_error=True)
    register_env("teastore", lambda config: Teastore())

    config_ppo = (ppo.PPOConfig()
        .rollouts(num_rollout_workers=number_of_rollout_workers, enable_connectors=False, num_envs_per_worker=1)
        .resources(num_gpus=number_of_gpus, num_cpus_per_worker=1)
        .environment(env="teastore")
        .exploration(exploration_config={"type": "EpsilonGreedy",
                                         "initial_epsilon": 1.0,
                                         "final_epsilon": 0.01,
                                         "epsilon_timesteps": int(1e6)})


        .training(train_batch_size=512,sgd_minibatch_size=64,
                  model={"fcnet_hiddens": [32,32]},
                  lr=0.00001,
                  entropy_coeff = 0.01)
        # .evaluation(evaluation_interval=evaluating_interval, evaluation_duration = 2)
        # .callbacks(MetricCallbacks)
        )

    config_dqn = (
        dqn.DQNConfig()
        .environment(env="teastore")
        .rollouts(num_rollout_workers=number_of_rollout_workers, enable_connectors=False, num_envs_per_worker=1)
        .resources(num_gpus=number_of_gpus, num_cpus_per_worker=1)
        .training(train_batch_size=64,
                  model={"fcnet_hiddens": [32,32]})
        # .callbacks(MetricCallbacks)
                    

    )

    algo = config_dqn.build()
    logdir = algo.logdir
    # policy_name = "/Users/hasan.nayir/ray_results/DQN_teastore_2024-04-25_12-22-41hfbkhwwp/checkpoint_010000"
    # algo.restore(policy_name)

    for i in range(num_iterations):
        print("------------- Iteration", i+1, "-------------")
        result = algo.train()
        if ((i+1) % save_interval) == 0:
            path_to_checkpoint = algo.save(checkpoint_dir = logdir) 
            print("----- Checkpoint -----")
            print(f"An Algorithm checkpoint has been created inside directory: {path_to_checkpoint}.")
        print(pretty_print(result))

        print("Episode Reward Mean: ", result["episode_reward_mean"])
        print("Episode Reward Min: ", result["episode_reward_min"])
        print("Episode Reward Max: ", result["episode_reward_max"])
    algo.save()
    algo.stop()
    ray.shutdown()
    
