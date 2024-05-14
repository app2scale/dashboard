from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.env import BaseEnv
from typing import Dict, Optional, Tuple, Union
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy import Policy
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
import numpy as np
import pandas as pd



class MetricCallbacks(DefaultCallbacks):
    column_name = ["replica","cpu", 'previous_tps', "expected_tps", "prev_action", 'reward', 'sum_reward']
    training_history = pd.DataFrame(columns= column_name)
    ct = 0
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))
        try:
            action = episode._agent_to_last_action["agent0"]
            reward = episode._agent_reward_history['agent0'][-1]
            sum_reward = 0
        except:
            action = -1
            reward = 99
        temp = np.concatenate([episode._agent_to_last_raw_obs["agent0"], [action, reward, episode.total_reward]])
        self.training_history.loc[self.ct, :] = temp
        self.ct += 1

        # episode.user_data["training_history"] = self.training_history


    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        """
        Custom metrics and reward will be added.
        user data'da her key için append episode sonunda ortalama bunu da custom metrics
        start fonksiyonunda last_infos gelmiyor
        """
        try:
            action = episode._agent_to_last_action["agent0"]
            reward = episode._agent_reward_history['agent0'][-1]
        except:
            action = -1
            reward = 99
        temp = np.concatenate([episode._agent_to_last_raw_obs["agent0"], [action, reward, episode.total_reward]])
        self.training_history.loc[self.ct, :] = temp
        self.ct += 1

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs):
        try:
            action = episode._agent_to_last_action["agent0"]
            reward = episode._agent_reward_history['agent0'][-1]
        except:
            action = -1
            reward = 99
        temp = np.concatenate([episode._agent_to_last_raw_obs["agent0"], [action, reward, episode.total_reward]])
        self.training_history.loc[self.ct, :] = temp
        self.ct += 1
        self.training_history.to_csv("training_history.csv", index=False)


    
