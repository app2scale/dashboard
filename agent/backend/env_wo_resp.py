#### No heap in state

from ray.rllib.env.policy_client import PolicyClient
import pandas as pd
from prometheus_api_client import PrometheusConnect
from kubernetes import client, config
import time
import numpy as np
from collections import OrderedDict
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, Tuple, Box
import ssl
import random
import logging
ssl._create_default_https_context = ssl._create_unverified_context
from itertools import product
import time
import gymnasium as gym
import math



class Teastore(gym.Env):
    #DATA_PATH = "agent/data/all_load_mpa_cpu_and_performance_without_average_payten_with_response.csv"
    #MAX_STEPS = 120


    def __init__(self, data, max_steps) -> None:
        self.data = data
        self.max_steps = max_steps
        # drop_rows = (df["cpu_usage"] != 0) | (df["memory_usage"] != 0)
        # self.data = df[drop_rows].reset_index(drop=True)
        # replica limit, cpu limit, cpu usage, num_request/exptected_tps, previous_tps/expected_tps, response_time
        self.action_space = Discrete(5)
        self.observation_space = Box(low=np.array([1, 4, 0, 0, 0, 0]), high=np.array([3, 9, 1, 1, 5, 5000]), dtype=np.float32)
        self.count = 0
        self.info = {}
        self.idx = 0
        self.up = None
        self.load_shape = self.load_generator()

    def load_generator(self):
        trx_load_data = pd.read_csv("agent/data/transactions.csv")
        trx_load = trx_load_data["transactions"].values.tolist()
        trx_load = (trx_load/np.max(trx_load)*30).astype(int)+1
        indexes = [(177, 184), (661, 685), (1143, 1152), (1498, 1524), (1858, 1900)]
        clipped_data = []
        for idx in indexes:
            start, end = idx
            clipped_data.extend(trx_load[start:end+1])

        payten_expected = np.array(clipped_data)*8
        payten_previous = np.roll(payten_expected, shift=1)
        # payten_load = pd.DataFrame(np.column_stack((payten_previous, payten_expected)), columns=["previous_tps", "expected_tps"])
        load = np.column_stack((payten_previous, payten_expected))
        return load


    def find_next_state(self, target):
        current_step = (self.count + self.offset -1) % self.load_shape.shape[0]
        new_previous_tps = self.load_shape[current_step][0]
        new_expected_tps = self.load_shape[current_step][1]
        # new_previous_tps = 152
        # new_expected_tps = 152
        next = np.concatenate([target, [new_previous_tps, new_expected_tps]])
        equal_rows = np.all(self.data.loc[:, ["replica", "cpu", "previous_tps", "expected_tps"]].values == next, axis=1)
        matched_indexes = np.where(equal_rows)[0]
        return matched_indexes.tolist(), new_previous_tps, new_expected_tps
    

    

    def reset(self, *, seed=None, options=None):
        self.idx = random.randint(0, len(self.data)-1)
        # replica limit, cpu limit, cpu usage, num_request/exptected_tps, previous_tps/expected_tps, response_time
        self.state = np.array(self.data.loc[self.idx, ["replica", "cpu", "cpu_usage","processed_rate","load_change_rate", "response_time"]])
        offset_loc = np.array(self.data.loc[self.idx, ["previous_tps", "expected_tps", "num_request"]])
        # self.state = np.array([3,9,152,152])
        self.offset = np.where((self.load_shape == np.array([offset_loc[0], offset_loc[1]])).all(axis=1))[0][0]
        self.truncated = False
        self.terminated = False
        self.reward = 0
        self.count = 0
        self.info["previous_tps"] = offset_loc[0]
        self.info["expected_tps"] = offset_loc[1]
        self.info["num_request"] = offset_loc[2]
        return self.state, self.info
    
    def step(self, action):
        selected_row_idx = 0
        self.count += 1

        if action == 0:
            temp_state = self.state[0:2] + np.array([0, 0])
        elif action == 1: # increase_replica
            temp_state = self.state[0:2] + np.array([1, 0])
        elif action == 2: # decrease_replica
            temp_state = self.state[0:2] + np.array([-1, 0])
        elif action == 3:
            temp_state = self.state[0:2] + np.array([0, 1])
        else:
            temp_state = self.state[0:2] + np.array([0 , -1])
        


        idx, new_previous_tps, new_expected_tps  = self.find_next_state(temp_state)
        self.info["previous_tps"] = new_previous_tps
        self.info["expected_tps"] = new_expected_tps

        if idx:
            selected_row_idx = random.choice(idx)
            selected_data = self.data.iloc[selected_row_idx]
            self.info["num_request"] = selected_data["num_request"]
            self.state = np.array(selected_data[["replica", "cpu", "cpu_usage","processed_rate","load_change_rate", "response_time"]])
            self.reward = selected_data['reward']

        else:

            idx, _, _  = self.find_next_state(self.state[0:2])
            selected_row_idx = random.choice(idx)
            selected_data = self.data.iloc[selected_row_idx]
            self.info["num_request"] = selected_data["num_request"]
            self.state = np.array(selected_data[["replica", "cpu", "cpu_usage","processed_rate","load_change_rate", "response_time"]])
            self.reward = -10

        
        self.terminated = (self.count >= self.max_steps)
        self.truncated = self.terminated
        return self.state, self.reward, self.terminated, self.truncated, self.info
    









