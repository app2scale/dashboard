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
    DATA_PATH = "./all_load_mpa_cpu_and_performance_without_average.csv"
    MAX_STEPS = 500


    def __init__(self) -> None:
        self.data = pd.read_csv(self.DATA_PATH)
        # drop_rows = (df["cpu_usage"] != 0) | (df["memory_usage"] != 0)
        # self.data = df[drop_rows].reset_index(drop=True)
        self.action_space = Discrete(5)
        self.observation_space = Box(low=np.array([1, 4, 0, 0]), high=np.array([3, 9, 1000,1000]), dtype=np.float32)
        self.count = 0
        self.info = {}
        self.previous_tps = 0
        self.idx = 0
        self.up = None
        self.load = 0
        self.response_time = 0
        self.num_request = 0




    def find_next_state(self, target, expected_tps):
        if expected_tps == 144:
            self.up = False
        elif expected_tps == 24:
            self.up = True
        
        if self.up == True:
            new_expected_tps = expected_tps + 24
        elif self.up == False:
            new_expected_tps = expected_tps - 24

        new_previous_tps = expected_tps
        # new_expected_tps = 48
        # new_previous_tps = 24
        next = np.concatenate([target, [new_previous_tps, new_expected_tps]])
        equal_rows = np.all(self.data.loc[:, ["replica", "cpu", "previous_tps", "expected_tps"]].values == next, axis=1)
        matched_indexes = np.where(equal_rows)[0]
        return matched_indexes.tolist(), new_previous_tps, new_expected_tps
    

    
    def reset(self, *, seed=None, options=None):
        self.idx = random.randint(0, len(self.data)-1)
        self.state = np.array(self.data.loc[self.idx, ["replica", "cpu", 'previous_tps', "expected_tps"]])
        # self.state = np.array([3,9,24,48])
        self.previous_tps = self.state[2]
        self.truncated = False
        self.terminated = False
        self.reward = 0
        self.count = 0
        self.info = {}
        self.up = True if self.state[3] - self.state[2] > 0 else False 
        self.load = self.state[-1]
        self.response_time = self.data.loc[self.idx, "response_time"]
        self.num_request = self.data.loc[self.idx, "num_request"]
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
        


        idx, new_previous_tps, new_expected_tps  = self.find_next_state(temp_state, self.state[3])

        if idx:
            selected_row_idx = random.choice(idx)
            selected_data = self.data.iloc[selected_row_idx]
            self.state = np.array(selected_data[["replica", "cpu", 'previous_tps',"expected_tps"]])
            self.reward = selected_data['reward']
            # self.reward = 1
            # print(f"state: {self.state} - previous_tps: {self.previous_tps}")
            self.previous_tps = selected_data["expected_tps"]
            self.num_request = self.data.loc[selected_row_idx, "num_request"]
            self.response_time = self.data.loc[selected_row_idx, "response_time"]
            
        else:
            self.state[2] = new_previous_tps
            self.state[3] = new_expected_tps
            self.previous_tps = new_expected_tps
            self.reward = -5
            self.num_request = 0
            self.response_time = 200

        self.load = self.state[-1]
        # self.response_time = 20
        # self.num_request = 20

        self.terminated = (self.count >= self.MAX_STEPS)
        self.truncated = self.terminated
        return self.state, self.reward, self.terminated, self.truncated, self.info







