import solara
import pandas as pd
import os
import zipfile
from ray.tune.registry import register_env
from ..environments.env import Teastore
from ray.rllib.algorithms import ppo, sac, dqn
from solara.components.file_drop import FileInfo
import time
from zipfile import ZipFile

test_plot_data = solara.reactive({'step': [], 'replica': [], 'cpu': [], "load": [], 
                                  "num_request": [], "response_time": []})
uploaded_algo = solara.reactive(None)
error_state = solara.reactive(None)
number_of_steps = solara.reactive(100)
available_checkpoint_names = solara.reactive([])
selected_checkpoint_name = solara.reactive(None)
uploaded_algo_status = solara.reactive(False)

@solara.component
def status_plot(data):
    options_replica = {
        "xAxis": {
            "type": "category",
            "data": data["step"],
        },
        "yAxis": {
            "type": "value",
        },
        "series": [ 
            {
                "name": "Replica",
                "data": data['replica'],
                "type": 'line'
            },      
        ],
        "title": {
            "text": 'Replica Number',
            "left": "center"
        },
        "legend": {
            "orient": 'vertical',
            "right": 0,
            # "top": 50,
            # "bottom": 50,
            "data": ["Replica"]
        },


    }

    options_cpu= {
        "xAxis": {
            "type": "category",
            "data": data["step"],
        },
        "yAxis": {
            "type": "value",
        },
        "series": [ 
            {
                "name": "CPU",
                "data": data['cpu'],
                "type": 'line'
            },         
        ],
        "title": {
            "text": 'CPU Limit',
            "left": "center"
            },
        "legend": {
            "orient": 'vertical',
            "right": 0,
            # "top": 50,
            # "bottom": 50,
            "data": ["CPU"]
        },
    }
    options_load= {
        "xAxis": {
            "type": "category",
            "data": data["step"],
        },
        "yAxis": {
            "type": "value",
        },
        "series": [ 
            {
                "name": "Processed req",
                "data": data['num_request'],
                "type": 'line'
            },
            {
                "name": "Load",
                "data": data['load'],
                "type": 'line'
            },

        ],
        "title": {
            "text": 'Number of Request (Tps) and Load (Tps)',
            "left": "center"
            },
        "legend": {
            "orient": 'vertical',
            "right": 0,
            # "top": 50,
            # "bottom": 50,
            "data": ["Processed req", "Load"]
        },
    }
    options_response_time= {
        "xAxis": {
            "type": "category",
            "data": data["step"],
        },
        "yAxis": {
            "type": "value",
        },
        "series": [ 
            {
                "name": "Response time",
                "data": data['response_time'],
                "type": 'line'
            },         
        ],
        "title": {
            "text": 'Response time (ms)',
            "left": "center"
            },
        "legend": {
            "orient": 'vertical',
            "right": 0,
            # "top": 50,
            # "bottom": 50,
            "data": ["Response time"]
        },        
    }

    with solara.GridFixed(columns=2):
        # with solara.Column():
        solara.FigureEcharts(option=options_replica)
        solara.FigureEcharts(option=options_cpu)
        
        
        solara.FigureEcharts(option=options_load)
        solara.FigureEcharts(option=options_response_time)

@solara.component
def ListAvailableCheckpoints():
    list_subfolders_names = [f.name for f in os.scandir("agent/checkpoints") if f.is_dir()]
    available_checkpoint_names.set(list_subfolders_names)

def load_agent():
    if selected_checkpoint_name.value is None:
        return None

    register_env("teastore", lambda config: Teastore())

    config_dqn = (
        dqn.DQNConfig()
        .environment(env="teastore")
        .rollouts(num_rollout_workers=1, enable_connectors=False, num_envs_per_worker=1)
        .resources(num_gpus=0, num_cpus_per_worker=1)
        .training(train_batch_size=256, model={"fcnet_hiddens": [32, 32]})
    )
    algo = config_dqn.build()
    checkpoint_dir = selected_checkpoint_name.value
    checkpoint_path = os.path.join("agent/checkpoints", checkpoint_dir)
    algo.restore(checkpoint_path)

    return algo

def load_test():
    algo = load_agent()
    if algo is None:
        error_state.set('Couldnt load checkpoint')
    else:
        uploaded_algo_status.set(True)
    uploaded_algo.set(algo)

def start_test():

    env = Teastore()
    obs, info = env.reset()   
    done = False
    truncated = False
    sum_reward = 0
    step_list = []
    replica_array = []
    cpu_array = []
    num_request_array = []
    load_array = []
    response_time_array = []

    for i in range(number_of_steps.value):
        step_list.append(i)
        replica_array.append(obs[0])
        cpu_array.append(obs[1])
        load_array.append(env.load)
        response_time_array.append(env.response_time)
        num_request_array.append(env.num_request)

        action = uploaded_algo.value.compute_single_action(obs, explore=False)
        next_state, reward, _, _, _ = env.step(action)
        obs = next_state


        test_plot_data.set(
                {'step':step_list.copy(),
                 'replica': replica_array.copy(),
                 'cpu': cpu_array.copy(),
                 "load": load_array.copy(),
                 "response_time": response_time_array.copy(),
                 "num_request": num_request_array.copy()
                 })
        # time.sleep(2)


@solara.component
def CheckpointUpload():
    filename, set_filename = solara.use_state("")

    def on_file(f: FileInfo):
        set_filename(f["name"])


        extracted_folder = os.path.join("agent/checkpoints")
        with ZipFile(f["file_obj"], "r") as zip_ref:
            zip_ref.extractall(extracted_folder)

    solara.FileDrop(label="Drang and drop checkpoint file as a zip", on_file=on_file, lazy=False)



@solara.component 
def Page():
    with solara.Sidebar():
        with solara.Card(title="Upload checkpoint or select"):
            CheckpointUpload()
        ListAvailableCheckpoints()
        solara.Select(label="Choose checkpoint", values=available_checkpoint_names.value, 
                      value=selected_checkpoint_name.value, on_value=selected_checkpoint_name.set)
        solara.Button(label="Run test", on_click=start_test, disabled=True if uploaded_algo.value is None else False)
        solara.Button(label="Load agent", on_click=load_test, disabled=True if selected_checkpoint_name.value is None else False)
        if uploaded_algo_status.value == False:
            solara.Info("Agent is not uploaded")
        else:
            solara.Info("Agent is ready for test")
        solara.SliderInt(label="choose number of steps", min=1, max=500, value=number_of_steps.value, on_value=number_of_steps.set)
        



    status_plot(test_plot_data.value)
