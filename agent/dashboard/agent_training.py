import asyncio
import solara
from solara.lab import task

from ray.tune.registry import register_env
from ray.rllib.algorithms import ppo, sac, dqn
import ray
from ray.tune.logger import pretty_print

from ..backend.env_wo_resp import Teastore
from ..backend.utils import read_data, Plot1D

from .data import state as data_state

local_state = solara.reactive(
    {
        'max_steps': solara.reactive(120),
        'num_iterations': solara.reactive(10),
        'number_of_rollout_workers': solara.reactive(8),
        'evaluating_interval': solara.reactive(3),
        'number_of_gpus': solara.reactive(0),
        'save_interval': solara.reactive(5),
        'num_envs_per_worker': solara.reactive(1),
        'num_cpus_per_worker': solara.reactive(1),
        'exploration_type': solara.reactive('EpsilonGreedy'),
        'initial_epsilon': solara.reactive(1.0),
        'final_epsilon': solara.reactive(0.01),
        'epsilon_timesteps': solara.reactive(int(1e6)),
        'learning_rate': solara.reactive(0.00001),
        'entropy_coeff': solara.reactive(0.01),
        'train_batch_size': solara.reactive(512),
        'sgd_minibatch_size': solara.reactive(64),
        'hidden_size_first': solara.reactive(32),
        'hidden_size_second': solara.reactive(32),
        'checkpoint_dirs': solara.reactive([]),
        'reward_mean': solara.reactive([]),
        'render_count': solara.reactive(0),
    }
    )
 

message = solara.reactive('')

@task
async def train_agent():
    message.set('Started training...')

    data = read_data(data_state.value['data_file'].value)
    max_steps = local_state.value['max_steps'].value
    num_iterations = local_state.value['num_iterations'].value
    number_of_rollout_workers = local_state.value['number_of_rollout_workers'].value
    evaluating_interval = local_state.value['evaluating_interval'].value
    number_of_gpus = local_state.value['number_of_gpus'].value
    save_interval = local_state.value['save_interval'].value
    num_envs_per_worker = local_state.value['num_envs_per_worker'].value
    num_cpus_per_worker = local_state.value['num_cpus_per_worker'].value
    exploration_type = local_state.value['exploration_type'].value
    initial_epsilon = local_state.value['initial_epsilon'].value
    final_epsilon = local_state.value['final_epsilon'].value
    epsilon_timesteps = local_state.value['epsilon_timesteps'].value
    learning_rate = local_state.value['learning_rate'].value
    entropy_coeff = local_state.value['entropy_coeff'].value
    train_batch_size = local_state.value['train_batch_size'].value
    sgd_minibatch_size = local_state.value['sgd_minibatch_size'].value
    hidden_size_first = local_state.value['hidden_size_first'].value
    hidden_size_second = local_state.value['hidden_size_second'].value

    ray.init(ignore_reinit_error=True)
    register_env("teastore", lambda config: Teastore(data, max_steps))

    config_ppo = (ppo.PPOConfig()
        .rollouts(num_rollout_workers=number_of_rollout_workers, enable_connectors=False, 
                  num_envs_per_worker=num_envs_per_worker)
        .resources(num_gpus=number_of_gpus, num_cpus_per_worker=num_cpus_per_worker)
        .environment(env="teastore")
        .exploration(exploration_config={"type": exploration_type,
                                         "initial_epsilon": initial_epsilon,
                                         "final_epsilon": final_epsilon,
                                         "epsilon_timesteps": epsilon_timesteps})


        .training(train_batch_size=train_batch_size,sgd_minibatch_size=sgd_minibatch_size,
                  model={"fcnet_hiddens": [hidden_size_first,hidden_size_second]},
                  lr=learning_rate,
                  entropy_coeff = entropy_coeff)
        # .evaluation(evaluation_interval=evaluating_interval, evaluation_duration = 2)
        # .callbacks(MetricCallbacks)
        )

    config_dqn = (
        dqn.DQNConfig()
        .environment(env="teastore")
        .rollouts(num_rollout_workers=number_of_rollout_workers, enable_connectors=False, 
                  num_envs_per_worker=num_envs_per_worker)
        .resources(num_gpus=number_of_gpus, num_cpus_per_worker=num_cpus_per_worker)
        .training(train_batch_size=train_batch_size,
                  model={"fcnet_hiddens": [hidden_size_first,hidden_size_second]})
        # .callbacks(MetricCallbacks)
                    

    )

    algo = config_dqn.build()
    logdir = algo.logdir
    # policy_name = "/Users/hasan.nayir/ray_results/DQN_teastore_2024-04-25_12-22-41hfbkhwwp/checkpoint_010000"
    # algo.restore(policy_name)

    
    reward_mean_list = []
    
    for i in range(num_iterations):

        print("------------- Iteration", i+1, "-------------")
        message.set(f'Iteration {i+1} of  {num_iterations}')
        result = algo.train()
        if ((i+1) % save_interval) == 0:
            path_to_checkpoint = algo.save(checkpoint_dir = logdir) 
            print("----- Checkpoint -----")
            print(f"An Algorithm checkpoint has been created inside directory: {path_to_checkpoint}.")
            existing_checkpoints = local_state.value['checkpoint_dirs'].value.copy()
            existing_checkpoints.append(path_to_checkpoint)
            local_state.value['checkpoint_dirs'].set(existing_checkpoints)


        print(pretty_print(result))

        print("Episode Reward Mean: ", result["episode_reward_mean"])
        print("Episode Reward Min: ", result["episode_reward_min"])
        print("Episode Reward Max: ", result["episode_reward_max"])
        reward_mean_list.append(result["episode_reward_mean"])
        local_state.value['reward_mean'].set(reward_mean_list.copy())
        if train_agent.cancelled:
            print('training is cancelled------------')
            break
    algo.save()
    algo.stop()
    ray.shutdown()




    message.set('Training is finished')

    return "Training is finished"

@solara.component
def Page():





    with solara.Sidebar():
            with solara.lab.Tabs():
                with solara.lab.Tab("SETTINGS"):
                    solara.InputInt(label="max_steps",
                                    value=local_state.value['max_steps'])
                    solara.InputInt(label="num_iterations",
                                    value=local_state.value['num_iterations'])                      
                    solara.InputInt(label="number_of_rollout_workers",
                                    value=local_state.value['number_of_rollout_workers'])                  
                    solara.InputInt(label="evaluating_interval",
                                    value=local_state.value['evaluating_interval'])  
                    solara.InputInt(label="number_of_gpus (disabled)",
                                    value=local_state.value['number_of_gpus'], disabled=True) 
                    solara.InputInt(label="save_interval",
                                    value=local_state.value['save_interval']) 
                    solara.InputInt(label="num_envs_per_worker",
                                    value=local_state.value['num_envs_per_worker']) 
                    solara.InputInt(label="num_cpus_per_worker",
                                    value=local_state.value['num_cpus_per_worker']) 
                    solara.Select(label='exploration_type', values=['EpsilonGreedy'],
                                  value=local_state.value['exploration_type'])
                    solara.InputFloat(label="initial_epsilon",
                                    value=local_state.value['initial_epsilon']) 
                    solara.InputFloat(label="final_epsilon",
                                    value=local_state.value['final_epsilon'])                     
                    solara.InputInt(label="epsilon_timesteps",
                                    value=local_state.value['epsilon_timesteps']) 
                    solara.InputFloat(label="learning_rate",
                                    value=local_state.value['learning_rate']) 
                    solara.InputFloat(label="entropy_coeff",
                                    value=local_state.value['entropy_coeff']) 
                    solara.InputInt(label="train_batch_size",
                                    value=local_state.value['train_batch_size']) 
                    solara.InputInt(label="sgd_minibatch_size",
                                    value=local_state.value['sgd_minibatch_size']) 
                    solara.InputInt(label="hidden_size_first",
                                    value=local_state.value['hidden_size_first']) 
                    solara.InputInt(label="hidden_size_second",
                                    value=local_state.value['hidden_size_second']) 


    solara.Button("Start Training", on_click=train_agent, disabled=True if train_agent.pending else False)
    if train_agent.pending:
        solara.Button("Cancel training", on_click=train_agent.cancel)
    solara.ProgressLinear(train_agent.pending)

    if train_agent.finished:
        message.set('Training finished')        
    elif train_agent.cancelled:
        message.set('training is cancelled')

    solara.Info(f'{message.value}')
    if  len(local_state.value['reward_mean'].value) > 0:
        y = local_state.value['reward_mean'].value
        x = [step for step in range(1, len(y)+1)]
        Plot1D(x, [y], 'Reward Mean', 'Episode', ['reward mean'])

    solara.Markdown(md_text=f'List of checkpoint dirs: {local_state.value["checkpoint_dirs"].value}')