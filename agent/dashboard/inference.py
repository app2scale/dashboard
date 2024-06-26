import solara
import pandas as pd
import time
import random
from ..backend.reward import RewardHighCPUUsage, RewardLowCPUUsage
from ..backend.policy import PolicyArgMax, PolicyHPA, PolicyDoNothing
from ..backend.load import ConstantLoad, SinusLoad, PaymentGateway113Load
from .training import local_state as training_state
from .data import state as data_state
from ..backend.utils import estimate_metrics, read_metrics, Gauge, Plot1D

local_state = solara.reactive(
    {
        'render_count': solara.reactive(0),
        'inference_plot_data': solara.reactive({}),

    }
    )

reward_objects = [RewardHighCPUUsage(), RewardLowCPUUsage()]
reward_labels = [r.label for r in reward_objects]
selected_reward_label = solara.reactive(reward_labels[0])

policy_objects = [PolicyDoNothing(), PolicyArgMax(), PolicyHPA(0.2), PolicyHPA(0.4), PolicyHPA(0.6), PolicyHPA(0.8)]
policy_labels = [p.label for p in policy_objects]
selected_policy_label = solara.reactive(policy_labels[0])

load_objects = [PaymentGateway113Load(), ConstantLoad(24), ConstantLoad(72), ConstantLoad(168), SinusLoad(125, 100)]
load_labels = [p.label for p in load_objects]
selected_load_label = solara.reactive(load_labels[0])

nsteps = solara.reactive(10)
initial_replica = solara.reactive(1)
initial_cpu = solara.reactive(4)
cpu_cost_per_hour = solara.reactive(0.031611)

animation_speed = solara.reactive(0.5)

use_model_to_estimate_metrics = solara.reactive(False)

inference_history = solara.reactive({})

def force_render():
    local_state.value['render_count'].set(1 + local_state.value['render_count'].value)




@solara.component
def InferencePlots(render_count):
    #print(local_state.value['render_count'].value)
    unavailable_states_in_data, set_unavailable_states_in_data = solara.use_state_or_update(0)
    total_cost, set_total_cost = solara.use_state_or_update(0)
    avg_tps_performance, set_avg_tps_performance = solara.use_state_or_update(0)

    def execute():
        #print(selected_policy_label, selected_reward_label, selected_load_label)

        model = training_state.value['model'].value
        input_cols = training_state.value['input_cols'].value
        output_cols = training_state.value['output_cols'].value
        chosen_load_index = load_labels.index(selected_load_label.value)
        chosen_load = load_objects[chosen_load_index]
        chosen_policy_index = policy_labels.index(selected_policy_label.value)
        chosen_policy = policy_objects[chosen_policy_index]
        chosen_reward_index = reward_labels.index(selected_reward_label.value)
        chosen_reward_fn = reward_objects[chosen_reward_index]
        ds = training_state.value['ds'].value


        df = ds.df
        #print(df.columns)
        # get all possible values for inputs
        input_ranges = {col: list(pd.unique(df[col])) for col in input_cols}
        
        # Step through load profile
        load_profile = chosen_load
        step = 0
        cur_hist = {}
        replica = initial_replica.value
        cpu = initial_cpu.value
        state_not_found_in_data = 0
        cum_cost = 0.0
        _avg_tps_performance = 0
        for load, eod in load_profile:
            if step == 0:
                prev_load = load
            if step > nsteps.value:
                break
            # the model uses load as an input, supply with it
            if 'expected_tps' in input_ranges.keys():
                input_ranges['expected_tps'] = [load]

            cur_state = {"replica": replica, "cpu": cpu, "expected_tps": load, "previous_tps": prev_load}
            cum_cost += cpu_cost_per_hour.value * 24 * replica * cpu / 10
            set_total_cost(cum_cost)
            if use_model_to_estimate_metrics.value:
                cur_metrics = estimate_metrics(model, ds, cur_state)
                #print(cur_metrics)
            else:
                cur_metrics = read_metrics(df, cur_state)
                if cur_metrics is None:
                    #print('there is no data for this state',cur_state)
                    state_not_found_in_data += 1
                    cur_metrics = estimate_metrics(model, ds, cur_state)

            combined_data = cur_state | cur_metrics 
            for state, value in combined_data.items():
                if state in cur_hist.keys():
                    cur_hist[state]['y'].append(value)
                    cur_hist[state]['x'].append(step)
                    cur_hist[state]['title'] = state
                    cur_hist[state]['xlabel'] = 'step'
                    cur_hist[state]['ylabel'] = state
                else:
                    cur_hist[state] = {}
                    cur_hist[state]['y'] = [value]
                    cur_hist[state]['x'] = [step]
                    cur_hist[state]['title'] = state
                    cur_hist[state]['xlabel'] = 'step'
                    cur_hist[state]['ylabel'] = state
            local_state.value['inference_plot_data'].set(cur_hist)

            # update tps performance
            if set(["expected_tps","num_request"]).issubset(set(combined_data.keys())):
                cur_tps_performance  = min(100, 100*combined_data['num_request']/combined_data["expected_tps"])
                _avg_tps_performance = ((1/(step+1)) * (step * _avg_tps_performance + cur_tps_performance))
                set_avg_tps_performance(_avg_tps_performance)
            force_render()
            time.sleep(animation_speed.value)

            next_state = chosen_policy.choose(model, ds, input_ranges, cur_state, cur_metrics, chosen_reward_fn)
            if 'replica' in next_state.keys():
                replica = next_state['replica']
            if 'cpu' in next_state.keys():
                cpu = next_state['cpu']
            
            step += 1
            set_unavailable_states_in_data(state_not_found_in_data)
            prev_load = load


#    with solara.GridFixed(columns=3):


    model = training_state.value['model'].value
    if model is None:
        solara.Warning("There is no trained model ye, please train one!")
    
    solara.Button(label="Execute", on_click=execute, disabled=model is None)
    if unavailable_states_in_data > 0:
        solara.Warning(f'There are {unavailable_states_in_data} unavailable states in data. Estimatated versions are used!')

    #print('Interence plots')
    if len(local_state.value['inference_plot_data'].value.items()) > 0:
        with solara.Row():
            solara.Text(f'Total cost is ${total_cost:.4f}')
            solara.Text(f'Avg. TPS Performance (%) is {avg_tps_performance:.4f}')
        
    plot_data = local_state.value['inference_plot_data'].value




    with solara.ColumnsResponsive(4, style="overflow: hidden;"):
        if "expected_tps" in plot_data.keys():
            Gauge(plot_data['expected_tps']['y'][-1], 250, 'incoming load')
            if "num_request" in plot_data.keys():
                performance = min(100, 100*plot_data['num_request']['y'][-1]/ plot_data['expected_tps']['y'][-1])
                Gauge(performance, 100, 'performance', style={"style": "width: 300px;"})

        for col, content in local_state.value['inference_plot_data'].value.items():
            if col == 'expected_tps' or col == 'num_request':
                continue
            if col == 'cpu_usage' or col == 'cpu':
                continue
            Plot1D(content['x'], [content['y']], content['title'], content['xlabel'],[content['ylabel']])
          
        # Let's plot num_request and expected_tps on the same plot
        if set(['num_request','expected_tps']).issubset(set(plot_data.keys())):
            x = plot_data['num_request']['x']
            xlabel = plot_data['num_request']['xlabel']
            y = [plot_data['expected_tps']['y'], plot_data['num_request']['y']]
            ylabel = ['expected tps', 'num request']
            title = 'num_request & expected_tps'
            Plot1D(x, y, title, xlabel, ylabel)

        # Let's plot cpu limit and cpu_usage on the same plot
        if set(['cpu_usage','cpu']).issubset(set(plot_data.keys())):
            x = plot_data['cpu']['x']
            xlabel = plot_data['cpu']['xlabel']
            y = [[_/10 for _ in plot_data['cpu']['y']], plot_data['cpu_usage']['y']]
            ylabel = ['cpu limit', 'cpu usage']
            title = 'cpu limit and cpu usaage'
            Plot1D(x, y, title, xlabel, ylabel)



@solara.component
def LoadProfilePlot(load_profile):
    x, y = [], []
    step = 0
    for load, eod in load_profile:
        if step > 150 or eod :
            break
        x.append(step)
        y.append(load)
        step += 1

    Plot1D(x,[y])


@solara.component
def Page():
    solara.Title("Inference")
    with solara.Sidebar():
        with solara.lab.Tabs():
            with solara.lab.Tab("REWARD"):
                with solara.Card(title="Reward Selection", subtitle="Choose an appropriate reward from the list."):
                    solara.Select(label="choose reward", value=selected_reward_label.value, values=reward_labels,
                                on_value=selected_reward_label.set)
                    chosen_reward_index = reward_labels.index(selected_reward_label.value)
                    chosen_reward = reward_objects[chosen_reward_index]
                    solara.Markdown(md_text=chosen_reward.__doc__)

            with solara.lab.Tab("POLICY"):
                with solara.Card(title="Policy Selection", subtitle="Choose an appropriate policy from the list."):
                    solara.Select(label="choose policy", value=selected_policy_label.value, values=policy_labels,
                                on_value=selected_policy_label.set)
                    chosen_policy_index = policy_labels.index(selected_policy_label.value)
                    chosen_policy = policy_objects[chosen_policy_index]
                    solara.Markdown(md_text=chosen_policy.__doc__)

            with solara.lab.Tab("LOAD"):
                with solara.Card(title="Load Profile Selection", subtitle="Choose an appropriate load profile from the list."):
                    solara.Select(label="choose load profile", value=selected_load_label.value, values=load_labels,
                                on_value=selected_load_label.set)
                    chosen_load_index = load_labels.index(selected_load_label.value)
                    chosen_load = load_objects[chosen_load_index]
                    solara.Markdown(md_text=chosen_load.__doc__)
                    LoadProfilePlot(chosen_load)
            with solara.lab.Tab("EXECUTE"):
                solara.InputInt(label='Number of steps', value=nsteps.value, on_value=nsteps.set)
                solara.InputInt(label="Initial replica", value=initial_replica)
                solara.InputInt(label="Initial CPU", value=initial_cpu)
                solara.InputFloat(label="CPU cost per hour ($)", value=cpu_cost_per_hour)
                solara.SliderFloat(label="Seconds per step (animation speed)", value=animation_speed, min=0, max=1, step=0.1)
                if set(training_state.value['input_cols'].value) == set(['replica','cpu','expected_tps','previous_tps']):
                    solara.Checkbox(label='Use twin model to estimate metrics', value=use_model_to_estimate_metrics)
                else:
                    with solara.Column():
                        solara.Checkbox(label='Use twin model to estimate metrics', value=use_model_to_estimate_metrics, disabled=True)
                        solara.Info('twin model is not suitable for metric estimation')

    InferencePlots(local_state.value['render_count'].value)