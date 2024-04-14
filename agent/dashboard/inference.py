import solara
import pandas as pd
from ..backend.reward import Reward1, Reward2
from ..backend.policy import Policy1
from ..backend.load import ConstantLoad
from .training import local_state as training_state
from .data import state as data_state
from ..backend.utils import predict_dict, Plot1D

local_state = solara.reactive(
    {
        'render_count': solara.reactive(0),
        'inference_plot_data': solara.reactive({}),

    }
    )

reward_objects = [Reward1(), Reward2()]
reward_labels = [r.label for r in reward_objects]
selected_reward_label = solara.reactive(reward_labels[0])

policy_objects = [Policy1()]
policy_labels = [p.label for p in policy_objects]
selected_policy_label = solara.reactive(policy_labels[0])

load_objects = [ConstantLoad(24), ConstantLoad(72), ConstantLoad(168)]
load_labels = [p.label for p in load_objects]
selected_load_label = solara.reactive(load_labels[0])

nsteps = solara.reactive(10)

inference_history = solara.reactive({})

def force_render():
    local_state.value['render_count'].set(1 + local_state.value['render_count'].value)




@solara.component
def InferencePlots(render_count):



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
        chosen_reward = reward_objects[chosen_reward_index]
        ds = training_state.value['ds'].value


        df = ds.df
        # get all possible values for inputs
        input = {col: list(pd.unique(df[col])) for col in input_cols}
        
        # Step through load profile
        load_profile = chosen_load
        step = 0
        cur_hist = {}
        for load in load_profile:
            if step > nsteps.value:
                break
            # the model uses load as an input, supply with it
            if 'expected_tps' in input.keys():
                input['expected_tps'] = [load]

        
            best_state = chosen_policy.choose(model, ds, input, chosen_reward)
            for state, value in best_state.items():
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
            #print(cur_hist)

            local_state.value['inference_plot_data'].set(cur_hist)
            force_render()

            step += 1
        #print(local_state.value['inference_plot_data'].value)


    solara.InputInt(label='Number of steps', value=nsteps.value, on_value=nsteps.set)
    model = training_state.value['model'].value
    if model is None:
        solara.Warning("Model is not ready yet!")
    
    solara.Button(label="Execute", on_click=execute, disabled=model is None)


    #print('Interence plots')
    for col, content in local_state.value['inference_plot_data'].value.items():
        options = {
            'title': {
                'text': content['title'],
                'left': 'center'},
            'tooltip': {
                'trigger': 'axis',
                'axisPointer': {
                    'type': 'cross'
                }
            },
            'xAxis': {
                'axisTick': {
                    'alignWithLabel': True
                },
                'data': content['x'],
                'name': content['xlabel'],
                'nameLocation': 'middle',
                'nameTextStyle': {'verticalAlign': 'top','padding': [10, 0, 0, 0]}
            },
            'yAxis': [
                {
                    'type': 'value',
                    'name': content['ylabel'],
                    'position': 'left',
                    'alignTicks': True,
                    'axisLine': {
                        'show': True,
                        'lineStyle': {'color': 'green'}}
                },
            ],
            'series': [
                {
                'name': content['ylabel'],
                'data': content['y'],
                'type': 'line',
                'yAxisIndex': 0
                },
            ],
        }
        solara.FigureEcharts(option=options)
        

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
    InferencePlots(local_state.value['render_count'].value)