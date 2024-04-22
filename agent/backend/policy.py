import pandas as pd
from .utils import predict_dict

class PolicyArgMax():
    '''Choose input for which max reward is obtained. Pseudocode:
    
* Determine the input features of the models a.k.a. state variables
* Determine all possible state combinations
* For each state, run the model, obtain the output and calculate reward
* Return the state giving the maximum reward

#### Formula

$$ \max_{state} reward(model(state)) $$

    '''

    def __init__(self):
        self.label = """PolicyArgMax: pursue max reward """

    def choose(self, model, ds, input_ranges, cur_state, cur_metrics, reward_fn):
        print(input_ranges)

        input_df, output_df = predict_dict(model, ds, input_ranges)

        io_df = pd.concat([input_df, output_df],axis=1)

        io_df['reward'] = io_df.apply(lambda row: reward_fn.calculate(row), axis=1)
        max_reward_index = io_df['reward'].argmax()
        next_state = io_df.loc[max_reward_index].to_dict()
        print('next state', next_state)
        return next_state


class PolicyHPA():
    '''Increase/decrease replica if cpu usage is above/below threshold
    
**Remarks**

* reward functions are not used.
    '''

    def __init__(self, threshold = 0.4):
        self.label = f"""Kubernetes HPA (cpu threshold={threshold})"""
        self.threshold = threshold

    def choose(self, model, ds, input_ranges, cur_state, cur_metrics, reward_fn):
        next_state = cur_state.copy()

        if cur_metrics['cpu_usage'] > self.threshold:
            next_state['replica'] = min(cur_state['replica'] + 1, max(input_ranges['replica']))
        elif cur_metrics['cpu_usage'] < self.threshold:
            next_state['replica'] = max(cur_state['replica'] - 1, min(input_ranges['replica']))
        
        return next_state