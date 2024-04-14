import pandas as pd
from .utils import predict_dict

class Policy1():
    '''Choose input for which max reward is obtained. Pseudocode:
    
* Determine the input features of the models a.k.a. state variables
* Determine all possible state combinations
* For each state, run the model, obtain the output and calculate reward
* Return the state giving the maximum reward

#### Formula

$$ \max_{state} reward(model(state)) $$

    '''

    def __init__(self):
        self.label = """PolicyArgMax: finds the best state giving max reward """

    def choose(self, model, ds, inputs, reward_fn):
        input_df, output_df = predict_dict(model, ds, inputs)

        print('Policy choose')
        print(input_df)
        print(output_df)
        
        io_df = pd.concat([input_df, output_df],axis=1)

        io_df['reward'] = io_df.apply(lambda row: reward_fn.calculate(row), axis=1)
        print(io_df)
        print(io_df.columns)
        max_reward_index = io_df['reward'].argmax()
        best_state = io_df.loc[max_reward_index].to_dict()
        print('best state', best_state)
        return best_state
