import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class ExplorationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, 
                input_cols,
                output_cols, transform_dict = None):
        super().__init__()
        self.df = df
        self.input_cols = input_cols
        self.output_cols = output_cols
        if transform_dict is None:
            self.transform_dict = self.transform_fit(df)
        else:
            self.transform_dict = transform_dict
        self.input_transformed = torch.tensor(self.input_transform(df[input_cols]).values).to(torch.float)
        self.output_transformed = torch.tensor(self.output_transform(df[output_cols]).values).to(torch.float)
                
    def __getitem__(self, idx: int) -> tuple[torch.tensor, torch.tensor]:
        inputs = self.input_transformed[idx]
        outputs = self.output_transformed[idx]
        return inputs, outputs
    
    def __len__(self) -> int:
        return len(self.df)
    
    def transform_fit(self, df):
        transform_dict = {}
        for f in self.input_cols + self.output_cols:
            if f in ['replica','cpu', 'heap']:
                shift = df[f].median()
                divide = 1
                logtransform = False  
            elif f in ['expected_tps']:
                shift = df[f].mean()
                divide = df[f].std()
                logtransform = False
            elif f in ['num_request']:
                shift = 0
                divide = df[f].max()
                logtransform = False
            elif f in ['response_time']:
                # shift/divide is only after log
                logtransform = True
                shift = 0
                divide = np.max(np.log10(df[f]))
            else:
                shift = 0
                divide = 1
                logtransform = False
                
            transform_dict[f] = {'shift': shift, 'divide': divide,'logtransform': logtransform}
        return transform_dict
        
    def transform(self, df, cols):
        df_transform = df.copy(deep=True)
        for f in df.columns:
            if f in cols:
                shift = self.transform_dict[f]['shift']
                divide = self.transform_dict[f]['divide']
                logtransform = self.transform_dict[f]['logtransform']
                if logtransform:
                    df_transform[f] = np.log10(df_transform[f])
                df_transform[f] =  (df_transform[f] - shift) / divide
        return df_transform
    
    def inv_transform(self, df, cols):
        df_transform = df.copy(deep=True)
        for f in df.columns:
            if f in cols:
                shift = self.transform_dict[f]['shift']
                divide = self.transform_dict[f]['divide']
                logtransform = self.transform_dict[f]['logtransform']
                df_transform[f] =  divide * df_transform[f] + shift
                if logtransform:
                    df_transform[f] = np.power(df_transform[f], 10)
        return df_transform
    
    def input_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df, self.input_cols)
    
    def output_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df, self.output_cols)
    
    def input_inv_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.inv_transform(df, self.input_cols)

    def output_inv_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.inv_transform(df, self.output_cols)
    

def data_replica_vs_cpu_usage():
    '''Replica versus cpu_usage'''
    df = pd.read_csv('averaged_full_state_data.csv')
    df.query('cpu == 5 and expected_tps == 88')
    input_cols = ['replica']
    output_cols = ['cpu_usage']
    other_cols = []
    
    
    df = df[input_cols + output_cols + other_cols]
    return df, input_cols, output_cols, other_cols