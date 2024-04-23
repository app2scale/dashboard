import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from functools import partial
import pandas as pd
from typing import Dict, Union, List
import itertools
import solara

from .data import ExplorationDataset
from .models import Perceptron, NetSingleHiddenLayer
from .loss import loss_mape

def predict_dict(model, ds, inputs: Dict[str, Union[List[int], List[float]]]):
    combinations = itertools.product(*inputs.values())
    input_cols = list(inputs.keys())

    input_for_df = {col: [] for col in input_cols}
    for combination in combinations:
        for input_col, value in zip(input_cols, combination):
            input_for_df[input_col].append(value)

    input_df = pd.DataFrame(input_for_df)
    input_df_transformed = ds.input_transform(input_df)
    #print(input_df)
    #print(input_df_transformed)
    ds_new = ExplorationDataset(input_df, input_cols=ds.input_cols, 
                output_cols=[], transform_dict=ds.transform_dict)
    dl_new = DataLoader(ds_new, batch_size=len(input_df), shuffle=False)
    for x, y in dl_new:
        output = model.forward(x)
    output_for_df = {col_name: output[:,col_index].detach().numpy() for col_index, col_name in enumerate(ds.output_cols)}
    output_df = pd.DataFrame(output_for_df)
    output_df_transformed = ds.output_inv_transform(output_df)
    #print(output_df)
    #print(output_df_transformed)
    return input_df, output_df_transformed

def estimate_metrics(model, ds, cur_state):
    input_ranges = {key: [value] for key, value in cur_state.items()}
    input_df, output_df = predict_dict(model, ds, input_ranges)
    est_metrics = {metric: output_df.loc[0,metric] for metric in output_df.columns}
    return est_metrics
                
def read_metrics(df, cur_state):
    cols = list(df.columns)

    dff = df[cols]
    for col, value in cur_state.items():
        #print(f'{col} = {value}', dff.columns)
        dff = dff.query(f'{col} == {value}')
    if len(dff) == 0:
        return None
    output_df = dff.sample(1)
    for feature in ['replica','cpu','expected_tps']:
        if feature in cols:
            cols.remove(feature)
    output_df = output_df[cols].reset_index(drop=True)
    metrics = {metric: output_df.loc[0,metric] for metric in output_df.columns}
    return metrics

def train(ds: ExplorationDataset, model_name, trn_ratio, 
          batch_size_trn, batch_size_val, optimizer_name, learning_rate,
          max_epoch, loss_name, seed):
    torch.manual_seed(seed)
    input_cols, output_cols = ds.input_cols, ds.output_cols
    df = ds.df
    if model_name == "Perceptron":
        model = Perceptron(in_features=len(input_cols), out_features=len(output_cols))
    elif model_name == "NetSingleHiddenLayer":
        # TODO: make hidden_size adjustable in ui
        model = NetSingleHiddenLayer(in_features=len(input_cols), out_features=len(output_cols), hidden_size=10)

    if loss_name == "mape":
        loss_fn = loss_mape

    trn_size = int(len(ds)*trn_ratio)
    val_size = len(ds) - trn_size
    generator = torch.Generator().manual_seed(seed)
    ds_trn, ds_val = torch.utils.data.random_split(ds, [trn_size, val_size], generator=generator)
    dl_trn = DataLoader(ds_trn, batch_size=batch_size_trn, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size_val, shuffle=True)

    if optimizer_name == "Adam":
        optimizer_fn = partial(torch.optim.Adam,lr=learning_rate)
    print('backend training ...')
    print('training in progress...', len(df))
    print('data columns', list(df.columns))
    print('input columns', input_cols)
    print('output columns', output_cols)
    print('training ratio', trn_ratio)
    print('batch size trainig', batch_size_trn)
    print('batch size validation', batch_size_val)
    print(f'Number of samples {len(ds)}')
    print(f'Number of samples in training {len(ds_trn)}')
    print(f'Number of samples in validation {len(ds_val)}')
    print(f'Learning rate: {learning_rate}')
    print(f'Optimizer {optimizer_name}')
    print(f'Max epoch: {max_epoch}')
    print(f'random seed',seed)

    x, y = ds[0]
    in_features = x.shape[0]
    out_features = y.shape[0]


    optimizer = optimizer_fn(model.parameters())

    #epochbar = tqdm(range(max_epoch))
    for ep in range(max_epoch):
        model.train()
        for x, y in dl_trn:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

        trn_loss = evaluate(model, dl_trn, loss_fn)
        val_loss = evaluate(model, dl_val, loss_fn)
        #epochbar.set_postfix(epoch=ep+1,loss=loss.item(),val_loss=val_loss)
        yield ep, trn_loss, val_loss, model
        
    return ep, trn_loss, val_loss, model

def predict(model, df, input_cols, output_cols, trn_ratio, 
            batch_size_trn, batch_size_val, seed):
    torch.manual_seed(seed)
    ds = ExplorationDataset(df, input_cols=input_cols, output_cols=output_cols)
    trn_size = int(len(ds)*trn_ratio)
    val_size = len(ds) - trn_size
    generator = torch.Generator().manual_seed(seed)
    ds_trn, ds_val = torch.utils.data.random_split(ds, [trn_size, val_size], generator=generator)
    dl_trn = DataLoader(ds_trn, batch_size=batch_size_trn, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size_val, shuffle=True)

    trn_pred, trn_target = predict_dataloader(model, dl_trn)
    val_pred, val_target = predict_dataloader(model, dl_val)

    results = {}
    for col, col_name in enumerate(output_cols):
        trn_df = pd.DataFrame(torch.cat([trn_pred[:,[col]], trn_target[:,[col]]],dim=1))
        trn_df = trn_df.rename(columns={0:'prediction',1:'target'})
        val_df = pd.DataFrame(torch.cat([val_pred[:,[col]], val_target[:,[col]]],dim=1))
        val_df = val_df.rename(columns={0:'prediction',1:'target'})
        results[col_name] = {'training': trn_df, 'validation': val_df}
    return results


    
def predict_dataloader(model, dataloader):
    with torch.no_grad():
        predictions = torch.empty(0, model.out_features)
        targets = torch.empty(predictions.shape)
        for x, y in dataloader:
            y_pred = model.forward(x)
            predictions = torch.cat([predictions, y_pred], dim=0)
            targets = torch.cat([targets, y], dim=0)
        return predictions, targets
        

def evaluate(model, dataloader, loss_fn):
    with torch.no_grad():
        avg_loss = 0
        for x, y in dataloader:
            y_pred = model.forward(x)
            loss = loss_fn(y_pred, y)
            avg_loss += loss.item()
        avg_loss = avg_loss / len(dataloader) 
        return avg_loss
    

def update_policy(model, rewards, log_probabilities, gamma, learning_rate, optimizer):
    discounted_rewards = []

    for t in range(len(rewards)):
        gt = 0
        pw = 0
        for r in rewards[t:]:
            gt = gt + gamma ** pw * r
            pw = pw + 1
        discounted_rewards.append(gt)

    discounted_rewards = torch.tensor(discounted_rewards)
    # normalize discounted rewards
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std(0) + 1e-9)

    policy_gradient = []
    for log_probability, gt in zip(log_probabilities, discounted_rewards):
        policy_gradient.append(-log_probability * gt)
        # policy_gradient.append(1.0 / log_probability * gt)

    model.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    # policy_gradient.backward()
    policy_gradient.backward(retain_graph=True)
    optimizer.step()


@solara.component
def Plot1D(x: List, y: List, title='title',xlabel='xlabel', ylabel='ylabel', force_render=0):
    options = {
        'title': {
            'text': title,
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
            'data': x,
            'name': xlabel,
            'nameLocation': 'middle',
            'nameTextStyle': {'verticalAlign': 'top','padding': [10, 0, 0, 0]}
        },
        'yAxis': [
            {
                'type': 'value',
                'name': ylabel,
                'position': 'left',
                'alignTicks': True,
                'axisLine': {
                    'show': True,
                    'lineStyle': {'color': 'green'}}
            },
        ],
        'series': [
            {
            'name': ylabel,
            'data': y,
            'type': 'line',
            'yAxisIndex': 1
            },
        ],
    }
    solara.FigureEcharts(option=options)
