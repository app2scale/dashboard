import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from functools import partial
import pandas as pd

from .data import ExplorationDataset
from .models import Perceptron
from .loss import loss_mape



def train(df, model_name, input_cols, output_cols, trn_ratio, 
          batch_size_trn, batch_size_val, optimizer_name, learning_rate,
          max_epoch, loss_name, seed):
    torch.manual_seed(seed)
    if model_name == "Perceptron":
        model = Perceptron(in_features=len(input_cols), out_features=len(output_cols))
    if loss_name == "mape":
        loss_fn = loss_mape
    ds = ExplorationDataset(df, input_cols=input_cols, output_cols=output_cols)

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
