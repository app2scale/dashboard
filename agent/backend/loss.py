import torch

def loss_mape(output, target):
    return torch.mean(torch.abs((target - output) / target))
