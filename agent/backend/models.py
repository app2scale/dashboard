import torch.nn as nn
import torch

class Perceptron(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer = nn.Sequential(nn.Linear(in_features=self.in_features,out_features=self.out_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer(x)
        return out


class NetSingleHiddenLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.layer1 = nn.Sequential(nn.Linear(in_features=self.in_features,out_features=self.hidden_size))
        self.layer2 = nn.Sequential(nn.Linear(in_features=self.hidden_size,out_features=self.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.relu(self.layer1(x))
        out = self.layer2(out)
        return out


class PolicyGradientNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(PolicyGradientNetwork, self).__init__()

        self.num_actions = num_actions
        self.norm0 = nn.BatchNorm1d(num_features=num_inputs)
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear_act = nn.Linear(hidden_size, int(num_actions))

    # a neural network two hidden layers with the same size
    def forward(self, state):
        x = self.norm0(state)
        x = torch.nn.functional.gelu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x_act = self.linear_act(x)
        x_act = torch.nn.functional.softmax(x_act, dim=1)

        # return the probability list of the actions
        return x_act