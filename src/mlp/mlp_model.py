import torch
import torch._dynamo
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: List[int], output_dim: int = 1, dropout: float = 0.5):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for n_dim in hidden_dim:
            layers.append(nn.Linear(prev_dim, n_dim))
            layers.append(nn.BatchNorm1d(n_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = n_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        # layers.append(nn.Sigmoid())


        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

