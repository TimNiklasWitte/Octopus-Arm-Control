import torch
import torch.nn as nn

from torchmetrics import MeanMetric

import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        self.loss_metric = MeanMetric()

    
    def forward(self, x):

        out = self.layers(x)

        action = torch.sigmoid(out)

        return action
    

  
