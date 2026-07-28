import torch
import torch.nn as nn

from torchmetrics import MeanMetric

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim: int = 64):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            #nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        self.loss_metric = MeanMetric()

    def forward(self, state):
        
        value = self.layers(state)
     
        return value
