import torch
import torch.nn as nn

from torchmetrics import MeanMetric

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim : int, hidden_dim: int = 256):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            #nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

     



        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        self.loss_metric = MeanMetric()

    def forward(self, state, action):
        x = torch.concat([state, action], dim=-1)
        value = self.layers(x)

        return value
