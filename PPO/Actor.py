import torch
import torch.nn as nn

from torchmetrics import MeanMetric
from torch.distributions.normal import Normal

import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            #nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.layer_mu = nn.Linear(hidden_dim, action_dim)
        self.layer_log_sigma = nn.Linear(hidden_dim, action_dim)


        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        self.loss_metric = MeanMetric()

    
    def forward(self, x):

        hidden = self.layers(x)


        mu = self.layer_mu(hidden)
        log_sigma = self.layer_log_sigma(hidden)
        log_sigma = torch.clamp(log_sigma, -20, 2)

        sigma = log_sigma.exp()

        probs = Normal(mu, sigma)

        actions = probs.rsample()

        log_probs = probs.log_prob(actions).sum(1)

        return actions, log_probs
    
    def get_logprobs(self, x, actions):

        hidden = self.layers(x)

        mu = self.layer_mu(hidden)
        log_sigma = self.layer_log_sigma(hidden)
        log_sigma = torch.clamp(log_sigma, -20, 2)
        sigma = log_sigma.exp()

        probs = Normal(mu, sigma)

        log_probs = probs.log_prob(actions).sum(1)

        return log_probs, probs.entropy()
    

  
