import torch
import torch.nn as nn

from torchmetrics import MeanMetric
from torch.distributions.normal import Normal


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_scale):
        super().__init__()

        self.action_scale = action_scale
        self.reparam_noise = 1e-6
        
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


        return mu, sigma


    def get_action(self, x, reparameterize=True):

        mu, sigma = self(x)
    
        probs = Normal(mu, sigma)

        if reparameterize:
            actions = probs.rsample()
        else:
            actions = probs.sample()


        y = torch.tanh(actions)
        action = y * self.action_scale
        action = (action + 1) / 2

        log_probs = probs.log_prob(actions)
        log_probs -= torch.log(1 - y.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1, keepdim=True)

        return action, log_probs
    