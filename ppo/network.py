import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, learning_rate, fc1_dims=256, fc2_dims=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.std = T.zeros(n_actions, device=self.device)

    def forward(self, state):
        mean = self.actor(state)
        std = F.softplus(self.std)
        dist = Normal(mean, std)
        return dist


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, learning_rate, fc1_dims=256, fc2_dims=256):
        super().__init__()

        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def forward(self, state):
        value = self.critic(state)
        return value
