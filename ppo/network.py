import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, learning_rate, fc1_dims=256, fc2_dims=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, learning_rate, fc1_dims=256, fc2_dims=256):
        super().__init__()

        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
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
