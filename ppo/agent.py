import numpy as np
import torch as T
import torch.nn.functional as F

from ppo.memory import Memory
from ppo.network import ActorNetwork, CriticNetwork


class Agent:

    def __init__(self, n_actions, input_dims, n_epochs, gamma=0.99, learning_rate=0.0003, policy_clip=0.2, batch_size=32, N=1024, gae_lambda=0.95):

        self.n_epochs = n_epochs
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, learning_rate)
        self.critic = CriticNetwork(input_dims, learning_rate)
        self.memory = Memory(batch_size)

    def push(self, state, action, prob, value, reward, done):
        self.memory.store_memory(state, action, prob, value, reward, done)

    def choose_action(self, state):
        state = T.tensor([state], dtype=T.float).to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        prob = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, prob, value

    def learn(self):
        for _ in range(self.n_epochs):  # Algorithm 1: outer loop
            state_arr, action_arr, old_probs_arr, values_arr, reward_arr, dones_arr = self.memory.get_memory()
            batches_idx = self.memory.generate_batch_index()
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):  # Algorithm1: inner loop
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    # \hat{A} = \delta_t + (\gamma \lambda)\delta_{t+1} + \cdots (in paper (11), (12))
                    a_t += discount * (reward_arr[k] + self.gamma * values_arr[k+1] * (
                        1-int(dones_arr[k])) - values_arr[k])  # (12)
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            advantage = T.tensor(advantage, dtype=T.float).to(
                self.actor.device)
            values_arr = T.tensor(
                values_arr, dtype=T.float).to(self.actor.device)

            for idxs in batches_idx:
                states = T.tensor(state_arr[idxs], dtype=T.float).to(
                    self.actor.device)
                old_probs = T.tensor(old_probs_arr[idxs]).to(self.actor.device)
                actions = T.tensor(action_arr[idxs]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[idxs] * prob_ratio
                weighted_clipped_probs = T.clamp(
                    prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[idxs]

                actor_loss = -T.min(weighted_probs, weighted_clipped_probs)

                returns = advantage[idxs] + values_arr[idxs]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + \
                    F.smooth_l1_loss(
                        critic_value, (advantage[idxs] + values_arr[idxs]).detach())

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                total_loss.mean().backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        self.memory.clear_memory()
