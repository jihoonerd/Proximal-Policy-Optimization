import numpy as np


class Memory:

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

        self.states = []
        self.actions = []
        self.probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def store_memory(self, state, action, prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def generate_batch_index(self):
        n_states = len(self.states)
        # This is for sampling a part of trajectory. (t_0, t_1, ..., t_{k+1})
        start_idx = np.arange(0, n_states, self.batch_size)
        idxs = np.arange(n_states, dtype=np.int32)
        np.random.shuffle(idxs)  # To mitigate correlation
        batches = np.split(idxs, len(start_idx))
        return batches

    def get_memory(self):
        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.values), np.array(self.rewards), np.array(self.dones),
