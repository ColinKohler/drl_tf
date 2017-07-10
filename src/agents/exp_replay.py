import numpy as np
import random

class ExpReplay(object):
    def __init__(self, state_shape, batch_size, exp_length, capacity=1e6):
        self.capacity = capacity
        self.actions = np.empty(self.capacity, dtype=np.uint8)
        self.rewards = np.empty(self.capacity, dtype=np.uint8)
        self.states = np.empty([self.capacity]+state_shape, dtype=np.float16)
        self.terminals = np.empty(self.capacity, dtype=np.bool)
        self.batch_size = batch_size
        self.exp_length = exp_length
        self.index = 0
        self.size = 0

        if self.exp_length != 1:
            self.batch_states = np.empty([self.batch_size, self.exp_length]+state_shape, dtype=np.float16)
            self.batch_states_ = np.empty([self.batch_size, self.exp_length]+state_shape, dtype=np.float16)
        else:
            self.batch_states = np.empty([self.batch_size]+state_shape, dtype=np.float16)
            self.batch_states_ = np.empty([self.batch_size]+state_shape, dtype=np.float16)

    # Store experience in memory
    def storeExperience(self, state, action, reward, terminal):
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.states[self.index, ...] = state
        self.terminals[self.index] = terminal

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    # Get batch from memory
    def getBatch(self):
        indexes = self._getBatchIndexes()
        for i, index in enumerate(indexes):
            self.batch_states[i, ...] = self._getState(index - 1)
            self.batch_states_[i, ...] = self._getState(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        return self.batch_states, actions, rewards, self.batch_states_, terminals

    # Get the experience state for a given index
    def _getState(self, index):
        return self.states[index-self.exp_length:index, ...]

    # Get valid indexes for a batch sample
    def _getBatchIndexes(self):
        indexes = list()
        while len(indexes) < self.batch_size:
            index = random.randint(self.exp_length+1, self.size-1)
            if self._isValidIndex(index, indexes):
                indexes.append(index)

        return indexes

    # Checks that a given index is valid
    def _isValidIndex(self, index, indexes):
        duplicate = index in indexes
        wrap_pointer = (index >= self.index and index - self.exp_length < self.index)
        bad_terminal = (self.terminals[(index - self.exp_length):index].any())

        return not duplicate and not wrap_pointer and not bad_terminal
