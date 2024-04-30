import numpy as np
import random
from collections import deque
from registry_models import Registry

class ReplayMemory:

    def __init__(self, size):
        self.buffer = deque(maxlen= size)
        self.maxSize = size
        self.len = 0

    def sample(self, count):
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        curr_state  = np.float32([arr[0] for arr in batch])
        curr_idx = np.float32([arr[1] for arr in batch])

        curr_action  = np.float32([arr[2] for arr in batch])

        set_actions = np.float32([arr[3] for arr in batch])

        reward  = np.float32([arr[4].numpy() for arr in batch])
        next_state  = np.float32([arr[5] for arr in batch])
        curr_performance  = np.float32([arr[6].numpy() for arr in batch])
        # curr_accuracy  = np.float32([arr[8].numpy() for arr in batch])
        curr_accuracy  = []

        prev_layers_arr = np.array([arr[8] for arr in batch])
        # prev_arch_arr = np.array([arr[9] for arr in batch])
        prev_arch_arr = []
        # done_arr  = np.array([arr[10] for arr in batch])
        done_arr  = np.array([arr[9] for arr in batch])

        curr_state = curr_state.reshape(-1, 1*curr_state.shape[2], curr_state.shape[3])
        next_state = next_state.reshape(-1, 1*next_state.shape[2], next_state.shape[3])


        return curr_state, curr_idx, curr_action, set_actions, reward, next_state, curr_performance, curr_accuracy, prev_layers_arr, prev_arch_arr, done_arr

    def len(self):
        return self.len
    
    # def find_arch(self, new_arch):
    #     return self.registry.find_arch(new_arch)
    
    def add(self, transitions):

        if self.len > self.maxSize:
            self.len = self.maxSize

        # print('Transitions before', self.len)

        for transition in transitions:
            for single_transition in transition:
                self.len += 1
                self.buffer.append(single_transition)

