"""
Data structure for implementing experience replay
Author: Patrick Emami
"""
import random
from collections import deque

import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque() # 0 index element is the first, -1 is the most recent one
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def refresh_rewards(self):
        if self.count == 0:
            return

        # Get the reward of the most recent experience
        last_reward = self.buffer[-1][2]  # Reward is the third element (index=2)

        # Check if the last reward is 100 or -100
        if last_reward == 100 or last_reward == -100:
            for i in range(2, self.count + 1):  # Start from i = 1
                # Calculate new reward based on last_reward and i
                rg = last_reward / i  # r(t-i)=r(t-i)+rg/i
                # Update reward in the buffer
                self.buffer[-i] = (
                    self.buffer[-i][0],
                    self.buffer[-i][1],
                    self.buffer[-i][2] + rg,
                    self.buffer[-i][3],
                    self.buffer[-i][4],
                )