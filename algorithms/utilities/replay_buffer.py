import random

import gym
import numpy as np


class ReplayBuffer:
    def __init__(self, x_shape, max_entries):
        self._max_entries = max_entries

        self._observations = np.full((max_entries, *x_shape), np.nan)
        self._actions = np.full(max_entries, np.nan)
        self._rewards = np.full(max_entries, np.nan)

        self._n_items = 0

    def store(self, observations, actions, rewards):
        n = observations.shape[0]

        self._observations = np.roll(self._observations, n, axis=0)
        self._actions = np.roll(self._actions, n, axis=0)
        self._rewards = np.roll(self._rewards, n, axis=0)

        self._observations[:n] = observations
        self._actions[:n] = actions
        self._rewards[:n] = rewards

        self._n_items = min(self._n_items + n, self._max_entries)

    def retrieve(self):
        return {
            "observations": self._observations[:self._n_items],
            "actions": self._actions[:self._n_items],
            "rewards": self._rewards[:self._n_items]
        }

    @classmethod
    def from_env(cls, env: gym.Env, max_entries):
        x_shape = env.observation_space.shape
        # y_shape = [env.action_space.n]

        return cls(x_shape, max_entries)


class SplitReplayBuffer:
    """
    A buffer split into sub-buffers according to the ratios.
    Storing will store into a random sub-buffer
    """
    def __init__(self, x_shape, max_entries, split_ratios=(0.8, 0.2)):
        self.buffers = [ReplayBuffer(x_shape, int(max_entries * ratio)) for ratio in split_ratios]
        self._split_ratios = split_ratios

    def store(self, observations, actions, rewards):
        buffer = random.choices(self.buffers, weights=self._split_ratios)
        buffer.store(observations, actions, rewards)

