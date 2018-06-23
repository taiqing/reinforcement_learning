# coding: utf-8

import os
from collections import deque, namedtuple
import numpy as np
import tensorflow as tf
import gym
from gym.spaces import Box, Discrete
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt


Transition = namedtuple('Transition', ['s', 'a', 'r', 's_next', 'done'])


def makedirs(name, exist_ok=True):
    if exist_ok:
        if os.path.exists(name) is False:
            os.makedirs(name)
    else:
        os.makedirs(name)


class ReplayMemory:
    def __init__(self, capacity=100000, replace=False, tuple_class=Transition):
        """
        :param capacity: #records in the queue
        :param replace:
        :param tuple_class:
        """
        self.buffer = []
        self.capacity = capacity
        self.replace = replace
        self.tuple_class = tuple_class
        self.fields = tuple_class._fields

    def add(self, record):
        """Any named tuple item."""
        if isinstance(record, self.tuple_class):
            self.buffer.append(record)
        elif isinstance(record, list):
            self.buffer += record

        while self.capacity and self.size > self.capacity:
            self.buffer.pop(0)

    def _reformat(self, indices):
        # Reformat a list of Transition tuples for training.
        # indices: list<int>
        return {
            field_name: np.array([getattr(self.buffer[i], field_name) for i in indices])
            for field_name in self.fields
        }

    def sample(self, batch_size):
        """
        :param batch_size:
        :return: the value of each field is a np.array object
        """
        assert len(self.buffer) >= batch_size
        idxs = np.random.choice(range(len(self.buffer)), size=batch_size, replace=self.replace)
        return self._reformat(idxs)

    def pop(self, batch_size):
        # Pop the first `batch_size` Transition items out.
        i = min(self.size, batch_size)
        batch = self._reformat(range(i))
        self.buffer = self.buffer[i:]
        return batch

    @property
    def size(self):
        return len(self.buffer)


def plot_learning_curve(filepath, value_dict, xlabel='step'):
    fig = plt.figure(figsize=(12, 4 * len(value_dict)))
    for i, (key, values) in enumerate(value_dict.items()):
        ax = fig.add_subplot(len(value_dict), 1, i + 1)
        ax.plot(range(len(values)), values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(key)
        ax.grid('k--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filepath)


class DiscretizedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_bins=10, low=None, high=None):
        super(DiscretizedObservationWrapper, self).__init__(env)
        assert isinstance(env.observation_space, Box)

        low = self.observation_space.low if low is None else low
        high = self.observation_space.high if high is None else high

        low = np.array(low)
        high = np.array(high)

        self.n_bins = n_bins
        self.val_bins = [np.linspace(l, h, n_bins + 1) for l, h in
                         zip(low.flatten(), high.flatten())]
        self.ob_shape = self.observation_space.shape
        # the actual #bins per dimension is self.n_bins + 2, counting the left and right outliers
        self.observation_space = Discrete((n_bins + 2) ** len(low))

    def _convert_to_one_number(self, digits):
        return sum([d * ((self.n_bins + 2) ** i) for i, d in enumerate(digits)])

    def observation(self, observation):
        digits = [np.digitize([x], bins)[0]
                  for x, bins in zip(observation.flatten(), self.val_bins)]
        return self._convert_to_one_number(digits)