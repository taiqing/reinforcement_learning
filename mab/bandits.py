# coding: utf-8

import numpy as np
import time


class Bandits(object):
    def __init__(self, thetas, rand_seed=None):
        self.thetas = thetas
        self.n = len(thetas)
        if rand_seed is not None:
            np.random.seed(rand_seed)

    def reward(self, a):
        if a < 0 or a > self.n - 1:
            raise Exception('invalid i: {}'.format(a))
        if np.random.rand() < self.thetas[a]:
            return 1
        else:
            return 0


if __name__ == '__main__':
    thetas = [0.1, 0.9, 0.5]
    bandits = Bandits(thetas, 12345)
    n_arm = len(thetas)
    n_trial_per_arm = 1000
    rewards = np.zeros(n_arm, np.float32)
    for a in range(n_arm):
        for i in range(n_trial_per_arm):
            rewards[a] += bandits.reward(a)
    rewards /= n_trial_per_arm
    print rewards
