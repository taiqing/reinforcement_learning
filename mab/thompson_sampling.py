# coding: utf-8

"""
Use Thompson sampling to solve the MAB problem
References:
    https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html
"""

import numpy as np
import matplotlib.pyplot as plt


from bandits import Bandits


if __name__ == '__main__':
    arm_theta = [0.1, 0.7, 0.8]
    n_arm = len(arm_theta)
    arm_alpha = np.ones(n_arm, np.float32)
    arm_beta = np.ones(n_arm, np.float32)
    n_trial = 1000000

    bandits = Bandits(arm_theta, 12345)
    action_history = []
    for t in xrange(n_trial):
        expected_rewards = [np.random.beta(arm_alpha[i], arm_beta[i]) for i in range(n_arm)]
        action = np.argmax(expected_rewards)
        action_history.append(action)
        reward = bandits.reward(action)
        arm_alpha[action] += reward
        arm_beta[action] += 1. - reward
    
    # visualization of action distributions over time
    action_distr = [[] for i in range(n_arm)]
    step_size = 100
    for t in xrange(0, n_trial - step_size + 1, step_size):
        actions = action_history[t: t + step_size]
        for arm in range(n_arm):
            action_distr[arm].append(np.mean(np.equal(actions, arm)))
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    for arm in range(n_arm):
        ax.plot(action_distr[arm], label='arm: {}'.format(arm))
    ax.legend()
    ax.grid()
    fig.show()
