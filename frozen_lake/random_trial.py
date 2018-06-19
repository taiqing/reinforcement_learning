# coding: utf-8

"""
Solving Frozen Lake with random trials
"""

import gym
import numpy as np


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    n_episodes_eval = 1000
    depth = 100

    env.render()
    r_list_eval = []
    for i in xrange(n_episodes_eval):
        s = env.reset()
        r_sum = 0
        for step in xrange(depth):
            a = env.action_space.sample()
            s_new, r, done, _ = env.step(a)
            s = s_new
            r_sum += r
            if done:
                break
        r_list_eval.append(r_sum)
    print "evaluation: score over time: {:.4f}".format(sum(r_list_eval) / n_episodes_eval)