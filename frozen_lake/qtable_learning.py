# coding: utf-8

"""
Solving Frozen Lake with Q-table learning
"""

import gym
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    lr = 0.8
    discount = 0.95
    n_episodes_learn = 2000
    n_episodes_eval = 1000
    depth = 100
    
    env.render()
    r_list = []
    for i in xrange(n_episodes_learn):
        s = env.reset()
        r_sum = 0
        for step in xrange(depth):
            a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n)*(1./(i+1)))
            s_new, r, done, _ = env.step(a)
            Q[s, a] = (1. - lr) * Q[s, a] + lr * (r + discount * np.max(Q[s_new, :]))
            r_sum += r
            s = s_new
            if done:
                break
        r_list.append(r_sum)
        if i % 100 == 0:
            print '{i}-th episode finished'.format(i=i)
    print "score over time: {:.4f}".format(sum(r_list) / n_episodes_learn)
    print "final Q-table values"
    print Q
    
    conv_bandwidth = 50
    smoothed_r_list = np.convolve(r_list, np.ones(conv_bandwidth, np.float32) / conv_bandwidth, mode='valid')
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    ax.plot(smoothed_r_list)
    ax.set_xlabel('episodes')
    ax.set_ylabel('smoothed reward summation')
    fig.show()
    
    # evalute the learnt Q-table
    r_list_eval = []
    for i in xrange(n_episodes_eval):
        s = env.reset()
        r_sum = 0
        for step in xrange(depth):
            a = np.argmax(Q[s, :])
            s_new, r, done, _ = env.step(a)
            s = s_new
            r_sum += r
            if done:
                break
        r_list_eval.append(r_sum)
    print "evaluation: score over time: {:.4f}".format(sum(r_list_eval) / n_episodes_eval)
