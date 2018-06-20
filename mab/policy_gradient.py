# coding: utf-8

"""
Solve the MAB problem with a policy gradient method
"""

import tensorflow as tf
import numpy as np

from bandits import Bandits


if __name__ == '__main__':
    thetas = [0.2, 0.1, 0.3, 0.01]
    bandits = Bandits(thetas)
    lr_init = 1e-2
    total_episodes = 1000
    epsilon = 0.1

    num_bandits = bandits.n
    tf.reset_default_graph()
    weights = tf.get_variable(
        name='weights',
        shape=num_bandits,
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(mean=-1, stddev=0.2, dtype=tf.float32))
    chosen_action = tf.argmax(weights)
    reward_holder = tf.placeholder(dtype=tf.float32)
    action_holder = tf.placeholder(dtype=tf.int32)
    lr_holder = tf.placeholder(dtype=tf.float32)
    loss = -tf.log(tf.sigmoid(weights[action_holder])) * (2*reward_holder-1)
    update = tf.train.GradientDescentOptimizer(learning_rate=lr_holder).minimize(loss)
    init = tf.global_variables_initializer()

    total_reward = np.zeros(num_bandits)
    total_pulls = np.zeros(num_bandits)
    with tf.Session() as sess:
        sess.run(init)
        lr = lr_init
        for i in xrange(total_episodes):
            if np.random.rand() < epsilon:
                action = np.random.randint(num_bandits)
            else:
                action = sess.run(chosen_action)
            reward = bandits.reward(action)
            sess.run(update, feed_dict={reward_holder: reward, action_holder: action, lr_holder: lr})
            total_reward[action] += reward
            total_pulls[action] += 1
            if i % 50 == 0:
                print "running rewards: {}".format(total_reward)
                print "total pulls: {}".format(total_pulls)
                print '* weights: {}'.format(sess.run(weights))
            if i % 100 == 0:
                lr /= 2
        final_weights = sess.run(weights)
    print 'final weights: {}'.format(final_weights)
    print "The agent thinks bandit {} is the most promising".format(np.argmax(final_weights))
