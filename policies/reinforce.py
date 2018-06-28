# coding: utf-8

"""
Solve the cart pole task with monte-carlo policy gradient
Ref:
https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html
"""


import os
import time
import numpy as np
import tensorflow as tf
import gym
import pandas as pd
import shutil

from utils import plot_learning_curve, makedirs
from nets import dense_nn
from base_tf_model import BaseTFModel


class ReinforcePolicy(BaseTFModel):
    def __init__(self, env, name,
                 model_path='./',
                 training=True,
                 gamma=0.99,
                 lr=0.001,
                 lr_decay=0.998,
                 layer_sizes=[32, 32],
                 baseline=True,
                 seed=None):
        self.name = name
        self.model_path = model_path
        self.env = env
        self.training = training
        self.gamma = gamma
        self.lr = lr
        self.lr_decay = lr_decay
        self.layer_sizes = layer_sizes
        self.baseline = baseline
        self.seed = seed

        self.action_size = self.env.action_space.n
        self.state_size = np.prod(list(self.env.observation_space.shape))
        print 'action_size: {a}, state_size: {s}'.format(a=self.action_size, s=self.state_size)

        if self.training:
            # clear existing model files
            if os.path.exists(self.model_path):
                print 'deleting existing model files at {}'.format(self.model_path)
                if os.path.isdir(self.model_path):
                    shutil.rmtree(self.model_path)
                else:
                    os.remove(self.model_path)

        BaseTFModel.__init__(self, self.name, self.model_path, saver_max_to_keep=5)

        print 'building graph ...'
        with self.graph.as_default():
            np.random.seed(self.seed)
            tf.set_random_seed(self.seed*3)
            self.__build_graph()

    def act(self, state):
        """
        :param state: 1d np.ndarray
        :return:
        """
        assert isinstance(state, np.ndarray) and state.ndim == 1
        return self.sess.run(self.sampled_actions, {self.states: state.reshape((1, -1))})

    def __build_graph(self):
        self.learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate')

        # inputs
        self.states = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='action')
        self.returns = tf.placeholder(tf.float32, shape=(None,), name='return')

        self.pi = dense_nn(self.states, self.layer_sizes + [self.action_size], name='pi_network')
        self.sampled_actions = tf.squeeze(tf.multinomial(self.pi, 1))
        self.max_action_proba = tf.reduce_max(tf.nn.softmax(self.pi), axis=-1)
        self.pi_vars = self.scope_vars('pi_network')

        if self.baseline:
            # state value estimation as the baseline
            self.v = dense_nn(self.states, self.layer_sizes + [1], name='v_network')
            # advantage
            self.target = self.returns - self.v

            with tf.variable_scope('v_optimize'):
                self.loss_v = tf.reduce_mean(tf.squared_difference(self.v, self.returns))
                self.optim_v = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_v, name='adam_optim_v')
        else:
            self.target = tf.identity(self.returns)

        with tf.variable_scope('pi_optimize'):
            self.loss_pi = tf.reduce_mean(
                tf.stop_gradient(self.target) * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pi, labels=self.actions),
                name='loss_pi')
            self.optim_pi = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_pi, name='adam_optim_pi')

        with tf.variable_scope('summary'):
            # max_action_proba reflects the level of exploration
            max_action_proba_summ = tf.summary.histogram("max_action_proba", self.max_action_proba)
            mean_of_max_action_proba_summ = tf.summary.scalar("mean_of_max_action_proba", tf.reduce_mean(self.max_action_proba))
            loss_pi_summ = tf.summary.scalar('loss_pi', self.loss_pi)
            summ_list = [mean_of_max_action_proba_summ, max_action_proba_summ, loss_pi_summ]
            if self.baseline:
                loss_v_summ = tf.summary.scalar('loss_v', self.loss_v)
                summ_list.append(loss_v_summ)
            self.merged_summary = tf.summary.merge(summ_list)

        if self.baseline:
            self.train_ops = [self.optim_pi, self.optim_v]
        else:
            self.train_ops = [self.optim_pi]
        self.init_vars = tf.global_variables_initializer()

    def train(self, n_episodes=800, every_episode=10):
        if self.training is False:
            raise Exception('prohibited to call train() for a non-training model')

        step = 0
        reward_history = []
        reward_averaged = []
        lr = self.lr
        self.sess.run(self.init_vars)
        for n_episode in range(n_episodes):
            ob = self.env.reset()

            done = False
            states = []
            actions = []
            rewards = []
            while not done:
                a = self.act(ob)
                new_ob, r, done, _ = self.env.step(a)
                step += 1
                states.append(ob)
                actions.append(a)
                rewards.append(r)
                ob = new_ob
            # one episode is complete
            reward_history.append(sum(rewards))
            reward_averaged.append(np.mean(reward_history[-10:]))

            # estimate returns backwards
            returns = []
            return_so_far = 0.0
            for r in rewards[::-1]:
                return_so_far = self.gamma * return_so_far + r
                returns.append(return_so_far)
            returns = returns[::-1]

            lr *= self.lr_decay
            _, summ_str = self.sess.run(
                [self.train_ops, self.merged_summary],
                feed_dict={
                    self.learning_rate: lr,
                    self.states: np.array(states),
                    self.actions: np.array(actions),
                    self.returns: np.array(returns),
                })
            self.writer.add_summary(summ_str, step)

            if reward_history and every_episode and n_episode % every_episode == 0:
                print("[episodes {}/step {}], best {}, avg10 {:.2f}:{}, lr {:.4f}".format(
                    n_episode, step, np.max(reward_history),
                    np.mean(reward_history[-10:]), reward_history[-5:],
                    lr,
                ))

        self.save_model(step=step)
        print "[training completed] episodes: {}, Max reward: {}, Average reward: {}".format(
            len(reward_history), np.max(reward_history), np.mean(reward_history))

        fig_path = os.path.join(self.model_path, 'figs')
        makedirs(fig_path)
        fig_file = os.path.join(fig_path, '{n}-{t}.png'.format(n=self.name, t=int(time.time())))
        plot_learning_curve(fig_file, {'reward': reward_history, 'reward_avg': reward_averaged}, xlabel='episode')

    def evaluate(self, n_episodes):
        if self.training:
            raise Exception('prohibited to call evaluate() for a training model')

        reward_history = []
        for episode in xrange(n_episodes):
            state = self.env.reset()
            reward_episode = 0.
            while True:
                action = self.act(state)
                new_state, reward, done, _ = self.env.step(action)
                reward_episode += reward
                state = new_state
                if done:
                    break
            reward_history.append(reward_episode)
        return reward_history


def main():
    env = gym.make("CartPole-v1")
    env.seed(12345)
    baseline = True
    n_episodes_train = 500
    n_episodes_eval = 100

    policy = ReinforcePolicy(env=env, name='ReinforcePolicy', model_path='result/ReinforcePolicy', baseline=baseline, seed=1234)
    policy.train(n_episodes=n_episodes_train)

    policy2 = ReinforcePolicy(env=env, name='ReinforcePolicy', model_path='result/ReinforcePolicy', baseline=baseline, training=False, seed=1234)
    policy2.load_model()
    reward_history = policy2.evaluate(n_episodes=n_episodes_eval)
    print 'reward history over {e} episodes: avg: {a:.4f}'.format(e=n_episodes_eval, a=np.mean(reward_history))
    print pd.Series(reward_history).describe()


if __name__ == '__main__':
    main()
