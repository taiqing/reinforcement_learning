# coding: utf-8

"""
Solve the cart pole task with deep q-network
Ref:
https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html
"""


import os
import time
import numpy as np
import tensorflow as tf
import gym
import pandas as pd

from utils import Transition, ReplayMemory, plot_learning_curve, makedirs
from nets import dense_nn
from base_tf_model import BaseTFModel


# TODO: add base class Policy
class DqnPolicy(BaseTFModel):
    def __init__(self, env, name,
                 model_path='./',
                 training=True,
                 gamma=0.99,
                 lr=0.001,
                 lr_decay=1.0,
                 epsilon=1.0,
                 epsilon_final=0.02,
                 batch_size=32,
                 memory_capacity=100000,
                 model_params={},
                 layer_sizes=[32, 32],
                 target_update_type='hard',
                 target_update_params={},
                 double_q=True,
                 dueling=True):
        self.name = name
        self.model_path = model_path
        BaseTFModel.__init__(self, self.name, self.model_path, saver_max_to_keep=5)

        self.env = env
        self.training = training
        self.gamma = gamma
        self.lr = lr
        self.lr_decay = lr_decay
        self.epsilon = epsilon
        self.epsilon_final = epsilon_final
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.model_params = model_params
        self.layer_sizes = layer_sizes
        self.double_q = double_q
        self.dueling = dueling

        self.target_update_type = target_update_type
        self.target_update_every_step = target_update_params.get('every_step', 100)
        self.target_update_tau = target_update_params.get('tau', 0.05)

        self.memory = ReplayMemory(capacity=memory_capacity)

        self.action_size = self.env.action_space.n
        self.state_size = np.prod(list(self.env.observation_space.shape))
        print 'action_size: {a}, state_size: {s}'.format(a=self.action_size, s=self.state_size)

        print 'building graph ...'
        with self.graph.as_default():
            self.build_graph()

    def build_graph(self):
        self.__create_q_networks()

        self.actions_selected_by_q = tf.argmax(self.q, axis=-1, name='action_selected')
        action_one_hot = tf.one_hot(self.actions, self.action_size, dtype=tf.float32, name='action_one_hot')
        pred = tf.reduce_sum(self.q * action_one_hot, axis=-1, name='pred')
        if self.double_q:
            action_next_one_hot = tf.one_hot(self.actions_next, self.action_size, dtype=tf.float32, name='action_next_one_hot')
            max_q_next_target = tf.reduce_sum(self.q_target * action_next_one_hot, axis=-1, name='max_q_next_target')
        else:
            max_q_next_target = tf.reduce_max(self.q_target, axis=-1)
        y = self.rewards + (1. - self.done_flags) * self.gamma * max_q_next_target
        self.loss = tf.reduce_mean(tf.square(pred - tf.stop_gradient(y)), name="loss_mse_train")
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, name="adam")
        self.init_vars = tf.global_variables_initializer()
        with tf.variable_scope('summary'):
            q_summ = []
            avg_q = tf.reduce_mean(self.q, 0)
            for idx in range(self.action_size):
                q_summ.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
            self.q_summ = tf.summary.merge(q_summ, 'q_summary')

            self.q_y_summ = tf.summary.histogram("batch/y", y)
            self.q_pred_summ = tf.summary.histogram("batch/pred", pred)
            self.loss_summ = tf.summary.scalar("loss", self.loss)

            self.merged_summary = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

    def __create_q_networks(self):
        # mini-batch
        self.states = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state')
        self.states_next = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state_next')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='action')
        # actions_next is not the actual actions in the next step;
        # it is used to predict the action value in the Bellman equation.
        self.actions_next = tf.placeholder(tf.int32, shape=(None,), name='action_next')
        self.rewards = tf.placeholder(tf.float32, shape=(None,), name='reward')
        self.done_flags = tf.placeholder(tf.float32, shape=(None,), name='done')
        self.learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate')

        if self.dueling:
            with tf.variable_scope('Q_primary'):
                self.q_hidden = dense_nn(self.states, self.layer_sizes[:-1], name='q_hidden', training=self.training)
                # advantage function A(s, a)
                self.adv = dense_nn(self.q_hidden, [self.layer_sizes[-1], self.action_size], name='adv', training=self.training)
                # state value function V(s)
                self.v = dense_nn(self.q_hidden, [self.layer_sizes[-1], 1], name='v', training=self.training)
                self.q = self.v + (self.adv - tf.reduce_mean(self.adv, reduction_indices=1, keep_dims=True))

            with tf.variable_scope('Q_target'):
                self.q_target_hidden = dense_nn(self.states_next, self.layer_sizes[:-1], name='q_hidden', training=self.training)
                self.adv_target = dense_nn(self.q_target_hidden, [self.layer_sizes[-1], self.action_size], name='adv', training=self.training)
                self.v_target = dense_nn(self.q_target_hidden, [self.layer_sizes[-1], 1], name='v', training=self.training)
                self.q_target = self.v_target + (self.adv_target - tf.reduce_mean(self.adv_target, reduction_indices=1, keep_dims=True))
        else:
            self.q = dense_nn(self.states, self.layer_sizes + [self.action_size], name='Q_primary', training=self.training)
            self.q_target = dense_nn(self.states_next, self.layer_sizes + [self.action_size], name='Q_target', training=self.training)

        self.q_vars = self.scope_vars('Q_primary')
        self.q_target_vars = self.scope_vars('Q_target')
        assert len(self.q_vars) == len(self.q_target_vars), "Two Q-networks are not same in structure."

    def __init_target_q_net(self):
        self.sess.run([v_t.assign(v) for v_t, v in zip(self.q_target_vars, self.q_vars)])

    def __update_target_q_net_hard(self):
        self.sess.run([v_t.assign(v) for v_t, v in zip(self.q_target_vars, self.q_vars)])

    def __update_target_q_net_soft(self, tau=0.05):
        self.sess.run([v_t.assign(v_t * (1. - tau) + v * tau)
                       for v_t, v in zip(self.q_target_vars, self.q_vars)])

    def __update_target_q_net(self, step):
        if self.target_update_type == 'hard':
            if step % self.target_update_every_step == 0:
                self.__update_target_q_net_hard()
        else:
            self.__update_target_q_net_soft(self.target_update_tau)

    def act(self, state, epsilon=0.1):
        """
        :param state: 1d np.ndarray
        :param epsilon:
        :return: int
        """
        assert isinstance(state, np.ndarray) and state.ndim == 1
        if self.training and np.random.random() < epsilon:
            return self.env.action_space.sample()

        with self.sess.as_default():
            return self.actions_selected_by_q.eval({self.states: state.reshape((1, -1))})[0]

    def train(self, n_episodes=500, annealing_episodes=450, every_episode=10):
        if self.training is False:
            raise Exception('prohibited to call train() for a non-training model')

        reward_history = [0.0]
        reward_averaged = []
        lr = self.lr
        eps = self.epsilon
        annealing_episodes = annealing_episodes or n_episodes
        eps_drop = (self.epsilon - self.epsilon_final) / annealing_episodes
        print "eps_drop: {}".format(eps_drop)
        step = 0

        # calling the property method of BaseTFModel to start a session
        self.sess.run(self.init_vars)
        self.__init_target_q_net()

        for n_episode in range(n_episodes):
            ob = self.env.reset()
            done = False
            traj = []
            reward = 0.
            while not done:
                a = self.act(ob, eps)
                new_ob, r, done, _ = self.env.step(a)
                step += 1
                reward += r
                traj.append(Transition(ob, a, r, new_ob, done))
                ob = new_ob

                # No enough samples in the buffer yet.
                if self.memory.size < self.batch_size:
                    continue
                # Training with a mini batch of samples
                batch_data = self.memory.sample(self.batch_size)
                feed_dict = {
                    self.learning_rate: lr,
                    self.states: batch_data['s'],
                    self.actions: batch_data['a'],
                    self.rewards: batch_data['r'],
                    self.states_next: batch_data['s_next'],
                    self.done_flags: batch_data['done']}

                if self.double_q:
                    actions_next = self.sess.run(self.actions_selected_by_q, {self.states: batch_data['s_next']})
                    feed_dict.update({self.actions_next: actions_next})

                _, q_val, q_target_val, loss, summ_str = self.sess.run(
                    [self.optimizer, self.q, self.q_target, self.loss, self.merged_summary], feed_dict=feed_dict)
                self.writer.add_summary(summ_str, step)
                self.__update_target_q_net(step)
            self.memory.add(traj)
            reward_history.append(reward)
            reward_averaged.append(np.mean(reward_history[-10:]))

            # Annealing the learning and exploration rate after every episode
            lr *= self.lr_decay
            if eps > self.epsilon_final:
                eps -= eps_drop

            if reward_history and every_episode and n_episode % every_episode == 0:
                print "[episodes: {}/step: {}], best: {}, avg: {:.2f}:{}, lr: {:.4f}, eps: {:.4f}".format(
                    n_episode, step,
                    np.max(reward_history), np.mean(reward_history[-10:]), reward_history[-5:],
                    lr, eps)

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
    n_episodes_eval = 100

    policy = DqnPolicy(env=env, name='DqnPolicy', model_path='result/DqnPolicy')
    policy.train(n_episodes=100)

    policy2 = DqnPolicy(env=env, name='DqnPolicy_eval', model_path='result/DqnPolicy', training=False)
    policy2.load_model()
    reward_history = policy2.evaluate(n_episodes=n_episodes_eval)
    print 'reward history over {e} episodes: avg: {a:.4f}'.format(e=n_episodes_eval, a=np.mean(reward_history))
    print pd.Series(reward_history).describe()


if __name__ == '__main__':
    main()
