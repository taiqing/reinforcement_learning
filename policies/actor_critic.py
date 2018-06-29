# coding: utf-8

"""
Solve the cart pole task with the Actor-Critic policy model
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
from collections import namedtuple

from utils import plot_learning_curve, makedirs
from nets import dense_nn
from base_tf_model import BaseTFModel
from utils import ReplayMemory


Record = namedtuple('Record', ['s', 'a', 'r', 's_next'])


class ActorCriticPolicy(BaseTFModel):
    def __init__(self, env, name,
                 model_path='./',
                 training=True,
                 gamma=0.9,
                 lr_a=0.01,
                 lr_a_decay=0.999,
                 lr_c=0.01,
                 lr_c_decay=0.999,
                 epsilon=1.0,
                 epsilon_final=0.05,
                 batch_size=16,
                 layer_sizes=[32],
                 grad_clip_norm=None,
                 act='bayesian',
                 seed=None):
        """
        :param env:
        :param name:
        :param model_path:
        :param training:
        :param gamma:
        :param lr_a:
        :param lr_a_decay:
        :param lr_c:
        :param lr_c_decay:
        :param epsilon:
        :param epsilon_final:
        :param batch_size:
        :param layer_sizes:
        :param grad_clip_norm:
        :param act: baysian or epsilon
        :param seed:
        """
        self.name = name
        self.model_path = model_path
        self.env = env
        self.training = training
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_a_decay = lr_a_decay
        self.lr_c = lr_c
        self.lr_c_decay = lr_c_decay
        self.epsilon = epsilon
        self.epsilon_final = epsilon_final
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.grad_clip_norm = grad_clip_norm
        self.seed = seed

        self.memory = ReplayMemory(tuple_class=Record)

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
            if self.seed is not None:
                np.random.seed(self.seed)
                tf.set_random_seed(int(self.seed/3))
            self.__build_graph()

        if act == 'bayesian':
            self.act = self.act_bayesian
        elif act == 'epsilon':
            self.act = self.act_epsilon
        else:
            raise Exception('not supported act {}'.format(act))

    def act_epsilon(self, state, **kwargs):
        """
        epsilon-greedy exploration is not effective in the case of large action spaces
        :param state:
        :param epsilon:
        :return:
        """
        if self.training and np.random.random() < kwargs['epsilon']:
            return self.env.action_space.sample()
        proba = self.sess.run(self.actor_proba, {self.states: state.reshape((1, -1))})[0]
        return np.argmax(proba)

    def act_bayesian(self, state, **kwargs):
        """
        :param state: 1d np.ndarray
        :return:
        """
        assert isinstance(state, np.ndarray) and state.ndim == 1
        return self.sess.run(self.sampled_actions, {self.states: state.reshape((1, -1))})
        # if self.training:
        #     return self.sess.run(self.sampled_actions, {self.states: state.reshape((1, -1))})
        # else:
        #     return self.sess.run(self.selected_actions, {self.states: state.reshape((1, -1))})

    def __build_graph(self):
        # c: critic, a: actor
        self.learning_rate_c = tf.placeholder(tf.float32, shape=None, name='learning_rate_c')
        self.learning_rate_a = tf.placeholder(tf.float32, shape=None, name='learning_rate_a')

        # inputs
        self.states = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state')
        self.states_next = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state_next')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='action')
        self.rewards = tf.placeholder(tf.float32, shape=(None,), name='reward')

        # actor: action probabilities
        self.actor = dense_nn(self.states, self.layer_sizes + [self.action_size], name='actor')
        # integer tensor
        self.sampled_actions = tf.squeeze(tf.multinomial(self.actor, 1))
        self.selected_actions = tf.squeeze(tf.argmax(self.actor, axis=-1))
        self.actor_proba = tf.nn.softmax(self.actor)
        self.actor_vars = self.scope_vars('actor')

        # critic: action value (Q-value)
        self.critic = dense_nn(self.states, self.layer_sizes + [1], name='critic')
        self.critic_vars = self.scope_vars('critic')
        self.td_targets = self.rewards \
                          + self.gamma * tf.squeeze(dense_nn(self.states_next, self.layer_sizes + [1], name='critic', reuse=True))
        # print the shape of td_targets
        # self.td_targets = tf.Print(self.td_targets, [tf.shape(self.td_targets)], first_n=1)

        action_ohe = tf.one_hot(self.actions, self.action_size, dtype=tf.float32, name='action_one_hot')
        self.pred_value = tf.reduce_sum(self.critic * action_ohe, axis=-1, name='q_action')
        self.td_errors = tf.stop_gradient(self.td_targets) - self.pred_value

        with tf.variable_scope('critic_train'):
            # self.reg_c = tf.reduce_mean([tf.nn.l2_loss(x) for x in self.critic_vars])
            self.loss_c = tf.reduce_mean(tf.square(self.td_errors))  # + 0.001 * self.reg_c
            self.optim_c = tf.train.AdamOptimizer(self.learning_rate_c)
            self.grads_c = self.optim_c.compute_gradients(self.loss_c, self.critic_vars)
            if self.grad_clip_norm:
                self.grads_c = [(tf.clip_by_norm(grad, self.grad_clip_norm), var) for grad, var in self.grads_c]
            self.train_op_c = self.optim_c.apply_gradients(self.grads_c)

        with tf.variable_scope('actor_train'):
            # self.reg_a = tf.reduce_mean([tf.nn.l2_loss(x) for x in self.actor_vars])
            self.loss_a = tf.reduce_mean(
                tf.stop_gradient(self.td_errors) * tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.actor, labels=self.actions),
                name='loss_actor')  # + 0.001 * self.reg_a
            self.optim_a = tf.train.AdamOptimizer(self.learning_rate_a)
            self.grads_a = self.optim_a.compute_gradients(self.loss_a, self.actor_vars)
            if self.grad_clip_norm:
                self.grads_a = [(tf.clip_by_norm(grad, self.grad_clip_norm), var) for grad, var in self.grads_a]
            self.train_op_a = self.optim_a.apply_gradients(self.grads_a)

        with tf.variable_scope('summary'):
            self.grads_a_summ = [tf.summary.scalar('grads/a_' + var.name, tf.norm(grad)) for
                                 grad, var in self.grads_a if grad is not None]
            self.grads_c_summ = [tf.summary.scalar('grads/c_' + var.name, tf.norm(grad)) for
                                 grad, var in self.grads_c if grad is not None]
            self.loss_c_summ = tf.summary.scalar('loss/critic', self.loss_c)
            self.loss_a_summ = tf.summary.scalar('loss/actor', self.loss_a)
            self.merged_summary = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

        self.train_ops = [self.train_op_a, self.train_op_c]
        self.init_vars = tf.global_variables_initializer()

    def train(self, n_episodes, annealing_episodes=None, every_episode=None, done_rewards=None):
        if self.training is False:
            raise Exception('prohibited to call train() for a non-training model')

        step = 0
        reward_history = []
        reward_averaged = []
        lr_c = self.lr_c
        lr_a = self.lr_a
        eps = self.epsilon
        annealing_episodes = annealing_episodes or n_episodes
        eps_drop = (eps - self.epsilon_final) / annealing_episodes
        print "eps_drop: {}".format(eps_drop)

        self.sess.run(self.init_vars)
        for n_episode in range(n_episodes):
            ob = self.env.reset()

            episode_reward = 0.
            done = False
            while not done:
                a = self.act(ob, epsilon=eps)
                ob_next, r, done, _ = self.env.step(a)
                step += 1
                episode_reward += r
                if done:
                    r = done_rewards or 0.
                self.memory.add(Record(ob, a, r, ob_next))
                ob = ob_next

                while self.memory.size >= self.batch_size:
                    batch = self.memory.pop(self.batch_size)
                    _, summ_str = self.sess.run(
                        [self.train_ops, self.merged_summary], feed_dict={
                            self.learning_rate_c: lr_c,
                            self.learning_rate_a: lr_a,
                            self.states: batch['s'],
                            self.actions: batch['a'],
                            self.rewards: batch['r'],
                            self.states_next: batch['s_next']
                        })
                    self.writer.add_summary(summ_str, step)
            reward_history.append(episode_reward)
            reward_averaged.append(np.mean(reward_history[-10:]))

            lr_c *= self.lr_c_decay
            lr_a *= self.lr_a_decay
            if eps > self.epsilon_final:
                eps -= eps_drop

            if reward_history and every_episode and n_episode % every_episode == 0:
                print(
                    "[episodes: {}/step: {}], best: {}, avg10: {:.2f}: {}, lr: {:.4f} | {:.4f} eps: {:.4f}".format(
                        n_episode, step, np.max(reward_history),
                        np.mean(reward_history[-10:]), reward_history[-5:],
                        lr_c, lr_a, eps
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
    n_episodes_train = 800
    n_episodes_eval = 100
    act = 'bayesian'

    policy = ActorCriticPolicy(env=env, name='ActorCriticPolicy', model_path='result/ActorCriticPolicy', act=act, seed=123)
    policy.train(n_episodes=n_episodes_train, annealing_episodes=720, every_episode=10, done_rewards=-100)

    # env.seed(101)
    policy2 = ActorCriticPolicy(env=env, name='ActorCriticPolicy', model_path='result/ActorCriticPolicy', act=act, training=False, seed=123)
    policy2.load_model()
    reward_history = policy2.evaluate(n_episodes=n_episodes_eval)
    print 'reward history over {e} episodes: avg: {a:.4f}'.format(e=n_episodes_eval, a=np.mean(reward_history))
    print pd.Series(reward_history).describe()


if __name__ == '__main__':
    main()

