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


Record = namedtuple('Record', ['s', 'a', 'r', 'td_target'])


class ActorCriticPolicy(BaseTFModel):
    def __init__(self, env, name,
                 model_path='./',
                 training=True,
                 gamma=0.9,
                 lr_a=0.02,
                 lr_a_decay=0.995,
                 lr_c=0.01,
                 lr_c_decay=0.995,
                 epsilon=1.0,
                 epsilon_final=0.05,
                 batch_size=32,
                 layer_sizes=[64],
                 grad_clip_norm=None,
                 seed=None):
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
                tf.set_random_seed(self.seed*3)
            self.__build_graph()
