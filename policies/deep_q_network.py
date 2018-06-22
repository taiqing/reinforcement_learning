# coding: utf-8

"""
Solve the cart pole task with deep q-network
Ref:
https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html
"""


import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete


