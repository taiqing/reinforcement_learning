# coding: utf-8

import tensorflow as tf
import numpy as np


def dense_nn(inputs, layers_sizes, name="mlp", reuse=None, dropout_keep_prob=None,
             batch_norm=False, training=True):
    print("Building mlp {} | sizes: {}".format(
        name, [inputs.shape[0]] + layers_sizes), "green")
    with tf.variable_scope(name):
        for i, size in enumerate(layers_sizes):
            print("Layer:", name + '_l' + str(i), size)
            if i > 0 and dropout_keep_prob is not None and training:
                # No dropout on the input layer.
                inputs = tf.nn.dropout(inputs, dropout_keep_prob)
            inputs = tf.layers.dense(
                inputs,
                size,
                # Add relu activation only for internal layers.
                activation=tf.nn.relu if i < len(layers_sizes) - 1 else None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                reuse=reuse,
                name=name + '_l' + str(i)
            )
            if batch_norm:
                inputs = tf.layers.batch_normalization(inputs, training=training)
    return inputs
