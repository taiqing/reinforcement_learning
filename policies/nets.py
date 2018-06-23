# coding: utf-8

import tensorflow as tf
import numpy as np


def dense_nn(inputs, layer_sizes, name="mlp", reuse=None, 
             dropout_keep_prob=None, batch_norm=False, training=True):
    """
    :param inputs: 
    :param layer_sizes: 
    :param name: 
    :param reuse: 
    :param dropout_keep_prob: 
    :param batch_norm: 
    :param training: 
    :return: 
    """
    assert isinstance(layer_sizes, list)
    print 'building mlp {n}, sizes: {s}'.format(n=name, s=[inputs.shape[-1]] + layer_sizes)
    with tf.variable_scope(name):
        for i, size in enumerate(layer_sizes):
            print "layer {n}_{i} size: {s}".format(n=name, i=i, s=size)
            if i > 0 and dropout_keep_prob is not None and training:
                # No dropout on the input layer.
                inputs = tf.nn.dropout(inputs, dropout_keep_prob)
            inputs = tf.layers.dense(
                inputs,
                size,
                # Add relu activation only for internal layers.
                activation=tf.nn.relu if i < len(layer_sizes) - 1 else None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                reuse=reuse,
                name=name + '_l' + str(i)
            )
            if batch_norm:
                inputs = tf.layers.batch_normalization(inputs, training=training)
    return inputs
