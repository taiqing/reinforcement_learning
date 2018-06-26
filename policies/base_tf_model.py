# coding: utf-8


import os
import tensorflow as tf

from utils import makedirs


class BaseTFModel(object):
    """Abstract object representing an Reader model.

    Code borrowed from: https://github.com/devsisters/DQN-tensorflow/blob/master/dqn/base.py
    with some modifications.
    """

    def __init__(self, model_name, model_path, saver_max_to_keep=5):
        self.__saver = None
        self.__saver_max_to_keep = saver_max_to_keep
        self.__writer = None
        self.__model_name = model_name
        self.__sess = None
        self.__model_path = model_path
        self.__graph = None

    def scope_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        assert len(res) > 0
        print("Variables in scope '%s'" % scope)
        for v in res:
            print("\t" + str(v))
        return res

    def save_model(self, step=None):
        print " [*] Saving checkpoints..."
        ckpt_file = os.path.join(self.checkpoint_dir, self.model_name)
        self.saver.save(self.sess, ckpt_file, global_step=step)

    def load_model(self):
        print " [*] Loading the latest checkpoint ..."

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        print(self.checkpoint_dir, ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print(ckpt_name)
            fname = os.path.join(self.checkpoint_dir, ckpt_name)
            print(fname)
            self.saver.restore(self.sess, fname)
            print " [*] Load SUCCESS: %s" % fname
            return True
        else:
            print " [!] Load FAILED: %s" % self.checkpoint_dir
            return False

    @property
    def checkpoint_dir(self):
        ckpt_path = os.path.join(self.__model_path, 'checkpoints')
        makedirs(ckpt_path)
        return ckpt_path

    @property
    def model_name(self):
        assert self.__model_name, "Not a valid model name."
        return self.__model_name

    @property
    def saver(self):
        if self.__saver is None:
            with self.graph.as_default():
                self.__saver = tf.train.Saver(max_to_keep=self.__saver_max_to_keep)
        return self.__saver

    @property
    def writer(self):
        if self.__writer is None:
            writer_path = os.path.join(self.__model_path, "logs")
            makedirs(writer_path)
            with self.graph.as_default():
                self.__writer = tf.summary.FileWriter(writer_path, self.sess.graph)
        return self.__writer

    @property
    def sess(self):
        if self.__sess is None:
            config = tf.ConfigProto()
            config.intra_op_parallelism_threads = 2
            config.inter_op_parallelism_threads = 2
            self.__sess = tf.Session(config=config, graph=self.graph)
        return self.__sess

    @property
    def graph(self):
        if self.__graph is None:
            self.__graph = tf.Graph()
        return self.__graph
