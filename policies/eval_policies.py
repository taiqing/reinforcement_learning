# coding: utf-8

import numpy as np
import pandas as pd
import gym

from actor_critic import ActorCriticPolicy
from deep_q_network import DqnPolicy
from q_table_learning import QLearnPolicy
from reinforce import ReinforcePolicy


if __name__ == '__main__':
    env_name = "CartPole-v1"
    n_episodes_eval = 100

    Policy = ActorCriticPolicy
    params = {
        'n_episodes': 800,
        'annealing_episodes': 720,
        'every_episode': 10,
        'done_rewards': -100,
        'act': 'bayesian',
        'seed': 123,
    }

    # training
    env = gym.make(env_name)
    env.seed(1)
    policy = Policy(env=env, training=True, **params)
    policy.train(**params)

    # evaluation
    env2 = gym.make(env_name)
    env2.seed(11)
    policy2 = Policy(env=env2, training=False, **params)
    policy2.load_model()
    reward_history = policy2.evaluate(n_episodes=n_episodes_eval)
    print 'reward history over {e} episodes: avg: {a:.4f}'.format(e=n_episodes_eval, a=np.mean(reward_history))
    print pd.Series(reward_history).describe()
