# coding: utf-8

"""
Bayesian Linear Probit Model
"""

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


def sample_norm_distr(mean, var, num):
    rv = stats.norm(loc=mean, scale=np.sqrt(var))
    return rv.rvs(size=(num, len(mean))).astype(np.float32)


def likelihood(w, x, beta):
    return stats.norm.cdf(np.matmul(x, w) / beta)


def simulate_reward(true_w, beta, x):
    """
    :param true_w:
    :param beta:
    :param x:
    :return: {0, 1}
    """
    assert true_w.ndim == 1
    assert x.ndim == 1
    # P(y=1 | x, w)
    p = stats.norm.cdf(np.matmul(true_w, x) / beta)
    y = stats.bernoulli(p=p).rvs(1)[0]
    return y


def f_v(t):
    return stats.norm.pdf(t) / stats.norm.cdf(t)


def f_w(t):
    return f_v(t) * (f_v(t) + t)


# def main():
if __name__ == '__main__':
    x_prior = 0.2
    # dimension
    d_x = 5
    # number of samples
    n_x = 2000
    true_w = np.array([-1, -1, -1, 1, 1], dtype=np.float32)
    w_init_mean = np.array([0] * d_x, dtype=np.float32)
    w_init_var = np.array([10.] * d_x, dtype=np.float32)
    n_iter = 1000
    beta = 1.
    probe_steps = 100

    # generate binary samples: 2d np.array
    X = stats.bernoulli.rvs(x_prior, size=[n_x, d_x])
    print 'note: {a} should be close to {p}'.format(p=x_prior, a=1.0 * X.sum() / X.size)

    w_mean = w_init_mean
    w_var = w_init_var
    reward_list = []
    for i in xrange(n_iter):
        # sample w
        w = sample_norm_distr(w_mean, w_var, 1)[0]
        # rank the samples according to likelihood of click
        p = likelihood(w, X, beta)
        # action: select the sample of the highest likelihood
        x = X[np.argmax(p)]
        # receive the reward
        y = simulate_reward(true_w, beta, x)
        reward_list.append(y)
        y = 2 * y - 1
        if i % probe_steps == 0:
            print '{}-th iter'.format(i)
            print 'action: {x}, reward: {y}'.format(x=x, y=y)
        # update the parameters of w
        sigma_square = beta * beta + np.matmul(x, w_var)
        sigma = np.sqrt(sigma_square)
        t = y * np.matmul(x, w_mean) / sigma
        w_mean_updated = w_mean + x * w_var * y * f_v(t) / sigma
        w_var_updated = w_var * (1. - x * w_var * f_w(t) / sigma_square)
        if i % probe_steps == 0:
            print 'w_mean: {a} -> {b}'.format(a=w_mean, b=w_mean_updated)
            print 'w_var: {a} -> {b}'.format(a=w_var, b=w_var_updated)
        w_mean = w_mean_updated
        w_var = w_var_updated
    ctr = np.convolve(reward_list, np.ones(50, np.float32) / 50, mode='valid')
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    ax.plot(ctr)
    fig.show()
    


# if __name__ == '__main__':
#     main()
