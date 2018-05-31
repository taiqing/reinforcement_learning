# coding: utf-8

"""
Bayesian Linear Probit Model
"""

from scipy import stats
import numpy as np


def sample_norm_distr(mean, var, num):
    rv = stats.norm(loc=mean, scale=np.sqrt(var))
    return rv.rvs(size=(num, len(mean))).astype(np.float32)

def likelihood(w, x, beta):
    return stats.norm.cdf(np.matmul(x, w)/ beta)


if __name__ == '__main__':
    x_prior = 0.1
    # dimension
    d_x = 10
    # number of samples
    n_x = 10
    w_init_mean = np.array([0] * d_x, dtype=np.float32)
    w_init_var = np.array([100] * d_x, dtype=np.float32)
    n_iter = 100
    beta = 1.

    # generate samples
    X = stats.bernoulli.rvs(x_prior, size=[n_x, d_x])
    print 1.0 * X.sum() / X.size
    
    w_mean = w_init_mean
    w_var = w_init_var
    for i in xrange(n_iter):
        # sample w
        w = sample_norm_distr(w_mean, w_var, 1)[0]
        # rank the samples according to likelihood of click
        p = likelihood(w, X, beta)
        break