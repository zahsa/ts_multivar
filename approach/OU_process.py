import numpy as np
import math
import scipy


def aic_fuc(L, K):
    # L is the maximum likelihood (log-likelihood)
    return 2 * K - 2 * L


def bic_fuc(L, K, n):
    # L is the maximum likelihood (log-likelihood)
    return K * np.log(n) - 2 * L


def ou_process(t, x, start=None):
    """ OU (Ornstein-Uhlenbeck) process
        dX = -A(X-alpha)dt + v dB
        Maximum-likelihood estimator
        Piece of code from:
        https://github.com/jwergieluk/ou_noise/tree/c5eee685c8a80a079dd32c759df3b97e05ef51ef
    """

    if start is None:
        v = est_v_quadratic_variation(t, x)
        start = (0.5, np.mean(x), v)

    def error_fuc(theta):
        return -loglik(t, x, theta[0], theta[1], theta[2])

    start = np.array(start)
    result = scipy.optimize.minimize(error_fuc, start, method='L-BFGS-B',
                                     bounds=[(1e-6, None), (None, None), (1e-8, None)],
                                     options={'maxiter': 500, 'disp': False})
    L = error_fuc(result.x)
    k = len(result.x)
    n = x.shape[1]

    aic = aic_fuc(L, k)
    bic = bic_fuc(L, k, n)
    return result.x, aic, bic


def est_v_quadratic_variation(t, x, weights=None):
    """ Estimate v using quadratic variation"""
    assert len(t) == x.shape[1]
    q = quadratic_variation(x, weights)
    return math.sqrt(q/(t[-1] - t[0]))


def quadratic_variation(x, weights=None):
    """ Realized quadratic variation of a path. The weights must sum up to one. """
    assert x.shape[1] > 1
    dx = np.diff(x)
    if weights is None:
        return np.sum(dx*dx)
    return x.shape[1]*np.sum(dx * dx * weights)


def loglik(t, x, mean_rev_speed, mean_rev_level, vola):
    """Calculates log likelihood of a path"""
    dt = np.diff(t)
    mu = mean(x[:, :-1], dt, mean_rev_speed, mean_rev_level)
    sigma = std(dt, mean_rev_speed, vola)
    return np.sum(scipy.stats.norm.logpdf(x[:, 1:], loc=mu, scale=sigma))


def mean(x0, t, mean_rev_speed, mean_rev_level):
    assert mean_rev_speed >= 0
    return x0 * np.exp(-mean_rev_speed * t) + (1.0 - np.exp(- mean_rev_speed * t)) * mean_rev_level


def std(t, mean_rev_speed, vola):
    return np.sqrt(variance(t, mean_rev_speed, vola))


def variance(t, mean_rev_speed, vola):
    assert mean_rev_speed >= 0
    assert vola >= 0
    return vola * vola * (1.0 - np.exp(- 2.0 * mean_rev_speed * t)) / (2 * mean_rev_speed)


def predict(x0, t, mean_rev_speed, mean_rev_level, vola):
    """ Simulates a sample path"""
    assert len(t) > 1
    x = scipy.stats.norm.rvs(size=len(t))
    x[0] = x0
    dt = np.diff(t)
    scale = std(dt, mean_rev_speed, vola)
    x[1:] = x[1:] * scale
    for i in range(1, len(x)):
        x[i] += mean(x[i - 1], dt[i - 1], mean_rev_speed, mean_rev_level)
    return x