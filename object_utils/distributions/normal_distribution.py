import numpy as np
import scipy


def get_normal_samples(mu, variance, n_samples, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return scipy.stats.norm.rvs(loc=mu, scale=variance, size=n_samples)


def get_normal_pdf(x, mu, variance):
    return scipy.stats.norm.pdf(x, loc=mu, scale=variance)


def get_multivaraite_normal_samples(mu, cov, n_samples, seed=None):
    return scipy.stats.multivariate_normal.rvs(mean=mu, cov=cov, size=n_samples, random_state=seed)


def get_multivaraite_normal_pdf(x, mu, cov):
    return scipy.stats.multivariate_normal.pdf(x=x, mean=mu, cov=cov)
