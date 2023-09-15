import numpy as np
import scipy


def get_negative_binomial_samples(n, p, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return scipy.stats.nbinom.rvs(n=n, p=p, seed=seed)


def get_negative_binomial_pdf(x, n, p):
    return scipy.stats.nbinom.pmf(x, n=n, p=p)
