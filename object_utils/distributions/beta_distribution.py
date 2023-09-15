import numpy as np
import scipy


def get_beta_samples(a, b, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return scipy.stats.beta.rvs(a=a, n=b, seed=seed)


def get_beta_pdf(x, a, b):
    return scipy.stats.beta.pdf(x, a, b)
