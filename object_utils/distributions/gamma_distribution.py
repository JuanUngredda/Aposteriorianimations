import numpy as np
import scipy


def get_gamma_samples(n_samples, a, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return scipy.stats.gamma.rvs(size=n_samples, a=a, random_state=seed)


def get_gamma_pdf(x, a):
    pdf = scipy.stats.gamma.pdf(x, a)
    return pdf
