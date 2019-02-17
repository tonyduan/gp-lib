import numpy as np
import scipy as sp
import scipy.stats


def gaussian_loglik(obs, mu, sigma):
    return sp.stats.multivariate_normal.logpdf(obs, mean=mu, cov=sigma)

def gaussian_entropy(sigma):
    n, n = sigma.shape
    return 0.5 * (n + np.log(np.linalg.det(sigma)) + np.log(2 * np.pi))

def r2_score(obs, pred):
    return 1 - np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

def ece_score(obs, mu, sigma):
    pass
