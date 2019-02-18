import numpy as np
import scipy as sp
import scipy.stats


def gaussian_loglik(obs, mu, sigma):
    m = mu.shape[0]
    return sp.stats.multivariate_normal.logpdf(obs, mean=mu, cov=sigma) / m

def gaussian_entropy(sigma):
    n, n = sigma.shape
    return 0.5 * (n + np.log(np.linalg.det(sigma)) + np.log(2 * np.pi))

def r2_score(obs, pred):
    return 1 - np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

def cal_error(obs, mu, sigma, bins=5):
    """
    Unweighted regression calibration error for GP predictions.

    We calculate the mean-squared error between predicted versus observed
    empirical CDFs, for the specified number of equally spaced bins on the
    interval [0,1].

    [Equation (9), Kuleshov et. al. 2018]

    Parameters
    ----------
    obs: m-length array of observations
    mu: m-length array of predicted means
    sigma: m x m array of predicted covariance
    bins: number of bins at which to evaluate

    Returns
    -------
    cal_error: float
    predicted: predicted CDFs corresponding to each bin
    empirical: observed CDFs corresponding to each bin
    """
    sigmas = np.diag(sigma)
    quantiles = sp.stats.norm.cdf(obs, mu, np.sqrt(sigmas))
    predicted = np.arange(1/bins, 1+1/bins, 1/bins)
    empirical = np.array([np.mean(quantiles < p) for p in predicted])
    return np.sum((predicted - empirical) ** 2) / bins, predicted, empirical
