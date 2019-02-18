import numpy as np
from gp.utils import *


class GaussianProcess(object):
    def __init__(self, kernel, noise_lvl):
        self.kernel = kernel
        self.noise_lvl = noise_lvl
        self.cache = {}


class ConstantMeanGP(GaussianProcess):
    """
    Deterministic mean (constant) Gaussian process.

    [ Equation (2.19) and (2.30), Rasmussen and Williams, 2006 ]

    Parameters
    ----------
    mean: float of constant mean to be subtracted from labels
    noise_lvl: float hyper-parameter, level of noise in observations
    """
    def __init__(self, mean, kernel, noise_lvl):
        super().__init__(kernel, noise_lvl)
        self.mean = mean

    def fit(self, x_tr, y_tr):
        """
        Parameters
        ----------
        x_tr: m x n array of training data
        y_tr: m-length array of training labels

        Returns
        -------
        loglik: marginal log-likelihood of observed data
        """
        y_tr = y_tr - self.mean
        k_tr = self.kernel(x_tr, x_tr) + self.noise_lvl * np.eye(x_tr.shape[0])
        l_tr = np.linalg.cholesky(k_tr)
        alpha = np.linalg.solve(l_tr.T, np.linalg.solve(l_tr, y_tr)).squeeze()
        self.cache.update({
            "x_tr": x_tr, "y_tr": y_tr, "l_tr": l_tr, "alpha": alpha
        })
        return (-np.log(np.linalg.det(l_tr)) \
                -0.5 * np.dot(y_tr, alpha) \
                -0.5 * y_tr.shape[0] * np.log(2 * np.pi)) / y_tr.shape[0]

    def predict(self, x_te):
        """
        Parameters
        ----------
        x_te: m x n array of test data

        Returns
        -------
        mean: m-length array of predicted mean
        var: m x m array of predicted variance
        """
        k_tr_te = self.kernel(self.cache["x_tr"], x_te)
        k_te = self.kernel(x_te, x_te)
        mean = np.dot(k_tr_te.T, self.cache["alpha"])
        v = np.linalg.solve(self.cache["l_tr"], k_tr_te)
        var = k_te - np.dot(v.T, v)
        return self.mean + mean, var + self.noise_lvl * np.eye(x_te.shape[0])


class StochasticMeanGP(GaussianProcess):
    """
    Stochastic mean GP, where the mean is given by a Bayesian linear model.

        g(x) = f(x) + h(x)ᵀβ, f(x) ~ GP(0, k(x, x')), β ~ N(b, B)

    [ Equation (2.41) and (2.43), Rasmussen and Williams, 2006 ]

    Parameters:
    -----------
    prior_mean:
    """
    def __init__(self, prior_mean, prior_var, kernel, noise_lvl):
        super().__init__(kernel, noise_lvl)
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.prior_prec = np.linalg.inv(prior_var)
        self.prior_gamma = np.dot(self.prior_prec, self.prior_mean).squeeze()

    def fit(self, x_tr, y_tr, h_tr):
        """
        Parameters
        ----------
        x_tr: m x n array of training data
        y_tr: m-length array of training labels
        h_tr: m x p array of training data

        Returns
        -------
        loglik: marginal log-likelihood of training data
        """
        k_tr = self.kernel(x_tr, x_tr) + self.noise_lvl * np.eye(x_tr.shape[0])
        l_tr = np.linalg.cholesky(k_tr)
        l_tr_inv = np.linalg.inv(l_tr)
        alpha = np.linalg.solve(l_tr.T, np.linalg.solve(l_tr, y_tr)).squeeze()
        zeta = np.linalg.inv(self.prior_prec + \
                             np.linalg.multi_dot([h_tr.T, l_tr_inv.T,
                                                  l_tr_inv, h_tr]))
        beta = np.dot(zeta, np.dot(h_tr.T, alpha) + self.prior_gamma)
        self.cache.update({
            "x_tr": x_tr, "y_tr": y_tr, "l_tr": l_tr, "l_tr_inv": l_tr_inv,
            "alpha": alpha, "beta": beta, "zeta": zeta, "h_tr": h_tr
        })
        return

    def predict(self, x_te, h_te):
        """
        """
        k_tr_te = self.kernel(self.cache["x_tr"], x_te)
        k_te = self.kernel(x_te, x_te)
        mean_gp = np.dot(k_tr_te.T, self.cache["alpha"])
        v = np.linalg.solve(self.cache["l_tr"], k_tr_te)
        var_gp = k_te - np.dot(v.T, v)
        r = h_te.T - np.linalg.multi_dot([self.cache["h_tr"].T,
                                          self.cache["l_tr_inv"].T,
                                          self.cache["l_tr_inv"], k_tr_te])
        mean = mean_gp + np.dot(r.T, self.cache["beta"])
        var = var_gp + np.linalg.multi_dot([r.T, self.cache["zeta"], r])
        return mean, var

    def get_posterior_beta(self):
        """

        """
        if not self.cache:
            raise ValueError("")
        return self.cache["beta"], self.cache["zeta"]
