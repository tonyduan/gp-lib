import numpy as np


class GaussianProcess(object):
    def __init__(self, kernel, noise_lvl):
        self.kernel = kernel
        self.noise_lvl = noise_lvl
        self.cache = {}
        self.trained = False


class ConstantMeanGP(GaussianProcess):
    """
    Constant mean (deterministic) Gaussian process.

    [ Equation (2.19) and (2.30), Rasmussen and Williams, 2006 ]

    Parameters
    ----------
    mean: float of constant mean to be subtracted from labels
    kernel: Kernel used (incororates hyperparameters)
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
        m, n = x_tr.shape
        y_tr = y_tr - self.mean
        k_tr = self.kernel(x_tr, x_tr) + self.noise_lvl * np.eye(m)
        l_tr = np.linalg.cholesky(k_tr)
        alpha = np.linalg.solve(l_tr.T, np.linalg.solve(l_tr, y_tr))
        self.cache.update({
            "x_tr": x_tr, "y_tr": y_tr, "l_tr": l_tr, "alpha": alpha,
            "k_tr": k_tr
        })
        self.trained = True
        return (-np.sum(np.log(np.diag(l_tr))) \
                -0.5 * y_tr @ alpha \
                -0.5 * m * np.log(2 * np.pi)) / m

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
        m, n = x_te.shape
        if not self.trained:
            return self.mean, self.kernel(x_te, x_te)
        k_tr_te = self.kernel(self.cache["x_tr"], x_te)
        k_te = self.kernel(x_te, x_te) + self.noise_lvl * np.eye(m)
        mean = k_tr_te.T @ self.cache["alpha"]
        v = np.linalg.solve(self.cache["l_tr"], k_tr_te)
        var = k_te - v.T @ v
        return self.mean + mean, var

    def gradient_update(self):
        """
        After fitting the GP, take a gradient descent step on the kernel.
        """
        if "k_tr_inv" not in self.cache:
            self.cache["k_tr_inv"] = np.linalg.inv(self.cache["k_tr"])
        self.kernel.gradient_update(self.cache)

class StochasticMeanGP(GaussianProcess):
    """
    Stochastic mean GP, where the mean is given by a Bayesian linear model.

        g(x) = f(x) + h(x)ᵀβ, f(x) ~ GP(0, k(x, x')), β ~ N(b, B)

    [ Equation (2.41) and (2.43), Rasmussen and Williams, 2006 ]

    Parameters:
    -----------
    prior_mean: p-length array of prior mean over β
    prior_var: p x p array of prior variance ovoer β
    kernel: Kernel used (incororates hyperparameters)
    noise_lvl: float hyper-parameter, level of noise in observations
    """
    def __init__(self, prior_mean, prior_var, kernel, noise_lvl):
        super().__init__(kernel, noise_lvl)
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.prior_prec = np.linalg.inv(prior_var)
        self.prior_gamma = (self.prior_prec @ self.prior_mean).squeeze()

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
        m, n = x_tr.shape
        k_tr = self.kernel(x_tr, x_tr) + self.noise_lvl * np.eye(m)
        l_tr = np.linalg.cholesky(k_tr)
        l_tr_inv = np.linalg.inv(l_tr)
        alpha = np.linalg.solve(l_tr.T, np.linalg.solve(l_tr, y_tr))
        zeta = np.linalg.inv(self.prior_prec + (h_tr.T @ l_tr_inv.T @
                                                l_tr_inv @ h_tr))
        beta = zeta @ (h_tr.T @ alpha + self.prior_gamma)
        self.cache.update({
            "x_tr": x_tr, "y_tr": y_tr, "l_tr": l_tr, "l_tr_inv": l_tr_inv,
            "alpha": alpha, "beta": beta, "zeta": zeta, "h_tr": h_tr,
            "k_tr": k_tr
        })
        delta = h_tr @ self.prior_mean - y_tr
        psi = k_tr + h_tr @ self.prior_var @ h_tr.T
        self.trained = True
        return (-np.sum(np.log(np.diag(np.linalg.cholesky(psi)))) \
                -0.5 * delta @ np.linalg.solve(psi, delta) \
                -0.5 * m * np.log(2 * np.pi)) / m

    def predict(self, x_te, h_te):
        """
        Parameters
        ----------
        x_te: m x n array of test data
        h_te: m x p array of test data

        Returns
        -------
        mean: m-length array of predicted mean
        var: m x m array of predicted variance
        """
        m, n = x_te.shape
        if not self.trained:
            return h_te @ self.prior_mean, \
                   self.kernel(x_te, x_te) + self.noise_lvl * np.eye(m) + \
                   h_te @ self.prior_var @ h_te.T
        k_tr_te = self.kernel(self.cache["x_tr"], x_te)
        k_te = self.kernel(x_te, x_te) + self.noise_lvl * np.eye(m)
        mean_gp = k_tr_te.T @ self.cache["alpha"]
        v = np.linalg.solve(self.cache["l_tr"], k_tr_te)
        var_gp = k_te - v.T @ v
        r = h_te.T - self.cache["h_tr"].T @ self.cache["l_tr_inv"].T @ \
                     self.cache["l_tr_inv"] @ k_tr_te
        mean = mean_gp + r.T @ self.cache["beta"]
        var = var_gp + r.T @ self.cache["zeta"] @ r
        return mean, var

    def get_beta(self):
        """
        After fitting the GP, return the posterior β in the model.

        Returns
        -------
        mean: p-length array for mean of β
        var: p x p array for variance of β
        """
        if not self.trained:
            return self.prior_mean, self.prior_var
        return self.cache["beta"], self.cache["zeta"]

    def gradient_update(self):
        """
        After fitting the GP, take a gradient descent step on the kernel.
        """
        if "k_tr_inv" not in self.cache:
            self.cache["k_tr_inv"] = np.linalg.inv(self.cache["k_tr"])
        self.kernel.gradient_update(self.cache)
