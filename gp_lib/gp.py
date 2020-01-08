import numpy as np
import scipy as sp


class GaussianProcess(object):

    def __init__(self, kernel, noise_lvl):
        self.kernel = kernel
        self.noise_lvl = noise_lvl
        self.cache = {}
        self.trained = False


class ConstantMeanGP(GaussianProcess):
    """
    Constant mean (deterministic) Gaussian process.

    [ Equation (2.30), Rasmussen and Williams, 2006 ]

    Parameters
    ----------
    mean: float of constant mean to be subtracted from labels
    kernel: Kernel used (incororates hyperparameters)
    noise_lvl: float hyper-parameter, level of noise in observations
    """
    def __init__(self, mean, kernel, noise_lvl):
        super().__init__(kernel, noise_lvl)
        self.mean = mean

    def fit(self, x_tr, y_tr, eval_gradient=False):
        """

        [ Equation (5.9), Rasmussen & Williams, 2006 ]

        Parameters
        ----------
        x_tr: m x n array of training data
        y_tr: m-length array of training labels
        eval_gradient: boolean whether to return the gradient wrt hyper-parameters

        Returns
        -------
        loglik: scalar marginal log-likelihood of observed data
        grad: p-length array of gradients with respect to kernel parameters theta
        """
        m, n = x_tr.shape
        y_tr = y_tr - self.mean

        # compute kernel and jacobian
        k_tr = self.kernel(x_tr, x_tr) + self.noise_lvl * np.eye(m)
        jac_k_tr = self.kernel.jacobian() if eval_gradient else None
        l_tr = np.linalg.cholesky(k_tr)
        alpha = np.linalg.solve(l_tr.T, np.linalg.solve(l_tr, y_tr))
        self.cache.update({
            "x_tr": x_tr, "l_tr": l_tr, "alpha": alpha,
        })
        self.trained = True

        # compute marginal log-likelihood
        loglik = (-np.sum(np.log(np.diag(l_tr))) \
                  -0.5 * (y_tr @ alpha + m * np.log(2 * np.pi))) / m
        if not eval_gradient:
            return loglik

        # compute gradient
        A = np.outer(alpha, alpha) - np.linalg.inv(k_tr)
        grad = 0.5 * np.trace(A @ jac_k_tr, axis1=1, axis2=2) / m
        return loglik, grad

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
            return self.mean * np.ones(m), self.kernel(x_te, x_te)
        k_tr_te = self.kernel(self.cache["x_tr"], x_te)
        k_te = self.kernel(x_te, x_te) + self.noise_lvl * np.eye(m)
        mean = k_tr_te.T @ self.cache["alpha"]
        v = np.linalg.solve(self.cache["l_tr"], k_tr_te)
        var = k_te - v.T @ v
        return self.mean + mean, var

    def tune(self, x_tr, y_tr, bounds=(-1e4, 1e4), maxiter=150, verbose=False):
        """
        Tune the kernel to maximize marginal likelihood, using `scipy.optimize.minimize`.

        Parameters
        ----------
        x_tr: m x n array of training data
        y_tr: m-length array of training labels

        Returns
        -------
        res: OptimizationResult from scipy
        """
        def obj_fn(theta):
            self.kernel.set_theta(theta)
            marginal_loglik, grad = self.fit(x_tr, y_tr, eval_gradient=True)
            return -marginal_loglik, -grad
        bounds = sp.optimize.Bounds(*bounds)
        theta = self.kernel.get_theta()
        return sp.optimize.minimize(obj_fn, theta, bounds=bounds, jac=True, method="L-BFGS-B",
                                    tol=1e-3, options={"disp": verbose, "maxiter": maxiter})


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

    def fit(self, x_tr, y_tr, h_tr, eval_gradient=False):
        """
        Parameters
        ----------
        x_tr: m x n array of training data
        y_tr: m-length array of training labels
        h_tr: m x p array of training data
        eval_gradient: boolean whether to return the gradient wrt hyper-parameters

        Returns
        -------
        loglik: marginal log-likelihood of training data
        grad: p-length array of gradients with respect to kernel parameters theta
        """
        m, n = x_tr.shape

        # compute kernel and jacobian
        k_tr = self.kernel(x_tr, x_tr) + self.noise_lvl * np.eye(m)
        jac_k_tr = self.kernel.jacobian() if eval_gradient else None
        l_tr = np.linalg.cholesky(k_tr)
        alpha = np.linalg.solve(l_tr.T, np.linalg.solve(l_tr, y_tr))
        v = np.linalg.solve(l_tr, h_tr)
        zeta = np.linalg.inv(self.prior_prec + v.T @ v)
        beta = zeta @ (h_tr.T @ alpha + self.prior_prec @ self.prior_mean)
        self.cache.update({
            "x_tr": x_tr, "l_tr": l_tr, "h_tr": h_tr, "alpha": alpha,
            "beta": beta, "zeta": zeta,
        })
        self.trained = True

        # compute marginal loglik
        mu = h_tr @ self.prior_mean - y_tr
        sigma = k_tr + h_tr @ self.prior_var @ h_tr.T
        l_sigma = np.linalg.cholesky(sigma)
        v = np.linalg.solve(l_sigma, mu)
        loglik = (-np.sum(np.log(np.diag(l_sigma))) \
                  -0.5 * v.T @ v \
                  -0.5 * m * np.log(2 * np.pi)) / m
        if not eval_gradient:
            return loglik

        # compute gradient
        A = np.outer(alpha, alpha) - np.linalg.inv(k_tr)
        grad = 0.5 * np.trace(A @ jac_k_tr, axis1=1, axis2=2) / m
        return loglik, grad


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
        r = h_te.T - np.linalg.solve(self.cache["l_tr"], self.cache["h_tr"]).T @ \
                     np.linalg.solve(self.cache["l_tr"], k_tr_te)
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

    def tune(self, x_tr, y_tr, h_tr, bounds=(-1e4, 1e4), maxiter=150, verbose=False):
        """
        Tune the kernel to maximize marginal likelihood, using `scipy.optimize.minimize`.

        Parameters
        ----------
        x_tr: m x n array of training data
        y_tr: m-length array of training labels
        h_tr: m x p array of training data

        Returns
        -------
        res: OptimizationResult from scipy
        """
        def obj_fn(theta):
            self.kernel.set_theta(theta)
            marginal_loglik, grad = self.fit(x_tr, y_tr, h_tr, eval_gradient=True)
            return -marginal_loglik, -grad
        bounds = sp.optimize.Bounds(*bounds)
        theta = self.kernel.get_theta()
        return sp.optimize.minimize(obj_fn, theta, bounds=bounds, jac=True, method="L-BFGS-B",
                                    tol=1e-3, options={"disp": verbose, "maxiter": maxiter})
