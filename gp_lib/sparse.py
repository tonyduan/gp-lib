import numpy as np
import scipy as sp
from gp_lib.gp import GaussianProcess


class SparseGP(GaussianProcess):
    """
    Sparse constant-mean GP fit with variational inference.

    [ Equations (6) and (10), Titsias 2009 ]

    Parameters
    ----------
    mean: float of constant mean to be subtracted from labels
    kernel: Kernel used (incororates hyperparameters)
    noise_lvl: float hyper-parameter, level of noise in observations
    """
    def __init__(self, mean, kernel, noise_lvl):
        super().__init__(kernel, noise_lvl)
        self.mean = mean

    def fit(self, x_tr, y_tr, x_ind, eval_gradient=False):
        """
        Parameters
        ----------
        x_tr: m x n array of training data
        y_tr: m-length array of training labels
        eval_gradient: boolean whether to return the gradient wrt hyper-parameters

        Returns
        -------
        lower_bound: scalar lower bound on the marginal log-likelihood of observed data
        grad: p-length array of gradients with respect to kernel parameters theta
        """
        m, n = x_tr.shape
        k, _ = x_ind.shape
        y_tr = y_tr - self.mean

        # compute kernels and jacobians
        k_ind = self.kernel(x_ind, x_ind)
        jac_k_ind = self.kernel.jacobian() if eval_gradient else None
        k_tr_ind = self.kernel(x_tr, x_ind)
        jac_k_tr_ind = self.kernel.jacobian() if eval_gradient else None

        # compute posterior mean, var over inducing points for downstream prediction
        psi_inv = k_ind + k_tr_ind.T @ k_tr_ind / self.noise_lvl
        l_psi = np.linalg.cholesky(psi_inv)
        y_ind = k_ind @ np.linalg.solve(l_psi.T, np.linalg.solve(l_psi, k_tr_ind.T @ y_tr))
        y_ind = y_ind / self.noise_lvl
        l_ind = np.linalg.cholesky(k_ind)
        alpha_ind = np.linalg.solve(l_ind.T, np.linalg.solve(l_ind, y_ind))
        self.cache.update({
            "x_ind": x_ind, "l_ind": l_ind, "alpha_ind": alpha_ind, "l_psi": l_psi,
        })
        self.trained = True

        # compute variational lower bound
        k_ind_inv = np.linalg.inv(k_ind)
        alpha_tr = np.linalg.solve(l_psi, k_tr_ind.T @ y_tr)
        lower_bound = (-0.5 * m * np.log(2 * np.pi) \
                       -0.5 * (m - k) * np.log(self.noise_lvl) \
                       +np.sum(np.log(np.diag(l_ind))) \
                       -0.5 * y_tr.T @ y_tr / self.noise_lvl \
                       -0.5 * self.kernel.trace_x_x(x_tr) / self.noise_lvl \
                       -0.5 * np.trace(k_ind_inv @ k_tr_ind.T @ k_tr_ind) / self.noise_lvl \
                       -np.sum(np.log(np.diag(self.noise_lvl ** 0.5 * l_psi))) \
                       +0.5 * alpha_tr.T @ alpha_tr / self.noise_lvl ** 2) / m
        if not eval_gradient:
            return lower_bound

        # compute gradient with respect to kernel hyper-parameters
        tmp1 = np.linalg.solve(l_psi.T, alpha_tr) / self.noise_lvl
        tmp2 = (k_ind_inv - np.linalg.inv(l_psi @ l_psi.T) - np.outer(tmp1, tmp1)) / self.noise_lvl
        tmp3 = k_tr_ind @ k_ind_inv / self.noise_lvl
        tmp4 = tmp2 - tmp3.T @ tmp3
        tmp5 = np.outer(tmp1, y_tr) / self.noise_lvl
        grad1 = 0.5 * self.noise_lvl * np.trace(jac_k_ind @ tmp4, axis1=1, axis2=2)
        grad2 = np.trace(jac_k_tr_ind @ (tmp2 @ k_tr_ind.T + tmp5), axis1=1, axis2=2)
        return lower_bound, (grad1 + grad2) / m

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
        k_ind_te = self.kernel(self.cache["x_ind"], x_te)
        k_te = self.kernel(x_te, x_te) + self.noise_lvl * np.eye(m)
        mean = k_ind_te.T @ self.cache["alpha_ind"]
        v = np.linalg.solve(self.cache["l_ind"], k_ind_te)
        w = np.linalg.solve(self.cache["l_psi"], k_ind_te)
        var = k_te - v.T @ v + w.T @ w
        return self.mean + mean, var

    def tune(self, x_tr, y_tr, x_ind, bounds=(-1e4, 1e4), maxiter=150, verbose=False):
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
            marginal_loglik, grad = self.fit(x_tr, y_tr, x_ind, eval_gradient=True)
            return -marginal_loglik, -grad
        bounds = sp.optimize.Bounds(*bounds)
        theta = self.kernel.get_theta()
        return sp.optimize.minimize(obj_fn, theta, bounds=bounds, jac=True, method="L-BFGS-B",
                                    tol=1e-3, options={"disp": verbose, "maxiter": maxiter})
