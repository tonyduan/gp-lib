import numpy as np
import scipy as sp
import scipy.spatial


class Kernel(object):
    def __call__(self, x, y):
        raise NotImplementedError

    def get_gradient_update(self, cache, grad_wrt_kernel_param):
        A = np.outer(cache["alpha"], cache["alpha"]) - cache["k_tr_inv"]
        m, m = cache["k_tr_inv"].shape
        return 0.5 * np.trace(A @ grad_wrt_kernel_param) / m


class SumKernel(Kernel):
    """
    Sum of two kernels.

    Parameters
    ----------
    k1: Kernel
    k2: Kernel
    """
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __call__(self, x, y):
        return k1(x, y) + k2(x, y)


class ProductKernel(Kernel):
    """
    Product of two kernels.

    Parameters
    ----------
    k1: Kernel
    k2: Kernel
    """
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __call__(self, x, y):
        return self.k1(x, y) * self.k2(x, y)


class SquaredExponentialKernel(Kernel):
    """
    Squared exponential kernel.
        k(x, y) = σ²exp(-0.5 || x - y ||² / l²)

    [ Equation (2.16), Rasmussen and Williams, 2006 ]

    Parameters
    ----------
    length_scale: float

    dims: array of indices
        specify which dimensions of x and y to include in kernel calculation;
        if None, default to all dimensions
    """
    def __init__(self, length_scale=1, sigma_sq=1, dims=None):
        self.length_scale = length_scale
        self.sigma_sq = sigma_sq
        self.dims = dims

    def __call__(self, x, y):
        i = self.dims if self.dims is not None else np.arange(x.shape[1])
        dists = sp.spatial.distance.cdist(x[:,i], y[:,i], metric="sqeuclidean")
        return self.sigma_sq * np.exp(-0.5 * dists / self.length_scale)

    def __repr__(self):
        return f"SquaredExponentialKernel(" \
               f"length_scale={self.length_scale:.2f}, " \
               f"sigma_sq={self.sigma_sq:.2f}, dims={self.dims})"

    def gradient_update(self, cache):
        g_sigma_sq = self.get_gradient_update(cache, cache["k_tr"])
        g_length_scale = self.get_gradient_update(cache, -cache["k_tr"] *
                np.log(cache["k_tr"] / self.sigma_sq))
        self.sigma_sq = np.exp(np.log(self.sigma_sq) + g_sigma_sq)
        self.length_scale = np.exp(np.log(self.length_scale) + g_length_scale)


class DotProductKernel(Kernel):
    """
    Dot product kernel.
        k(x, y) = σ²xᵀy

    Parameters
    ----------
    dims: array of indices
        specify which dimensions of x and y to include in kernel calculation;
        if None, default to all dimensions
    """
    def __init__(self, sigma_sq=1, dims=None):
        self.sigma_sq = sigma_sq
        self.dims = dims

    def __call__(self, x, y):
        i = self.dims if self.dims is not None else np.arange(x.shape[1])
        return self.sigma_sq * np.dot(x[:,i], y[:,i].T)

    def __repr__(self):
        return f"DotProductKernel(sigma_sq={self.sigma_sq:.2f}, " \
               f"dims={self.dims})"

    def gradient_update(self, cache):
        g_sigma_sq = self.get_gradient_update(cache, cache["k_tr"])
        self.sigma_sq = np.exp(np.log(self.sigma_sq) + g_sigma_sq)
