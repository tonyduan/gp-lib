import numpy as np
import scipy as sp
import scipy.spatial


class Kernel(object):
    def __call__(self, x, y):
        raise NotImplementedError


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

    def __call__(x, y):
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

    def __call__(x, y):
        return k1(x, y) * k2(x, y)


class SquaredExponentialKernel(Kernel):
    """
    Squared exponential kernel.
        k(x, y) = exp(-0.5 || x - y ||² / l²)

    [ Equation (2.16), Rasmussen and Williams, 2006 ]

    Parameters
    ----------
    length_scale: float

    dims: array of indices
        specify which dimensions of x and y to include in kernel calculation;
        if None, default to all dimensions
    """
    def __init__(self, length_scale, sigma_sq=1, dims=None):
        self.length_scale = length_scale
        self.sigma_sq = sigma_sq
        self.dims = dims

    def __call__(self, x, y):
        i = self.dims if self.dims is not None else np.arange(x.shape[1])
        dists = sp.spatial.distance.cdist(x[:,i], y[:,i], metric="sqeuclidean")
        return self.sigma_sq * np.exp(-0.5 * dists / self.length_scale ** 2)


class DotProductKernel(Kernel):
    """
    Dot product kernel.
        k(x, y) = xᵀy

    Parameters
    ----------
    dims: array of indices
        specify which dimensions of x and y to include in kernel calculation;
        if None, default to all dimensions
    """
    def __init__(self, dims=None):
        self.dims = dims

    def __call__(self, x, y):
        i = self.dims if self.dims is not None else np.arange(x.shape[1])
        return np.dot(x[:,i], y[:,i].T)
