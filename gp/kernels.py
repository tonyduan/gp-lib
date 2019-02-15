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


class DotProductKernel(Kernel):
    """
    Dot product

    Parameters
    """
    def __init__(self, dims=None):
        self.dims = dims

    def __call__(self, x, y):
        i = self.dims if self.dims is not None else np.arange(x.shape[1])
        return np.dot(x[:,i], y[:,i].T)


class SquaredExponentialKernel(Kernel):
    """
    Squared Exponential Kernel

    [ Equation (2.16), Rasmussen and Williams, 2006 ]

    Parameters
    ----------
    length_scale:
    dims: array
    """
    def __init__(self, length_scale, dims=None):
        self.length_scale = length_scale
        self.dims = dims

    def __call__(self, x, y):
        i = self.dims if self.dims is not None else np.arange(x.shape[1])
        dists = sp.spatial.distance.cdist(x[:,i], y[:,i], metric="sqeuclidean")
        return np.exp(-0.5 * dists / self.length_scale ** 2)
