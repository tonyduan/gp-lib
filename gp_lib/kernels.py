import numpy as np
import scipy as sp
import scipy.spatial
from functools import reduce


class Kernel(object):

    def __call__(self, x, y):
        """
        Returns
        -------
        kernel: m x n array
        """
        raise NotImplementedError

    def trace_x_x(self, x):
        """
        Returns tr(k(x, x))
        -------
        trace: scalar
        """
        raise NotImplementedError

    def get_theta(self):
        """
        Returns
        -------
        theta: p-length array of kernel hyperparameters
        """
        raise NotImplementedError

    def set_theta(self, theta):
        """
        Parameters
        ----------
        theta: p-length array of kernel hyperparameters
        """
        raise NotImplementedError

    def jacobian(self):
        """
        Returns
        -------
        jacobian: p x m x m array of kernel gradient with respect to each of the p parameters
        """
        raise NotImplementedError


class SumKernel(Kernel):
    """
    Sum of multiple kernels.

    Parameters
    ----------
    kernels: list of Kernel objects
    """
    def __init__(self, kernels):
        self.kernels = kernels
        self.n_parameters = sum([k.n_parameters for k in kernels])

    def __call__(self, x, y):
        return np.sum([k(x, y) for k in self.kernels], axis=0)

    def __repr__(self):
        return "(" + " + ".join([k.__repr__() for k in self.kernels]) + "])" + ")"

    def trace_x_x(self, x):
        return np.sum([k.trace_x_x(x) for k in self.kernels])

    def get_theta(self):
        return np.hstack([k.get_theta() for k in self.kernels if k.n_parameters > 0])

    def set_theta(self, theta):
        ptr = 0
        for k in filter(lambda k: k.n_parameters > 0, self.kernels):
            k.set_theta(theta[ptr:ptr + k.n_parameters])
            ptr += k.n_parameters

    def jacobian(self):
        return np.vstack([k.jacobian() for k in self.kernels if k.n_parameters > 0])


class ProductKernel(Kernel):
    """
    Product of multiple kernels.

    Parameters
    ----------
    kernels: list of Kernel objects
    """
    def __init__(self, kernels):
        self.kernels = kernels
        self.n_parameters = sum([k.n_parameters for k in kernels])
        self.cache = {}

    def __call__(self, x, y):
        self.cache["k"] = np.prod([k(x, y) for k in self.kernels], axis=0)
        return self.cache["k"]

    def __repr__(self):
        return "(" + " * ".join([k.__repr__() for k in self.kernels]) + "])" + ")"

    def trace_x_x(self, x):
        return np.prod([k.trace_x_x(x) for k in self.kernels])

    def get_theta(self):
        return np.hstack([k.get_theta() for k in self.kernels if k.n_parameters > 0])

    def set_theta(self, theta):
        ptr = 0
        for k in filter(lambda k: k.n_parameters > 0, self.kernels):
            k.set_theta(theta[ptr:ptr + k.n_parameters])
            ptr += k.n_parameters

    def jacobian(self):
        return np.vstack([self.cache["k"] / (k.cache["k"] + 1e-4) * k.jacobian() \
                          for k in self.kernels if k.n_parameters > 0])


class ConstantKernel(Kernel):
    """
    Constant kernel.
        k(x, y) = c

    Parameters
    ----------
    c: float
    """
    n_parameters = 1

    def __init__(self, c=1.0):
        self.c = c
        self.cache = {}

    def __call__(self, x, y):
        self.cache["k"] = np.ones((len(x), len(y))) * self.c
        return self.cache["k"]

    def __repr__(self):
        return f"ConstantKernel({self.c:.2f})"

    def trace_x_x(self, x):
        return x.shape[0] * self.c

    def get_theta(self):
        return np.array([np.log(self.c)])

    def set_theta(self, theta):
        self.c = np.exp(theta)

    def jacobian(self):
        return np.array([self.cache["k"]])


class SEKernel(Kernel):
    """
    Squared exponential kernel.
        k(x, y) = exp(-0.5 || x - y ||² / l²)

    [ Equation (2.16), Rasmussen and Williams, 2006 ]

    Parameters
    ----------
    length_scale: float

    dims: variable-length array of indices to specify which dimensions to include in calculation;
          if None, default to all dimensions
    """
    n_parameters = 1

    def __init__(self, length_scale=1.0, dims=None):
        self.length_scale = length_scale
        self.dims = dims
        self.cache = {}

    def __call__(self, x, y):
        i = self.dims if self.dims is not None else np.arange(x.shape[1])
        dists = sp.spatial.distance.cdist(x[:,i], y[:,i], metric="sqeuclidean")
        self.cache["k"] = np.exp(-0.5 * dists / self.length_scale)
        return self.cache["k"]

    def __repr__(self):
        return f"SEKernel(length_scale={self.length_scale:.2f}, " \
               f"dims={self.dims})"

    def trace_x_x(self, x):
        return x.shape[0]

    def get_theta(self):
        return np.array([np.log(self.length_scale)])

    def set_theta(self, theta):
        self.length_scale = np.exp(theta[0])

    def jacobian(self):
        return np.array([-self.cache["k"] * np.log(self.cache["k"] + 1e-4)])


class AnisotropicSEKernel(Kernel):
    """
    Anisotropic squared exponential kernel.

    Parameters
    ----------
    length_scale: p-length array

    dims: variable-length array of indices to specify which dimensions to include in calculation;
          if None, default to all dimensions
    """
    def __init__(self, length_scale=None):
        self.length_scale = length_scale
        self.length_scale_ext = self.length_scale[:, np.newaxis, np.newaxis]
        self.n_parameters = len(length_scale)
        self.cache = {}

    def __call__(self, x, y):
        self.cache["dists"] = np.stack([(x[:, i, np.newaxis] - y[:, i, np.newaxis].T) ** 2 \
                                        for i in range(self.n_parameters)])
        self.cache["k"] = np.exp(-0.5 * (self.cache["dists"] / self.length_scale_ext).sum(axis=0))
        return self.cache["k"]

    def __repr__(self):
        return f"AnisotropicSEKernel(length_scale={np.array2string(self.length_scale, precision=1)})"

    def trace_x_x(self, x):
        return x.shape[0]

    def get_theta(self):
        return np.log(self.length_scale)

    def set_theta(self, theta):
        self.length_scale = np.exp(theta) + 1e-4
        self.length_scale_ext = self.length_scale[:, np.newaxis, np.newaxis]

    def jacobian(self):
        return 0.5 * self.cache["k"] * self.cache["dists"] / self.length_scale_ext


class DotProductKernel(Kernel):
    """
    Dot product kernel.
        k(x, y) = xᵀy

    Parameters
    ----------
    dims: variable-length array of indices to specify which dimensions to include in calculation;
          if None, default to all dimensions
    """
    n_parameters = 0

    def __init__(self, dims=None):
        self.dims = dims

    def __call__(self, x, y):
        i = self.dims if self.dims is not None else np.arange(x.shape[1])
        return np.dot(x[:,i], y[:,i].T)

    def __repr__(self):
        return f"DotProductKernel(sigma_sq={self.sigma_sq:.2f}, " \
               f"dims={self.dims})"

    def trace_x_x(self, x):
        return np.sum(x ** 2)

    def get_theta(self):
        raise ValueError("DotProductKernel takes no parameters.")

    def set_theta(self, theta):
        raise ValueError("DotProductKernel takes no parameters.")

    def jacobian(self):
        raise ValueError("DotProductKernel takes no parameters.")
