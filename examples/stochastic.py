import numpy as np
import matplotlib.pyplot as plt
from gp.gp import *
from gp.kernels import *


def gen_data(n=50, bound=1, deg=2, beta=1, noise=0.1, intcpt=-0.5):
    x = np.linspace(-bound, bound, n)[:, np.newaxis]
    h = np.linspace(-bound, bound, n)[:, np.newaxis]
    y = x ** deg + h * beta + noise * np.random.randn(*x.shape) + intcpt
    y = y.squeeze()
    return x, y, h


def plot_predictions(x_tr, y_tr, x_te, mean, var):
    plt.figure(figsize=(8, 4))
    sds = np.sqrt(np.diag(var))
    plt.axhline(0, linestyle="--", color="grey")
    plt.axvline(0, linestyle="--", color="grey")
    plt.plot(x_te.squeeze(), mean, color="black")
    plt.plot(x_te.squeeze(), mean - 1.96 * sds, linestyle="--", color="black")
    plt.plot(x_te.squeeze(), mean + 1.96 * sds, linestyle="--", color="black")
    plt.scatter(x_tr.squeeze(), y_tr, s=3, marker="x", color="black")


if __name__ == "__main__":

    x_tr, y_tr, h_tr = gen_data(100, noise=0.01)
    x_te, y_te, h_te = gen_data(100, bound=2)

    print("== Squared Exponential Kernel")
    gp = StochasticMeanGP(np.array([0]), 5 * np.eye(1),
                          SquaredExponentialKernel(1), 0.1)
    marginal_loglik = gp.fit(x_tr, y_tr, h_tr)
    mean, var = gp.predict(x_te, h_te)

    gp.get_posterior_beta()
    print("Posterior beta:", )

    plot_predictions(x_tr, y_tr, x_te, mean, var)
    plt.show()
