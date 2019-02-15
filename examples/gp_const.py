import numpy as np
import matplotlib.pyplot as plt
from gp.gp import *
from gp.kernels import *


def gen_data(n=50, bound=1, deg=2, noise=0.1, intcpt=-0.5):
    x = np.linspace(-bound, bound, n)[:, np.newaxis]
    y = (x ** deg + noise * np.random.randn(*x.shape) + intcpt).squeeze()
    return x, y


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

    print("== Data Set 1")
    x_tr, y_tr = gen_data()
    x_te, _ = gen_data(100, bound=2)

    print("== Squared Exponential Kernel")
    gp = ConstantMeanGP(0, SquaredExponentialKernel(1), 0.1)
    marginal_loglik = gp.fit(x_tr, y_tr)
    mean, var = gp.predict(x_te)

    print("Marginal log-likelihood:", marginal_loglik)
    plot_predictions(x_tr, y_tr, x_te, mean, var)
    plt.show()

    print("== Dot Product Kernel")
    gp = ConstantMeanGP(0, DotProductKernel(), 0.1)
    marginal_loglik = gp.fit(x_tr, y_tr)
    mean, var = gp.predict(x_te)

    print("Marginal log-likelihood:", marginal_loglik)
    plot_predictions(x_tr, y_tr, x_te, mean, var)
    plt.show()

    print("== Data Set 2")
    x_tr, y_tr = gen_data(deg=1)
    x_te, _ = gen_data(100, bound=2)

    print("== Dot Product Kernel")
    gp = ConstantMeanGP(0, DotProductKernel(), 0.1)
    marginal_loglik = gp.fit(x_tr, y_tr)
    mean, var = gp.predict(x_te)

    print("Marginal log-likelihood:", marginal_loglik)
    plot_predictions(x_tr, y_tr, x_te, mean, var)
    plt.show()

    print("== Dot Product Kernel")
    gp = ConstantMeanGP(-0.5, DotProductKernel(), 0.1)
    marginal_loglik = gp.fit(x_tr, y_tr)
    mean, var = gp.predict(x_te)

    print("Marginal log-likelihood:", marginal_loglik)
    plot_predictions(x_tr, y_tr, x_te, mean, var)
    plt.show()
