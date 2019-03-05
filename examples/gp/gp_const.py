import numpy as np
import matplotlib.pyplot as plt
from gp.gp import *
from gp.kernels import *
from gp.utils import *


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


def plot_calibration(predicted, empirical):
    slope, intcpt = np.polyfit(predicted, empirical, deg=1)
    x = np.linspace(0, 1)
    y = x * slope + intcpt
    plt.figure(figsize=(6, 6))
    plt.scatter(predicted, empirical, marker="x", color="black")
    plt.plot(x, x, linestyle="--", color="grey")
    plt.plot(x, y, linestyle="--", color="darkgreen")
    plt.xlim(0, 1)
    plt.ylim(0, 1)


def summarize_gp_fit(gp, x_tr, y_tr, x_te, y_te):
    marginal_loglik = gp.fit(x_tr, y_tr)
    mean, var = gp.predict(x_te)
    cal, predicted, empirical = cal_error(y_te, mean, var)
    print("Marginal log-likelihood:", marginal_loglik)
    print("Test log-likelihoood:", gaussian_loglik(y_te, mean, var))
    print("Calibration error:", cal)
    plot_predictions(x_tr, y_tr, x_te, mean, var)
    plt.show()
    plot_calibration(predicted, empirical)
    plt.show()


if __name__ == "__main__":

    print("== Data Set 1")
    x_tr, y_tr = gen_data(500, deg=2, noise=0.1, bound=1)
    x_te, y_te = gen_data(500, deg=2, noise=0.1, bound=1)

    print("== Squared Exponential Kernel")
    gp = ConstantMeanGP(0, SquaredExponentialKernel(1, 1), 0.01)
    summarize_gp_fit(gp, x_tr, y_tr, x_te, y_te)

    print("== Data Set 2")
    x_tr, y_tr = gen_data(5, deg=1, bound=1, intcpt=-1)
    x_te, y_te = gen_data(100, deg=1, bound=2, intcpt=-1)

    print("== Dot Product Kernel")
    gp = ConstantMeanGP(-1, DotProductKernel(), 0.01)
    summarize_gp_fit(gp, x_tr, y_tr, x_te, y_te)
