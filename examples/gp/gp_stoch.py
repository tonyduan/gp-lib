import numpy as np
import matplotlib.pyplot as plt
from gp_lib.gp import *
from gp_lib.kernels import *
from gp_lib.utils import *


def gen_data(n=50, bound=1, deg=2, beta=1, noise=0.1, intcpt=-1):
    x = np.linspace(-bound, bound, n)[:, np.newaxis]
    h = np.linspace(-bound, bound, n)[:, np.newaxis]
    y = x ** deg + h * beta + noise * np.random.randn(*x.shape) + intcpt
    y = y.squeeze()
    return x, y, np.c_[h, np.ones_like(h)]


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


if __name__ == "__main__":

    x_tr, y_tr, h_tr = gen_data(200, deg=2, noise=0.1, intcpt=-2)
    x_te, y_te, h_te = gen_data(200, deg=2, noise=0.1, bound=1, intcpt=-2)

    print("== Squared Exponential Kernel")
    gp = StochasticMeanGP(np.array([0, 0]), 5 * np.eye(2),
                          SquaredExponentialKernel(1), 0.01)
    marginal_loglik = gp.fit(x_tr, y_tr, h_tr)
    mean, var = gp.predict(x_te, h_te)
    print(f"Train likelihood: {marginal_loglik:.4f}")
    print(f"Test likelihood: {gaussian_loglik(y_te, mean, var):.4f}")

    beta_mean, beta_var = gp.get_beta()
    print("Posterior beta:", )
    print(f"Mean: {beta_mean.round(3)}")
    print(f"Var:\n{beta_var.round(3)}")

    plot_predictions(x_tr, y_tr, x_te, mean, var)
    plt.show()

    cal, pred, obs = cal_error(y_te, mean, var)
    print(f"Calibration: {cal:.4f}")
    plot_calibration(pred, obs)
    plt.show()
