import numpy as np
import matplotlib.pyplot as plt
from gp_lib.gp import *
from gp_lib.sparse import *
from gp_lib.kernels import *
from gp_lib.utils import *


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
    idxs = np.random.choice(np.arange(len(x_tr)), 10)
    result = gp.tune(x_tr, y_tr, x_tr[idxs], verbose=False)
    mean, var = gp.predict(x_te)
    cal, predicted, empirical = cal_error(y_te, mean, var)
    print(f"== Marginal log-likelihood lower bound: {-result.fun.squeeze():.2f}")
    print(f"== Test log-likelihoood: {gaussian_loglik(y_te, mean, var):.2f}")
    print(f"== Calibration error: {cal:.4f}")
    print(f"== Kernel: {gp.kernel}")
    plot_predictions(x_tr, y_tr, x_te, mean, var)
    ind_mean, ind_var = gp.predict(x_tr[idxs])
    ind_sd = np.sqrt(np.diag(ind_var))
    plt.scatter(x_tr[idxs].squeeze(), ind_mean, marker="x", color="red", label="Inducing")
    plt.scatter(x_tr[idxs].squeeze(), ind_mean + 1.96 * ind_sd, marker="o", color="red")
    plt.scatter(x_tr[idxs].squeeze(), ind_mean - 1.96 * ind_sd, marker="o", color="red")
    plt.scatter(x_tr[idxs].squeeze(), y_tr[idxs], marker="x", color="blue", label="Naive")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    np.random.seed(123)

    x_tr, y_tr = gen_data(100, deg=2, noise=0.1, bound=1)
    x_te, y_te = gen_data(100, deg=2, noise=0.1, bound=1)

    x_tr = x_tr / np.std(x_tr)
    x_te = x_te / np.std(x_te)
    y_tr = y_tr / np.std(y_tr)
    y_te = y_te / np.std(y_te)

    print("== Squared Exponential Kernel")
    gp = SparseGP(0, SumKernel([SEKernel(1), WhiteKernel(0.1)]), 0.01)
    summarize_gp_fit(gp, x_tr, y_tr, x_te, y_te)

    print("== Squared Exponential Kernel")
    gp = SparseGP(0, ProductKernel([SEKernel(1), ConstantKernel()]), 0.1)
    summarize_gp_fit(gp, x_tr, y_tr, x_te, y_te)

    gp = ConstantMeanGP(0, ProductKernel([SEKernel(1), ConstantKernel()]), 0.1)
    result = gp.tune(x_tr, y_tr, verbose=False)
    mean, var = gp.predict(x_te)
    print(f"== Marginal log-likelihood lower bound: {-result.fun.squeeze():.2f}")
    print(f"== Test log-likelihoood: {gaussian_loglik(y_te, mean, var):.2f}")
    print(f"== Kernel: {gp.kernel}")
    plot_predictions(x_tr, y_tr, x_te, mean, var)
    plt.show()

