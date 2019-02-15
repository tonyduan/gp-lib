import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from gp.gp import *
from gp.kernels import *
from gp.algs import *
from gp.utils import *


def gen_data(n=50, bound=1, deg=2, noise=0.1, intcpt=0):
    x = np.linspace(-bound, bound, n)[:, np.newaxis]
    y = (x ** deg + noise * np.random.randn(*x.shape)).squeeze()
    return x, y


def plot_predictions(x_tr, y_tr, x_te, mean, var):
    plt.figure(figsize=(8, 4))
    sds = np.sqrt(np.diag(var))
    plt.plot(x_te.squeeze(), mean, color="black")
    plt.plot(x_te.squeeze(), mean - 1.96 * sds, linestyle="--", color="grey")
    plt.plot(x_te.squeeze(), mean + 1.96 * sds, linestyle="--", color="grey")
    plt.scatter(x_tr.squeeze(), y_tr, s=10, marker="x", color="darkred")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--num-samples", default=10, type=int)
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--noise-lvl", default=0.01, type=float)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig()

    x_tr, y_tr = gen_data(n=100, deg=3, noise=args.noise_lvl)
    x_te, y_te = gen_data(n=200, deg=3, noise=0)

    # kernel = SquaredExponentialKernel(1)
    kernel = DotProductKernel()
    gp_inst = lambda: ConstantMeanGP(0, kernel, args.noise_lvl)
    idxs = pick_idxs_greedy(x_tr, y_tr, args.num_samples, gp_inst)

    gp = gp_inst()
    marginal_loglik = gp.fit(x_tr[idxs,:], y_tr[idxs])
    mean, var = gp.predict(x_te)

    print("== Greedy")
    print("R2:", r2_score(y_te, mean))
    print("Marginal log-likelihood:", marginal_loglik)

    if args.show_plots:
        for i in range(1, len(idxs)):
            plot_predictions(x_tr[idxs[:i],:], y_tr[idxs[:i]], x_te, mean, var)
            plt.show()

    idxs = np.random.choice(np.arange(x_tr.shape[0]), args.num_samples)

    gp = gp_inst()
    marginal_loglik = gp.fit(x_tr[idxs,:], y_tr[idxs])
    mean, var = gp.predict(x_te)
    print("== Random")
    print("R2:", r2_score(y_te, mean))
    print("Marginal log-likelihood:", marginal_loglik)

    if args.show_plots:
        for i in range(1, len(idxs)):
            plot_predictions(x_tr[idxs[:i],:], y_tr[idxs[:i]], x_te, mean, var)
            plt.show()
