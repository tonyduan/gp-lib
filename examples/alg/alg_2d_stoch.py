import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gp.algs import *
from gp.gp import *
from gp.kernels import *
from argparse import ArgumentParser


def gen_data(n=20, bound=1, deg=2, beta=1, noise=0.1, intcpt=-1):
    x1 = np.linspace(-bound, bound, n)
    x2 = np.linspace(-bound, bound, n)
    x = np.transpose([np.tile(x1, len(x2)), np.repeat(x2, len(x2))])
    h = np.linspace(-bound, bound, n ** 2)
    y = x[:,0] * x[:,1] + h * beta + noise * np.random.randn(*h.shape) + intcpt
    y = y.squeeze()
    return x, y, np.c_[h, np.ones_like(h)]

def plot_predictions(x_tr, y_tr, x_te, mean, var):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x_te[:,0], x_te[:,1], mean, color="lightblue", alpha=0.5,
                    linewidth=0.1)
    ax.scatter(x_tr[:,0], x_tr[:,1], y_tr, color="darkred")

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

    parser = ArgumentParser()
    parser.add_argument("--num-samples", default=20, type=int)
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--noise-lvl", default=0.25, type=float)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig()

    x_tr, y_tr, h_tr = gen_data(n=20, deg=2, noise=args.noise_lvl)
    x_te, y_te, h_te = gen_data(n=20, deg=2, noise=args.noise_lvl)

    kernel = SquaredExponentialKernel(1)
    prior_mean = np.array([0, 0])
    prior_var = np.eye(2) * 5
    gp = StochasticMeanGP(prior_mean, prior_var, kernel, args.noise_lvl ** 2)

    idxs = pick_idxs(x_tr, y_tr, args.num_samples, gp, h=h_tr, epsilon_pctle=75)

    train_loglik = gp.fit(x_tr[idxs,:], y_tr[idxs], h_tr[idxs])
    mean, var = gp.predict(x_te, h_te)
    _, pred, obs = cal_error(y_te, mean, var)
    mu, sigma = gp.get_beta()

    print("== Greedy")
    print("R2:", r2_score(y_te, mean))
    print("Train log-likelihood:", train_loglik)
    print("Test log-likelihood:", gaussian_loglik(y_te, mean, var))
    print("Posterior mu:", mu)
    print("Posterior sigma:\n", sigma)

    if args.show_plots:
        for i in range(1, len(idxs)):
            plot_predictions(x_tr[idxs[:i],:], y_tr[idxs[:i]], x_te, mean, var)
            plt.show()
        plot_calibration(pred, obs)
        plt.show()

    idxs = np.random.choice(np.arange(x_tr.shape[0]), args.num_samples)

    train_loglik = gp.fit(x_tr[idxs,:], y_tr[idxs], h_tr[idxs,:])
    mean, var = gp.predict(x_te, h_te)
    _, pred, obs = cal_error(y_te, mean, var)
    mu, sigma = gp.get_beta()

    print("== Random")
    print("R2:", r2_score(y_te, mean))
    print("Train log-likelihood:", train_loglik)
    print("Test log-likelihood:", gaussian_loglik(y_te, mean, var))
    print("Posterior mu:", mu)
    print("Posterior sigma:\n", sigma)

    if args.show_plots:
        for i in range(1, len(idxs)):
            plot_predictions(x_tr[idxs[:i],:], y_tr[idxs[:i]], x_te, mean, var)
            plt.show()
        plot_calibration(pred, obs)
        plt.show()
