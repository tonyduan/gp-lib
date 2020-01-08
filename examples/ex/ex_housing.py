import numpy as np
import scipy as sp
from argparse import ArgumentParser
from sklearn.datasets import load_breast_cancer, load_iris, load_boston, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score
from gp_lib.gp import ConstantMeanGP
from gp_lib.kernels import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

if __name__ == "__main__":

    np.random.seed(123)

    argparser = ArgumentParser()
    argparser.add_argument("--lr", type=float, default=1.0)
    argparser.add_argument("--verbose", action="store_true")
    argparser.add_argument("--tol", type=float, default=1e-3)
    argparser.add_argument("--noise-lvl", type=float, default=10)
    args = argparser.parse_args()

    # standardize all features so that squared exponential kernel length scales are closer in
    # magnitude to each other, and dot product kernels are scale-independent
    x, y = load_boston(True)
    m, n = x.shape
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    x = np.c_[x, np.ones(len(x))]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y)

    # scikit-learn implementation
    # gp = GaussianProcessRegressor(kernel=RBF(np.ones(14)), alpha=args.noise_lvl, normalize_y=True)
    # gp.fit(x_tr, y_tr)
    # mean, var = gp.predict(x_te, return_cov=True)
    # print(f"R2: {r2_score(y_te, mean):.2f}")
    # print(f"RMSE: {((mean - y_te) ** 2).mean() ** 0.5:.2f}")

    # anisotropic squared exponential kernel
    # no constant is needed because most we expect most of the data to fit within the
    # interpolation range anyway
    kernel = AnisotropicSEKernel(np.ones(14))
    # kernel = ProductKernel([DotProductKernel(), ConstantKernel()])
    gp = ConstantMeanGP(np.mean(y_tr), kernel, args.noise_lvl)

    # scipy optimization
    gp.tune(x_tr, y_tr, verbose=False)
    mean, var = gp.predict(x_te)
    print(f"R2: {r2_score(y_te, mean):.2f}")
    print(f"RMSE: {((mean - y_te) ** 2).mean() ** 0.5:.2f}")

    # manual optimization loop
    kernel = AnisotropicSEKernel(np.ones(14))
    gp = ConstantMeanGP(np.mean(y_tr), kernel, args.noise_lvl)
    prev_marginal_loglik = float("-inf")
    theta = kernel.get_theta()

    for i in range(1000):
        marginal_loglik, grad = gp.fit(x_tr, y_tr, eval_gradient=True)
        print(f"Iteration {i}: {marginal_loglik:.4f}")
        if marginal_loglik - prev_marginal_loglik < args.tol:
            break
        prev_marginal_loglik = marginal_loglik
        theta = theta + args.lr * grad
        kernel.set_theta(theta)

    mean, var = gp.predict(x_te)
    print(f"R2: {r2_score(y_te, mean):.2f}")
    print(f"RMSE: {((mean - y_te) ** 2).mean() ** 0.5:.2f}")
