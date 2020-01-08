import numpy as np
import scipy as sp
from argparse import ArgumentParser
from sklearn.datasets import load_breast_cancer, load_iris, load_boston, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score
from gp_lib.gp import ConstantMeanGP
from gp_lib.sparse import SparseGP
from gp_lib.kernels import *


if __name__ == "__main__":

    np.random.seed(123)

    argparser = ArgumentParser()
    argparser.add_argument("--lr", type=float, default=0.1)
    argparser.add_argument("--verbose", action="store_true")
    argparser.add_argument("--tol", type=float, default=1e-3)
    args = argparser.parse_args()

    # standardize all features so that squared exponential kernel length scales are closer in
    # magnitude to each other, and dot product kernels are scale-independent
    x, y = load_boston(True)
    idxs = np.arange(len(x))
    np.random.shuffle(idxs)
    x = x[idxs]
    y = y[idxs]
    m, n = x.shape
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    x = np.c_[x, np.ones(len(x))]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y)

    # anisotropic squared exponential kernel
    # no constant is needed because most we expect most of the data to fit within the
    # interpolation range anyway
    noise_lvl = 1.0
    kernel = AnisotropicSEKernel(np.ones(14))
    gp = ConstantMeanGP(np.mean(y_tr), kernel, noise_lvl)

    # train using all the data
    result = gp.tune(x_tr, y_tr, verbose=False)
    mean, var = gp.predict(x_te)
    print(f"Loglik: {-result.fun:.2f}")
    print(f"R2: {r2_score(y_te, mean):.2f}")
    print(f"RMSE: {((mean - y_te) ** 2).mean() ** 0.5:.2f}")

    # train using 1/10-th of the data as variational inputs, using the previous kernel
    print("===")
    gp = SparseGP(np.mean(y_tr), kernel, noise_lvl)
    lower_bound = gp.fit(x_tr, y_tr, x_tr[:20], eval_gradient=False)
    mean, var = gp.predict(x_te)
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"R2: {r2_score(y_te, mean):.2f}")
    print(f"RMSE: {((mean - y_te) ** 2).mean() ** 0.5:.2f}")

    # train using 1/10-th of the data as variational inputs, learn kernel from scratch
    print("===")
    kernel = AnisotropicSEKernel(np.ones(14))
    gp = SparseGP(np.mean(y_tr), kernel, noise_lvl)
    result = gp.tune(x_tr, y_tr, x_tr[:20], verbose=False)
    mean, var = gp.predict(x_te)
    print(f"Lower bound: {-result.fun:.2f}")
    print(f"R2: {r2_score(y_te, mean):.2f}")
    print(f"RMSE: {((mean - y_te) ** 2).mean() ** 0.5:.2f}")
