import numpy as np
from argparse import ArgumentParser
from sklearn.datasets import load_breast_cancer, load_iris, load_boston, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score
from gp_lib.gp import ConstantMeanGP
from gp_lib.kernels import *



if __name__ == "__main__":

    np.random.seed(123)

    argparser = ArgumentParser()
    argparser.add_argument("--lr", type=float, default=1.0)
    argparser.add_argument("--r2", type=float, default=0.75)
    argparser.add_argument("--tol", type=float, default=1e-3)
    argparser.add_argument("--m", type=int, default=1000)
    argparser.add_argument("--n", type=int, default=100)
    argparser.add_argument("--verbose", action="store_true")
    args = argparser.parse_args()

    x = np.random.randn(args.m, args.n)
    x = np.c_[x, np.ones(args.m)]
    theta = np.random.randn(args.n + 1)
    y = x @ theta / args.n ** 0.5 + (1 / args.r2 - 1) ** 0.5 * np.random.randn(args.m)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y)

    # kernel = SumKernel([SEKernel(dims=np.array([i])) for i in range(x.shape[1])])
    # kernel = SEKernel()
    kernel = ProductKernel([DotProductKernel(), ConstantKernel()])
    gp = ConstantMeanGP(0, kernel, 1 - args.r2)

    prev_marginal_loglik = float("-inf")

    theta = gp.kernel.get_theta()
    for i in range(500):
        marginal_loglik, grad = gp.fit(x_tr, y_tr, eval_gradient=True)
        theta = theta + 1.0 * grad
        gp.kernel.set_theta(theta)
        print(f"Iteration {i}: {marginal_loglik:.2f}")
        if marginal_loglik - prev_marginal_loglik < args.tol:
            break
        prev_marginal_loglik = marginal_loglik

    mean, var = gp.predict(x_te)

    print(f"R2: {r2_score(y_te, mean):.2f}")
    print(f"RMSE: {((mean - y_te) ** 2).mean() ** 0.5:.2f}")
