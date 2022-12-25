### Gaussian processes

Last update: January 2020.

---

Lightweight Python library implementing Gaussian processes for regression [1].

```
pip3 install gp-lib
```

A Gaussian process specifies a collection of jointly Gaussian random variables specified by a mean (which below we assume to be zero) and covariance function between two data points.

```math
f \sim N(0, \Sigma) \quad\quad y \sim N(f, \sigma^2 I)\quad\quad \Sigma[i,j] = k(x_i, x_j)
```

Predictions are made by conditioning on a subset of variables.

```math
\begin{align*}
f|x,y,x_\ast & \sim N(\mu, \Sigma)\\
\mu & = k(x_\ast,x)(k(x,x)+\sigma^2I)^{-1}y\\
\Sigma &= k(x_\ast,x_\ast) - k(x_\ast,x)(k(x,x)+ \sigma^2I)^{-1}k(x,x_\ast),
\end{align*}
```
where $\{x, y\}$ and $\{x_\ast,y_\ast\}$ denote the train and test split respectively. 

Notice this requires inversion of an $M \times M$ matrix where $M$ is the size of the training set. Runtime is $O(M^3)$.

Suppose we use a dot product kernel $k(x_i,x_j) = x_i^\intercal x_j$. Then we recover Bayesian Linear Regression.
```math
\begin{align*}
\mu|x,y,x_\ast & = x_\ast x^\intercal(xx^\intercal + \sigma^2 I)^{-1}y & \triangleq x_\ast w
\end{align*}
```

This expression is equivalent to the "kernel trick" variant of regularized least squares (a.k.a. "ridge regression").
```math
\begin{align*}
\underset{z}{\arg\min} \frac{1}{2}\|Az- b\|^2+ \frac{\lambda}{2} \|z\|^2 & = (A^\intercal A + \lambda I)^{-1}b\\
 & = A^\intercal (AA^\intercal +\lambda I)^{-1}b,
\end{align*}
```

where the equivalency above arises from the Matrix Inversion Lemma.

#### Explicit basis functions

One unique aspect of our code is it provides out-of-the-box support for GPs with *explicit basis functions*, and the corresponding closed form solutions described in Chapter 2.7 of [1].

Specifically, we model a GP with mean given by $h(x)^\intercal\beta$, where $\beta$ is a parameter with normal prior. It can be particularly useful to specify basis functions $h(x)$ with representations learned from deep neural networks [2].
```math
g(x) = f(x) + h(x)^\intercal\beta, \quad f(x) \sim \mathrm{GP}(0, k(x,x')), \quad \beta\sim \mathcal{N}(b,B)
```
Note that this is equivalent to the following GP. However, for issues of numerical stability it is recommended to use the explicit closed form solutions that we provide instead of naively implementing the below.
```math
g(x) \sim \mathrm{GP}(h(x)^\intercal b, k(x,x') + h(x)^\intercal B h(x'))
```
Often times the mean parameter itself will be of interest. After a set of observations, its posterior distribution will be normally distributed (thanks to conjugacy of the normal distribution) with the following parameters.
```math
\begin{align*}
\mathbb{E}[\beta] & = (B^{-1} + HK^{-1}H^\intercal)^{-1}(HK^{-1}y + B^{-1}b)\\
\mathrm{Var}[\beta] & = (B^{-1}+HK^{-1}H^\intercal)^{-1}
\end{align*}
```
Below we show an example of fitting to a linear relationship with quadratic residuals.

```python
import numpy as np
from gp_lib.gp import StochasticMeanGP
from gp_lib.kernels import SEKernel

# generate some data
x = np.linspace(-1, 1, 500)[:, np.newaxis]
h = np.linspace(-1, 1, 500)[:, np.newaxis]
y = (x ** 2 + h + 0.1 * np.random.randn(*x.shape) -2).squeeze()
h = np.c_[h, np.ones_like(h)]

# zero-mean, vague prior
kernel = SEKernel(1, 1)
gp = StochasticMeanGP(prior_mean=np.array([0, 0]),
                      prior_var=5 * np.eye(2), kernel=kernel,
                      noise_lvl=0.01)

# fit and make predictions
marginal_loglik = gp.fit(x, y, h)
mean, var = gp.predict(x, y)
beta_mean, beta_var = gp.get_beta()
```

This results in the following posterior estimate.
```math
\mathbb{E}[\beta] =\begin{bmatrix} 0.844 \\-0.188\end{bmatrix}
\quad\quad
\mathrm{Var}[\beta] = \begin{bmatrix} 0.261 & -0.\\-0. &     0.464\end{bmatrix}
```

#### Supported kernels

For now we have the SE kernel and dot product kernel.
```math
K_\mathrm{SE}(x,y) = \exp\left(-\frac{1}{2\ell^2}||x-y||^2_2\right) \quad\quad K_\mathrm{dot}(x,y) = x^\intercal y
```
Hyper-parameters can be tuned via gradient ascent on the marginal log-likelihood, or cross-validation on the marginal log-likelihood.

**Sparse GPs**

We support variational learning of sparse GPs [4].

#### Optimal sensor placement

We implement as well a greedy selection algorithm for near-optimal sensor placement with Gaussian processes [3]. The intuition is that we want to pick a set of fixed size to maximize the *mutual information* between selected data points and non-selected data points.
```math
\mathcal{A} = \underset{\mathcal{A} \subset \mathcal{V}:|\mathcal{A}| = k}{\arg\max}\enspace I(\mathcal{A}; \mathcal{V} \setminus \mathcal{A})
```
This process is approximated in a greedy manner, picking out the next data point that maximizes mutual information between the selection and all remaining items.

#### Usage

The above examples results in the following prediction intervals.

![ex_model](examples/ex.png)

For further details the `examples/` folder.

#### References

[1] Carl Edward Rasmussen and Christopher K. I. Williams. 2005. Gaussian Processes for Machine Learning (Adaptive Computation and Machine Learning). The MIT Press.

[2] Jiaxuan You, Xiaocheng Li, Melvin Low, David Lobell, Stefano Ermon. Deep Gaussian Process for Crop Yield Prediction Based on Remote Sensing Data. in *Thirty-First AAAI Conference on Artificial Intelligence* (2017).

[3] Andreas Krause, Ajit Singh, and Carlos Guestrin. 2008. Near-Optimal Sensor Placements in Gaussian Processes: Theory, Efficient Algorithms and Empirical Studies. J. Mach. Learn. Res. 9 (June 2008), 235-284.

[4] Titsias, M. (2009). Variational Learning of Inducing Variables in Sparse Gaussian Processes. In International Conference on Artiﬁcial Intelligence and Statistics, pp. 567–574.

#### License

This library is available under the MIT License.
