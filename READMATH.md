### Gaussian processes

Last update: April 2019.

---

Lightweight Python implementation of Gaussian processes [1] for regression.

A Gaussian process is a collection of jointly Gaussian random variables specified by a mean (which below we assume to be zero) and covariance function between two data points.

$$
p(Y|X) \sim \mathcal{N}(0, \Sigma) \quad \quad \Sigma[i,j] = K(x_i, x_j)
$$

Predictions are made by conditioning on a subset of variables.

$$
p(Y|X',Y',X) \sim \mathcal{N}(\mu, \Sigma)\quad\quad \mu = K(X,X')K(X',X')^{-1}Y, \quad\quad\Sigma = K(X,X) - K(X,X')K(X',X')^{-1}K(X',X)
$$

#### Explicit basis functions

What sets our code apart is it provides out-of-the-box support for GPs with *explicit basis functions*, and the corresponding closed form solutions described in Chapter 2.7 of [1]. 

Specifically, we model a GP with mean given by $h(x)^\intercal\beta$, where $\beta$ is a parameter with normal prior. It can be particularly useful to specify basis functions $h(x)$ with representations learned from deep neural networks [2].
$$
g(x) = f(x) + h(x)^\intercal\beta, \quad f(x) \sim GP(0, k(x,x')), \quad \beta\sim \mathcal{N}(b,B)
$$
Note that this is equivalent to the following GP. However, for issues of numerical stability it is recommended to use the explicit closed form solutions that we provide instead of naively implementing the below.
$$
g(x) \sim GP(h(x)^\intercal b, k(x,x') + h(x)^\intercal B h(x'))
$$
Often times the mean parameter itself will be of interest. After a set of observations, its posterior distribution will be normally distributed (thanks to conjugacy of the normal distribution) with the following parameters.
$$
\begin{align*}
\mathbb{E}[\beta] & = (B^{-1} + HK^{-1}H^\intercal)^{-1}(HK^{-1}y + B^{-1}b)\\
\mathrm{Var}[\beta] & = (B^{-1}+HK^{-1}H^\intercal)^{-1}.
\end{align*}
$$

#### Supported kernels

Squared exponential kernel.
$$
K_\mathrm{SE}(x,y) = \sigma^2\exp\left(-\frac{1}{2\ell^2}||x-y||^2_2\right)
$$
Dot product kernel.
$$
K_\mathrm{dot}(x,y) = \sigma^2 x^\intercal y
$$
Hyper-parameters can be tuned via gradient ascent on the marginal log-likelihood, or cross-validation on the marginal log-likelihood.

#### Optimal sensor placement

We implement as well a greedy selection algorithm for near-optimal sensor placement with Gaussian processes [3]. The intuition is that we want to pick a set of fixed size to maximize the *mutual information* between selected data points and non-selected data points.
$$
\mathcal{A} = \underset{\mathcal{A} \subset \mathcal{V}:|\mathcal{A}| = k}{\arg\max}\enspace I(\mathcal{A}; \mathcal{V} \setminus \mathcal{A})
$$
This process is approximated in a greedy manner, picking out the next data point that maximizes mutual information between the selection and all remaining items. See the file `gp/algs.py` for implementation details.

#### Usage

```python
from gp.gp import ConstantMeanGP

# parameters are mean, kernel, noise level
gp = ConstantMeanGP(0, SquaredExponentialKernel(10, 1), noise_lvl=0.01)

# tune for optimal hyper-parameters via gradient ascent
for i in range(20):
    marginal_loglik = gp.fit(x_tr, y_tr)
    gp.gradient_update()
    print(f"Iteration {i}: {marginal_loglik:.2f}")
print("== Kernel:", gp.kernel)

# predicted posterior mean and variance
mean, var = gp.predict(x_te)
```

Below we show an example of fitting a quadratic function with a squared exponential kernel and the corresponding confidence interval.

![ex_model](svgs/ex.png)

For further details the `examples/` folder.

#### References

[1] Carl Edward Rasmussen and Christopher K. I. Williams. 2005. Gaussian Processes for Machine Learning (Adaptive Computation and Machine Learning). The MIT Press.

[2] Jiaxuan You, Xiaocheng Li, Melvin Low, David Lobell, Stefano Ermon. Deep Gaussian Process for Crop Yield Prediction Based on Remote Sensing Data. in *Thirty-First AAAI Conference on Artificial Intelligence* (2017).

[3] Andreas Krause, Ajit Singh, and Carlos Guestrin. 2008. Near-Optimal Sensor Placements in Gaussian Processes: Theory, Efficient Algorithms and Empirical Studies. J. Mach. Learn. Res. 9 (June 2008), 235-284.

#### License

This library is available under the MIT License.
