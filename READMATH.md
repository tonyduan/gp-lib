### Gaussian processes in Python

Last update: February 2019.

---

Lightweight implementation of Gaussian processes [1] in Python.

At the core, a Gaussian process is a collection of jointly Gaussian random variables specified by a mean (which below we assume to be zero) and covariance function.

$$
p(Y|X) \sim \mathcal{N}(0, \Sigma) \quad \quad \Sigma[i,j] = K(x_i, x_j)
$$

Predictions are made by conditioning on a subset of variables.

$$
p(Y|X',Y',X) \sim \mathcal{N}(\mu, \Sigma)\quad\quad \mu = K(X,X')K(X',X')^{-1}Y, \quad\quad\Sigma = K(X,X) - K(X,X')K(X',X')^{-1}K(X',X)
$$

We implement as well a greedy selection algorithm for near-optimal sensor placement in Gaussian processes [2]. 

#### Usage

Todo.

For further details the `examples/` folder.

#### References

[1] Carl Edward Rasmussen and Christopher K. I. Williams. 2005. Gaussian Processes for Machine Learning (Adaptive Computation and Machine Learning). The MIT Press.

[2] Andreas Krause, Ajit Singh, and Carlos Guestrin. 2008. Near-Optimal Sensor Placements in Gaussian Processes: Theory, Efficient Algorithms and Empirical Studies. J. Mach. Learn. Res. 9 (June 2008), 235-284.

#### License

This library is available under the MIT License.
