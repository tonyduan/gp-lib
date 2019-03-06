### Gaussian processes in Python

Last update: February 2019.

---

Lightweight implementation of Gaussian processes [1] for regression in Python.

A Gaussian process is a collection of jointly Gaussian random variables specified by a mean (which below we assume to be zero) and covariance function between two data points.

<p align="center"><img alt="$$&#10;p(Y|X) \sim \mathcal{N}(0, \Sigma) \quad \quad \Sigma[i,j] = K(x_i, x_j)&#10;$$" src="svgs/50927beb4a6dc77c7356429e8b72204e.svg" align="middle" width="294.63350235pt" height="17.031940199999998pt"/></p>

Predictions are made by conditioning on a subset of variables.

<p align="center"><img alt="$$&#10;p(Y|X',Y',X) \sim \mathcal{N}(\mu, \Sigma)\quad\quad \mu = K(X,X')K(X',X')^{-1}Y, \quad\quad\Sigma = K(X,X) - K(X,X')K(X',X')^{-1}K(X',X)&#10;$$" src="svgs/4b5b660767d05920b7df1d15686f169e.svg" align="middle" width="802.3917957pt" height="18.312383099999998pt"/></p>

---

We implement as well a greedy selection algorithm for near-optimal sensor placement in Gaussian processes [2]. The intuition is that we want to pick a set of fixed size to maximize the *mutual information* between selected data points and remaining items.
<p align="center"><img alt="$$&#10;\mathcal{A} = \underset{\mathcal{A} \subset \mathcal{V}:|\mathcal{A}| = k}{\arg\max}\enspace I(\mathcal{A}; \mathcal{V} \setminus \mathcal{A})&#10;$$" src="svgs/da12add3bd7a6c02b827fc1db32c4183.svg" align="middle" width="194.59628759999998pt" height="29.771669399999997pt"/></p>
This process is approximated in a greedy manner.

#### Usage

Todo.

For further details the `examples/` folder.

#### References

[1] Carl Edward Rasmussen and Christopher K. I. Williams. 2005. Gaussian Processes for Machine Learning (Adaptive Computation and Machine Learning). The MIT Press.

[2] Andreas Krause, Ajit Singh, and Carlos Guestrin. 2008. Near-Optimal Sensor Placements in Gaussian Processes: Theory, Efficient Algorithms and Empirical Studies. J. Mach. Learn. Res. 9 (June 2008), 235-284.

#### License

This library is available under the MIT License.
