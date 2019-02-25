### Gaussian processes in Python

Last update: February 2019.

---

Lightweight implementation of Gaussian processes [1] in Python.

At the core, a Gaussian process is a collection of jointly Gaussian random variables specified by a mean (which below we assume to be zero) and covariance function.

<p align="center"><img src="svgs/50927beb4a6dc77c7356429e8b72204e.svg?invert_in_darkmode" align=middle width=294.63350235pt height=17.031940199999998pt/></p>

Predictions are made by conditioning on a subset of variables.

<p align="center"><img src="svgs/4b5b660767d05920b7df1d15686f169e.svg?invert_in_darkmode" align=middle width=802.3917957pt height=18.312383099999998pt/></p>

We implement as well a greedy selection algorithm for near-optimal sensor placement in Gaussian processes [2]. 

#### Usage

Todo.

For further details the `examples/` folder.

#### References

[1] Carl Edward Rasmussen and Christopher K. I. Williams. 2005. Gaussian Processes for Machine Learning (Adaptive Computation and Machine Learning). The MIT Press.

[2] Andreas Krause, Ajit Singh, and Carlos Guestrin. 2008. Near-Optimal Sensor Placements in Gaussian Processes: Theory, Efficient Algorithms and Empirical Studies. J. Mach. Learn. Res. 9 (June 2008), 235-284.

#### License

This library is available under the MIT License.
