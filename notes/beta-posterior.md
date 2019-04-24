#### Stochastic Mean GP: Posterior over Beta 

Suppose we have the following model (Rasmussen & Williams 2002).
$$
g(x) = f(x) + h(x)^\intercal\beta, \quad f(x) \sim GP(0, k(x,x')), \quad \beta\sim \mathcal{N}(b,B)
$$
Specifically, this is a GP with mean given by $h(x)^\intercal\beta$, where $\beta$ is a parameter with normal prior. 

Note that this is equivalent to the following GP, but we use close form solutions instead of naive implementation for numerical stability purposes.
$$
g(x) \sim GP(h(x)^\intercal b, k(x,x') + h(x)^\intercal B h(x'))
$$
After observing $\{(x^{(i)},y^{(i)},h^{(i)}) \}_{i=1}^n$, we have
$$
\begin{align*}
\mathbb{E}[\beta] & = (B^{-1} + HK^{-1}H^\intercal)^{-1}(HK^{-1}y + B^{-1}b)\\
\mathrm{Var}[\beta] & = (B^{-1}+HK^{-1}H^\intercal)^{-1}.
\end{align*}
$$

This gives us a principled way to compute the posterior over $\beta$.

