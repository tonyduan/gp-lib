### Variational Learning of Sparse Gaussian Processes

---

Suppose the GP has function values $f$ and we are given training data $\{(x^{(i)},y^{(i)})\}_{i=1}^m$.

We want to maximize the marginal likelihood of the given data, optimizing over inducing points $\{(z^{(i)},u^{(i)})\}_{i=1}^k$. We'll hypothesize the latent variable model $u \leftarrow f \rightarrow y$, noting that $u \perp y | f$.

Standard results show that with this model,
```math
\begin{align*}
p(y|f) & = N(y;f,\sigma^2 I)\\
p(f|u) & = N(f;\mu, K),
\end{align*}
```
where $K=K_{m,m}-K_{m,k}K_{k,k}^{-1} K_{k,m}$ and $\mu = K_{m,k}K_{k,k}^{-1}u$.

**Sparse Gaussian Processes**

If we assume that $u$ is sufficient for $f$ (and therefore sufficient for $y$),
```math
\begin{align*}
p(z|y) & = \int_u \int_f p(z|u,f,y)p(f|u,y)p(u|y)df dfu \tag{introduce latent $u,f$}\\
 & = \int_u\int_f p(z|u)p(f|y)p(u|y)df du \\
 & = \int_up(z|u)p(u|y)du\\
 & = \mathbb{E}_{u\sim p(u|y)}[p(z|u)]
\end{align*}
```
In practice we'll use a variational distribution $q(u)$ to approximate $p(u|y)$.

Then prediction will proceed as if we trained using labels $u$ and training points $z$ with kernel
```math
\begin{align*}
& = -K_{xm}K_{mm}^{-1}K_{mx}+ K_{xm}(K_{m,m}+\sigma^{-2}K_{m,n}K_{n,m})^{-1}K_{mx} \\
& = -K_{xm}(K_{mm}^{-1} - (K_{m,m}+\sigma^{-2}K_{m,n}K_{n,m})^{-1})K_{mx}\\
& = -K_{xm}(K_{mm}^{-1} - \Psi^{-1})K_{mx}\\
\end{align*}
```
This is like using the kernel $(K_{mm}^{-1}-\Psi^{-1})^{-1}$ for prediction, and labels $\mu^\ast$.

**Titsias [2009]**

We maximize a lower bound on the log-likelihood of observations, over a variational distribution $q(u)$.
```math
\begin{align*}
\log p(y) & = \log \int_f p(y,f)df \tag{introduce latent $f$}\\
& \geq \int_f \log p(y,f)df \tag{Jensen's inequality}\\
& = \int_f \log \int_up(y,f,u)dudf \tag{introduce latent $u$}\\
& = \int_f\log \int_u q(u)\frac{p(y,f,u)}{q(u)}dudf \tag{importance weights}\\
& = \int_f\log \int_u q(u)p(f|u)\frac{p(u)p(y|f)}{q(u)}dudf \tag{re-arrange terms}\\
& \geq \int_f\int_u q(u)p(f|u)\log \frac{p(u)p(y|f)}{q(u)}dudf \tag{Jensen's inequality}\\
& = \mathbb{E}_{u \sim q(u)}\left[\mathbb{E}_{f\sim p(f|u)}\left[\log \frac{p(u)p(y|f)}{q(u)}\right]\right]
\end{align*}
```
Optimizing over $q(u)$ is equivalent to minimizing the following KL divergence to the true posterior over the latent variables after observing all the data (see my note on the ELBO); i.e.
```math
\min_{q(u)}\ D_\mathrm{KL}(\ q(u,f)\ ||\ p(f,u|y)\ ),\quad q(u,f)=q(u)p(f|u).
```
[Titsias 2009] shows that a closed form expression for the last line is available, after solving for the optimal variational distribution $q^\ast(u)$:
```math
\begin{align*}
\log p(y) & \geq \log N(y;0,K_{m,k}K_{k,k}^{-1}K_{k,m} + \sigma^2 I) - \frac{1}{2\sigma^2}\mathrm{tr}(K)\\
 & = \frac{1}{2}y^\top (K_{m,k}K_{k,k}^{-1}K_{k,m}+\sigma^2I)^{-1}y - \frac{1}{2}\log |K_{m,k}K_{k,k}^{-1}K_{k,m}+\sigma^2I|-\frac{m}{2}\log 2\pi-\frac{1}{2\sigma^2}(\mathrm{tr}(K_{m,m}) - \mathrm{tr}(K_{k,k}^{-1}K_{k,m}K_{m,k}))\\
 & = -\frac{m}{2}\log 2\pi-\frac{m-k}{2}\log\sigma^2+\frac{1}{2}\log |K_{k,k}| -\frac{1}{2}\log|\sigma^2K_{k,k}+K_{k,m}K_{m,k}| - \frac{1}{2\sigma^2}y^\top y \\&\quad\ + \frac{1}{2\sigma^2}y^\top K_{mk}(\sigma^2 K_{k,k}+K_{k,m}K_{mk})^{-1}K_{km}y -\frac{1}{2\sigma^2}\mathrm{tr}(K_{m,m})-\frac{1}{2\sigma^2}\mathrm{tr}(K_{k,k}^{-1}(K_{k,m}K_{m,k}))
\end{align*}
```
We need gradients of this expression with respect to hyper-parameters. See [Titsias 2009].

Furthermore note that the optimal variational distribution is the following.
```math
\begin{align*}
q^\ast(u) & = N(u;\mu^\ast,\Sigma^\ast)\\
\mu^\ast & = \frac{1}{\sigma^2}K_{k,k}\Psi K_{k,m}y\\
\Sigma^\ast & = K_{k,k}\Psi K_{k,k}\\
\Psi & = (K_{k,k} + \sigma^{-2}K_{k,m}K_{m,k})^{-1}.
\end{align*}
```
The closed form expression can be optimized over the choice of inducing points $z^{(i)}$, if the appropriate gradients are available. [Titsias 2009] recommends as well a greedy algorithm (instead of gradient-based optimization) for selection of the inducing points. The idea is to incrementally pick the next point from a set, that optimizes the lower bound.

**Hensman et. al. [2012]**

This paper introduces stochastic variational inference (SVI) over mini-batches of data to optimize instead of using the closed-form expressions derived by [Titsias 2009].

Observe the following lower bound due to Jensen's inequality:
```math
\begin{align*}
\log p(y|u) & = \log \mathbb{E}_{f\sim p(f|u)}[p(y|f)]\\
& \geq \mathbb{E}_{f\sim p(f|u)}[\log p(y|f)]\\
& = \sum_{i=1}^m \left(\log N(y_i;\mu_i,\sigma^2)-\frac{1}{2\sigma^2} \tilde{K}_{i,i}\right)
\end{align*}
```
The difference between this lower bound and the original expression is the following KL divergence.
```math
\begin{align*}
D_{\mathrm{KL}}(\ p(f|u)\ ||\ p(f|u,y)\ ) & = \mathbb{E}_{f\sim p(f|u)}\left[\log \frac{p(f|u)}{p(f|u,y)}\right]\\
& = \mathbb{E}_{f\sim p(f|u)}\left[\log \frac{p(f,u)p(u,y)}{p(f,u,y)p(u)}\right] \tag{expand terms}\\
& = \mathbb{E}_{f\sim p(f|u)}\left[\log \frac{p(y|u)}{p(y|f)}\right] \tag{due to $y\perp u | f$}\\
& = \log p(y|u) - \mathbb{E}_{f\sim p(f|u)}[\log p(y|f)]
\end{align*}
```
Now recall that in the previous section we introduced the lower bound,
```math
\log p(y) \geq \mathbb{E}_{u \sim q(u)}\left[\mathbb{E}_{f\sim p(f|u)}\left[\log \frac{p(u)p(y|f)}{q(u)}\right]\right]
```

