### Kernel Gradients

---

Recall Equation 5.9 from Rasmussen & Williams, 2006:
```math
\frac{\partial}{\partial\theta_j}\log p(y|X,\theta) = \frac{1}{2}\mathrm{tr}\left((\alpha\alpha^\top -K^{-1})\frac{\partial K}{\partial \theta_j}\right),\ \alpha=K^{-1}y.
```
Below we list $\frac{\partial K}{\partial \theta_j}$ for various kernels.

---

**Squared Exponential Kernel**

We have one parameter $\ell = e^\theta > 0$, the length scale. Note $\frac{d\ell}{d\theta}=\ell$.
```math
\begin{align*}
k(x,y) &= \exp\left(-\frac{1}{2\ell}||x-y||_2^2\right)\\
\frac{d}{d\ell}k(x,y) & = \frac{1}{2\ell^2}||x-y||^2_2\exp\left(-\frac{1}{2\ell}||x-y||_2^2\right)\\
& = -k(x,y)\log k(x,y)/ \ell\\
\frac{d}{d\theta}k(x,y) & = -k(x,y)\log k(x,y)
\end{align*}
```
*Anisotropic Version*

We have a $p$-length parameter $\ell = e^\theta>0$, the length scale.
```math
\begin{align*}
k(x,y) & = \exp \left(-\frac{1}{2} \sum_{i=1}^p \frac{1}{\ell_i}(x_i-y_i)^2\right)\\
\frac{d}{d\ell_j}k(x,y) & = \frac{1}{2\ell_j^2}(x_i-y_i)^2\exp\left(-\frac{1}{2}\sum_{i=1}^p \frac{1}{\ell_i}(x_i-y_i)^2\right)\\
\frac{d}{d\theta_j}k(x,y) & = \frac{1}{2\ell_j}(x_i-y_i)^2 k(x,y)
\end{align*}
```

---

**Sum Kernel**
```math
\begin{align*}
k(x,y) &= k_1(x,y) + k_2(x,y)\\
\frac{d}{d\theta_1} & = \frac{d}{d\theta_1}k_1(x,y)
\end{align*}
```

---

**Product Kernel**
```math
\begin{align*}
k(x,y) & = k_1(x,y) k_2(x,y)\\
\frac{d}{d\theta_1} & = k_2(x,y)\frac{d}{d\theta_1}k_1(x,y)
\end{align*}
```

---

**Constant Kernel**
```math
\begin{align*}
k(x,y) & = c \triangleq e^\theta\\
\frac{d}{d\theta} k(x,y)& = c = e^\theta
\end{align*}
```
