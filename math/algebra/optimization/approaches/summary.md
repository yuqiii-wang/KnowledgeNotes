# Summary of various methods

## Gauss-Newton vs Steepest Descent

Both are used to solve the minimization problem given the residual $\bold{r}$:
$$
arg \space \underset{\bold{r}}{min} \space
\bold{r}(\bold{x})^\text{T} \bold{r}(\bold{x})
$$

* Gradient descent
$$
\begin{align*}
\bold{x}_{n+1}
&=
\bold{x}_{n}
-
\lambda \Delta \big(\frac{1}{2} \bold{r}(\bold{x}_n)^\text{T} \bold{r}(\bold{x}_n)\big)
\\ &=
\bold{x}_{n}
-
\lambda \bold{J}^\text{T}_r \bold{r} (\bold{x}_n)
\end{align*}
$$
where $\lambda$ can be set to $\lambda=\frac{\bold{r}_k^T \bold{r}_k}{\bold{r}_k^T A \bold{r}_k}$ for steepest descent.

* Gauss-Newton

$$
\bold{x}_{n+1}
=
\bold{x}_{n}
-
(\bold{J}^\text{T}_r \bold{J}_r)^{-1} \bold{J}^\text{T}_r \bold{r} (\bold{x}_n)
$$
where $\bold{H}=\bold{J}^\text{T}_r \bold{J}_r$ is the Hessian matrix that defines the second order derivative.

## Levenberg-Marquardt

