# Summary of various methods

## Gauss-Newton vs Steepest Descent

Both are used to solve the minimization problem given the residual $\mathbf{r}$:

$$
\space \underset{\mathbf{r}}{min} \space
\mathbf{r}(\mathbf{x})^\text{T} \mathbf{r}(\mathbf{x})
$$

* Gradient descent

$$
\begin{align*}
\mathbf{x}\_{n+1}&=
\mathbf{x}\_{n} -
\lambda \Delta \big(\frac{1}{2} \mathbf{r}(\mathbf{x}_n)^\text{T} \mathbf{r}(\mathbf{x}_n)\big)
\\\\ &=
\mathbf{x}\_{n} -
\lambda \mathbf{J}^\text{T}_r \mathbf{r} (\mathbf{x}_n)
\end{align*}
$$

where $\lambda$ can be set to $\lambda=\frac{\mathbf{r}_k^T \mathbf{r}_k}{\mathbf{r}_k^T A \mathbf{r}_k}$ for steepest descent.

* Gauss-Newton

$$
\mathbf{x}\_{n+1}=
\mathbf{x}\_{n} -
(\mathbf{J}^\text{T}_r \mathbf{J}_r)^{-1} \mathbf{J}^\text{T}_r \mathbf{r} (\mathbf{x}_n)
$$

where $\mathbf{H}=\mathbf{J}^\text{T}_r \mathbf{J}_r$ is the Hessian matrix that defines the second order derivative.

## Levenberg-Marquardt vs Doglet

