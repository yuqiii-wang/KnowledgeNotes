# Maximum Likelihood Estimation

Maximum likelihood estimation (MLE) is a method of estimating the parameters of an assumed probability distribution, given some observed data.

We model a set of observations $\mathbf{y}=(y_1, y_2, ..., y_n)$ as a random sample from an unknown joint probability distribution which is expressed in terms of a set of parameters $\theta=[\theta_1, \theta_2, ..., \theta_k]^T$. Thsi distribution falls within a parameteric family $f(\space \cdot \space; \theta | \theta \in \Theta)$, where $\Theta$ is called parameter space.

Likelihood function is expressed as

$$
L_n(\mathbf{y};\theta)
$$

To best model the observations $\mathbf{y}$ by finding the optimal $\hat{\theta}$:

$$
\hat{\theta} = arg \space \underset{\theta \in \Theta}{max} \space L_n(\mathbf{y};\theta)
$$

In practice, it is often convenient to work with the natural logarithm of the likelihood function, called the log-likelihood:

$$
ln \space L_n(\mathbf{y};\theta)
$$
since the logarithm is a monotonic function. 

Max value is located at where derivatives are zeros
$$
\frac{\partial ln \space L_n(\mathbf{y};\theta)}{\partial \theta_1} = 0,
\frac{\partial ln \space L_n(\mathbf{y};\theta)}{\partial \theta_2} = 0,
...,
\frac{\partial ln \space L_n(\mathbf{y};\theta)}{\partial \theta_k} = 0
$$

## MLE in Gaussian Distribution

Given random variable $\mathbf{y} \in \mathbb{R}^n$ following Gaussian distribution $\mathbf{y} \sim N(\mu, \Sigma)$, the probability density function is

$$
P(\mathbf{y};\theta) = 
\frac{1}{\sqrt{(2\pi)^ndet(\Sigma)}}
e^{-\frac{1}{2}(\mathbf{y}-\mu)^\text{T}\Sigma^{-1}(\mathbf{y}-\mu)}
$$

Take the negative logarithm of the equation:

$$
\begin{align*}
-ln(P(\mathbf{y};\theta)) &=
-ln\bigg(
    \frac{1}{\sqrt{(2\pi)^ndet(\Sigma)}}
    e^{-\frac{1}{2}(\mathbf{y}-\mu)^\text{T}\Sigma^{-1}(\mathbf{y}-\mu)}
\bigg)
\\\\ &=
-ln\bigg(\frac{1}{\sqrt{(2\pi)^ndet(\Sigma)}}\bigg)
+\frac{1}{2}(\mathbf{y}-\mu)^\text{T}\Sigma^{-1}(\mathbf{y}-\mu)
\end{align*}
$$

$-ln\bigg(\frac{1}{\sqrt{(2\pi)^ndet(\Sigma)}}\bigg)$ is discarded since optimization to finding optimum is by the change of hidden parameter $\mathbf{\theta}$ that does not affect the source data distribution $\mathbf{y}$.

Since $-ln(P(\mathbf{y};\theta))$ is monotonically decreasing, finding the maximum of $L_n(\mathbf{y};\theta)$ is same as find the minimum of the negative logarithm mapping.

$$
\begin{align*}
\hat{\theta} &=
 arg \space \underset{\theta \in \Theta}{max} \space L_n(\mathbf{y};\theta)
\\\\ &=
arg\space \underset{\theta \in \Theta}{min}\space
\frac{1}{2}(\mathbf{y}-\mu)^\text{T}\Sigma^{-1}(\mathbf{y}-\mu)
\end{align*}
$$