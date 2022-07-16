# Maximum likelihood estimation

Maximum likelihood estimation (MLE) is a method of estimating the parameters of an assumed probability distribution, given some observed data.

We model a set of observations $\bold{y}=(y_1, y_2, ..., y_n)$ as a random sample from an unknown joint probability distribution which is expressed in terms of a set of parameters $\theta=[\theta_1, \theta_2, ..., \theta_k]^T$. Thsi distribution falls within a parameteric family $f(\space \cdot \space; \theta | \theta \in \Theta)$, where $\Theta$ is called parameter space.

Likelihood function is expressed as 
$$
L_n(\bold{y};\theta)
$$

To best model the observations $\bold{y}$ by finding the optimal $\hat{\theta}$:
$$
\hat{\theta} = arg \space \underset{\theta \in \Theta}{max} \space L_n(\bold{y};\theta)
$$

In practice, it is often convenient to work with the natural logarithm of the likelihood function, called the log-likelihood:
$$
ln \space L_n(\bold{y};\theta)
$$
since the logarithm is a monotonic function. 

Max value is located at where derivatives are zeros
$$
\frac{\partial ln \space L_n(\bold{y};\theta)}{\partial \theta_1} = 0,
\frac{\partial ln \space L_n(\bold{y};\theta)}{\partial \theta_2} = 0,
...,
\frac{\partial ln \space L_n(\bold{y};\theta)}{\partial \theta_k} = 0
$$