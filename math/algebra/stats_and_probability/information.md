# Information

## Score

In statistics, score (or informant) $s(\theta)$ is a $1 \times m$ vector of the gradient of a log-likelihood $log L(\theta)$ function with respect to its config parameter vector $\theta$ (size of $1 \times m$). 

$$
s(\theta)=\frac{\partial \space log L(\theta)}{\partial \theta}
$$

### Mean

Given observations $\bold{x}=[x_1, x_2, ..., x_n]$ in the sample space $X$, the expected value of score is
$$
\begin{align*}
E(s|\theta)
&=
\int_X \frac{\partial \space log L(\theta;x)}{\partial \theta} f(x;\theta) dx
\\ &=
\int_X \frac{\partial \space f(x; \theta)}{\partial \theta} \frac{1}{f(x;\theta)} f(x;\theta) dx
\\ &=
\int_X \frac{\partial \space f(x; \theta)}{\partial \theta} dx
\end{align*}
$$

By Leibniz integral rule which allows for interchange of derivative and integral, there is
$$
\begin{align*}
E(s|\theta) &=
\int_X \frac{\partial \space f(x; \theta)}{\partial \theta} dx
\\ &=
\frac{\partial }{\partial \theta} \int_X f(x;\theta) dx
\\ &=
\frac{\partial }{\partial \theta} 1 
\\ &= 
0
\end{align*}
$$

### Intuition

The optimal configuration $\theta$ to fit sample space distribution is by minimizing its log-likelihood function $logL(\theta;x)$. 

$logL(\theta;x)$ by ideal $\theta$ should see its minima along side with its derivative zero, hence $E(s|\theta)=0$.

## Fisher information

Fisher information is a way of measuring the amount of information that an observable random variable $\bold{x} \in X$ carries about an unknown parameter $\theta$.

Let $f(X;\theta)$ be the probability density function for $\bold{x} \in X$ conditioned on $\theta$.

Fisher information $\bold{I}(\theta)$ is defined to be the variance of score:
$$
\begin{align*}
\bold{I}(\theta)
&=
E\bigg[
    \bigg(
        \frac{\partial \space log L(\bold{x};\theta)}{\partial \theta}  
    \bigg)^2
    \bigg| \theta
\bigg]
\\ &=
\int_X \bigg( \frac{\partial \space log L(\theta;x)}{\partial \theta} \bigg)^2 f(x;\theta) dx
\end{align*}
$$

### Twice differentiable with respect to $\theta$

For $\bold{I}(\theta)$ being twice differentiable with repssect to $\theta$, $\bold{I}(\theta)$ can be expressed as

$$
\begin{align*}
\bold{I}(\theta)
&=
-E\bigg[
        \frac{\partial^2 \space log L(\bold{x};\theta)}{\partial \theta^2}  
    \bigg| \theta
\bigg]
\end{align*}
$$

Thus, the Fisher information may be seen as the curvature of the support curve (the graph of the log-likelihood).

### Fisher information matrix

For a multi-dimensional observation vector $x_k=(x_{k,1}, x_{k,2}, ..., x_{k,l})$ in the dataset $[x_1, x_2, ..., x_k, ..., x_n] \in \bold{x}$, a Fisher information matrix is the covariance of score, in which each entry is defined
$$
\bold{I}_{i,j}(\theta)=
E\bigg[
    \bigg(\frac{\partial \space log\space f(\bold{x};\theta)}{\partial \theta_i}\bigg)
    \cdot
    \bigg(\frac{\partial \space log\space f(\bold{x};\theta)}{\partial \theta_j}\bigg)^\text{T}
    \bigg| \theta
\bigg]
$$

### Intuition

Fisher information is defined by computing the covariance of the gradient of log-likelihood function $log L(\theta)$.

$\bold{I}_{i,j}(\theta)$ says that, the greater the one covariance element value, the greater the gradient of the partial derivative direction $\angle (\theta_i +  \theta_j)$, indicating optimization directions and volumes.