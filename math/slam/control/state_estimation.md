# State Estimation

A robot's state $\bold{x}$ is measured collectively by its own motion dynamics $\bold{f}$ plus noises $\bold{w}$, and observation $\bold{z}$ by landmark $\bold{y}$ plus noises $\bold{v}$.

For the $k$-step given a number of landmark observations $j=1,2,...$, there is
$$
\begin{align*}
    \bold{x}_k &= \bold{f}(\bold{x}_{k-1}, \bold{u}_k)+\bold{w}_k
    \\
    \bold{z}_{k,j} &= \bold{h}(\bold{x}_{k}, \bold{y}_j)+\bold{v}_k
\end{align*}
$$

By Bayes equation, there is
$$
\begin{align*}
P(\bold{x}, \bold{y} | \bold{z}, \bold{u}) &=
\frac{P(\bold{z}, \bold{u}| \bold{x}, \bold{y})P(\bold{x}, \bold{y})}{P(\bold{z}, \bold{u})}
\\ & \propto
P(\bold{z}, \bold{u}| \bold{x}, \bold{y})P(\bold{x}, \bold{y})
\end{align*}
$$
where $P(\bold{z}, \bold{u}| \bold{x}, \bold{y})$ is the likelihood and $P(\bold{x}, \bold{y})$ is the prior.

A Maximum Likelihood Estimation problem is defined as
$$
arg \space \underset{}{max} \space
P(\bold{z}, \bold{u} | \bold{x}, \bold{y} )
$$

If the noises follow Gaussian distribution $\bold{w}_k \sim N(0, \bold{R}_k)$ and $\bold{v}_{k,j} \sim N(0, \bold{Q}_{k,j})$, for observation model, there is

$$
P(\bold{z}_{k,j} | \bold{x}_k, \bold{y}_j) = 
N(\bold{h}(\bold{y}_j, \bold{x}_k), \bold{Q}_k)
$$

Denote $(\bold{x}_k, \bold{y}_j)^*$ as the optimized 
$$
\begin{align*}
(\bold{x}_k, \bold{y}_j)^* &=
arg \space \underset{\bold{z}}{max}\space N (\bold{h}(\bold{y}_j, \bold{x}_k), \bold{Q}_k)
\\ &=
arg \space \underset{\bold{z}}{min}\space \bigg(
    \big(\bold{z}_{k,j}-\bold{h}(\bold{y}_j, \bold{x}_k)\big)^\text{T}
    \bold{Q}_{k,j}^{-1}
    \big(\bold{z}_{k,j}-\bold{h}(\bold{y}_j, \bold{x}_k)\big)
\bigg)
\end{align*}
$$

This quadratic form is called *Mahalanobis distance*. 
It can also be regarded as the Euclidean distance of $\mathcal{L}_2$-norm weighted by $\bold{Q}_{k,j}^{-1}$, 
where $\bold{Q}_{k,j}$ is also called the *information matrix*, which is exactly the *inverse* of the Gaussian covariance matrix.

It is usually assumed that the inputs
and observations are independent of each other, so that the joint distribution can be factorized such as
$$
P(\bold{z}, \bold{u} | \bold{x}, \bold{y} )
=
\prod_k P(\bold{u}_k | \bold{x}_{k-1}, \bold{x}_k )
\prod_{k,j} P(\bold{z}_{k,j} | \bold{x}_k, \bold{y}_j) 
$$

The error $\bold{e}$ can be measured separately since dynamics $\bold{u}$ and observations $\bold{z}$ are independent.
$$
\begin{align*}
\bold{e}_{u,k} &= 
\bold{x}_k - \bold{f}(\bold{x}_k, \bold{u}_k)
\\
\bold{e}_{z,j,k} &= 
\bold{z}_{k,j} - \bold{h}(\bold{z}_{k,j}, \bold{y}_j)
\end{align*}
$$

The problem of find the optimal state estimate $\hat{\bold{x}}_k$ can be transformed into a least square problem; the cost function is shown as below
$$
min \space J(\bold{x}, \bold{y}) = 
\sum_k \bold{e}_{u,k}^\text{T} \bold{R}_k^{-1} \bold{e}_{u,k}
+
\sum_k \bold{e}_{z,j,k}^\text{T} \bold{Q}_{k,j}^{-1} \bold{e}_{z,j,k} 
$$