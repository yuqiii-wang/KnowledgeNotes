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
P(\bold{z}, \bold{u} | \bold{x}, \bold{y} )=
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
\sum_k\sum_j \bold{e}_{z,j,k}^\text{T} \bold{Q}_{k,j}^{-1} \bold{e}_{z,j,k} 
$$

### SLAM Constraints in Least Square Problem

* the $k$-th motion error only relates to $\bold{x}_k$ and $\bold{x}_{k-1}$, and observation error only concerns $\bold{x}_k$ and $\bold{y}_j$.
This relationship will give a sparse least-square problem for all $k$ steps.
* rotation matrix/transformation matrix has the properties such as $\bold{R}^\text{T}\bold{R}=\bold{I}$ and $det(\bold{R})=1$
* the employment of $\mathcal{L}_2$-norm error amplifies covariance matrices, that is prone to bias prediction when covariance matrices are imbalanced, such as having large values of $\bold{R}_k$ but small values of $\bold{Q}_k$

## Example: Batch State Estimation

Consider a very simple linear system that describes a vehicle running on a 1-d space either forward or backward.
$$
\begin{align*}
x_k &= x_{k-1}+u_{k}+w_k, & \quad w_k & \sim N(0,Q_k)
\\
z_k &= x_k + n_k, & \quad n_k & \sim N(0,R_k)
\end{align*}
$$

Define a batch state variable $\bold{x}=[x_0,x_1,x_2,x_3,...,x_n]^\text{T}$, batch observation $\bold{z}=[z_1,z_2,z_3,...,z_n]^\text{T}$, batch action/dynamics $\bold{u}=[u_1,u_2,u_3,...,u_n]^\text{T}$

The optimal state estimate $\bold{x}^*$ derives from this maximum likelihood optimization.
$$
\begin{align*}
\bold{x}^* &= 
arg \space \underset{}{max} \bold{P} (\bold{x}|\bold{u},\bold{z})
\\ &=
arg \space \underset{}{max} \bold{P} (\bold{u},\bold{z}|\bold{x})
\\ &=
\prod^n_{k=1} P(u_k|x_{k-1},x_k)
\prod^n_{k=1} P(z_k|x_k)
\end{align*}
$$
where each probability is given by a Gaussian distribution, such as
$$
\begin{align*}
P(u_k|x_{k-1},x_k) &\sim N(x_k-x_{k-1},R_k)
\\
P(z_k|x_k) &\sim N(x_k,Q_k)
\end{align*}
$$

The errors to the motion and observation are
$$
\begin{align*}
e_{u,k} &= x_k-x_{k-1}-u_k
\\
e_{z,k} &= z_k-x_k
\end{align*}
$$

Here defines the objective function
$$
min \space J(\bold{x}) = 
\sum_k^n \bold{e}_{u,k}^\text{T} \bold{R}_k^{-1} \bold{e}_{u,k}
+
\sum_k^n \bold{e}_{z,k}^\text{T} \bold{Q}_{k}^{-1} \bold{e}_{z,k} 
$$

Define $\bold{y}=[\bold{u},\bold{z}]^\text{T}$, the error can be rewritten as

$$
\bold{y} - \bold{H}\bold{x}=
\bold{e} \sim N(0,\Sigma)
$$
where $\bold{H}$ is
$$
\bold{H}=
\begin{bmatrix}
    1 & -1 & 0 & 0 &  & 0 & 0 \\
    0 & 1 & -1 & 0 &  & 0 & 0 \\
    0 & 0 & 1 & -1 &  & 0 & 0 \\
     &  &  &  & \ddots & &  \\
    0 & 0 & 0 & 0 &  & 1 & -1 \\
    \hline
    0 & 1 & 0 & 0 &  & 0 & 0 \\
    0 & 0 & 1 & 0 &  & 0 & 0 \\
    0 & 0 & 0 & 1 &  & 0 & 0 \\
     &  &  &  & \ddots & &  \\
    0 & 0 & 0 & 0 &  & 1 & 0 \\
\end{bmatrix}
$$
and the covariance matrix is $\Sigma=diag(R_1, R_2, R_3,...,R_n,Q_1,Q_2,Q_3,...,Q_n)$.

The Maximum Likelihood optimization given the objective function $\bold{e}^\text{T} \Sigma^{-1} \bold{e}$ (this is a quadratic function that has a global minimum) can be approximated by the Gauss-Newton method, so that the optimal state $\bold{x}^*$ can be computed via the below expression.
$$
\begin{align*}
\bold{x}^* &=
arg \space \underset{\bold{x}}{min}\space \bold{e}^\text{T} \Sigma^{-1} \bold{e}
\\ &=
\big(\bold{H}^{\text{T}}\Sigma^{-1}\bold{H}\big)^{-1}
\bold{H}^{\text{T}}\Sigma^{-1} \bold{y}
\end{align*}
$$

