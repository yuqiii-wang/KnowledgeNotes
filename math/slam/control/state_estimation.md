# State Estimation

A robot's state $\mathbf{x}$ is measured collectively by its own motion dynamics $\mathbf{f}$ plus noises $\mathbf{w}$, and observation $\mathbf{z}$ by landmark $\mathbf{y}$ plus noises $\mathbf{v}$.

For the $k$-step given a number of landmark observations $j=1,2,...$, there is

$$
\begin{align*}
    \mathbf{x}_k &= \mathbf{f}(\mathbf{x}_{k-1}, \mathbf{u}_k)+\mathbf{w}_k
    \\\\
    \mathbf{z}_{k,j} &= \mathbf{h}(\mathbf{x}_{k}, \mathbf{y}_j)+\mathbf{v}_k
\end{align*}
$$

By Bayes equation, there is

$$
\begin{align*}
P(\mathbf{x}, \mathbf{y} | \mathbf{z}, \mathbf{u}) &=
\frac{P(\mathbf{z}, \mathbf{u}| \mathbf{x}, \mathbf{y})P(\mathbf{x}, \mathbf{y})}{P(\mathbf{z}, \mathbf{u})}
\\\\ & \propto
P(\mathbf{z}, \mathbf{u}| \mathbf{x}, \mathbf{y})P(\mathbf{x}, \mathbf{y})
\end{align*}
$$

where $P(\mathbf{z}, \mathbf{u}| \mathbf{x}, \mathbf{y})$ is the likelihood and $P(\mathbf{x}, \mathbf{y})$ is the prior.

A Maximum Likelihood Estimation problem is defined as

$$
arg \space \underset{}{max} \space
P(\mathbf{z}, \mathbf{u} | \mathbf{x}, \mathbf{y} )
$$

If the noises follow Gaussian distribution $\mathbf{w}_k \sim N(0, \mathbf{R}_k)$ and $\mathbf{v}_{k,j} \sim N(0, \mathbf{Q}_{k,j})$, for observation model, there is

$$
P(\mathbf{z}_{k,j} | \mathbf{x}_k, \mathbf{y}_j) = 
N(\mathbf{h}(\mathbf{y}_j, \mathbf{x}_k), \mathbf{Q}_k)
$$

Denote $(\mathbf{x}_k, \mathbf{y}_j)^*$ as the optimized 
$$
\begin{align*}
(\mathbf{x}_k, \mathbf{y}_j)^* &=
arg \space \underset{\mathbf{z}}{max}\space N (\mathbf{h}(\mathbf{y}_j, \mathbf{x}_k), \mathbf{Q}_k)
\\\\ &=
arg \space \underset{\mathbf{z}}{min}\space \bigg(
    \big(\mathbf{z}_{k,j}-\mathbf{h}(\mathbf{y}_j, \mathbf{x}_k)\big)^\text{T}
    \mathbf{Q}_{k,j}^{-1}
    \big(\mathbf{z}_{k,j}-\mathbf{h}(\mathbf{y}_j, \mathbf{x}_k)\big)
\bigg)
\end{align*}
$$

This quadratic form is called *Mahalanobis distance*. 
It can also be regarded as the Euclidean distance of $\mathcal{L}_2$-norm weighted by $\mathbf{Q}_{k,j}^{-1}$, 
where $\mathbf{Q}_{k,j}$ is also called the *information matrix*, which is exactly the *inverse* of the Gaussian covariance matrix.

It is usually assumed that the inputs
and observations are independent of each other, so that the joint distribution can be factorized such as

$$
P(\mathbf{z}, \mathbf{u} | \mathbf{x}, \mathbf{y} )=
\prod_k P(\mathbf{u}_k | \mathbf{x}_{k-1}, \mathbf{x}_k )
\prod_{k,j} P(\mathbf{z}_{k,j} | \mathbf{x}_k, \mathbf{y}_j) 
$$

The error $\mathbf{e}$ can be measured separately since dynamics $\mathbf{u}$ and observations $\mathbf{z}$ are independent.

$$
\begin{align*}
\mathbf{e}_{u,k} &= 
\mathbf{x}_k - \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k) \\\\
\mathbf{e}_{z,j,k} &= 
\mathbf{z}_{k,j} - \mathbf{h}(\mathbf{z}_{k,j}, \mathbf{y}_j)
\end{align*}
$$

The problem of find the optimal state estimate $\hat{\mathbf{x}}_k$ can be transformed into a least square problem; the cost function is shown as below

$$
min \space J(\mathbf{x}, \mathbf{y}) = 
\sum_k \mathbf{e}_{u,k}^\text{T} \mathbf{R}_k^{-1} \mathbf{e}_{u,k}
+
\sum_k\sum_j \mathbf{e}_{z,j,k}^\text{T} \mathbf{Q}_{k,j}^{-1} \mathbf{e}_{z,j,k} 
$$

### SLAM Constraints in Least Square Problem

* the $k$-th motion error only relates to $\mathbf{x}_k$ and $\mathbf{x}_{k-1}$, and observation error only concerns $\mathbf{x}_k$ and $\mathbf{y}_j$.
This relationship will give a sparse least-square problem for all $k$ steps.
* rotation matrix/transformation matrix has the properties such as $\mathbf{R}^\text{T}\mathbf{R}=\mathbf{I}$ and $det(\mathbf{R})=1$
* the employment of $\mathcal{L}_2$-norm error amplifies covariance matrices, that is prone to bias prediction when covariance matrices are imbalanced, such as having large values of $\mathbf{R}_k$ but small values of $\mathbf{Q}_k$

## Example: Batch State Estimation

Consider a very simple linear system that describes a vehicle running on a 1-d space either forward or backward.

$$
\begin{align*}
x_k &= x_{k-1}+u_{k}+w_k, & \quad w_k & \sim N(0,Q_k) \\\\
z_k &= x_k + n_k, & \quad n_k & \sim N(0,R_k)
\end{align*}
$$

Define a batch state variable $\mathbf{x}=[x_0,x_1,x_2,x_3,...,x_n]^\text{T}$, batch observation $\mathbf{z}=[z_1,z_2,z_3,...,z_n]^\text{T}$, batch action/dynamics $\mathbf{u}=[u_1,u_2,u_3,...,u_n]^\text{T}$

The optimal state estimate $\mathbf{x}^*$ derives from this maximum likelihood optimization.

$$
\begin{align*}
\mathbf{x}^* &= 
arg \space \underset{}{max} \mathbf{P} (\mathbf{x}|\mathbf{u},\mathbf{z})
\\\\ &=
arg \space \underset{}{max} \mathbf{P} (\mathbf{u},\mathbf{z}|\mathbf{x})
\\\\ &=
\prod^n\_{k=1} P(u_k|x_{k-1},x_k)
\prod^n\_{k=1} P(z_k|x_k)
\end{align*}
$$

where each probability is given by a Gaussian distribution, such as

$$
\begin{align*}
P(u_k|x_{k-1},x_k) &\sim N(x_k-x_{k-1},R_k) \\\\
P(z_k|x_k) &\sim N(x_k,Q_k)
\end{align*}
$$

The errors to the motion and observation are

$$
\begin{align*}
e_{u,k} &= x_k-x_{k-1}-u_k \\\\
e_{z,k} &= z_k-x_k
\end{align*}
$$

Here defines the objective function

$$
min \space J(\mathbf{x}) = 
\sum_k^n \mathbf{e}_{u,k}^\text{T} \mathbf{R}_k^{-1} \mathbf{e}_{u,k}
+
\sum_k^n \mathbf{e}_{z,k}^\text{T} \mathbf{Q}_{k}^{-1} \mathbf{e}_{z,k} 
$$

Define $\mathbf{y}=[\mathbf{u},\mathbf{z}]^\text{T}$, the error can be rewritten as

$$
\mathbf{y} - \mathbf{H}\mathbf{x}=
\mathbf{e} \sim N(0,\Sigma)
$$

where $\mathbf{H}$ is

$$
\mathbf{H}=
\begin{bmatrix}
    1 & -1 & 0 & 0 &  & 0 & 0 \\\\
    0 & 1 & -1 & 0 &  & 0 & 0 \\\\
    0 & 0 & 1 & -1 &  & 0 & 0 \\\\
     &  &  &  & \ddots & &  \\\\
    0 & 0 & 0 & 0 &  & 1 & -1 \\\\
    \hline
    0 & 1 & 0 & 0 &  & 0 & 0 \\\\
    0 & 0 & 1 & 0 &  & 0 & 0 \\\\
    0 & 0 & 0 & 1 &  & 0 & 0 \\\\
     &  &  &  & \ddots & &  \\\\
    0 & 0 & 0 & 0 &  & 1 & 0 \\\\
\end{bmatrix}
$$

and the covariance matrix is $\Sigma=diag(R_1, R_2, R_3,...,R_n,Q_1,Q_2,Q_3,...,Q_n)$.

The Maximum Likelihood optimization given the objective function $\mathbf{e}^\text{T} \Sigma^{-1} \mathbf{e}$ (this is a quadratic function that has a global minimum) can be approximated by the Gauss-Newton method, so that the optimal state $\mathbf{x}^*$ can be computed via the below expression.

$$
\begin{align*}
\mathbf{x}^* &=
arg \space \underset{\mathbf{x}}{min}\space \mathbf{e}^\text{T} \Sigma^{-1} \mathbf{e}
\\\\ &=
\big(\mathbf{H}^{\text{T}}\Sigma^{-1}\mathbf{H}\big)^{-1}
\mathbf{H}^{\text{T}}\Sigma^{-1} \mathbf{y}
\end{align*}
$$

