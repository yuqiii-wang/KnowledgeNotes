# Kalman filters for non-linear problems


## Extended Kalman filter (EKF)

For non-linear dynamic and observation transformation:

$$
\begin{align*}
\mathbf{x}_{k} &= f(
    \mathbf{x}_{k-1|k-1}, \mathbf{u}_k
) + \mathbf{w}_k \\\\
\mathbf{z}_{k} &= h(\mathbf{x}_k)+\mathbf{v}_k
\end{align*}
$$

where $f$ and $h$ denote the non-linear transformation.

Redefine $\mathbf{F}_k$ and $\mathbf{H}_k$ to be the Jacobian matrices of $f$ and $h$, respectively.

$$
\begin{align*}
\mathbf{F}_k&=
\frac{\partial f}{\partial \mathbf{x}}
\bigg|_{\mathbf{x}_{k-1|k-1}, \mathbf{u}_k} \\\\
\mathbf{H}_k&=
\frac{\partial h}{\partial \mathbf{x}}
\bigg|_{\mathbf{x}_{k|k-1}}
\end{align*}
$$

The computation for $\mathbf{K}_k$ is identical to its linear Kalman counterpart.

Predicted (a priori) estimate covariance
$$
\mathbf{\hat{P}}_{k|k-1}=
\mathbf{F}_k\mathbf{P}_{k-1|k-1} \mathbf{F}^\text{T}_k + \mathbf{Q}_k
$$

Innovation (or pre-fit residual) covariance

$$
\mathbf{{S}}_{k} =
\mathbf{H}_k \mathbf{\hat{P}}_{k|k-1} \mathbf{H}^\text{T}_k + \mathbf{R}_k
$$

Optimal Kalman gain

$$
\mathbf{K}_k =
\mathbf{\hat{P}}_{k|k-1} \mathbf{H}^\text{T}_k \mathbf{{S}}_{k}^{-1}
$$


### Convergence discussions

For non-linear transformation, EKF uses first- or second-order derivatives as approximations. The Gaussian noises $\mathbf{Q}$ and $\mathbf{R}$ are applied to the whole non-linear transformations $f$ and $h$, whereas EKF Kalman gain $\mathbf{K}_k$ only covers the first- or second-order derivative. 

This leads to precision loss when $k \rightarrow \infty$, since higher order derivatives are not considered, and the lost precision errors accumulate over the time.

Besides, sampling intervals should be small, otherwise, the first- or second-order Taylor expansion does not provide good approximations.

## Unscented Kalman filter (UKF)

To resolve the time-consuming Jacobian computation as well as first-order derivative induced loss of precision, UKF instead directly samples from history data (the selected points are called *sigma points*), by which the covariances are formed.

### Sampling

Define a random variable $\mathbf{x} \in \mathbb{R}^d$ assumed exhibited normal distribution (of a mean $\mathbf{\overline{x}}$ and covariance $\mathbf{P_\mathbf{x}}$ ) sampling behavior.

Define $g$ as a linear/non-linear transformation on normal distribution random variables $\mathbf{x}$ such as $\mathbf{y}=g(\mathbf{x})$. The transformed result is $\mathbf{y}$.

![unscented_transform](imgs/unscented_transform.png "unscented_transform")


Define a matrix $\mathbf{X}$ consisted of $2d+1$ *sigma* vectors $X_i$ with corresponding weight $W_i$.

$$
\begin{align*}
X_0 &= \mathbf{\overline{x}} \\\\
X_i &= 
\mathbf{\overline{x}} + (\sqrt{(d+\lambda)\mathbf{P_\mathbf{x}}})_i 
\quad
\quad i=1,2,...,d \\\\
X_i &= 
\mathbf{\overline{x}} - (\sqrt{(d+\lambda)\mathbf{P_\mathbf{x}}})_{i-d} 
\quad i=d+1,d+2,...,2d \\\\
W_0^{(m)} &=\frac{\lambda}{d+\lambda} \\\\
W_0^{(c)} &=\frac{\lambda}{d+\lambda} + (1+\alpha^2+\beta) \\\\
W_i^{(m)}=W_i^{(m)} &=
\frac{1}{2(d+\lambda)}
\quad\quad\quad\quad\quad\quad i=1,2,...,2d
\end{align*}
$$

where 

* $\lambda=\alpha^2(d+\kappa)-d$ is a scaling parameter
* $\alpha \in (0,1]$ determines the spread of the sigma points, typically $\alpha=e^{-0.001}$
* $\kappa \ge 0$ is a secondary scaling parameter, typically $\kappa = 1$ 
* $\beta$ is used to incorporate prior knowledge of Gaussian distribution ($\beta=2$ is optimal by experience).
* $(\sqrt{(d+\lambda)\mathbf{P_\mathbf{x}}})_i$ is the $i$-th row of the matrix square root

### Transform Sigma Points

The expectation of $\mathbf{y}$ can be approximated via Gauss-Hermite quadrature:

$$
\begin{align*}
\mathbf{\overline{x}}
&\approx
\sum^{2d}_{i=0}
W^{(m)} x_i \\\\
\mathbf{\overline{y}} 
&\approx
\sum^{2d}_{i=0}
W^{(m)} y_i \\\\
\mathbf{P_x} 
&\approx
\sum^{2d}_{i=0}
W^{(c)} (x_i-\mathbf{\overline{x}}) (x_i-\mathbf{\overline{x}})^\text{T} \\\\
\mathbf{P_y} 
&\approx
\sum^{2d}_{i=0}
W^{(c)} (y_i-\mathbf{\overline{y}}) (y_i-\mathbf{\overline{y}})^\text{T} \\\\
\mathbf{P_{xy}} 
&\approx
\sum^{2d}_{i=0}
W^{(c)} (x_i-\mathbf{\overline{x}}) (y_i-\mathbf{\overline{y}})^\text{T}
\end{align*}
$$ 

### Kalman Gain $\mathbf{K}$

$$
\mathbf{K} = \mathbf{P_{xy}} \mathbf{P_{yy}}^{-1}
$$

### Gauss-Hermite quadrature discussions

Sampling follows *Gauss-Hermite quadrature*. Given each dimension having $3$ sampling points, the polynomial precision is a degree of $5$.

## Example

Given a vehicle state composed of a distance $p$ and velocity $v$. 
Its init estimates: init state and covariances are known as below.

$$
\mathbf{x} =
\begin{bmatrix}
p \\\\ v
\end{bmatrix} , \quad
\mathbf{\hat{x}}_0 \sim N
\bigg(
\begin{bmatrix}
0 \\\\ 5
\end{bmatrix} ,
\begin{bmatrix}
0.01 & 0\\\\
0 & 1.0
\end{bmatrix} \bigg)
$$

and vehicle motion model

$$
\begin{align*}
\mathbf{\hat{x}}_k &= 
f(\mathbf{\hat{x}}_{k-1}, \mathbf{u}_k, \mathbf{w}_k)
\\\\ &=
\begin{bmatrix}
1 & \Delta t \\\\
0 & 1
\end{bmatrix}
\mathbf{x}_{k-1}
+
\begin{bmatrix}
0 \\\\
\Delta t
\end{bmatrix}
\mathbf{u}_k
+
\mathbf{w}_k
\\\\ &=
\begin{bmatrix}
1 & 0.5 \\\\
0 & 1
\end{bmatrix}
\mathbf{x}_{k-1}
-2
\begin{bmatrix}
0 \\\\
0.5
\end{bmatrix}
+
\mathbf{w}_k
\end{align*}
$$

where $\mathbf{u}_k = a = -2 \space m/s^2$ is the acceleration.

Vehicle measurement model is defined such that we can only observe the distance

$$
\begin{align*}
y_k &= h(\mathbf{x}) + \mathbf{v}_k
\\\\ &=
\begin{bmatrix}
1 & 0\\\\
0 & 0
\end{bmatrix}
\mathbf{x} + \mathbf{v}_k
\end{align*}
$$

Both $\mathbf{w}_k$ and $\mathbf{v}_k$ follow Gaussian distribution

$$
\begin{align*}
\mathbf{w}_k &\sim
N
\bigg(
\begin{bmatrix}
0 \\\\ 0
\end{bmatrix} ,
\begin{bmatrix}
0.01 & 0\\\\
0 & 0.01
\end{bmatrix}
\bigg) \\\\
\mathbf{v}_k &\sim
N
\bigg(
\begin{bmatrix}
0 \\\\ 0
\end{bmatrix} ,
\begin{bmatrix}
0.01 & 0\\\\
0 & 0.0
\end{bmatrix} \bigg)
\end{align*}
$$

### Computation

Use Cholesky to solve $\mathbf{P}_0=\begin{bmatrix} 0.01 & 0 \\\\ 0 & 1.0 \end{bmatrix}$, there is

$$
{\Sigma}_0 = 
\begin{bmatrix}
0.1 & 0\\\\
0 & 1.0
\end{bmatrix}
$$

Compute $2$-dimensional sigma points:

$$
\begin{align*}
\sqrt{d+\lambda} &=
\sqrt{d+\alpha^2(d+\kappa)-d}
\\\\ &=
\alpha \sqrt{d+\kappa}
\\\\ & \approx
\sqrt{3}
\end{align*}
$$

At the time $k=0$, compute sigma points by $\mathbf{P}_0$: on each dimension compute $3$ points.

$$
\begin{align*}
\check{x}_0^{(0)} &=
\begin{bmatrix}
0 \\\\ 5
\end{bmatrix} \\\\
\check{x}_0^{(1)} &=
\begin{bmatrix}
0 \\\\ 5
\end{bmatrix} +
\sqrt{3}
\begin{bmatrix}
0.1 \\\\ 0
\end{bmatrix} =
\begin{bmatrix}
\frac{\sqrt{3}}{10} \\\\ 5
\end{bmatrix} \\\\
\check{x}_0^{(2)} &=
\begin{bmatrix}
0 \\\\ 5
\end{bmatrix} +
\sqrt{3}
\begin{bmatrix}
0 \\\\ 1.0
\end{bmatrix} =
\begin{bmatrix}
0 \\\\ 5+\sqrt{3}
\end{bmatrix} \\\\
\check{x}_0^{(3)} &=
\begin{bmatrix}
0 \\\\ 5
\end{bmatrix} - \sqrt{3}
\begin{bmatrix}
0.1 \\\\ 0
\end{bmatrix} =
\begin{bmatrix}
-\frac{\sqrt{3}}{10} \\\\ 5
\end{bmatrix} \\\\
\check{x}_0^{(4)} &=
\begin{bmatrix}
0 \\\\ 5
\end{bmatrix} -
\sqrt{3}
\begin{bmatrix}
0 \\\\ 1.0
\end{bmatrix} =
\begin{bmatrix}
0 \\\\ 5-\sqrt{3}
\end{bmatrix}
\end{align*}
$$

For the given $5$ sigma points at the time $k=0$, by vehicle motion update, update the $5$ corresponding sigma points.

$$
\begin{align*}
\hat{x}_1^{(0)} &=
\begin{bmatrix}
1 & 0.5 \\\\
0 & 1
\end{bmatrix}
\begin{bmatrix}
0 \\\\
5
\end{bmatrix}
-2
\begin{bmatrix}
0 \\\\
0.5
\end{bmatrix}=
\begin{bmatrix}
2.5 \\\\
4.0
\end{bmatrix}\\\\
\hat{x}_1^{(1)} &=
\begin{bmatrix}
1 & 0.5 \\\\
0 & 1
\end{bmatrix}
\begin{bmatrix}
\frac{\sqrt{3}}{10} \\\\
5
\end{bmatrix}
-2
\begin{bmatrix}
0 \\\\
0.5
\end{bmatrix}=
\begin{bmatrix}
\frac{\sqrt{3}}{10}+2.5 \\\\
4
\end{bmatrix}
\approx
\begin{bmatrix}
2.67 \\\\
4.0
\end{bmatrix}\\\\
\hat{x}_1^{(2)} &=
\begin{bmatrix}
1 & 0.5 \\\\
0 & 1
\end{bmatrix}
\begin{bmatrix}
0 \\\\
5+\sqrt{3}
\end{bmatrix}
-2
\begin{bmatrix}
0 \\\\
0.5
\end{bmatrix}=
\begin{bmatrix}
\frac{\sqrt{3}}{2}+2.5 \\\\
\sqrt{3}+4
\end{bmatrix}
\approx
\begin{bmatrix}
3.4 \\\\
5.67
\end{bmatrix}\\\\
\hat{x}_1^{(3)} &=
\begin{bmatrix}
1 & 0.5 \\\\
0 & 1
\end{bmatrix}
\begin{bmatrix}
-\frac{\sqrt{3}}{10} \\\\
5
\end{bmatrix}
-2
\begin{bmatrix}
0 \\\\
0.5
\end{bmatrix}=
\begin{bmatrix}
-\frac{\sqrt{3}}{10}+2.5 \\\\
4
\end{bmatrix}
\approx
\begin{bmatrix}
2.33 \\\\
4.0
\end{bmatrix}\\\\
\hat{x}_1^{(4)} &=
\begin{bmatrix}
1 & 0.5 \\\\
0 & 1
\end{bmatrix}
\begin{bmatrix}
0 \\\\
5-\sqrt{3}
\end{bmatrix}
-2
\begin{bmatrix}
0 \\\\
0.5
\end{bmatrix}=
\begin{bmatrix}
-\frac{\sqrt{3}}{2}+2.5 \\\\
-\sqrt{3}+4
\end{bmatrix}
\approx
\begin{bmatrix}
1.6 \\\\
2.3
\end{bmatrix}
\end{align*}
$$

Compute the weight $W$

$$
\begin{align*}
\frac{\lambda}{d+\lambda}&=
\frac
{\alpha^2(d+\kappa)-d}
{d+\alpha^2(d+\kappa)-d}
&
\frac{1}{2(d+\lambda)}&=
\frac{1}{2(d+\alpha^2(d+\kappa)-d)}\\\\ &=
1-\frac{d}{\alpha^2(d+\kappa)}
& &=
\frac{1}{2\alpha^2(d+\kappa)}\\\\ &=
\frac{1}{3}
& &=
\frac{1}{6}
\end{align*}
$$

Compute the mean of motion update $x_1^-$

$$
\begin{align*}
\hat{x}^-_1 &=
\sum^{2d}_{i=0} W_i^{(m)} \hat{x}_1^{(i)}
\\\\ &=
\frac{1}{3}
\begin{bmatrix}
2.5 \\\\
4.0
\end{bmatrix}
+
\frac{1}{6}
\begin{bmatrix}
2.67 \\\\
4.0
\end{bmatrix}
+
\frac{1}{6}
\begin{bmatrix}
3.4 \\\\
5.67
\end{bmatrix}
+
\frac{1}{6}
\begin{bmatrix}
2.33 \\\\
4.0
\end{bmatrix}
+
\frac{1}{6}
\begin{bmatrix}
1.6 \\\\
2.3
\end{bmatrix}\\\\ &=
\begin{bmatrix}
2.5 \\\\
4.0
\end{bmatrix}
\end{align*}
$$

Compute the covariance of motion update $\mathbf{\hat{P}}_{1,x}$ 

$$
\begin{align*}
\mathbf{\hat{P}}_{1,x} &=
\sum^{2d}_{i=0}
W^{(c)} (\hat{x}^{(i)}_1-{\hat{x}_1^-}) (\hat{x}^{(i)}_1-{\hat{x}_1^-})^\text{T}+\mathbf{Q}_0
\\\\ &=
\frac{1}{3}
\bigg(
    \begin{bmatrix}
      2.5 \\\\
      4.0
    \end{bmatrix}
    -
    \begin{bmatrix}
      2.5 \\\\
      4.0
    \end{bmatrix}
\bigg)
\bigg(
    \begin{bmatrix}
      2.5 \\\\
      4.0
    \end{bmatrix}
    -
    \begin{bmatrix}
      2.5 \\\\
      4.0
    \end{bmatrix}
\bigg)^\text{T}\\\\ & \quad +
\frac{1}{6}
\bigg(
    \begin{bmatrix}
      2.67 \\\\
      4.0
    \end{bmatrix}
    -
    \begin{bmatrix}
      2.5 \\\\
      4.0
    \end{bmatrix}
\bigg)
\bigg(
    \begin{bmatrix}
      2.67 \\\\
      4.0
    \end{bmatrix}
    -
    \begin{bmatrix}
      2.5 \\\\
      4.0
    \end{bmatrix}
\bigg)^\text{T}\\\\ & \quad + ... + 
\begin{bmatrix}
      0.1 & 0\\\\
      0 & 0.1
\end{bmatrix}\\\\ &=
\begin{bmatrix}
      0.36 & 0.5 \\\\
      0.5 & 1.1
\end{bmatrix}
\end{align*}
$$

By Cholesky decomposition to find the solution to the covariance matrix

$$
\begin{align*}
\mathbf{\hat{P}}_{1,x}&=
\mathbf{L}_1 \mathbf{L}_1^\text{T} \\\\
\begin{bmatrix}
      0.36 & 0.5 \\\\
      0.5 & 1.1
\end{bmatrix}&=
\begin{bmatrix}
      0.60 & 0.0 \\\\
      0.83 & 0.64
\end{bmatrix}
\begin{bmatrix}
      0.60 & 0.0 \\\\
      0.83 & 0.64
\end{bmatrix}^\text{T}
\end{align*}
$$

Given the covariance solutions, update the computed motion results with applied Gaussian noises.

$$
\begin{align*}
\check{x}_1^{(0)} &=
\begin{bmatrix}
2.5 \\\\
4
\end{bmatrix}\\\\
\check{x}_1^{(1)} &=
\begin{bmatrix}
2.5 \\\\
4
\end{bmatrix}
+
\sqrt{3}
\begin{bmatrix}
      0.60 \\\\
      0.83
\end{bmatrix}
\approx
\begin{bmatrix}
3.54 \\\\
5.44
\end{bmatrix}\\\\
\check{x}_1^{(2)} &=
\begin{bmatrix}
2.5 \\\\
4
\end{bmatrix}
+
\sqrt{3}
\begin{bmatrix}
      0.0 \\\\
      0.64
\end{bmatrix}
\approx
\begin{bmatrix}
2.5 \\\\
5.10
\end{bmatrix}\\\\
\check{x}_1^{(3)} &=
\begin{bmatrix}
2.5 \\\\
4
\end{bmatrix}
-
\sqrt{3}
\begin{bmatrix}
      0.60 \\\\
      0.83
\end{bmatrix}
\approx
\begin{bmatrix}
1.46 \\\\
2.56
\end{bmatrix}\\\\
\check{x}_1^{(4)} &=
\begin{bmatrix}
2.5 \\\\
4
\end{bmatrix}
-
\sqrt{3}
\begin{bmatrix}
      0.60 \\\\
      0.83
\end{bmatrix}
\approx
\begin{bmatrix}
2.5 \\\\
2.90
\end{bmatrix}
\end{align*}
$$

The observation $y_1^{(i)}$ is updated (without applied Gaussian noises).

$$
\begin{align*}
\hat{y}^{(i)}_1 &= h(\hat{x}^{(i)}_1) + \mathbf{v}_k
\\\\ &=
\begin{bmatrix}
1 & 0\\\\
0 & 0
\end{bmatrix}
\hat{x}^{(i)}_1
 + \mathbf{v}_k\\\\
\hat{y}^{(0)}_1&=
\begin{bmatrix}
1 & 0\\\\
0 & 0
\end{bmatrix}
\begin{bmatrix}
2.5 \\\\
4.0
\end{bmatrix}=
\begin{bmatrix}
2.5 \\\\
0
\end{bmatrix}\\\\
\hat{y}^{(1)}_1&=
\begin{bmatrix}
1 & 0\\\\
0 & 0
\end{bmatrix}
\begin{bmatrix}
3.54 \\\\
5.44
\end{bmatrix}=
\begin{bmatrix}
3.54 \\\\
0
\end{bmatrix}\\\\
\hat{y}^{(2)}_1&=
\begin{bmatrix}
1 & 0\\\\
0 & 0
\end{bmatrix}
\begin{bmatrix}
2.5 \\\\
5.10
\end{bmatrix}=
\begin{bmatrix}
2.5 \\\\
0
\end{bmatrix}\\\\
\hat{y}^{(3)}_1&=
\begin{bmatrix}
1 & 0\\\\
0 & 0
\end{bmatrix}
\begin{bmatrix}
1.46 \\\\
2.56
\end{bmatrix}=
\begin{bmatrix}
1.46 \\\\
0
\end{bmatrix}\\\\
\hat{y}^{(4)}_1&=
\begin{bmatrix}
1 & 0\\\\
0 & 0
\end{bmatrix}
\begin{bmatrix}
2.5 \\\\
2.90
\end{bmatrix}=
\begin{bmatrix}
2.5 \\\\
0
\end{bmatrix}
\end{align*}
$$

Computed the mean of observation. It is the same as directly measuring ${x}_1^-$.

$$
\begin{align*}
\hat{y}^-_1&= \sum^{2d}_{i=0}
W^{(m)}_i \hat{y}^{(i)}_1\\\\ &=
\frac{1}{3} 
\begin{bmatrix}
2.5 \\\\
0
\end{bmatrix}
+
\frac{1}{6} 
\begin{bmatrix}
3.54 \\\\
0
\end{bmatrix}
+
\frac{1}{6}
\begin{bmatrix}
2.5 \\\\
0
\end{bmatrix}
+
\frac{1}{6}
\begin{bmatrix}
1.46 \\\\
0
\end{bmatrix}
+
\frac{1}{6}
\begin{bmatrix}
2.5 \\\\
0
\end{bmatrix}
\\\\ &=
\begin{bmatrix}
2.5 \\\\
0
\end{bmatrix}
\end{align*}
$$

The covariance for the observation model $\hat{y}_1$ is computed.

$$
\begin{align*}
\mathbf{\hat{P}}_{1,y} &=
\sum^{2d}_{i=0}
W^{(c)} (\hat{y}^{(i)}_1-{\hat{y}_1^-}) (\hat{y}^{(i)}_1-{\hat{y}_1^-})^\text{T}+\mathbf{R}_0\\\\ &=
\frac{1}{3}
\bigg(
    \begin{bmatrix}
      2.5 \\\\
      0.0
    \end{bmatrix}
    -
    \begin{bmatrix}
      2.5 \\\\
      0.0
    \end{bmatrix}
\bigg)
\bigg(
    \begin{bmatrix}
      2.5 \\\\
      0.0
    \end{bmatrix}
    -
    \begin{bmatrix}
      2.5 \\\\
      0.0
    \end{bmatrix}
\bigg)^\text{T}\\\\ & \quad +
\frac{1}{6}
\bigg(
    \begin{bmatrix}
      3.54 \\\\
      0.0
    \end{bmatrix}
    -
    \begin{bmatrix}
      2.5 \\\\
      0.0
    \end{bmatrix}
\bigg)
\bigg(
    \begin{bmatrix}
      3.54 \\\\
      0.0
    \end{bmatrix}
    -
    \begin{bmatrix}
      2.5 \\\\
      0.0
    \end{bmatrix}
\bigg)^\text{T}
\\\\ & \quad + ... +
\begin{bmatrix}
  0.01 \\\\
  0.0
\end{bmatrix}\\\\ &=
0 + 
\frac{1}{6}
\begin{bmatrix}
  1.04 \\\\
  0.0
\end{bmatrix}^2+
\frac{1}{6}
\begin{bmatrix}
  0.0 \\\\
  0.0
\end{bmatrix}^2+
\frac{1}{6}
\begin{bmatrix}
  -1.04 \\\\
  0.0
\end{bmatrix}^2+
\frac{1}{6}
\begin{bmatrix}
  -0.0 \\\\
  0.0
\end{bmatrix}^2
+
\begin{bmatrix}
  0.01 \\\\
  0.0
\end{bmatrix}\\\\ &\approx
\begin{bmatrix}
  0.333 \\\\
  0.0
\end{bmatrix} + 
\begin{bmatrix}
  0.01 \\\\
  0.0
\end{bmatrix}\\\\ &=
0.343
\end{align*}
$$

Now compute the covariance between $\hat{x}_1$ and $\hat{y}_1$

$$
\begin{align*}
\mathbf{\hat{P}}_{1,xy} &=
\sum^{2d}_{i=0}
W^{(c)} (\hat{x}^{(i)}_1-{\hat{x}_1^-}) (\hat{y}^{(i)}_1-{\hat{y}_1^-})^\text{T}\\\\ &=
\frac{1}{3}
\bigg(
    \begin{bmatrix}
      2.5 \\\\
      4.0
    \end{bmatrix}
    -
    \begin{bmatrix}
      2.5 \\\\
      4.0
    \end{bmatrix}
\bigg)
\bigg(
    \begin{bmatrix}
      2.5 \\\\
      0.0
    \end{bmatrix}
    -
    \begin{bmatrix}
      2.5 \\\\
      0.0
    \end{bmatrix}
\bigg)^\text{T}\\\\ & \quad +
\frac{1}{6}
\bigg(
    \begin{bmatrix}
      3.54 \\\\
      4.0
    \end{bmatrix}
    -
    \begin{bmatrix}
      2.5 \\\\
      4.0
    \end{bmatrix}
\bigg)
\bigg(
    \begin{bmatrix}
      3.54 \\\\
      0.0
    \end{bmatrix}
    -
    \begin{bmatrix}
      2.5 \\\\
      0.0
    \end{bmatrix}
\bigg)^\text{T}
\\\\ & \quad + ...\\\\ & \approx
\begin{bmatrix}
  0.333 \\\\
  0.0
\end{bmatrix}
\end{align*}
$$

Then we can know the Kalman Gain $\mathbf{K}_1$

$$
\begin{align*}
\mathbf{K}_1 &= 
\mathbf{\hat{P}}_{1,xy} \mathbf{\hat{P}}_{1,y}^{-1} 
\\\\ &=
\frac{333}{1000} \times \frac{1000}{343}
\\\\ &=
0.970
\end{align*}
$$

The final result for $\mathbf{x}_1$ is known via the applied Kalman gain.

$$
\begin{align*}
\mathbf{x}_1&=
\mathbf{\hat{x}}_1 + \mathbf{K}_1 (y_1 - \hat{y}^-_1)
\\\\ &=
\begin{bmatrix}
2.5 \\\\
4
\end{bmatrix}
+
0.97
\begin{bmatrix}
2.5 - 2.5 \\\\
0
\end{bmatrix}
\\\\ &= 
\begin{bmatrix}
2.5 \\\\
4
\end{bmatrix}
\end{align*}
$$

### Discussion

Here we notice that Kalman Gain $\mathbf{K}$ is large. This is attributed to the beginning where $\mathbf{P}_0$ is large as well as the time interval $\Delta t = 0.5 s$. This gives a result of big spreads. The observation function $h$ is a simple directs measurement of the distance in $\mathbf{x}_k$, which accounts for value change sensitivity. If $h$ is not sensitive to $\mathbf{P}_{1,y}$, $\mathbf{K}$ approaches to $0.5$. 