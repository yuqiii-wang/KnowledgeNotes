# Kalman Filter

**Summary**: assumed a system having Gaussian noise covariances $\bold{Q}_k$ and $\bold{R}_k$ on state transformation $\bold{F}_k$ on $\bold{x}_k$ and state observation $\bold{z}_k$, respectively, 
given a sufficient number of iterations $k \rightarrow \infty$, by optimizing Kalman gain $\bold{K}$, 
the expected mean-squared error between the ground truth state $\bold{x}_k$ and estimation state $\bold{\hat{x}}_{k}$ should be minimized 
$$
arg \space \underset{\bold{K}}{min} \space 
E \big(
    ||\bold{x}_k-\bold{\hat{x}}_{k|k}||^2
    \big)
$$ 
with fluctuations compliant with its covariances $cov(\bold{x}_k-\bold{\hat{x}}_{k|k})$.

### Model assumptions

* Dynamic transformation

Given a state transformation $\bold{F}_k$ on the previous state $\bold{x}_{k-1}$, and added dynamic $\bold{B}_k \bold{u}_k$ ($\bold{B}_k$ is the action transformation and $\bold{u}_k$ is the action input), plus Gaussian noises $\bold{w}_k \sim N(0, \bold{Q}_k)$ 

$$
\bold{x}_{k}=
\bold{F}_k\bold{x}_{k-1} + \bold{B}_k \bold{u}_k + \bold{w}_k
$$

* Observation 

Given theoretical state observation/measurement transform $\bold{H}_k$ and the observed state $\bold{z}_k$, plus Gaussian noises $\bold{v}_k \sim N(0, \bold{R}_k)$

$$
\bold{z}_k=
\bold{H}_k \bold{x}_k + \bold{v}_k
$$

### Predict phase

Predicted (a priori) state estimate
$$
\bold{\hat{x}}_{k|k-1}=
\bold{F}_k\bold{x}_{k-1|k-1} + \bold{B}_k \bold{u}_k
$$

Predicted (a priori) estimate covariance
$$
\bold{\hat{P}}_{k|k-1}=
\bold{F}_k\bold{P}_{k-1|k-1} \bold{F}^\text{T}_k + \bold{Q}_k
$$

### Update phase

Innovation or measurement pre-fit residual
$$
\bold{\hat{y}}_k=
\bold{z}_k-\bold{H}_k \bold{\hat{x}}_{k|k-1}
$$

Innovation (or pre-fit residual) covariance
$$
\bold{{S}}_{k}=
\bold{H}_k \bold{\hat{P}}_{k|k-1} \bold{H}^\text{T}_k + \bold{R}_k
$$

Optimal Kalman gain
$$
\bold{K}_k=
\bold{\hat{P}}_{k|k-1} \bold{H}^\text{T}_k \bold{{S}}_{k}^{-1}
$$

Updated (a posteriori) state estimate
$$
\bold{x}_{k|k}=
\bold{\hat{x}}_{k|k-1} + \bold{K}_k \bold{\hat{y}}_k
$$

Updated (a posteriori) estimate covariance
$$
\bold{P}_{k|k}=
(\bold{I}-\bold{K}_k \bold{H}) \bold{\hat{P}}_{k|k-1}
$$

Measurement post-fit residual
$$
\bold{\hat{y}}_{k|k}=
\bold{z}_k - \bold{H}_k \bold{x}_{k|k}
$$

## Derivations

### Deriving the *posteriori* estimate covariance matrix

Starting with invariant on the error covariance:
$$
\begin{align*}
\bold{P}_{k|k}&=
cov(\bold{x}_k - \bold{\hat{x}}_{k|k})
\\ &=
cov \big(
        \bold{x}_k - (\bold{\hat{x}}_{k|k-1} + \bold{K}_{k} \bold{\hat{y}}_k)
    \big)
\\ &=
cov \big(
        \bold{x}_k - 
        (\bold{\hat{x}}_{k|k-1} + \bold{K}_{k} 
            (\bold{z}_k - \bold{H}_k \bold{\hat{x}}_{k|k-1})
        )
    \big)
\\ &=
cov \big(
        \bold{x}_k - 
        (\bold{\hat{x}}_{k|k-1} + \bold{K}_{k} 
            (\bold{H}_k\bold{x}_k + \bold{v}_k - \bold{H}_k \bold{\hat{x}}_{k|k-1}
            )
        )
    \big)
\\ &=
cov \big(
        (\bold{I}-\bold{K}_k\bold{H}_k)
        (\bold{x}_k - \bold{\hat{x}_{k|k-1}})
        - \bold{K}_k \bold{v}_k
    \big)
\\
\bold{v}_k & \text{ is uncorrelated with the other terms} 
\\ &=
cov \big(
        (\bold{I}-\bold{K}_k\bold{H}_k)
        (\bold{x}_k - \bold{\hat{x}_{k|k-1}})
    \big)
-
cov(\bold{K}_k \bold{v}_k)
\\
\text{by} & \text{ the properties of vector covariance}
\\ &=
(\bold{I}-\bold{K}_k\bold{H}_k)
cov(\bold{x}_k - \bold{\hat{x}_{k|k-1}})
(\bold{I}-\bold{K}_k\bold{H}_k)^\text{T}
+
\bold{K}_k
cov(\bold{v}_k)
\bold{K}_k^\text{T}
\\ &=
(\bold{I}-\bold{K}_k\bold{H}_k)
\bold{P}_{k|k-1}
(\bold{I}-\bold{K}_k\bold{H}_k)^\text{T}
+
\bold{K}_k
\bold{R}_k
\bold{K}_k^\text{T}
\end{align*}
$$

### Deriving Kalman gain

Starting from the minimization problem
$$
arg \space \underset{\bold{K}}{min} \space 
E \big(
    ||\bold{x}_k-\bold{\hat{x}}_{k|k}||^2
    \big)
$$ 

For $k=0,1,2,...,n$, 
given $\bold{x}_k \in \mathbb{R}^m$, 
by vector dot product as the squared operation, 
there is 
$$
\begin{align*}
\sum^n_{k=0}
||\bold{x}_k-\bold{\hat{x}}_{k|k}||^2&=
\sum^n_{k=0}
(\bold{x}_k-\bold{\hat{x}}_{k|k})^\text{T}
(\bold{x}_k-\bold{\hat{x}}_{k|k})
\\ &=
\sum^n_{k=0}
\bigg(
\begin{bmatrix}
x_{k,1} \\
x_{k,2} \\
\vdots \\
x_{k,m}
\end{bmatrix}
-
\begin{bmatrix}
\hat{x}_{k,1} \\
\hat{x}_{k,2} \\
\vdots \\
\hat{x}_{k,m}
\end{bmatrix}
\bigg)^\text{T}
\bigg(
\begin{bmatrix}
x_{k,1} \\
x_{k,2} \\
\vdots \\
x_{k,m}
\end{bmatrix}
-
\begin{bmatrix}
\hat{x}_{k,1} \\
\hat{x}_{k,2} \\
\vdots \\
\hat{x}_{k,m}
\end{bmatrix}
\bigg)
\end{align*}
$$ 

So that, the expected error is 
$$
E \big(
    ||\bold{x}_k-\bold{\hat{x}}_{k|k}||^2
    \big)=
\frac{\sum^n_{k=0}
||\bold{x}_k-\bold{\hat{x}}_{k|k}||^2}
{n}
$$

$cov(\bold{x}_k - \bold{\hat{x}}_{k|k})$ (the formal writing should be $cov(\bold{x}_k - \bold{\hat{x}}_{k|k}, \bold{x}_k - \bold{\hat{x}}_{k|k})$, here is a shorthand note) describes a covariance of a vector $\bold{x}_k - \bold{\hat{x}}_{k|k}$ with the vector itself, so that 
$$
cov(\bold{x}_k - \bold{\hat{x}}_{k|k},)=
\begin{bmatrix}
\sigma^2_{1} & 0 &  & 0 \\
0 & \sigma^2_{2} &  & 0 \\
 &  & \ddots & 0 \\
0 & 0 &  &  \sigma^2_{m} \\
\end{bmatrix}
$$

Each covariance's entry $\sigma_i^2$ is the mean of each error vector element's squared sum $\frac{1}{n} \sum_{k=0}^n(x_{k,i}-\hat{x}_{k,i})^2$. Therefore, 
$$
E \big(
    ||\bold{x}_k-\bold{\hat{x}}_{k|k}||^2
    \big)=
tr \big(
    cov(\bold{x}_k - \bold{\hat{x}}_{k|k})
    \big)
$$ 
where $tr$ denotes the trace of the covariance matrix.

Consequently, the minimization problem becomes:
$$
arg \space \underset{\bold{K}}{min} \space
tr \big(
    cov(\bold{x}_k - \bold{\hat{x}}_{k|k})
    \big)
$$

Remember, we have obtained the covariance expression for this error, the *posteriori* estimate covariance matrix $\bold{P}_{k|k}=cov(\bold{x}_k - \bold{\hat{x}}_{k|k})$, with the engagement of Kalman gain $\bold{K}$, so that, by setting its first order derivative to zero, there is
$$
\begin{align*}
\frac{\partial tr(\bold{P}_{k|k})}{\partial \bold{K}}&=
\frac{\partial }{\partial \bold{K}}
tr
\big(
    cov(\bold{x}_k - \bold{\hat{x}}_{k|k})
\big)
\\ &=
\frac{\partial }{\partial \bold{K}}
tr
\big(
    (\bold{I}-\bold{K}_k\bold{H}_k)
    \bold{P}_{k|k-1}
    (\bold{I}-\bold{K}_k\bold{H}_k)^\text{T}
    +
    \bold{K}_k
    \bold{R}_k
    \bold{K}_k^\text{T}
\big)
\\ &=
\frac{\partial }{\partial \bold{K}}
tr
\big(
    \bold{P}_{k|k-1}-\bold{K}_k\bold{H}_k\bold{P}_{k|k-1}
    - \bold{P}_{k|k-1}\bold{H}_k^\text{T}\bold{K}_k^\text{T}
    + \bold{K}_k \bold{S}_k \bold{K}_k^\text{T}
\big)
\\ &=
2(\bold{H}_k\bold{P}_{k|k-1})^\text{T}
+
2 \bold{K}_k \bold{S}_k
\\ &= 0
\end{align*}
$$

The Kalman gain $\bold{K}_k$ can be computed:
$$
\bold{K}_k = 
-(\bold{H}_k\bold{P}_{k|k-1})^\text{T} \bold{S}_k^{-1}
$$

## Convergence

In the long term, the mean squared error should be nearly zero given a sufficient number of iterations.  
$$
E \big(
    ||\bold{x}_k-\bold{\hat{x}}_{k|k}||^2
    \big)
\approx 0
$$

The $cov(\bold{x}_k - \bold{\hat{x}}_{k|k})$ 's expression taking into consideration $\bold{P}$ (accounts for dynamic transformation covariance $\bold{Q}$) and $\bold{H}$ (accounts for observation transformation covariance $\bold{R}$) is viewed as the ground truth.

This means, the ratio Kalman filter $\bold{K}_k$ is a compromise between the dynamic model's and measurement's Gaussian distribution samplings. The correction by $\bold{K}_k$ can only be optimal when $\bold{Q}$ and $\bold{R}$ are accurate (the fluctuations of $\bold{\hat{x}}_{k|k-1}$ and $\bold{z_k}$ are contained in $\bold{Q}$ and $\bold{R}$).

In other words, $\bold{K}_k \bold{\hat{y}}_k$ can be a good compensation to $\bold{\hat{x}}_{k|k-1}$ when $\bold{\hat{y}}_k$ is contained in $\bold{R}$, and $\bold{K}_k$ in $\bold{R}$ and $\bold{Q}$, respectively.
$$
\bold{x}_{k|k}=
\bold{\hat{x}}_{k|k-1} + \bold{K}_k \bold{\hat{y}}_k
$$

![kalman_filter](imgs/kalman_filter.png "kalman_filter")

## Example

Distance $x$ and velocity $\dot{x}$ of a vehicle is given below
$$
\bold{x} = 
\begin{bmatrix}
x \\
\dot{x}
\end{bmatrix}
$$

Vehicle drives with a constant acceleration $a_k$ between two timesteps $k-1$ and $k$, following normal distribution with mean $0$ and standard deviation $\sigma_a$. Given Newton's laws of motion:
$$
\bold{x}_k = \bold{F} \bold{x}_{k-1} + \bold{B} a_k
$$
where
$$
\bold{F} = 
\begin{bmatrix}
1 & \Delta t \\
0 & 1
\end{bmatrix}
, \space
\bold{B} = 
\begin{bmatrix}
\frac{1}{2} \Delta t^2 \\
\Delta t
\end{bmatrix}
$$

Given $a_k$ following normal distribution, there is (remember $E(a_k)=0$, so that in the dynamic model, $\bold{B}\bold{u}$ is removed)
$$
\bold{x}_k = \bold{F} \bold{x}_{k-1} + \bold{w}_k
$$
where $\bold{w}_k \sim N(0, \bold{Q})$ (remember the noise $\bold{w}_k$ is associated with the acceleration, so only $\sigma_{a_k}\bold{B}$ is considered as the standard deviation, not included $\bold{F}_k$), in which 
$$
\bold{Q} = \sigma_{a_k}\bold{B} \bold{B}^\text{T} \sigma_{a_k} =
\begin{bmatrix}
\frac{1}{4}\Delta t^4 & \frac{1}{2}\Delta t^3 \\
\frac{1}{2}\Delta t^3 & \Delta t^2
\end{bmatrix}
\sigma_{a_k}^2
$$

Since $\bold{B}\bold{B}^\text{T}$ is not full ranked ($R_1 = [\frac{1}{4}\Delta t^4, \frac{1}{2}\Delta t^3] = \frac{1}{2}\Delta t^3 R_2$) hence 
$\bold{w}_k \sim \bold{B} \cdot N(0, \bold{Q}) \sigma_{a_k}^2 \sim \bold{B} \cdot N (0, \sigma_{a_k}^2)$

Here defines observation 
$$
\bold{z}_k = \bold{H} \bold{x}_k + \bold{v}_k
$$

where $\bold{H}=[1 \quad 0]$, that only the traveled distance is measured.

Here $\bold{R} = E[\bold{v}_k \bold{v}_k^T] = [\sigma_{z}^2]$, since $\bold{z}_k$ is one-dimensional only measuring traveled distance noises.

$\bold{P}_{0|0}$ is the initial covariance matrix when $k=0$. In this case, assume we have high confidence of the initial vehicle state, so that $\sigma_x=0$ and $\sigma_{\dot{x}}=0$
$$
\bold{P}_{0|0} = 
\begin{bmatrix}
\sigma_x^2 & 0 \\
0 & \sigma_{\dot{x}}^2
\end{bmatrix}=
\begin{bmatrix}
0 & 0 \\
0 & 0
\end{bmatrix}
$$

Here assumes a vehicle starts from $0$ distance with a velocity $v_0$
$$
\bold{\hat{x}}_{0|0} = 
\begin{bmatrix}
x \\
\dot{x}
\end{bmatrix}=
\begin{bmatrix}
0 \\
v_0
\end{bmatrix}
$$


### One iteration

The below computation expressions removed the subscript $k$ if a matrix is constant for all $k=0,1,2,...,\infty$.

* Prediction

$$
\begin{align*}

\bold{\hat{x}}_{1|0}&=
\bold{F} \bold{x}_{0|0} + \bold{B}_k \bold{u}_k
\\ &=
\begin{bmatrix}
1 & \Delta t \\
0 & 1
\end{bmatrix}
\begin{bmatrix}
0 \\
v_0
\end{bmatrix}
\\ &=
\begin{bmatrix}
\Delta t v_0 \\
v_0
\end{bmatrix}
\end{align*}
$$

$$
\begin{align*}
\bold{\hat{P}}_{1|0}&=
\bold{F} \bold{P}_{0|0} \bold{F}^\text{T} + \bold{Q}
\\ &=
\begin{bmatrix}
1 & \Delta t \\
0 & 1
\end{bmatrix}
\begin{bmatrix}
\sigma_x^2 & 0 \\
0 & \sigma_{\dot{x}}^2
\end{bmatrix}
\begin{bmatrix}
1 & 0 \\
\Delta t & 1
\end{bmatrix}
+ \bold{Q}
\\ &=
\begin{bmatrix}
\sigma_x^2 & \Delta t \sigma_{\dot{x}}^2 \\
0 & \sigma_{\dot{x}}^2 
\end{bmatrix}
\begin{bmatrix}
1 & 0 \\
\Delta t & 1
\end{bmatrix}
+ \bold{Q}
\\ &=
\begin{bmatrix}
\sigma_x^2+\Delta t^2 \sigma_{\dot{x}}^2  & \Delta t \sigma_{\dot{x}}^2 \\
\Delta t \sigma_{\dot{x}}^2 & \sigma_{\dot{x}}^2 
\end{bmatrix}
+ \bold{Q}
\\ &=
\begin{bmatrix}
\sigma_x^2+\Delta t^2 & \Delta t \sigma_{\dot{x}}^2 \\
\Delta t^2 \sigma_{\dot{x}}^2 & \sigma_{\dot{x}}^2 
\end{bmatrix}
+
\begin{bmatrix}
\frac{1}{4}\Delta t^4 & \frac{1}{2}\Delta t^3 \\
\frac{1}{2}\Delta t^3 & \Delta t^2
\end{bmatrix}
\sigma_{a_k}^2
\\
\text{If} & \text{ both } \sigma_x=0 \text{ and } \sigma_{\dot{x}}=0
\\ &=
\begin{bmatrix}
\frac{1}{4}\Delta t^4 & \frac{1}{2}\Delta t^3 \\
\frac{1}{2}\Delta t^3 & \Delta t^2
\end{bmatrix}
\sigma_{a_k}^2
\\ &=
\bold{Q}
\end{align*}
$$

* Update

Assume that observation is 
$$
\bold{z}_1=
\begin{bmatrix}
\Delta t v_0 \pm \sigma_z\\
0
\end{bmatrix}
$$

So that
$$
\begin{align*}

\bold{\hat{y}}_1&=
\bold{z}_1-\bold{H} \bold{\hat{x}}_{1|0}
\\ &=
\begin{bmatrix}
\Delta t v_0 \pm \sigma_z\\
0
\end{bmatrix}
-
\begin{bmatrix}
\Delta t v_0 \\
0
\end{bmatrix}
\\ &=
\begin{bmatrix}
\pm \sigma_z \\
0
\end{bmatrix}
\end{align*}
$$

$$
\begin{align*}
\bold{{S}}_{1}&=
\bold{H} \bold{\hat{P}}_{1|0} \bold{H}^\text{T} + \bold{R}
\\ &=
\begin{bmatrix}
1 & 0 
\end{bmatrix}
\begin{bmatrix}
\frac{1}{4}\Delta t^4 & \frac{1}{2}\Delta t^3 \\
\frac{1}{2}\Delta t^3 & \Delta t^2
\end{bmatrix}
\sigma_{a_k}^2
\begin{bmatrix}
1 \\
0 
\end{bmatrix}
+
\sigma_{z}^2
\\ &=
\begin{bmatrix}
\frac{1}{4}\Delta t^4 & \frac{1}{2}\Delta t^3 
\end{bmatrix}
\sigma_{a_k}^2
\begin{bmatrix}
1 \\
0 
\end{bmatrix}
+
\sigma_{z}^2
\\ &=
\frac{1}{4}\Delta t^4 \sigma_{a_k}^2
+
\sigma_{z}^2
\end{align*}
$$

$$
\begin{align*}
\bold{K}_1&=
\bold{\hat{P}}_{1|0} \bold{H}^\text{T} \bold{{S}}_{1}^{-1}
\\ &=
\frac{1}{
    \frac{1}{4}\Delta t^4 \sigma_{a_k}^2
    +
    \sigma_{z}^2
    }
\begin{bmatrix}
\frac{1}{4}\Delta t^4 & \frac{1}{2}\Delta t^3 \\
\frac{1}{2}\Delta t^3 & \Delta t^2
\end{bmatrix}
\sigma_{a_k}^2
\begin{bmatrix}
1 \\
0 
\end{bmatrix}
\\ &=
\frac{1}{
    \frac{1}{4}\Delta t^4 \sigma_{a_k}^2
    +
    \sigma_{z}^2
    }
\begin{bmatrix}
\frac{1}{4}\Delta t^4 & \frac{1}{2}\Delta t^3 
\end{bmatrix}
\sigma_{a_k}^2
\end{align*}
$$

$$
\begin{align*}

\bold{x}_{1|1}&=
\bold{\hat{x}}_{1|0} + \bold{K}_1 \bold{\hat{y}}_1
\\ &=
\begin{bmatrix}
\Delta t v_0 \\
v_0
\end{bmatrix}
+
\frac{1}{
    \frac{1}{4}\Delta t^4 \sigma_{a_k}^2
    +
    \sigma_{z}^2
    }
\begin{bmatrix}
\frac{1}{4}\Delta t^4 & \frac{1}{2}\Delta t^3 
\end{bmatrix}
\sigma_{a_k}^2
\begin{bmatrix}
\pm \sigma_z \\
0
\end{bmatrix}
\\ &=
\begin{bmatrix}
\Delta t v_0 \\
v_0
\end{bmatrix}
+
\frac
{
    1
}
{
    \frac{1}{4}\Delta t^4 \sigma_{a_k}^2
    +
    \sigma_{z}^2
}
\begin{bmatrix}
        \pm \sigma_z
        \frac{1}{4}\Delta t^4
        \sigma_{a_k}^2
    \\
        0
    \end{bmatrix}
\end{align*}
$$
