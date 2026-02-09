# Pre-integration in IMU

IMU's data comes in with much higher frequencies than SLAM recording/computing keyframe. 
Pre-integration can integrate/process/fuse recent history IMU data within the time range covered by a keyframe, or between two keyframes $K_i$ and $K_j$.

There are noises in IMU: measurement noises between two keyframes $\mathbf{\eta}_{ij}$ and zero offset/biases to angular and linear movements $\mathbf{b}_{\omega}$ and $\mathbf{b}_{a}$. During integration, such noises (assumed Gaussian noises) can be approximated by linear increments, hence avoided repeated noise computation.

Covariances $\mathbf{\Sigma}$ for the aforementioned noises are used to dynamically scale the percentage of the noise components in estimating the IMU's state.

In other words, IMU reading $\hat{W}_{ij}$ contains a true reading $\overline{W}_{ij}$ and a reading noise $\delta W$. 
Observation (such as pose estimate from point cloud) $W_{ij}$ is to be adjusted to make residual $\mathbf{r}_W$ minimized.

$\delta W$ can be approximated by linear increment. 
The covariance $\mathbf{\Sigma}_W$ of $\delta W$ scales the percentage of noise that, a large $\mathbf{\Sigma}_W$ means a large noise $\delta W$; accordingly, $W_{ij}$ should be updated to a larger value- to compensate this noise.

$$
\begin{align*}
    \mathbf{r}_W &= W_{ij} - \hat{W}_{ij} \\\\
      &= W_{ij} - (\overline{W}_{ij} + \delta W) \\\\
\end{align*}
$$

## Preliminaries and Some Notations

Lie algebra to Lie group mapping is an exponential mapping $so(3) \rightarrow SO(3)$:

$$
e^{\mathbf{\phi}^\wedge} = \exp(\mathbf{\phi}^\wedge) = \text{Exp}(\mathbf{\phi}) = \mathbf{R}
$$

and its inverse is

$$
\mathbf{\phi} = \ln(\mathbf{R}^\vee) = \text{Log}(\mathbf{R})
$$

Baker-Campbell-Hausdorff (BCH) formula describes the relationship between Lie algebra plus/minus operations and Lie group multiplication.

$$
e^{(\mathbf{\phi}+\Delta\mathbf{\phi})^\wedge}
\approx
e^{(\mathbf{J}_l\Delta\mathbf{\phi})^\wedge}
e^{\mathbf{\phi}^\wedge}=
e^{\mathbf{\phi}^\wedge}
e^{(\mathbf{J}_r\Delta\mathbf{\phi})^\wedge}
$$

where $\mathbf{J}_l(\mathbf{\phi})=\mathbf{J}_l(\theta \mathbf{n})=\frac{\sin\theta}{\theta} I + (1 - \frac{sin\theta}{\theta})\mathbf{n}\mathbf{n}^\text{T} + \frac{1-cos\theta}{\theta}\mathbf{n}^{\wedge}$,
inside $\mathbf{\phi}=\theta \mathbf{n}$ describes the length and direction. Its inverse is $\mathbf{J}_l^{-1}(\theta \mathbf{n}) = \frac{\theta}{2}cot\frac{\theta}{2}I + (1-\frac{\theta}{2}cot\frac{\theta}{2})\mathbf{n}\mathbf{n}^\text{T} - \frac{\theta}{2}\mathbf{n}^{\wedge}$.

When $\delta\mathbf{\phi}$ is small, there is approximation $\exp(\delta\mathbf{\phi}^\wedge) \approx \mathbf{I}+\delta\mathbf{\phi}^\wedge$

The left Jacobian and right Jacobian relationship is $\mathbf{J}_l(\mathbf{\phi})=\mathbf{J}_r(-\mathbf{\phi})$.

The adjoint relationship in Lie group is $\mathbf{R}^\top \text{Exp}(\mathbf{\phi})\mathbf{R}=\text{Exp}(\mathbf{R}^\top\mathbf{\phi})$.


## Summary and Approximation Error Discussions

Start from kinematics for rotation ${\mathbf{R}}$, velocity ${\mathbf{v}}$ and position ${\mathbf{p}}$, assumed the true values can be decomposed of subtracting zero drifts and Gaussian noises from reading.

$$
\begin{align*}
    {\mathbf{R}}(t_0+\Delta t) 
    &= \mathbf{R}(t_0) e^{\mathbf{\omega}^\wedge(t_0)\Delta t} \\\\
    &= \mathbf{R}(t_0)
    e^{ \big(\hat{\mathbf{\omega}}(t_0)- \mathbf{b}\_\mathbf{\omega} - \mathbf{\eta}\_\mathbf{\omega} \big)^\wedge \Delta t} 
    \\\\
    {\mathbf{v}}(t_0+\Delta t) &= \mathbf{v}(t_0) + \mathbf{g}\_\text{earth} \Delta t
    + \mathbf{a}\Delta t \\\\
    &= \mathbf{v}(t_0) + \mathbf{g}\_\text{earth} \Delta t
    + \mathbf{R}(t_0) \cdot \big( \hat{\mathbf{a}}(t_0) - \mathbf{b}\_\mathbf{a}(t_0) - \mathbf{\eta}\_\mathbf{a}(t_0) \big) \Delta t
    \\\\
    {\mathbf{p}}(t_0+\Delta t) 
    &= \mathbf{p}(t_0) + \mathbf{v}(t_0)\Delta t + \frac{1}{2} \mathbf{a}(t_0) \Delta t^2 \\\\
    &= \mathbf{p}(t_0) + \mathbf{v}(t_0)\Delta t +
    \frac{1}{2} \mathbf{g}_{\text{earth}} \Delta t^2
    + \mathbf{R}(t_0) \cdot \big( \hat{\mathbf{a}}(t_0) - \frac{1}{2} \mathbf{b}\_\mathbf{a}(t_0) - \mathbf{\eta}\_\mathbf{a}(t_0) \big) \Delta t^2
\end{align*}
$$

Approximate the true value by first order (BCH in Lie algebra), so that the true value for rotation change ${\Delta{\mathbf{R}}_{ij}}$ can be expressed as the reading ${\Delta\hat{\mathbf{R}}_{ij}}$ removing the noise $\text{Exp} \big( -\delta\mathbf{\phi}_{ij} \big)$.

$$
    {\Delta{\mathbf{R}}_{ij}}=
    {\Delta\hat{\mathbf{R}}_{ij}}
    \prod^{j-1}_{k=i} 
    \text{Exp} \big(
    -{\Delta\hat{\mathbf{R}}_{k+1,k+2}}  \mathbf{J}_{rk} \mathbf{\eta}_{\mathbf{\omega},k} \Delta t \big)=
    {\Delta\hat{\mathbf{R}}_{ij}}
    \text{Exp} \big( -\delta\mathbf{\phi}_{ij} \big)
, \qquad
  \text{where } \Delta\hat{\mathbf{R}}_{ij} =
  \prod^{j-1}_{k=i}
  e^{ \big( (\hat{\mathbf{\omega}}_k- \mathbf{b}_{\mathbf{\omega},k})\Delta t 
  \big)^\wedge }
$$

Similarly, velocity and position changes can be expressed as $\Delta{\mathbf{v}}_{ij} = \Delta\hat{\mathbf{v}}_{ij} - \delta \mathbf{v}_{ij}$ and $\Delta{\mathbf{p}}_{ij} = \Delta\hat{\mathbf{p}}_{ij} + \delta \mathbf{p}_{ij}$.

The approximation error is shown as below:

* The rotation is approximated by the first order BCH $e^{(\mathbf{\phi}+\Delta\mathbf{\phi})^\wedge} \approx e^{(\mathbf{J}_l\Delta\mathbf{\phi})^\wedge} e^{\mathbf{\phi}^\wedge}$.
The higher order errors are discarded.

* When $\delta\mathbf{\phi}$ is small, there is approximation $\exp(\delta\mathbf{\phi}^\wedge) \approx \mathbf{I}+\delta\mathbf{\phi}^\wedge$.
This approximation derives from $\sin(||\overrightarrow{\mathbf{\omega}}||t) \rightarrow ||\overrightarrow{\mathbf{\omega}}||$ and $\cos(||\overrightarrow{\mathbf{\omega}}||t) \rightarrow 1$, hence the error grows as $\delta\mathbf{\phi}$ grows, which means that if IMU reading interval is large, the approximation result is not accurate.

Further dissect the error term, that indicates the added error is proportional to the rotation first order rate ($\mathbf{J}_{rk}$ is computable by Rodrigue's formula as the first order rotation rate): 
* an accumulated Gaussian noise term, $\sum_{k=i}^{j-2} \Delta\hat{\mathbf{R}}_{k+1,j-1}^\top \mathbf{J}_{rk} \mathbf{\eta}_{\mathbf{\omega},k} \Delta t$
* the last time Wiener process error $\Delta\hat{\mathbf{R}}_{j,j}^\top \mathbf{J}_{r,j-1} \mathbf{\eta}_{\mathbf{\omega},j-1} \Delta t = \mathbf{J}_{r,j-1} \mathbf{\eta}_{\mathbf{\omega},j-1} \Delta t$

$$
\begin{align*}
\text{Exp} \big( \delta\mathbf{\phi}_{ij} \big)&=
    \prod^{j-1}_{k=i} 
    \text{Exp} \big(
    -{\Delta\hat{\mathbf{R}}_{k+1}^\top}  \mathbf{J}_{rk} \mathbf{\eta}_{\mathbf{\omega},k} \Delta t \big)&=
    \sum_{k=i}^{j-1}
    {\Delta\hat{\mathbf{R}}_{k+1,j}^\top}  \mathbf{J}_{rk} \mathbf{\eta}_{\mathbf{\omega},k} \Delta t&=
    \sum_{k=i}^{j-2} \Big(
    \underbrace{\Delta\hat{\mathbf{R}}_{k+1,j-1}^\top}_{
    \big( \Delta\hat{\mathbf{R}}_{k+1,j-1} \Delta\hat{\mathbf{R}}_{j-1,j} \big)^\top
    }  \mathbf{J}_{rk} \mathbf{\eta}_{\mathbf{\omega},k} \Delta t \Big)
    + \underbrace{\Delta\hat{\mathbf{R}}_{j,j}^\top}_{=\mathbf{I}}  \mathbf{J}_{r,j-1} \mathbf{\eta}_{\mathbf{\omega},j-1} \Delta t
\end{align*}
$$

Covariance is propagated via Wiener process

$$
\mathbf{\Sigma}_j = 
\Delta\hat{\mathbf{R}}_{j-1,j}^\top \mathbf{\Sigma}_{j-1} \Delta\hat{\mathbf{R}}_{j-1,j}+
\mathbf{J}_{r,j-1}^\top \mathbf{\Sigma}_{\mathbf{\eta}_{\mathbf{\omega},j-1}} \mathbf{J}_{r,j-1} 
$$

For dynamic biases, convert the biases to a true value plus Gaussian noise: $\hat{\mathbf{b}}_{\mathbf{\omega},k} \rightarrow \mathbf{b}_{\mathbf{\omega},k} + \delta\mathbf{b}_{\mathbf{\omega},k}$ and $\hat{\mathbf{b}}_{\mathbf{a},k} \rightarrow \mathbf{b}_{\mathbf{a},k} + \delta\mathbf{b}_{\mathbf{a},k}$.
Then, compute the corrected $\Delta\mathbf{R}, \Delta\mathbf{v}, \Delta\mathbf{p}$ considered linearized acceleration and gyro biases.

In other words, $\delta\mathbf{b}_{\mathbf{\omega},k}$ and $\delta\mathbf{b}_{\mathbf{a},k}$ are approximated by first order derivative increment.

$$
\Delta\hat{\mathbf{R}}  = \Delta\mathbf{R} \frac{\partial \mathbf{R}}{\partial \mathbf{b}\_\omega} \Delta\mathbf{b}\_\omega
,\qquad
\Delta\hat{\mathbf{v}} = \Delta\mathbf{v} + \frac{\partial \mathbf{v}}{\partial \mathbf{b}_a} \Delta\mathbf{b}_a + \frac{\partial \mathbf{v}}{\partial \mathbf{b}\_\omega} \Delta\mathbf{b}\_\omega
, \qquad
\Delta\hat{\mathbf{p}} = \Delta\mathbf{p} + \frac{\partial \mathbf{p}}{\partial \mathbf{b}_a} \Delta\mathbf{b}_a + \frac{\partial \mathbf{p}}{\partial \mathbf{b}\_\omega} \Delta\mathbf{b}\_\omega
$$

## Definition

Typically, there are five variables to consider in an IMU system: rotation $\mathbf{R}$, translation $\mathbf{p}$, angular velocity $\mathbf{\omega}$, linear velocity $\mathbf{v}$ and acceleration $\mathbf{a}$:

$$
\begin{align*}
    \dot{\mathbf{R}} &= \mathbf{R} \mathbf{\omega}^\wedge \\\\
    \dot{\mathbf{p}} &= \mathbf{v} \\\\
    \dot{\mathbf{v}} &= \mathbf{a}
\end{align*}
$$

Take integral starting from $t_0$ to $t_0+\Delta t$:

$$
\begin{align*}
    {\mathbf{R}}(t_0+\Delta t) &= \mathbf{R}(t_0) e^{\mathbf{\omega}^\wedge(t_0) \Delta t} \\\\
    {\mathbf{v}}(t_0+\Delta t) &= \mathbf{v}(t_0) + \mathbf{a}\Delta t \\\\
    {\mathbf{p}}(t_0+\Delta t) &= \mathbf{p}(t_0) + \mathbf{v}(t_0)\Delta t + \frac{1}{2} \mathbf{a}(t_0) \Delta t^2
\end{align*}
$$

The estimate of angular velocity $\hat{\mathbf{\omega}}$ and acceleration $\hat{\mathbf{a}}$ can be affected by earth's gravity $\mathbf{g}_{\text{earth}}$ 
and Gaussian noises $\mathbf{\eta}\_\mathbf{a}, \mathbf{\eta}\_\mathbf{\omega}$. Consider IMU calibration zero offset $\mathbf{b}\_\mathbf{\omega}, \mathbf{b}\_\mathbf{a}$, there are

$$
\begin{align*}
\hat{\mathbf{\omega}}(t) &= 
\mathbf{\omega}(t) + \mathbf{b}\_\mathbf{\omega}(t) + \mathbf{\eta}\_\mathbf{\omega}(t) \\\\
\hat{\mathbf{a}}(t) &= 
\mathbf{R}(t) \cdot \big( \mathbf{a}(t) - \mathbf{g}_{\text{earth}} \big) + \mathbf{b}\_\mathbf{a}(t) + \mathbf{\eta}\_\mathbf{a}(t) \\\\
\end{align*}
$$

* The continuous integral model is

$$
\begin{align*}
    {\mathbf{R}}(t_0+\Delta t) 
    &= \mathbf{R}(t_0) e^{\mathbf{\omega}^\wedge(t_0)\Delta t} \\\\
    &= \mathbf{R}(t_0)
    e^{ \big(\hat{\mathbf{\omega}}(t_0)- \mathbf{b}\_\mathbf{\omega} - \mathbf{\eta}\_\mathbf{\omega} \big)^\wedge \Delta t} 
    \\\\
    {\mathbf{v}}(t_0+\Delta t) &= \mathbf{v}(t_0) + \mathbf{g}\_\text{earth} \Delta t
    + \mathbf{a}\Delta t \\\\
    &= \mathbf{v}(t_0) + \mathbf{g}\_\text{earth} \Delta t
    + \mathbf{R}(t_0) \cdot \big( \hat{\mathbf{a}}(t_0) - \mathbf{b}\_\mathbf{a}(t_0) - \mathbf{\eta}\_\mathbf{a}(t_0) \big) \Delta t
    \\\\
    {\mathbf{p}}(t_0+\Delta t) 
    &= \mathbf{p}(t_0) + \mathbf{v}(t_0)\Delta t + \frac{1}{2} \mathbf{a}(t_0) \Delta t^2 \\\\
    &= \mathbf{p}(t_0) + \mathbf{v}(t_0)\Delta t +
    \frac{1}{2} \mathbf{g}_{\text{earth}} \Delta t^2
    + \mathbf{R}(t_0) \cdot \big( \hat{\mathbf{a}}(t_0) - \frac{1}{2} \mathbf{b}\_\mathbf{a}(t_0) - \mathbf{\eta}\_\mathbf{a}(t_0) \big) \Delta t^2
\end{align*}
$$

where $\mathbf{g}_{\text{earth}}$ is a constant only having non-zero value at the vertical direction dimension, hence, irrelevant of rotation by $\mathbf{R}(t)$.

* The discrete model (replace the continuous interval $[t_0, t_0+\Delta t]$ with $k=i,i+1,i+2,...,j$, where $k$ is the IMU sensor reading index between the $i$-th and $j$-th keyframes; in other words, $\Delta t_{ij}=\sum_{k=i}^{j-1}\Delta t$, where $\Delta t$ is the IMU sensor reading interval)

$$
\begin{align*}
\mathbf{R}_j &= 
\mathbf{R}\_i \prod^{j-1}_{k=i}
e^{ \big(\hat{\mathbf{\omega}}_k- \mathbf{b}_{\mathbf{\omega},k} - \mathbf{\eta}_{\mathbf{\omega},k} \big)^\wedge \Delta t} \\\\
\mathbf{v}_j &= \mathbf{v}\_i + \mathbf{g}\_\text{earth} \Delta t_{ij} + 
\sum_{k=i}^{j-1} \mathbf{R}_{k} \cdot \big( \hat{\mathbf{a}}_k - \mathbf{b}_{\mathbf{a},k} - \mathbf{\eta}_{\mathbf{a},k} \big) \Delta t \\\\
\mathbf{p}_j &= \mathbf{p}\_i + \sum_{k=i}^{j-1} \mathbf{v}_k \Delta t + 
\frac{1}{2} \mathbf{g}\_\text{earth} \Delta t_{ij}^2 + 
\frac{1}{2} \sum_{k=i}^{j-1} \mathbf{R}_{k} \cdot \big( \hat{\mathbf{a}}_k - \mathbf{b}_{\mathbf{a},k} - \mathbf{\eta}_{\mathbf{a},k} \big) \Delta t^2
\end{align*}
$$

* The differential model is

$$
\begin{align*}
\Delta\mathbf{R}_{ij} & \overset{\Delta}{=} \mathbf{R}\_i^\top \mathbf{R}_j =
\prod^{j-1}_{k=i}
e^{ \big(\hat{\mathbf{\omega}}_k- \mathbf{b}_{\mathbf{\omega},k} - \mathbf{\eta}_{\mathbf{\omega},k} \big)^\wedge \Delta t} \\\\
\Delta \mathbf{v}_{ij} &\overset{\Delta}{=}
\mathbf{R}\_i^\top( \mathbf{v}_{j} - \mathbf{v}_{i} + \mathbf{g}\_\text{earth} \Delta t_{ij}) &=
\sum_{k=i}^{j-1} \Delta\mathbf{R}_{ik} \cdot \big( \hat{\mathbf{a}}_k - \mathbf{b}_{\mathbf{a},k} - \mathbf{\eta}_{\mathbf{a},k} \big) \Delta t \\\\
\Delta\mathbf{p}_{ij} & \overset{\Delta}{=}
\mathbf{R}\_i^\top \big( \mathbf{p}_j - \mathbf{p}\_i - \mathbf{v}_{i} \Delta t_{ij} - 
\frac{1}{2} \mathbf{g}\_\text{earth} \Delta t_{ij}^2 \big)&=
\mathbf{p}\_i + \sum_{k=i}^{j-1} 
\bigg(
\mathbf{v}_k \Delta t + 
\frac{1}{2} \Delta\mathbf{R}_{ik} \cdot \big( \hat{\mathbf{a}} - \mathbf{b}_{\mathbf{a},k} - \mathbf{\eta}_{\mathbf{a},k} \big) \Delta t^2
\bigg)
\end{align*}
$$

The differential model removes computation for the $i$-th 

## Pre-integration Measurement Modelling

Should consider zero offset in IMU, here propose three assumptions:
* Zero offset at the $i$-th keyframe timestamp is fixed, and is a constant throughout the whole computation
* Approximate measurement in linear form only (first order Taylor expansion)
* If zero offset estimate changes, use this linear model for correction

By these assumptions, only need to care about the Gaussian noises $\mathbf{\eta}_{\mathbf{\omega}}$ and $\mathbf{\eta}_{\mathbf{a}}$.

$$
\begin{align*}
    &&
    \Delta\mathbf{R}_{ij} &= 
    \prod^{j-1}_{k=i}
    e^{ \big(\hat{\mathbf{\omega}}_k- \mathbf{b}_{\mathbf{\omega},k} - \mathbf{\eta}_{\mathbf{\omega},k} \big)^\wedge \Delta t}
\\\\ && &=
    \prod^{j-1}_{k=i}
    e^{ \big( (\hat{\mathbf{\omega}}_k- \mathbf{b}_{\mathbf{\omega},k})\Delta t - \mathbf{\eta}_{\mathbf{\omega},k} \Delta t
    \big)^\wedge }
\\\\  \text{By BCH approximation}
    && &\approx
    \prod^{j-1}_{k=i}
    e^{ \big( (\hat{\mathbf{\omega}}_k- \mathbf{b}_{\mathbf{\omega},k})\Delta t 
    \big)^\wedge }
    \underbrace{e^{ \big(-\mathbf{J}_{rk} \mathbf{\eta}_{\mathbf{\omega},k} \Delta t \big)^\wedge}}_{\text{noise term}}
\\\\ \text{Simply expand the multiplications}
&& &=
    \underbrace{e^{ \big( (\hat{\mathbf{\omega}}_k- \mathbf{b}_{\mathbf{\omega},k})\Delta t 
    \big)^\wedge }}_{\Delta\hat{\mathbf{R}}_{k,k+1}}
    e^{ \big(-\mathbf{J}_{rk} \mathbf{\eta}_{\mathbf{\omega},k} \Delta t \big)^\wedge}
    \underbrace{e^{ \big( (\hat{\mathbf{\omega}}_{k+1}- \mathbf{b}_{\mathbf{\omega},k+1})\Delta t 
    \big)^\wedge }}_{\Delta\hat{\mathbf{R}}_{k+1,k+2}}
    e^{ \big(-\mathbf{J}_{r,k+1} \mathbf{\eta}_{\mathbf{\omega},k+1} \Delta t \big)^\wedge}
    \underbrace{e^{ \big( (\hat{\mathbf{\omega}}_{k+2}- \mathbf{b}_{\mathbf{\omega},k+2})\Delta t 
    \big)^\wedge }}_{\Delta\hat{\mathbf{R}}_{k+2,k+3}}
    e^{ \big(-\mathbf{J}_{r,k+2} \mathbf{\eta}_{\mathbf{\omega},k+2} \Delta t \big)^\wedge}
    \dots
\\\\ \text{Insert identity matrices } 
&& &=
    {e^{ \big( (\hat{\mathbf{\omega}}_k- \mathbf{b}_{\mathbf{\omega},k})\Delta t 
    \big)^\wedge }}
    \underbrace{
    {e^{ \big( (\hat{\mathbf{\omega}}_{k+1}- \mathbf{b}_{\mathbf{\omega},k+1})\Delta t 
    \big)^\wedge }}
    \bigg( {e^{ \big( (\hat{\mathbf{\omega}}_{k+1}- \mathbf{b}_{\mathbf{\omega},k+1})\Delta t 
    \big)^\wedge }} \bigg)^\top}_{=I}
    e^{ \big(-\mathbf{J}_{rk} \mathbf{\eta}_{\mathbf{\omega},k} \Delta t \big)^\wedge}
    {e^{ \big( (\hat{\mathbf{\omega}}_{k+1}- \mathbf{b}_{\mathbf{\omega},k+1})\Delta t 
    \big)^\wedge }}
    \dots
\\\\  \text{By adjoint relationship in Lie group} 
\\\\   \text{such that } \mathbf{R}^\top \text{Exp}(\mathbf{\phi})\mathbf{R}=\text{Exp}(\mathbf{R}^\top\mathbf{\phi})
&& &=
    {e^{ \big( (\hat{\mathbf{\omega}}_k- \mathbf{b}_{\mathbf{\omega},k})\Delta t 
    \big)^\wedge }}
    e^{ \big( (\hat{\mathbf{\omega}}_{k+1}- \mathbf{b}_{\mathbf{\omega},k+1})\Delta t 
    \big)^\wedge }
    \underbrace{
    \bigg( \underbrace{
    {e^{ \big( (\hat{\mathbf{\omega}}_{k+1}- \mathbf{b}_{\mathbf{\omega},k+1})\Delta t 
    \big)^\wedge }}}_{\Delta\hat{\mathbf{R}}_{k+1,k+2}} 
    \bigg)^\top
    \underbrace{e^{ \big(-\mathbf{J}_{rk} \mathbf{\eta}_{\mathbf{\omega},k} \Delta t \big)^\wedge}}_{
    \text{Exp}(-\mathbf{J}_{rk} \mathbf{\eta}_{\mathbf{\omega},k} \Delta t )}
    e^{ \big( (\hat{\mathbf{\omega}}_{k+1}- \mathbf{b}_{\mathbf{\omega},k+1})\Delta t 
    \big)^\wedge }}_{
    \text{Exp}(-{\Delta\hat{\mathbf{R}}_{k+1,k+2}}  \mathbf{J}_{rk} \mathbf{\eta}_{\mathbf{\omega},k} \Delta t )}
    \dots
\\\\ && &=
    \prod^{j-1}_{k=i}
    \bigg(
    e^{ \big( (\hat{\mathbf{\omega}}_k- \mathbf{b}_{\mathbf{\omega},k})\Delta t 
    \big)^\wedge }
    \text{Exp}(-{\Delta\hat{\mathbf{R}}_{k+1,k+2}}  \mathbf{J}_{rk} \mathbf{\eta}_{\mathbf{\omega},k} \Delta t )
    \bigg)
\\\\ \text{Define }
\Delta\hat{\mathbf{R}}_{ij} =
\prod^{j-1}_{k=i}
e^{ \big( (\hat{\mathbf{\omega}}_k- \mathbf{b}_{\mathbf{\omega},k})\Delta t 
\big)^\wedge }
\text{, so that }
&& &=
    {\Delta\hat{\mathbf{R}}_{ij}}
    \prod^{j-1}_{k=i} 
    \text{Exp} \big(
    -{\Delta\hat{\mathbf{R}}_{k+1,k+2}}  \mathbf{J}_{rk} \mathbf{\eta}_{\mathbf{\omega},k} \Delta t \big)
\\\\ \text{Collectively define the noise term}
&& &=
    {\Delta\hat{\mathbf{R}}_{ij}}
    \text{Exp} \big( -\delta\mathbf{\phi}_{ij} \big)
\end{align*}
$$

Similarly, $\Delta\mathbf{v}_{ij}$ can be computed with the derived $\Delta\hat{\mathbf{R}}_{ij}$

$$
\begin{align*}
&&
    \Delta\mathbf{v}_{ij} &=
    \sum_{k=i}^{j-1} 
    \underbrace{\Delta\mathbf{R}_{ik}}_{{\Delta\hat{\mathbf{R}}_{ij}}
    \text{Exp} \big( -\delta\mathbf{\phi}_{ij} \big)}
    \cdot \big( \hat{\mathbf{a}}_k - \mathbf{b}_{\mathbf{a},k} - \mathbf{\eta}_{\mathbf{a},k} \big) \Delta t \\\\
&& &=
    \sum_{k=i}^{j-1} 
    {\Delta\hat{\mathbf{R}}_{ik}}
    \underbrace{\text{Exp} \big( -\delta\mathbf{\phi}_{ik} \big)}_{ \begin{matrix}
        \footnotesize{\text{First order exponential approximation}} \\\\
        \approx I-\delta\mathbf{\phi}^{\wedge}_{ik}
    \end{matrix}
    }
    \cdot \big( \hat{\mathbf{a}}_k - \mathbf{b}_{\mathbf{a},k} - \mathbf{\eta}_{\mathbf{a},k} \big) \Delta t
\\\\ && &\approx
    \sum_{k=i}^{j-1} 
    {\Delta\hat{\mathbf{R}}_{ik}}
    {\big( I-\delta\mathbf{\phi}^{\wedge}_{ik} \big)}
    \cdot \big( \hat{\mathbf{a}}_k - \mathbf{b}_{\mathbf{a},k} - \mathbf{\eta}_{\mathbf{a},k} \big) \Delta t
\\\\ && &=
    \underbrace{\sum_{k=i}^{j-1} 
    {\Delta\hat{\mathbf{R}}_{ik}}
    \cdot \big( \hat{\mathbf{a}}_k - \mathbf{b}_{\mathbf{a},k}  \big) \Delta t}_{
    \overset{\Delta}{=}  \Delta\hat{\mathbf{v}}_{ij}   }
    + \sum_{k=i}^{j-1} {\Delta\hat{\mathbf{R}}_{ik}} \cdot \mathbf{\eta}_{\mathbf{a},k} \cdot \Delta t
    - \underbrace{\sum_{k=i}^{j-1} {\Delta\hat{\mathbf{R}}_{ik}} \cdot \delta\mathbf{\phi}^{\wedge}_{ik} \cdot \big( \hat{\mathbf{a}}_k - \mathbf{b}_{\mathbf{a},k} - \mathbf{\eta}_{\mathbf{a},k} \big) \Delta t}_{ \footnotesize{\text{Second order noise be zero }}
    \delta\mathbf{\phi}^{\wedge}_{ik} \cdot \mathbf{\eta}_{\mathbf{a},k} \approx 0}
\\\\ && &=
    \Delta\hat{\mathbf{v}}_{ij} 
    + \underbrace{
        \sum_{k=i}^{j-1} {\Delta\hat{\mathbf{R}}_{ik}} \cdot \mathbf{\eta}_{\mathbf{a},k} \cdot \Delta t
        - \sum_{k=i}^{j-1} {\Delta\hat{\mathbf{R}}_{ik}} \cdot \delta\mathbf{\phi}^{\wedge}_{ik} \cdot \big( \hat{\mathbf{a}}_k - \mathbf{b}_{\mathbf{a},k} \big) \Delta t}_{
    \overset{\Delta}{=} -\delta \mathbf{v}_{ij}}
\\\\ && &=
    \Delta\hat{\mathbf{v}}_{ij} - \delta \mathbf{v}_{ij}
\end{align*}
$$

Similarly, translation differential $\Delta{\mathbf{p}}_{ij}$ can be computed as

$$
\begin{align*}
\Delta{\mathbf{p}}_{ij} &=
\underbrace{\sum^{j-1}_{k=i} \Big(
\mathbf{v}_{ik} \Delta t + \frac{1}{2} {\Delta\hat{\mathbf{R}}_{ik}} \big( \hat{\mathbf{a}} - \mathbf{b}_{\mathbf{a},k}) \Delta t^2 \Big)  }_{
\Delta \hat{\mathbf{p}}_{ij}}
\\\\ & \quad +
\underbrace{\sum^{j-1}_{k=i} \Big( 
\delta \mathbf{v}_{ij} \Delta t +\frac{1}{2} \Delta \hat{\mathbf{R}}_{ik} \cdot \big( \hat{\mathbf{a}}_k - \mathbf{b}_{\mathbf{a},k} \big) \cdot \delta\mathbf{\phi}_{ij} \Delta t^2 -\frac{1}{2} \Delta\hat{\mathbf{R}}_{ik} \cdot \mathbf{\eta}_{\mathbf{a},k} \cdot \Delta t^2 \Big)}_{
\delta {\mathbf{p}}_{ij} }&=
\Delta \hat{\mathbf{p}}_{ij} + \delta {\mathbf{p}}_{ij}
\end{align*}
$$

Rearrange the expressions of the original definitions with the obtained noises: 
$$
\begin{align*}
    \Delta\hat{\mathbf{R}}_{ij} &= 
    \mathbf{R}\_i^\top \mathbf{R}_j \text{Exp} \big( -\delta\mathbf{\phi}_{ij} \big) \\\\
    \Delta\hat{\mathbf{v}}_{ij} &= 
    \mathbf{R}\_i^\top( \mathbf{v}_{j} - \mathbf{v}_{i} + \mathbf{g}\_\text{earth} \Delta t_{ij}) + \delta \mathbf{v}_{ij} \\\\
    \Delta\hat{\mathbf{p}}_{ij} &= 
    \mathbf{R}\_i^\top \big( \mathbf{p}_j - \mathbf{p}\_i - \mathbf{v}_{i} \Delta t_{ij} - 
    \frac{1}{2} \mathbf{g}\_\text{earth} \Delta t_{ij}^2 \big) + \delta {\mathbf{p}}_{ij}
\end{align*}
$$

$\Delta\hat{\mathbf{R}}_{ij}, \Delta\hat{\mathbf{v}}_{ij}, \Delta\hat{\mathbf{p}}_{ij}$ can be computed from taking the sum of all IMU's sensor readings indexed at $k=i,i+1,i+2,...,j$.
The right hand side expressions are direct computation on the "gap" between two timestamp $k=i$ and $k=j$ plus some noises.

## Pre-integration Zero Drift Noise Modelling

Recall the noise definition for rotation:

$$
\text{Exp} \big( \delta\mathbf{\phi}_{ij} \big)=
\prod^{j-1}_{k=i} 
\text{Exp} \big(
-{\Delta\hat{\mathbf{R}}_{k+1}^\top}  \mathbf{J}_{rk} \mathbf{\eta}_{\mathbf{\omega},k} \Delta t \big)
$$

Notice here $\mathbf{\eta}_{\mathbf{\omega},k}$ is white noise so that the mean of $\delta\mathbf{\phi}_{ij}$ across $k=i,i+1,i+2,...,j$ is $\mu_{\delta\mathbf{\phi}_{ij}}=0$.

Then to analyze its covariance: to remove the exponential term, do $\log$ operation, then by BCH linear approximation, since the noise is Gaussian, and the right Jacobian of Gaussian noise should be very small, hence $\mathbf{J}_{r,k}\approx \mathbf{I}$.

The below expression shows how the noise is computed from the ($j-1$)-th timestamp to the $j$-th timestamp given already computed accumulated noise to the ($j-1$)-th timestamp ${ \delta \mathbf{\phi}_{i,j-1} }$.

Apparently, this expression is linear.

$$
\begin{align*}
&&
    \delta\mathbf{\phi}_{ij} &= 
    -\text{Log} \Big( \underbrace{
    \prod^{j-1}_{k=i}
    \text{Exp} \big(
    -{\Delta\hat{\mathbf{R}}_{k+1}^\top}  \mathbf{J}_{rk} \mathbf{\eta}_{\mathbf{\omega},k} \Delta t \big)}_{
    \footnotesize{\text{approximation by }}
    \exp(\delta\mathbf{\phi}^\wedge) \approx \mathbf{I}+\delta\mathbf{\phi}^\wedge
    } \Big)
\\\\ && &\approx
    -\text{Log} \Big(-\text{Exp} \big(\sum_{k=i}^{j-1}
    {\Delta\hat{\mathbf{R}}_{k+1}^\top}  \mathbf{J}_{rk} \mathbf{\eta}_{\mathbf{\omega},k} \Delta t
    \big) \Big)
\\\\ && &=
    \sum_{k=i}^{j-1}
    {\Delta\hat{\mathbf{R}}_{k+1}^\top}  \mathbf{J}_{rk} \mathbf{\eta}_{\mathbf{\omega},k} \Delta t
\\\\ \text{Introduce the subscript } j \text{ to } {\Delta\hat{\mathbf{R}}_{k+1,j}^\top}
\\\\ \text{to control the computation between } i \text{ and } j
&& &=
    \sum_{k=i}^{j-1}
    {\Delta\hat{\mathbf{R}}_{k+1,j}^\top}  \mathbf{J}_{rk} \mathbf{\eta}_{\mathbf{\omega},k} \Delta t
\\\\ && &=
    \sum_{k=i}^{j-2} \Big(
    \underbrace{\Delta\hat{\mathbf{R}}_{k+1,j-1}^\top}_{
    \big( \Delta\hat{\mathbf{R}}_{k+1,j-1} \Delta\hat{\mathbf{R}}_{j-1,j} \big)^\top
    }  \mathbf{J}_{rk} \mathbf{\eta}_{\mathbf{\omega},k} \Delta t \Big)
    + \underbrace{\Delta\hat{\mathbf{R}}_{j,j}^\top}_{=\mathbf{I}}  \mathbf{J}_{r,j-1} \mathbf{\eta}_{\mathbf{\omega},j-1} \Delta t
\\\\ \text{Define a noise term}
&& &=
    \Delta\hat{\mathbf{R}}_{j-1,j}
    \underbrace{\sum_{k=i}^{j-2} \Big(
    {\Delta\hat{\mathbf{R}}_{k+1,j-1}^\top}  \mathbf{J}_{rk} \mathbf{\eta}_{\mathbf{\omega},k} \Delta t \Big)}_{
    \delta \mathbf{\phi}_{i,j-1}}
    + \mathbf{J}_{r,j-1} \mathbf{\eta}_{\mathbf{\omega},j-1} \Delta t   
\\\\ && &=
    \Delta\hat{\mathbf{R}}_{j-1,j}
    { \delta \mathbf{\phi}_{i,j-1} }
    + \mathbf{J}_{r,j-1} \mathbf{\eta}_{\mathbf{\omega},j-1} \Delta t       
\end{align*}
$$

Given $\mathbf{\eta}_{\mathbf{\omega},k}$ being a Gaussian type of noise, the sum of many $\mathbf{\eta}_{\mathbf{\omega},k}$ should be a Gaussian type of noise as well.
Then, compute the covariances:

Set the covariance of ${ \delta \mathbf{\phi}_{i,j-1} }$ to $\mathbf{\Sigma}_{j-1}$, and $\mathbf{\eta}_{\mathbf{\omega},j-1}$ to $\mathbf{\Sigma}_{\mathbf{\eta}_{\mathbf{\omega},j-1}}$, there is

$$
\mathbf{\Sigma}_j = 
\Delta\hat{\mathbf{R}}_{j-1,j}^\top \mathbf{\Sigma}_{j-1} \Delta\hat{\mathbf{R}}_{j-1,j}+
\mathbf{J}_{r,j-1}^\top \mathbf{\Sigma}_{\mathbf{\eta}_{\mathbf{\omega},j-1}} \mathbf{J}_{r,j-1} 
$$
that $\mathbf{\Sigma}_j$ continuously grows as it adds $\mathbf{J}_{r,j-1}^\top \mathbf{\Sigma}_{\mathbf{\eta}_{\mathbf{\omega},j-1}} \mathbf{J}_{r,j-1}$ at every $k$-th step.

Similarly, velocity noise $\delta \mathbf{v}_{i,j}$ and translation noise $\delta \mathbf{p}_{i,j}$ can be computed by

$$
\begin{align*}
    \delta \mathbf{v}_{i,j} &=
        \sum_{k=i}^{j-1} {\Delta\hat{\mathbf{R}}_{ik}} \cdot \mathbf{\eta}_{\mathbf{a},k} \cdot \Delta t
        - \sum_{k=i}^{j-1} {\Delta\hat{\mathbf{R}}_{ik}} \cdot \delta\mathbf{\phi}^{\wedge}_{ik} \cdot \big( \hat{\mathbf{a}}_k - \mathbf{b}_{\mathbf{a},k} \big) \Delta t&= 
    \underbrace{
    \sum_{k=i}^{j-2} \Big(
    {\Delta\hat{\mathbf{R}}_{ik}} \cdot \mathbf{\eta}_{\mathbf{a},k} \cdot \Delta t
    - {\Delta\hat{\mathbf{R}}_{ik}} \cdot \delta\mathbf{\phi}^{\wedge}_{ik} \cdot \big( \hat{\mathbf{a}}_k - \mathbf{b}_{\mathbf{a},k} \big) \Delta t
    \Big)}_{\delta \mathbf{v}_{i,j-1}}
\\\\ & \quad +
    {\Delta\hat{\mathbf{R}}_{i,j-1}} \cdot \mathbf{\eta}_{\mathbf{a},j-1} \cdot \Delta t
    - {\Delta\hat{\mathbf{R}}_{i,j-1}} \cdot \big( \hat{\mathbf{a}}_{j-1} - \mathbf{b}_{\mathbf{a},i} \big) \Delta t&=
    \delta \mathbf{v}_{i,j-1} +
    \Big(
    {\Delta\hat{\mathbf{R}}_{i,j-1}} \cdot \mathbf{\eta}_{\mathbf{a},j-1} \cdot \Delta t
    - {\Delta\hat{\mathbf{R}}_{i,j-1}} \cdot \big( \hat{\mathbf{a}}_{j-1} - \mathbf{b}_{\mathbf{a},i} \big) \Delta t
    \Big)\\\\ \space\\\\ \delta \mathbf{p}_{i,j} &=
    \sum^{j-1}_{k=i} \Big( 
    \delta \mathbf{v}_{ik} \Delta t 
    + \frac{1}{2} \Delta \hat{\mathbf{R}}_{ik} \cdot \big( \hat{\mathbf{a}}_k - \mathbf{b}_{\mathbf{a},k} \big) \cdot \delta\mathbf{\phi}_{ij} \Delta t^2 -\frac{1}{2} \Delta\hat{\mathbf{R}}_{ik} \cdot \mathbf{\eta}_{\mathbf{a},k} \Delta t^2 \Big)&=
    \underbrace{
    \sum_{k=i}^{j-2} \Big( 
    \delta \mathbf{v}_{ik} \Delta t 
    + \frac{1}{2} \Delta \hat{\mathbf{R}}_{ik} \cdot \big( \hat{\mathbf{a}}_k - \mathbf{b}_{\mathbf{a},k} \big) \cdot \delta\mathbf{\phi}_{ij} \Delta t^2 -\frac{1}{2} \Delta\hat{\mathbf{R}}_{ik} \cdot \mathbf{\eta}_{\mathbf{a},k}\Delta t^2
    \Big) }_{\delta \mathbf{p}_{i,j-1}}
\\\\ & \quad+
    \delta \mathbf{v}_{i,j-1} \Delta t 
    + \frac{1}{2} \Delta \hat{\mathbf{R}}_{i,j-1} \cdot \big( \hat{\mathbf{a}}_{j-1} - \mathbf{b}_{\mathbf{a},i} \big) \cdot \delta\mathbf{\phi}_{i,j-1} \Delta t^2 -\frac{1}{2} \Delta\hat{\mathbf{R}}_{i,j-1} \cdot \mathbf{\eta}_{\mathbf{a},j-1}  \Delta t^2&= 
    \delta \mathbf{p}_{i,j-1} + \Big(
    \frac{1}{2} \Delta \hat{\mathbf{R}}_{i,j-1} \cdot \big( \hat{\mathbf{a}}_{j-1} - \mathbf{b}_{\mathbf{a},i} \big) \cdot \delta\mathbf{\phi}_{i,j-1} \Delta t^2 -\frac{1}{2} \Delta\hat{\mathbf{R}}_{i,j-1} \cdot \mathbf{\eta}_{\mathbf{a},j-1}  \Delta t^2 \Big)
\end{align*}
$$ 

To summarize, define an accumulated noise vector $\mathbf{\eta}_{ik}$ and the $j$-th step noise vector $\mathbf{\eta}_{j}$:

$$
\mathbf{\eta}_{ik} = 
\begin{bmatrix}
    \delta\mathbf{\phi}_{ij} \\\\
    \delta \mathbf{v}_{i,j} \\\\
    \delta \mathbf{p}_{i,j}
\end{bmatrix}
, \quad
\mathbf{\eta}_{j} = 
\begin{bmatrix}
    \mathbf{\eta}_{\mathbf{\omega},j} \\\\
    \mathbf{\eta}_{\mathbf{a},j}
\end{bmatrix}
$$

The noise propagation model that linearly adds on the next timestamp noise $\mathbf{\eta}_{j}$ to form the total noise between the $i$-th timestamp and $j$-th timestamp is

$$
\mathbf{\eta}_{ij} = 
A_j\space \mathbf{\eta}_{ik} + B_j\space \mathbf{\eta}_{j}
$$

where

$$
\begin{align*}
    A_j &= \begin{bmatrix}
        \Delta\hat{\mathbf{R}}_{j-1,j}^\top 
        & \mathbf{0} & \mathbf{0} \\\\
        {\Delta\hat{\mathbf{R}}_{i,j-1}} \cdot \big( \hat{\mathbf{a}}_{j-1} - \mathbf{b}_{\mathbf{a},i} \big) \Delta t 
        & \mathbf{I} & \mathbf{0} \\\\
        \frac{1}{2} {\Delta\hat{\mathbf{R}}_{i,j-1}} \cdot \big( \hat{\mathbf{a}}_{j-1} - \mathbf{b}_{\mathbf{a},i} \big) \Delta t^2 
        & \Delta t & \mathbf{I}
    \end{bmatrix}
    \\\\
    B_j &= \begin{bmatrix}
        \mathbf{J}_{r,j-1}\Delta t & \mathbf{0} \\\\
        \mathbf{0} & {\Delta\hat{\mathbf{R}}_{i,j-1}} \cdot \Delta t \\\\
        \mathbf{0} & \frac{1}{2} {\Delta\hat{\mathbf{R}}_{i,j-1}} \cdot \Delta t^2 \\\\
    \end{bmatrix}
\end{align*}
$$

## Zero Offset Update

Previously, zero offset is assumed to be constant, however, it is a naive assumption that often zero offset shows Gaussian noises.

In this discussion, zero offset noises are **approximated by linear increment**, so that $\mathbf{b}_{\mathbf{\omega},k} \rightarrow \mathbf{b}_{\mathbf{\omega},k} + \delta\mathbf{b}_{\mathbf{\omega},k}$ and $\mathbf{b}_{\mathbf{a},k} \rightarrow \mathbf{b}_{\mathbf{a},k} + \delta\mathbf{b}_{\mathbf{a},k}$, where $\delta\mathbf{b}_{\mathbf{\omega},k}$ and $\delta\mathbf{b}_{\mathbf{a},k}$ are the white noises.

Then formulate $\Delta\hat{\mathbf{R}}_{ij}$, $\Delta\hat{\mathbf{v}}_{ij}$ and $\Delta\hat{\mathbf{p}}_{ij}$ as functions of the arguments $(\mathbf{b}_{\mathbf{\omega},k} + \delta\mathbf{b}_{\mathbf{\omega},k})$ and $(\mathbf{b}_{\mathbf{a},k} + \delta\mathbf{b}_{\mathbf{a},k})$

$$
\begin{align*}
    \Delta\hat{\mathbf{R}}_{ij} (\mathbf{b}_{\mathbf{\omega},i} + \delta\mathbf{b}_{\mathbf{\omega},i}) &= 
    \Delta\hat{\mathbf{R}}_{ij} \text{Exp} \big( \frac{\partial\space \Delta\hat{\mathbf{R}}_{ij}}{\partial\space \mathbf{b}_{\mathbf{\omega},i}} \delta\mathbf{b}_{\mathbf{\omega},i} \big) \\\\
    \Delta\hat{\mathbf{v}}_{ij} (\mathbf{b}_{\mathbf{\omega},i} + \delta\mathbf{b}_{\mathbf{\omega},i}, \mathbf{b}_{\mathbf{a},k} + \delta\mathbf{b}_{\mathbf{a},k}) &= 
    \Delta\hat{\mathbf{v}}_{ij} (\mathbf{b}_{\mathbf{\omega},i}, \mathbf{b}_{\mathbf{a},k})
    + \frac{\partial\space \Delta\hat{\mathbf{v}}_{ij}}{\partial\space \mathbf{b}_{\mathbf{\omega},i}} \delta\mathbf{b}_{\mathbf{\omega},i}
    + \frac{\partial\space \Delta\hat{\mathbf{v}}_{ij}}{\partial\space \mathbf{b}_{\mathbf{a},i}} \delta\mathbf{b}_{\mathbf{a},i} \\\\
    \Delta\hat{\mathbf{p}}_{ij} (\mathbf{b}_{\mathbf{\omega},i} + \delta\mathbf{b}_{\mathbf{\omega},i}, \mathbf{b}_{\mathbf{a},k} + \delta\mathbf{b}_{\mathbf{a},k}) &= 
    \Delta\hat{\mathbf{p}}_{ij} (\mathbf{b}_{\mathbf{\omega},i}, \mathbf{b}_{\mathbf{a},k})
    + \frac{\partial\space \Delta\hat{\mathbf{p}}_{ij}}{\partial\space \mathbf{b}_{\mathbf{\omega},i}} \delta\mathbf{b}_{\mathbf{\omega},i}
    + \frac{\partial\space \Delta\hat{\mathbf{p}}_{ij}}{\partial\space \mathbf{b}_{\mathbf{a},i}} \delta\mathbf{b}_{\mathbf{a},i}
\end{align*}
$$

Then, compute the Jacobians of the noises simply by replacing $\mathbf{b}_{\mathbf{\omega},i}$ with $\mathbf{b}_{\mathbf{\omega},i} + \delta\mathbf{b}_{\mathbf{\omega},i}$. Take $\text{Exp} \frac{\partial\space \Delta\hat{\mathbf{R}}_{ij}}{\partial\space \mathbf{b}_{\mathbf{\omega},i}} \delta\mathbf{b}_{\mathbf{\omega},i}$ for example:

$$
\begin{align*}
    \Delta\hat{\mathbf{R}}_{ij} ( \mathbf{b}_{\mathbf{\omega},i} + \delta\mathbf{b}_{\mathbf{\omega},i} ) &= 
    \prod^{j-1}_{k=i}
    \text{Exp}{ \Big( \big( \hat{\mathbf{\omega}}_k - 
    \underbrace{(\mathbf{b}_{\mathbf{\omega},k} + \delta\mathbf{b}_{\mathbf{\omega},k}) }_{\mathbf{b}_{\mathbf{\omega},k} \footnotesize{\text{ with noise}}} \big)
    \Delta t \Big) }&=
    \prod^{j-1}_{k=i}
    \text{Exp}{ \Big( \big( \hat{\mathbf{\omega}}_k - 
    { \mathbf{b}_{\mathbf{\omega},k} } \big)
    \Delta t \Big) }
    \text{Exp} \Big(-\mathbf{J}_{r,k} \delta\mathbf{b}_{\mathbf{\omega},k} \Delta t \Big)&=
    \underbrace{ \text{Exp}{ \Big( \big( \hat{\mathbf{\omega}}\_i - 
    { \mathbf{b}_{\mathbf{\omega},i} } \big)
    \Delta t \Big) }}_{\Delta \hat{\mathbf{R}}_{i,i+1}}
    \text{Exp} \Big(-\mathbf{J}_{r,i} \delta\mathbf{b}_{\mathbf{\omega},i} \Delta t \Big)
    \underbrace{ \text{Exp}{ \Big( \big( \hat{\mathbf{\omega}}_{i+1} - 
    { \mathbf{b}_{\mathbf{\omega},{i+1}} } \big)
    \Delta t \Big) }}_{\Delta \hat{\mathbf{R}}_{i+1,i+2}}
    \text{Exp} \Big(-\mathbf{J}_{r,i+1} \delta\mathbf{b}_{\mathbf{\omega},i+1} \Delta t \Big)
    \dots&=
    {\Delta \hat{\mathbf{R}}_{i,i+1}} 
    \underbrace{ {\Delta \hat{\mathbf{R}}_{i,i+1}}
    {\Delta \hat{\mathbf{R}}^\top_{i,i+1}}}_{=\mathbf{I}}
    \text{Exp} \Big(-\mathbf{J}_{r,i} \delta\mathbf{b}_{\mathbf{\omega},i} \Delta t \Big)
    \underbrace{ {\Delta \hat{\mathbf{R}}_{i+1,i+2}}
    {\Delta \hat{\mathbf{R}}^\top_{i+1,i+2}}}_{=\mathbf{I}}
    \text{Exp} \Big(-\mathbf{J}_{r,i+1} \delta\mathbf{b}_{\mathbf{\omega},i+1} \Delta t \Big)
    \dots&=
    {\Delta \hat{\mathbf{R}}_{i,i+1}} 
    \underbrace{{\Delta \hat{\mathbf{R}}^\top_{i,i+1}}
    \text{Exp} \Big(-\mathbf{J}_{r,i} \delta\mathbf{b}_{\mathbf{\omega},i} \Delta t \Big)
    {\Delta \hat{\mathbf{R}}_{i,i+1}}}_{
        \footnotesize{\text{By adjoint rule }} 
        \mathbf{R}^\top \text{Exp}(\mathbf{\phi})\mathbf{R}=\text{Exp}(\mathbf{R}^\top\mathbf{\phi})
    }
    \cdot
    \underbrace{ {\Delta \hat{\mathbf{R}}^\top_{i+1,i+2}}
    \text{Exp} \Big(-\mathbf{J}_{r,i+1} \delta\mathbf{b}_{\mathbf{\omega},i+1} \Delta t \Big)
    {\Delta \hat{\mathbf{R}}_{i+1,i+2}}}_{
        \footnotesize{\text{By adjoint rule }} 
        \mathbf{R}^\top \text{Exp}(\mathbf{\phi})\mathbf{R}=\text{Exp}(\mathbf{R}^\top\mathbf{\phi}) } \dots&=
    \prod^{j-1}_{k=i}
    \Delta \hat{\mathbf{R}}_{k,k+1}
    \underbrace{\prod^{j-1}_{k=i}
    \text{Exp} \Big(-\Delta \hat{\mathbf{R}}_{k,k+1}^\top \mathbf{J}_{r,k} \delta\mathbf{b}_{\mathbf{\omega},k} \Delta t \Big)}_{
    \footnotesize{\text{approximation by }}
    \exp(\delta\mathbf{\phi}^\wedge) \approx \mathbf{I}+\delta\mathbf{\phi}^\wedge
    }  \\\\ &\approx
    \Delta \hat{\mathbf{R}}_{ij} \cdot
    \text{Exp} \Big(
        -\sum^{j-1}_{k=i}
        \Delta \hat{\mathbf{R}}_{k,k+1}^\top
        \mathbf{J}_{r,k} \delta\mathbf{b}_{\mathbf{\omega},k} \Delta t \Big)
\end{align*}
$$

So that 
$$
\text{Exp} \frac{\partial\space \Delta\hat{\mathbf{R}}_{ij}}{\partial\space \mathbf{b}_{\mathbf{\omega},i}} \delta\mathbf{b}_{\mathbf{\omega},i} =
\text{Exp} \Big(
    -\sum^{j-1}_{k=i}
    \Delta \hat{\mathbf{R}}_{k,k+1}^\top
    \mathbf{J}_{r,k} \delta\mathbf{b}_{\mathbf{\omega},k} \Delta t
\Big)
$$

where the Jacobian is $\frac{\partial\space \Delta\hat{\mathbf{R}}_{ij}}{\partial\space \mathbf{b}_{\mathbf{\omega},i}}=-\sum^{j-1}_{k=i} \Delta \hat{\mathbf{R}}_{k,k+1}^\top \mathbf{J}_{r,k} \Delta t$.

Similarly, the Jacobians for $\Delta\hat{\mathbf{v}}_{ij} (\mathbf{b}_{\mathbf{\omega},i} + \delta\mathbf{b}_{\mathbf{\omega},i}, \mathbf{b}_{\mathbf{a},k} + \delta\mathbf{b}_{\mathbf{a},k})$ and $\Delta\hat{\mathbf{p}}_{ij} (\mathbf{b}_{\mathbf{\omega},i} + \delta\mathbf{b}_{\mathbf{\omega},i}, \mathbf{b}_{\mathbf{a},k} + \delta\mathbf{b}_{\mathbf{a},k})$ can be computed.
Below are the finished results.

$$
\begin{align*}
    &&
    \frac{\partial\space \Delta\hat{\mathbf{v}}_{ij}}{\partial\space \mathbf{b}_{\mathbf{a},i}}
    &= 
    -\sum^{j-1}_{k=i} \Delta \hat{\mathbf{R}}_{k,k+1} \Delta t
    &\quad&&
    \frac{\partial\space \Delta\hat{\mathbf{v}}_{ij}}{\partial\space \mathbf{b}_{\mathbf{\omega},i}}
    &=
    -\sum^{j-1}_{k=i} \Delta \hat{\mathbf{R}}_{k,k+1} (\hat{\mathbf{a}}_{k}-{\mathbf{b}}_{\mathbf{a},k}) 
    \frac{\partial\space \Delta\hat{\mathbf{R}}_{ij}}{\partial\space \mathbf{b}_{\mathbf{\omega},i}} \Delta t \\\\
    &&
    \frac{\partial\space \Delta\hat{\mathbf{p}}_{ij}}{\partial\space \mathbf{b}_{\mathbf{a},i}}
    &= 
    \sum^{j-1}_{k=i} \Big(
    \frac{\partial\space \Delta\hat{\mathbf{v}}_{ij}}{\partial\space \mathbf{b}_{\mathbf{a},i}} \Delta t
    - \frac{1}{2} \Delta \hat{\mathbf{R}}_{k,k+1} \Delta t^2
    \Big)
    &\quad&&
    \frac{\partial\space \Delta\hat{\mathbf{p}}_{ij}}{\partial\space \mathbf{b}_{\mathbf{\omega},i}}
    &=
    \sum^{j-1}_{k=i} \Big(
    \frac{\partial\space \Delta\hat{\mathbf{v}}_{ij}}{\partial\space \mathbf{b}_{\mathbf{\omega},i}} \Delta t
    - \frac{1}{2} \sum^{j-1}_{k=i} \Delta \hat{\mathbf{R}}_{k,k+1} (\hat{\mathbf{a}}_{k}-{\mathbf{b}}_{\mathbf{a},k}) 
    \frac{\partial\space \Delta\hat{\mathbf{R}}_{ij}}{\partial\space \mathbf{b}_{\mathbf{\omega},i}} \Delta t^2
    \Big)
\end{align*}
$$

## Pre-integration in Graph Optimization

Define the IMU pre-integration graph optimization problem:
a one-time IMU reading should contain such information $\mathbf{x}_k=[\mathbf{R} \quad \mathbf{p} \quad \mathbf{v} \quad \mathbf{b}\_\mathbf{a} \quad \mathbf{b}_{\mathbf{\omega}}]_k$.
Such information can help compute, or by direct IMU reading $\Delta\hat{\mathbf{R}}_{ij},  \Delta\hat{\mathbf{v}}_{ij}, \Delta\hat{\mathbf{p}}_{ij}$, whose expressions are inclusive of zero offset noises $\delta\mathbf{b}_{\mathbf{a},k}$ and $\delta\mathbf{b}_{\mathbf{\omega},k}$.

In other words, IMU reading $\Delta\hat{\mathbf{R}}_{ij},  \Delta\hat{\mathbf{v}}_{ij}, \Delta\hat{\mathbf{p}}_{ij}$ have true reading plus noises.

Here formulate the residuals for graph optimization

$$
\begin{align*}
\mathbf{r}_{\Delta \mathbf{R}_{ij}} &=
    \text{Log}\Big(
    \Delta\hat{\mathbf{R}}^\top_{ij}
    (\mathbf{R}\_i^\top \mathbf{R}_j)\Big)
\\\\ \mathbf{r}_{\Delta \mathbf{v}_{ij}} &=
    \mathbf{R}\_i^\top( \mathbf{v}_{j} - \mathbf{v}_{i} + \mathbf{g}\_\text{earth} \Delta t_{ij}) - 
    \Delta \hat{\mathbf{v}}_{ij}
\\\\ \mathbf{r}_{\Delta \mathbf{p}_{ij}} &=
    \mathbf{R}\_i^\top \big( \mathbf{p}_j - \mathbf{p}\_i - \mathbf{v}_{i} \Delta t_{ij} - 
    \frac{1}{2} \mathbf{g}\_\text{earth} \Delta t_{ij}^2 \big) -
    \Delta \hat{\mathbf{p}}_{ij}
\end{align*}
$$

where $\mathbf{R}\_i, \mathbf{v}\_i, \mathbf{p}\_i$ are observations (inclusive of measurement errors $\mathbf{\eta}_{ij}$) that should be adjusted to reduce residuals.

In conclusion, the residual $[\mathbf{r}_{\Delta \mathbf{R}_{ij}}  \quad \mathbf{r}_{\Delta \mathbf{v}_{ij}} \quad \mathbf{r}_{\Delta \mathbf{p}_{ij}}]^\top$ contain measurement errors $\mathbf{\eta}_{ij}$ and zero offset noises $\delta\mathbf{b}_{\mathbf{a},k}$ and $\delta\mathbf{b}_{\mathbf{\omega},k}$,
and these errors are collectively optimized by graph optimization by updating $\mathbf{R}\_i, \mathbf{v}\_i, \mathbf{p}\_i$.

To make the graph optimization better converge, need to compute the Jacobians of the residual $[\mathbf{r}_{\Delta \mathbf{R}_{ij}}  \quad \mathbf{r}_{\Delta \mathbf{v}_{ij}} \quad \mathbf{r}_{\Delta \mathbf{p}_{ij}}]^\top$. The Jacobian can be computed by Lie algebra perturbation model.

If used dynamic bias model, the biases should be computed as an optimization item in the above least squares problem.
Its Jacobians should be computed for convergence.

