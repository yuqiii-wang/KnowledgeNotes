# VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator

### Preliminaries and Notations

Notations: subscript $\tiny{W}$ means being in a world frame; subscript $\tiny{B}$ means being in a rigid body frame, also defined the same as IMU's; $\tiny{C}$ means being in a camera frame.

Subscript $\tiny{K}$ means keyframe index, while lower case $\tiny{k}$ means ordinary updating index, such as a shot by a camera $k_c$ or one IMU reading $k_i$.

The transform between IMU frame $\tiny{B}$ and camera frame $\tiny{C}$ is $[\bold{\phi}_{\tiny{BC}}|\bold{p}_{\tiny{BC}}]$:
$$
\begin{align*}
    \bold{\phi}_{\tiny{CBk_c}} &= 
    \bold{\phi}_{\tiny{Ck_c}} \otimes \bold{\phi}_{\tiny{BC}}^{-1}
\\
    s\bold{p}_{\tiny{CBk_c}} &=
    s \bold{p}_{\tiny{Ck_c}} - \bold{R}_{\tiny{Ck_c}} \bold{p}_{\tiny{BC}}
\end{align*}
$$

This pair $[s\bold{p}_{\tiny{CBk_c}} | \bold{\phi}_{\tiny{CBk_c}}]$ describes the IMU pose in the camera frame; $s$ is a scale factor.

## Measurement Preprocessing

### Vision Processing Front-end

* Visual Feature Point Process

For each new image, existing features are tracked by the KLT sparse optical flow algorithm.

Meanwhile, new corner features are detected to maintain a minimum number ($100$-$300$) of features in each image.

Outlier rejection is performed using RANSAC with fundamental matrix model.

* Keyframe Selection

Parallax tracked feature points different from the last keyframe: if the number greater than a threshold, launch a new keyframe.

If the new keyframe has too few feature points, discard it.

Use IMU to assist triangulation (rotation only mono vision cannot triangulate new feature points) when camera is rotating.

### IMU Pre-integration

For IMU haves much higher update frequencies than keyframe updates', should pre-integrate IMU readings and align the integrated results to corresponding keyframes.

IMU noises are assumed Gaussian and approximated by linear increments.

Between two neighbor keyframes $K_K$ and ${K+1}$, the pre-integrated IMU reading defines position difference estimate $\Delta\hat{\bold{p}}_{\tiny{{K+1}}}$, velocity difference estimate $\Delta\hat{\bold{v}}_{{K+1}}$ and orientation difference estimate $\Delta\hat{\bold{\phi}}_{{K+1}}$.

$$
\begin{align*}
    \Delta\hat{\bold{p}}_{\tiny{{K+1}}} &=
\bold{R}_{\tiny{BK+1}}
\big(\bold{p}_{\tiny{WK+1}}-\bold{p}_{\tiny{WK}}
+\frac{1}{2}\bold{g}_{earth}\Delta t^2
-\bold{v}_{\tiny{WK}}\Delta t \big)
+\bold{n}_{\tiny{Bp}}
\\
    \Delta\hat{\bold{v}}_{\tiny{{K+1}}} &=
\bold{R}_{\tiny{W{K+1}}}
\big(\bold{v}_{\tiny{W{K+1}}}-\bold{v}_{\tiny{WK}}
+\bold{g}_{earth}\Delta t \big)
+\bold{n}_{\tiny{Bv}}
\\
    \Delta\hat{\bold{\phi}}_{\tiny{{K+1}}} &=
\bold{\phi}_{\tiny{WK}}^{-1} \otimes
\bold{\phi}_{\tiny{W{K+1}}}
+\bold{n}_{\tiny{B\phi}}
\\
    \bold{b}_{\tiny{ak_i}}-\bold{b}_{\tiny{ak_i+1}} &=
\bold{0} + \bold{n}_{\tiny{Bb_a}}
\\
    \bold{b}_{\tiny{\omega k_i}}-\bold{b}_{\tiny{\omega k_i+1}} &=
\bold{0} + \bold{n}_{\tiny{Bb_\omega}}
\end{align*}
$$
where $\otimes$ represents the multiplication operation between two quaternions; $\bold{g}_{earth}$ is a constant vector for earth's gravity.

$\bold{n}_{\tiny{Bp}}$, $\bold{n}_{\tiny{Bv}}$ and $\bold{n}_{\tiny{B\phi}}$ are Gaussian noises (measurement errors) 
such that $\bold{n}_{\tiny{Bp}}\sim N(\bold{0}, \bold{\Sigma}_{\tiny{Bp}})$, $\bold{n}_{\tiny{Bv}}\sim N(\bold{0}, \bold{\Sigma}_{\tiny{Bv}})$ and $\bold{n}_{\tiny{B\phi}}\sim N(\bold{0}, \bold{\Sigma}_{\tiny{B\phi}})$.

$\bold{n}_{\tiny{Bb_a}} \sim N(\bold{0}, \bold{\Sigma}_{\tiny{Bb_a}})$ 
and $\bold{n}_{\tiny{Bb_\omega}} \sim N(\bold{0}, \bold{\Sigma}_{\tiny{Bb_\omega}})$ are the acceleration and rotation zero offset difference prone to drifting noises.

The difference vectors $\Delta\hat{\bold{p}}_{\tiny{{K+1}}}, \Delta\hat{\bold{v}}_{\tiny{{K+1}}}, \Delta\hat{\bold{\phi}}_{\tiny{{K+1}}}$ has no subscript $\tiny{W}$, since they only describe two keyframes' transform differences.

The estimates are inclusive of gyroscope bias $\bold{b}_\omega$ and acceleration bias $\bold{b}_a$.

## Estimator Preprocessing

With obtained pre-integrated IMU readings and some initial keyframes plus visual feature points from measurement preprocessing, 
estimator initialization fuses such data to align IMU readings to keyframe poses.

### Vision-Only Sliding Window for Structure from Motion (SfM)

Maintain a sliding window of frames: 
within this windows, check if there are adequate corresponding matched features; then triangulate the feature points.

Based on these triangulated features, a perspective-n-point (PnP) method is performed to estimate poses of all other frames in the window.

Finally,
a global full bundle adjustment is applied to minimize the total reprojection error of all feature observations.

### Visual-Inertial Alignment

* Gyroscope Bias Calibration

For frames in a sliding window containing some camera frames $k_c \in \mathcal{B}$, attempt to update gyroscope bias $\bold{b}_\omega$, 
so that IMU orientation reading difference $\Delta\hat{\bold{\phi}}_{\tiny{K}}$ can align to camera estimated orientation transform $\bold{\phi}^{-1}_{\tiny{C{K+1}}} \otimes \bold{\phi}_{\tiny{CK}}$.

$$
\argmin_{\delta \bold{b}_\omega}
\sum_{k_c \in \mathcal{B}}
\Big|\Big|
\big( \bold{\phi}^{-1}_{\tiny{C{k+1}}} \otimes
\bold{\phi}_{\tiny{Ck}} \big) \otimes
\Delta\hat{\bold{\phi}}_{\tiny{k}}
\Big|\Big|^2
$$
where subscript $\tiny{C}$ indicates being in a camera frame.

* Velocity, Gravity Vector and Metric Scale Initialization

For $n$ initial IMU readings $k_i \in \bold{\mathcal{B}}$, define velocity states accordingly. The subscript $\tiny{B}$ represent a body rigid transform, typically aligned to the same orientation as IMU device.

$$
\bold{\mathcal{X}_V} = 
[\bold{v}_{\tiny{Bk_{1}}},\quad
\bold{v}_{\tiny{Bk_{2}}},\quad
\dots,\quad
\bold{v}_{\tiny{Bk_{n}}},\quad]
\in \mathbb{R}^{3 \times n}
$$

Set $\Delta\hat{\bold{z}}_{\tiny{BK}}$ to represent the difference estimate for position and velocity change, 
that can be said a sum of true observations $\bold{H}_{\tiny{BK}} \bold{\mathcal{X}_V}$ and noises $\delta\bold{n}_{\tiny{BK}}$.

This term $\bold{R}_{\tiny{BC}}\bold{R}_{\tiny{BCK}} \bold{p}_{\tiny{CK}}$ describes alignment of camera position to the rigid body frame $\tiny{B}$.

$$
\begin{align*}
\Delta\hat{\bold{z}}_{\tiny{BK}}
&=
\begin{bmatrix}
    \Delta \bold{p}_{\tiny{BK}}
    + \bold{R}_{\tiny{BC}}\bold{R}_{\tiny{CK}} \bold{p}_{\tiny{BCK}}
    - \bold{p}_{\tiny{BCK}}
\\
    \Delta \bold{v}_{\tiny{BK}}
\end{bmatrix}
\\ &=
\bold{H}_{\tiny{BK}} \bold{\mathcal{X}_V}
+ \delta\bold{n}_{\tiny{BK}}
\end{align*}
$$

The optimization attempts to reduce the noise by updating $\bold{\mathcal{X}_V}$
$$
\min_{\bold{\mathcal{X}_V}}
\sum_{k_i \in \bold{\mathcal{B}}}
\Big|\Big|
\Delta\hat{\bold{z}}_{\tiny{BK}}
- 
\bold{H}_{\tiny{BK}} \bold{\mathcal{X}_V}
\Big|\Big|^2
$$

## Tightly-Coupled Monocular VIO

Proceed with a sliding window-based tightly-coupled monocular VIO for high-accuracy and robust state estimation.

The full state vector in the sliding window having $n$  is defined as:
$$
\begin{align*}
    \bold{\mathcal{X}} &= [
\bold{x}_0, \quad \bold{x}_1, \quad, ... , \quad \bold{x}_n, \quad
\lambda_0, \quad \lambda_1, \quad, ... , \quad \lambda_m]
\\
\bold{x}_{\tiny{k}} &= [
\bold{p}_{\tiny{WBk}}, \quad \bold{v}_{\tiny{WBk}}, \quad \bold{\phi}_{\tiny{WBk}}, \quad \bold{b}_{{\omega}}, \quad \bold{p}_{{a}}]
\\
\bold{x}_{\tiny{BC}} &= [\bold{\phi}_{\tiny{BC}}|\bold{p}_{\tiny{BC}}]
\end{align*}
$$
where $\bold{x}_k, k\in[0,n]$ is the IMU state at the time that the $k$-th image is captured. It contains position, velocity, and orientation of the IMU in the world frame, and acceleration bias $\bold{b}_{{a}}$ and gyroscope bias $\bold{b}_{{\omega}}$ in the IMU body frame.

$n$ is the total number of keyframes, and $m$ is the total number of features in the sliding window. 
$\lambda_l$ is the inverse depth of the $l$-th feature from its first observation.

$$
\min_{\bold{\mathcal{X}}}
\underbrace{\big|\big|
    \bold{r}_p - H_p \bold{\mathcal{X}}
\big|\big|^2}_{
\text{Marginalization residual}}
+
\underbrace{\sum_{k_i \in \mathcal{B}} 
\Big|\Big|
    \bold{r}_\mathcal{B} ( \hat{\bold{z}}_{\tiny{BK}} ,\bold{\mathcal{X}} )
\Big|\Big|^2}_{
\text{IMU measurement residuals}}
+  
\underbrace{\sum_{(j,l) \in \mathcal{C}} 
\rho\Big( \big|\big|
    \bold{r}_\mathcal{C} ( \hat{\bold{z}}_{\tiny{C_jl}},\bold{\mathcal{X}} )
\big|\big|^2 \Big)}_{
\text{Visual measurement residuals}}
$$
where $\rho(e)$ is a Huber norm. 
$$
\rho(e) = \left\{
    \begin{align*}
        & 1 && e \ge 1 \\
        & 2\sqrt{s}-1 && e < 1
    \end{align*}
\right.
$$

$\hat{\bold{z}}_{\tiny{BK}}$ describes the estimate of $K$-th keyframe's IMU pre-integrated state; 
$\hat{\bold{z}}_{\tiny{C_jl}}$ describes in the $j$-th camera frame the estimate of the $l$-th visual feature.

### IMU Measurement Residual

Consider the IMU measurements within two consecutive
frames $k_i$ and $k_{i+1}$

$$
\bold{r}_\mathcal{B} ( \hat{\bold{z}}_{\tiny{Bk_i+1}} ,\bold{\mathcal{X}} )
=
\begin{bmatrix}
    \delta \Delta \bold{p}_{\tiny{Bk_i+1}} \\
    \delta \Delta \bold{v}_{\tiny{Bk_i+1}} \\
    \delta \Delta \bold{\phi}_{\tiny{Bk_i+1}} \\
    \delta \Delta \bold{b}_{\bold{a}} \\
    \delta \Delta \bold{b}_{\bold{\omega}} \\
\end{bmatrix}
=
\begin{bmatrix}
\begin{align*}
&
    \Delta\hat{\bold{p}}_{\tiny{{K+1}}} -
\bold{R}_{\tiny{BK+1}}
\big(\bold{p}_{\tiny{WK+1}}-\bold{p}_{\tiny{WK}}
+\frac{1}{2}\bold{g}_{earth}\Delta t^2
-\bold{v}_{\tiny{WK}}\Delta t \big)
\\ &
    \Delta\hat{\bold{v}}_{\tiny{{K+1}}} -
\bold{R}_{\tiny{W{K+1}}}
\big(\bold{v}_{\tiny{W{K+1}}}-\bold{v}_{\tiny{WK}}
+\bold{g}_{earth}\Delta t \big)
\\ &
    \Delta\hat{\bold{\phi}}_{\tiny{{K+1}}} 
\bold{\phi}_{\tiny{WK}}^{-1} \otimes
\bold{\phi}_{\tiny{W{K+1}}}
\\ &
    \bold{b}_{\tiny{ak_i}}-\bold{b}_{\tiny{ak_i+1}} 
\\ &
    \bold{b}_{\tiny{\omega k_i}}-\bold{b}_{\tiny{\omega k_i+1}} 
    
\end{align*}
\end{bmatrix}
$$

### Visual Measurement Residual

Consider the $l$-th feature $(\frac{1}{\lambda_l}u_{\tiny{Cil}},\frac{1}{\lambda_l}v_{\tiny{Cil}})$ that is observed in the $i$-th image, 
the perfect reprojection of this the feature observation in the $j$-th image can be computed by

$$
\begin{bmatrix}
    x_{\tiny{Cjl}} \\
    y_{\tiny{Cjl}} \\
    z_{\tiny{Cjl}} \\
    1 \\
\end{bmatrix}
=
\bold{T}_{\tiny{BC}}^{-1}
\bold{T}_{\tiny{WBj}}^{-1}
\bold{T}_{\tiny{WBi}}
\bold{T}_{\tiny{BC}}
\begin{bmatrix}
    \frac{1}{\lambda_l}u_{\tiny{Cil}} \\
    \frac{1}{\lambda_l}v_{\tiny{Cil}} \\
    \frac{1}{\lambda_l} \\
    1 \\
\end{bmatrix}
$$
where $\bold{T}_{\tiny{BC}}\bold{T}_{\tiny{WBi}}$ describes the transform first from camera frame to body frame, then from body frame to world frame.

For non-perfect reprojection, the residual of one shared-observed feature can be defined as
$$
\bold{r}_{\mathcal{C}\tiny{ijl}} = 
\begin{bmatrix}
    \frac{x_{\tiny{Cjl}}}{z_{\tiny{Cjl}}}
    - \frac{1}{\lambda_l}u_{\tiny{Cil}} \\
    - \frac{y_{\tiny{Cjl}}}{z_{\tiny{Cjl}}}
    - \frac{1}{\lambda_l}v_{\tiny{Cil}}
\end{bmatrix}
$$

### Marginalization 

The proposed sliding window method is a compromise between :
* applying full bundle adjustment on all camera poses and feature points, but would be time-consuming
* just use two neighbor keyframes' computed triangulation, but results would be coarse

The sliding window needs to continuously add new keyframes and feature points, as well as removing old ones.
This process is coined *Marginalization*

<div style="display: flex; justify-content: center;">
      <img src="imgs/vins_sliding_window.png" width="40%" height="40%" alt="vins_sliding_window" /> 
</div>
<br/>

Recall Schur trick that
attempts to remove $\Delta \bold{x}_{\bold{\xi}'}$ (old $\bold{x}_k$) and retain $\Delta \bold{x}_{\bold{\xi}}$ (existing $\bold{x}_k$),
and takes into consideration the Gaussian noises $\bold{g}=[\bold{v} \quad \bold{w}]^\text{T}$, there is
$$
\begin{bmatrix}
    \bold{B} & \bold{E} \\
    \bold{E}^\text{T} & \bold{C}
\end{bmatrix}
\begin{bmatrix}
    \Delta \bold{x}_{\bold{\xi}} \\
    \Delta \bold{x}_{\bold{\xi}'}
\end{bmatrix}
=
\begin{bmatrix}
    \bold{v} \\
    \bold{w}
\end{bmatrix}
$$
that has a result
$$
\underbrace{(\bold{B}-\bold{E}\bold{C}^{-1}\bold{E}^\text{T})}_{
H_{\tiny{p}}}
\underbrace{\Delta \bold{x}_{\bold{\xi}}}_{
\bold{\mathcal{X}}}
=
\underbrace{
\bold{v} - \bold{E}\bold{C}^{-1} \bold{w}}_{
\bold{r}_{\tiny{p}}}
$$

Back to VINS that marginalizes out old IMU states $\bold{x}_k$ and features $\lambda_l$ from the sliding window. The subscript $\tiny{p}$ means prior.

For pose optimization such as 
$\min_{\bold{\mathcal{X}}} \sum_{\bold{x}_k \in \bold{\mathcal{X}}} ||(\bold{x}_{ki}^{-1}\bold{x}_{kj}) \Delta\hat{\bold{x}}_{kij}||^2$ 
where $\Delta\hat{\bold{x}}_{kij}$ describes the transform between two frames $\bold{x}_{kj}$ and $\bold{x}_{ki}$.
Hessian is defined $H_{\tiny{p}}=J \Sigma^{-1} J^\top$ for the least squares problem.

## Global Pose Graph Optimization

### Relocalization

The relocalization process effectively aligns the current sliding window maintained by the monocular VIO to the graph of past poses.

IMU introduces accumulated drifts to the system.
To counter this issue, define another error term that accounts for loop closure frames ${o}$:
$$
\sum_{l,o \in \mathcal{L}} \rho \big(
|| \bold{r}_{\bold{\mathcal{C}}}(
\hat{\bold{z}}_{\tiny{{o}l}},
\bold{\mathcal{X}},
\hat{\bold{\phi}}_{\tiny{W{o}}},
\hat{\bold{p}}_{\tiny{W{o}}},
) ||\big)
$$
where $\mathcal{L}$ is the set of the observation of retrieved features in
the loop closure frames. $(l, o)$ means $l$-th feature observed in the loop closure frame $o$.

### Pose Optimization

VINS measures four degrees-of-freedom (x, y, z and yaw angle).

In graph optimization, there are two edges:

1) *Sequential Edge*: 
A keyframe will establish several sequential edges to its previous keyframes.

2) *Loop Closure Edge*: 
If the newly marginalized keyframe has a loop connection, it will be connected with the loop closure frame by a loop closure edge in the pose graph.