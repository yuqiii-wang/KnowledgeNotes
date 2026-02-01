# Direct Tracking and Mapping (DTAM)

### Dense Mapping Problem Formulation

Direct method starts with the idea of using all the pixels from camera frame to camera frame to resolve the world around sensors, relying on principles from photogrammetry, different from feature-based methods that take interest in image features.

Define the pose of a camera $c$ with respect to the world frame of reference $w$ as $T_{wc} \in SE(3)$:
$$
T_{wc} = 
\begin{bmatrix}
    R_{wc} & \mathbf{c}\_{w} \\\\
    \mathbf{0}^\top & 1
\end{bmatrix}
$$

Define a world point such that $\mathbf{x}_w = T_{wc}\mathbf{x}_c$. Camera pre-calibrated intrinsic matrix is denoted as $K$. 
Perspective projection of a
3D point is denoted as $\mathbf{x}_c = (x, y, z)^\top$,
and its homogeneous projection is $\pi(\mathbf{x}_c) = (x/z, y/z)^\top$. Besides, define $\pi^{-1}(\mathbf{u}, d) = \frac{1}{d}K^{-1}\dot{\mathbf{u}}$ to represent the operation reversed the homogeneous projection to include the depth axis, where $\dot{\mathbf{u}}:=(u,v,1)^\top$.


Define a keyframe $r$ consisted of a reference image $I_r$ with pose $T_{rw}$. Inside this image, each pixel is denoted as $\mathbf{u}_r=(u_r,v_r)^\top \in \Omega$, where $\Omega$ is a 3D RGB space.

Define a camera movement $m \in \mathcal{I}(r)$ and the new camera pose is $T_{mw}$.
The camera movement set $\mathcal{I}(r)$ consists of frames nearby and overlapping
$r$.

Indicated in the figure below, each pixel of the reference frame $\mathbf{u}_r$ has an associated row of entries $C_r(\mathbf{u})$ (shown in red) that store the average photometric error $C_r(\mathbf{u}, d)$ computed for each inverse depth $d \in D$, where $D$ is constrained in $S \in [\xi_{min}, \xi_{max}]$.

The $C_r$ space is discretized into $M \times N \times S$ voxels that help optimization solution search.

<div style="display: flex; justify-content: center;">
      <img src="imgs/dense_tracking_and_mapping.png" width="40%" height="40%" alt="dense_tracking_and_mapping" />
</div>
</br>

The average photometric error $C_r(\mathbf{u}, d)$ is computed by projecting a 3D
point in the volume into each of the overlapping images and
summing the $\mathcal{L}_1$ norm of the individual photometric errors
obtained:

* Each photometric error to a keyframe $r$ given the movement $m$:
$$
\rho_r(I_m, \mathbf{u}, d) = 
I_{r}(\mathbf{u}) - 
I_m\big(
    \pi (K T_{wc} \pi^{-1}(\mathbf{u}, d))
\big)
$$

* Collectively, the normalized $\mathcal{L}_1$ norm error over a number for movements $m \in \mathcal{I}(r)$ is
$$
C_r(\mathbf{u}, d) = \frac{1}{|\mathcal{I}(r)|}
\sum_{m \in \mathcal{I}(r)}
\big|\big|
\rho_r(I_m, \mathbf{u}, d)
\big|\big|_1
$$

### Mapping Noise Problem

Directly solving $C_r(\mathbf{u}, d)$ can lead to noisy results.

Exemplified in the figure below, there exist three points $a,b,c$, where $a$ is a point on a plain black plane (totally featureless); $c$ is on an edge, and $a$ is a corner feature.

The three points' normalized costs $C_r(\mathbf{u}, d)$ are plotted shown as bold red curves, computed/normalized from their individual photometric errors $\rho_r(I_m, \mathbf{u}, d)$ which are marked in various color thinner lines.

Point $a$'s photometric errors are very chaotic while $b$ exhibits a clear error minima. 
This shows that feature points' depth are easy to estimate, while featureless points are noisy.

<div style="display: flex; justify-content: center;">
      <img src="imgs/noisy_inverse_depth.png" width="80%" height="40%" alt="noisy_inverse_depth" />
</div>
</br>

### Regularization

Regularization can be implemented to address the aforementioned issue, that in real world, often is the assumption that plain/featureless surface object should have smooth decaying/growing depth.
In other words, depth changes at the point $a$ should be smooth.

Huber loss can be used to contain large error terms by $\mathcal{L}_1$ norm and amplify small error terms by $\mathcal{L}_2$.

$$
L\_{\delta}(e)=
\left\{
    \begin{array}{c}
        \frac{1}{2}e^2 &\quad \text{for} |e|\le \delta
        \\\\
        \delta \cdot (|e|-\frac{1}{2}\delta) &\quad \text{otherwise}
    \end{array}
\right.
$$

The regularizer comprises a weighted Huber norm over the gradient of the inverse depth map $g(\mathbf{u}) \cdot \big|\big|\nabla \mathbf{\xi}(\mathbf{u})\big|\big|_\delta$.

Replace Huber loss error term $L\_{\delta}(e)$ with the gradient of the inverse depth map such as $L\_{\delta}\big(\nabla \mathbf{\xi}(\mathbf{u})\big)=\big|\big|\nabla \mathbf{\xi}(\mathbf{u})\big|\big|_\delta$. 
Here $\nabla \mathbf{\xi}(\mathbf{u})$ describes depth changes.

As a result, if there is any sudden change in depth (for this pixel's projected depth lower than $\delta$ such as $\big|\mathbf{\xi}(\mathbf{u})\big|<\delta$), the depth changes are smoothed by $\mathcal{L}_2$; otherwise, smoothed by $\mathcal{L}_1$.

Per pixel weight $g(\mathbf{u})$ is defined as
$$
g(\mathbf{u}) = 
e^{-\alpha \big|\big| 
    \nabla I_r (\mathbf{u})
\big|\big|^\beta_2}
$$

This term $g(\mathbf{u})$ reduces the regularization strength where the edge magnitude is high, 
thereby limiting solution smoothing across region boundaries (making edge/corner transition obvious). 
$\nabla I_r (\mathbf{u})$ describes in-image pixel changes (increase weight for visual feature pixels). $\alpha$ is a scaling factor and $\beta$ is the exponential controlling factor.

Below is the energy function that takes into account the regularization and photometric errors.

$$
E_{\mathbf{\xi}} = 
\int_{\Omega} \big\{
    g(\mathbf{u}) \cdot \big|\big|\nabla \mathbf{\xi}(\mathbf{u})\big|\big|_\delta
    +
    \lambda C\big(\mathbf{u}, \mathbf{\xi}(\mathbf{u})\big)
\big\} d\mathbf{u}
$$

where regularizer $g(\mathbf{u}) \cdot \big|\big|\nabla \mathbf{\xi}(\mathbf{u})\big|\big|_\delta$ is a convex function while $C\big(\mathbf{u}, \mathbf{\xi}(\mathbf{u})\big)$ is non-convex.

In order to optimize the non-convex function, the below approximation is proposes.

$$
E_{\mathbf{\xi}, \mathbf{\alpha}} = 
\int_{\Omega} \big\{
    g(\mathbf{u}) \cdot \big|\big|\nabla \mathbf{\xi}(\mathbf{u})\big|\big|_\delta
    +
    \frac{1}{2\theta} \big( \mathbf{\xi}(\mathbf{u})-\mathbf{\alpha}(\mathbf{u}) \big)^2
    +
    \lambda C\big(\mathbf{u}, \mathbf{\alpha}(\mathbf{u})\big)
\big\} d\mathbf{u}
$$

where, the coupling term $Q(\mathbf{u})=\frac{1}{2\theta} \big( \mathbf{\xi}(\mathbf{u})-\mathbf{\alpha}(\mathbf{u}) \big)^2$ serves to drive the original and auxiliary variables together in optimization, enforcing $\mathbf{\xi}=\mathbf{\alpha}$ as $\theta \rightarrow 0$.

$g(\mathbf{u}) \cdot \big|\big|\nabla \mathbf{\xi}(\mathbf{u})\big|\big|_\delta + Q(\mathbf{u})$ can be solved via a primal-dual approach. $Q(\mathbf{u})+\lambda C\big(\mathbf{u}, \mathbf{\alpha}(\mathbf{u})\big)$ is trivially point-wise optimizable and can be solved using an exhaustive search over a finite range of discretely sampled inverse depth values.

### Dense Tracking and Pose Estimation

DTAM proposes the pose of a live camera estimation by finding the parameters of motion 
which generate a synthetic view which best matches the live video image.

The live camera pose estimation is essentially finding a camera pose transform ${T}\_{wl}$ in the world frame. 
The subscript $l$ means "live" and $w$ means "world". 
Set ${T}\_{wl}=T_{wv}T_{vl}$, where the subscript $v$ means "virtual", basically as if a virtual camera taking a shot at this pose $T_{wv}$ to represent any camera pose near to the true pose ${T}\_{wl}$. 

$T_{lv}$ is a correction to $T_{wv}$ to form the true pose ${T}\_{wl}$, and $T_{lv}$ is function of $\psi \in \mathbb{R}^6$  belonging to the Lie algebra $se3$.


Pose estimation is to minimize this cost function
$$
F(\psi)=
\frac{1}{2} \sum_{\mathbf{u}\in\Omega}
\big(
    f_{\mathbf{u}}(\psi)
\big)^2
$$

where $f_{\mathbf{u}}(\psi)$ is the overall difference between two images $I_l$ and $I_v$, in which $I_l$ takes into account $T_{lv}(\psi)$ that adjusts $I_l$ to reduce the difference to $I_v$.

$$
f_{\mathbf{u}}(\psi) = 
I_l \big( \pi\big( 
    KT_{lv}(\psi) \pi^{-1} (\mathbf{u}, \xi_v(\mathbf{u}))
\big)\big) - I_v(\mathbf{u})
$$