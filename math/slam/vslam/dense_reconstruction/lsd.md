# LSD-SLAM: Large-Scale Direct Monocular SLAM

LSD-SLAM generates a consistent global map, using direct image alignment and probabilistic, semi-dense depth maps instead of keypoints.

Overall, LSD-SLAM includes three major tasks:
* The *tracking* component continuously tracks new camera images by estimating their rigid body pose $\mathbf{\xi} \in se(3)$.
* The *depth map estimation* component uses tracked frames to either refine
or replace the current keyframe $K_i$ by minimizing photometric errors with applied regularization.
* Incorporate depth map into a global map with refined loop closures and scale-drift ($sim(3)$).

### Lie Algebra for 3D Rigid Body Transform

A 3D rigid body transform $\mathbf{G} \in SE(3)$ (or $\mathbf{G} \in \mathbb{R}^{4 \times 4}$) is defined as
$$
\mathbf{G} = 
\begin{bmatrix}
    \mathbf{R} & \mathbf{t} \\\\
    \mathbf{0} & 1
\end{bmatrix}
, \quad
\mathbf{R} \in SO(3)
, \space
\mathbf{t} \in \mathbb{R}^3
$$

Define the transform corresponding element as $\mathbf{\xi} \in se(3)$, or $\mathbf{\xi} \in \mathbb{R}^6$. Elements are mapped to $\mathbf{G}$ by $\mathbf{G}=\exp_{se(3)}(\mathbf{\xi})$,
and its inverse is $\mathbf{\xi}=\log_{SE(3)}(\mathbf{G})$.
The transformation moving a point from
frame $i$ to frame $j$ is written as $\mathbf{\xi}\_{ij}$.

Define the pose concatenation operator $\circ : se(3) \times se(3) \rightarrow se(3)$ as
$$
\mathbf{\xi}\_{ki} := 
\mathbf{\xi}\_{kj} \circ \mathbf{\xi}\_{ji} :=
\log_{SE(3)} \bigg(
    \exp_{se(3)}(\mathbf{\xi}\_{kj})
    \cdot 
    \exp_{se(3)}(\mathbf{\xi}\_{ji})
\bigg)
$$

Define the 3D projective warp function $\omega$, which projects an
image point $\mathbf{p}$ and its inverse depth $d$ into a by $\mathbf{\xi}$ transformed camera frame.

$$
\omega(\mathbf{p}, d, \mathbf{\xi}) :=
\begin{bmatrix}
    \frac{x'}{z'} \\\\
    \frac{y'}{z'} \\\\
    \frac{1}{z'} \\\\
\end{bmatrix}
, \quad
\begin{bmatrix}
    x' \\\\
    y' \\\\
    z' \\\\
    1
\end{bmatrix}
:=
\exp_{se(3)}(\mathbf{\xi})
\cdot 
\begin{bmatrix}
    \frac{\mathbf{p}_x}{d} \\\\
    \frac{\mathbf{p}_y}{d} \\\\
    \frac{1}{d} \\\\
    1
\end{bmatrix}
$$

A 3D similarity transform $\mathbf{S} \in Sim(3)$ denotes rotation, scaling and translation:
$$
\mathbf{S} = 
\begin{bmatrix}
    s\mathbf{R} & \mathbf{t} \\\\
    \mathbf{0}  & 1
\end{bmatrix}
, \quad
\mathbf{R} \in SO(3),
\space \mathbf{t} \in \mathbb{R}^3,
s \in \mathbb{R}^+
$$

For associated Lie-algebra $\mathbf{\xi} \in sim(3)$, or $\mathbf{\xi} \in \mathbb{R}^7$, now has one additional DOF: scaling to rotation.

Two images $I_A$ and $I_B$ are aligned by Gauss-Newton minimization of the photometric error:
$$
\underset{\mathbf{\xi} \in \mathbb{R}^7}{\min} \space E(\mathbf{\xi}) = 
\sum_i 
    \underbrace{\bigg(
    I_A(\mathbf{p}\_i) 
    - I_B \big(\omega(\mathbf{p}\_i, D_A(\mathbf{p}\_i), \mathbf{\xi}) \big)
\bigg)^2}\_{:= r_i^2(\mathbf{\xi})}
$$

whose iterative step update $\mathbf{\xi}^*=\Delta\mathbf{\xi}^*+\mathbf{\xi}_0$ by Gauss-Newton method can be computed via

$$
\Delta\mathbf{\xi}^* = 
\frac{J^\top \mathbf{r}(\mathbf{\xi}_0)}{J^\top J}
$$
where $\mathbf{r}(\mathbf{\xi}_0)$ is the initial residual for $\mathbf{\xi}_0$, and $J$ is the Jacobian.

To apply weights on residuals to down-weight large error (Huber loss should have the same result), there is
$$
E(\mathbf{\xi}) = 
\sum_i w_i(\mathbf{\xi}) r_i^2(\mathbf{\xi})
$$

and update is

$$
\Delta\mathbf{\xi}^* = 
\frac{J^\top W \mathbf{r}(\mathbf{\xi}_0)}{J^\top W J}
$$

The inverse of the weighted Hessian $({J^\top W J})^{-1}$ is an estimate for the covariance $\Sigma_{\mathbf{\xi}}$ (only hold true when pose error $\epsilon$ follows Gaussian distribution) such that
$$
\mathbf{\xi}^* = 
\mathbf{\epsilon} \circ \mathbf{\xi}
, \quad
\mathbf{\epsilon} \sim N(0, \Sigma_{\mathbf{\xi}})
$$

### Tracking: Direct $se(3)$ Image Alignment

LSD's proposed normalized-variance that takes into account varying noise on the depth estimates.

Starting from an existing keyframe $K_i=(I_i, D_i, V_i)$, 
the relative 3D pose $\mathbf{\xi}\_{ij} \in se(3)$ of a new image $I_j$ is computed by minimizing the variance-Huber-normalized photometric error:
$$
\min_{\mathbf{\xi} \in se(3)} E_\mathbf{p}(\mathbf{\xi_{ij}})= 
\sum_{\mathbf{p} \in \Omega_{D_i}}
\bigg|\bigg|
    \frac{r_p^2 (\mathbf{p}, \mathbf{\xi}\_{ij})}{\sigma_{r_p (\mathbf{p}, \mathbf{\xi}\_{ij})}^2}
\bigg|\bigg|_\delta
$$

where
$$
||e||_\delta=
\left\{
    \begin{array}{c}
        \frac{1}{2}e^2 &\quad \text{for} |e|\le \delta
        \\\\
        \delta \cdot (|e|-\frac{1}{2}\delta) &\quad \text{otherwise}
    \end{array}
\right.
$$

$r_p (\mathbf{p}, \mathbf{\xi}\_{ij})$ is the Lucas-Kanade style photometric error
$$
r_p (\mathbf{p}, \mathbf{\xi}\_{ij}) :=
I_i(\mathbf{p}\_i)  - I_j \big(\omega(\mathbf{p}\_i, D_A(\mathbf{p}\_i), \mathbf{\xi}\_{ij})\big)
$$

The residual’s variance $\sigma_{r_p (\mathbf{p}, \mathbf{\xi}\_{ij})}$ is computed using covariance propagation $\Sigma_{r_p}=J_{r_p} \Sigma_{r_p} J^\top_{r_p}$, where $J^\top_{r_p}$ is the Jacobian of $r_p$, and utilizing the inverse depth variance $V_i$:
$$
\sigma^2_{r_p (\mathbf{p}, \mathbf{\xi}\_{ij})} :=
2\sigma_I^2 +
\bigg( \frac{\partial r_p(\mathbf{p}, \mathbf{\xi}\_{ij})}
{\partial D_i(\mathbf{p})}
\bigg)^2 V_i(\mathbf{p})
$$

### Depth Map Estimation

* Keyframe Selection

Define the distance between two keyframes from $K_i$ to $K_j$ as
$$
dist(\mathbf{\xi}\_{ij}) := 
\mathbf{\xi}\_{ij}^\top W \mathbf{\xi}\_{ij}
$$

where $W$ is a diagonal matrix containing the weights.

Each keyframe is scaled and its mean inverse depth
is final one distance.

If two keyframes with $dist(\mathbf{\xi}\_{ij})$ greater than a threshold, a new keyframe is created. 

* Depth Map Creation

Once a new frame is chosen to become a keyframe, its
depth map is initialized by projecting points from the previous keyframe into it.

* Depth Map Refinement. 

Tracked frames that do not become a keyframe
are used to refine the current keyframe.

### Direct $sim(3)$ Image Alignment

Monocular SLAM is – in contrast to
RGB-D or Stereo-SLAM – inherently scale-ambivalent. To address this issue, LSD SLAM proposes *direct, scale-drift aware image alignment* on $sim(3)$.

In addition to the photometric residual $r_p$ , LSD SLAM considers a depth residual $r_d$ which
penalizes deviations in inverse depth between keyframes, allowing to directly estimate the scaled transformation between them.

$$
\min_{\mathbf{\xi} \in se(3)} E(\mathbf{\xi_{ij}})
:= 
\sum_{\mathbf{p} \in \Omega_{D_i}}
\bigg|\bigg|
    \frac{r_p^2 (\mathbf{p}, \mathbf{\xi}\_{ij})}{\sigma_{r_p (\mathbf{p}, \mathbf{\xi}\_{ij})}^2}
    +
    \frac{r_d^2 (\mathbf{p}, \mathbf{\xi}\_{ij})}{\sigma_{r_d (\mathbf{p}, \mathbf{\xi}\_{ij})}^2}   
\bigg|\bigg|_\delta
$$

where $r_d$ is defined as the gap between $3$-axis depth and the estimated depth from $1$- and $2$- axises of the transformed pixel $\mathbf{p}':= \omega_s \big(\mathbf{p}, D_i(\mathbf{p}), \mathbf{\xi}\_{ij} \big)$.

$$
r_d(\mathbf{p}, \mathbf{\xi}\_{ij})
:=
[\mathbf{p}']_3 - D_j([\mathbf{p}']_{1,2})
$$

Accordingly, the variance for $r_d$ talks about two parts: one for $D_j([\mathbf{p}']_{1,2})$ and $D_j(\mathbf{p]})$
$$
\sigma^2_{r_d(\mathbf{p}, \mathbf{\xi}\_{ij})} := 
V_j([\mathbf{p}']_{1,2})
\bigg( \frac{\partial \space r_d(\mathbf{p}, \mathbf{\xi}\_{ij})}
{\partial \space D_j([\mathbf{p}']_{1,2})}
\bigg)^2
+
V_j(\mathbf{p})
\bigg( \frac{\partial \space r_d(\mathbf{p}, \mathbf{\xi}\_{ij})}
{\partial \space D_j(\mathbf{p})}
\bigg)^2
$$

### Loop Closure

After a new keyframe $K_i$ is added to the map, a number of possible loop closure keyframes $K_{i1}, K_{i2}, ..., K_{in}$ are collected.

* LSD SLAM uses closest $10$ keysframes.

* Suitable candidate are proposed by an appearance-based mapping algorithm

Fast Appearance-Based Mapping (FAB-MAP) leverages the high information content of visual images to detect loop closure.

FAB-MAP matches the appearance between the current
visual scene and a past location by representing the image with bag-of-words techniques.

An appearance-based feature detector is used to find interesting points in the image and represent the appearance of the local region using a 64 or 128 dimensional vector.

The observation, $Z$, of the image at time $k$ is
then reduced to a binary vector, indicating what feature
words are present in the image.

$$
Z_k = \{ z_1, z_2, ..., z_n \}
$$

* Reciprocal tracking check to prevent false loop closures

For each keyframe $K_{i}$, independently forward and reverse track $\mathbf{\xi}\_{ij}$ and $\mathbf{\xi}\_{ji}$. The detected loop closure is good only if the
two estimates are statistically similar (by $\mathcal{L}_2$ norm err):
$$
e(\mathbf{\xi}\_{ij}, \mathbf{\xi}\_{ji}) = 
(\mathbf{\xi}\_{ij} \circ \mathbf{\xi}\_{ji})^\top
\big( \Sigma_{ji} + Adj_{ji}\Sigma_{ij}Adj_{ji}^\top \big)^{-1}
(\mathbf{\xi}\_{ij} \circ \mathbf{\xi}\_{ji})
$$

### Optimizations

* Initialization

Use a small amount of known keyframes to constrain $\mathbf{\xi} \in se(3)$ for $\min_{\mathbf{\xi} \in se(3)} E(\mathbf{\xi_{ij}})$

* Efficient Second Order Minimization

A sum-of-squared-difference (SSD) problem attempts to 
$$
\min_{\mathbf{\xi} \in se(3)} 
\frac{1}{2} \big|\big|
    I_A(\Delta \mathbf{\xi} \circ \mathbf{\xi}_c)
    - I_A(\mathbf{\xi}_0)
\big|\big|^2
$$

where $\mathbf{\xi}_c$ is the current camera pose and $\mathbf{\xi}_0$ is the previous. Through $\Delta \mathbf{\xi}$, the current camera pose $\mathbf{\xi}_c$ should be transformed to its previous, and the resultant image pixel $I_A(.)$'s difference should be zero.

However, Newton method for SSD problem is time-consuming for Hessian computation.

Efficient Second Order Minimization approximates the problem by only taking $\Delta \mathbf{\xi}$, and it see faster computation without engaging a Hessian.

$$
\min_{\mathbf{\xi} \in se(3)} 
\frac{1}{2} \big|\big|
    I_A(\Delta \mathbf{\xi})
    - I_A(\mathbf{\xi}_0)
\big|\big|^2
$$

* Coarse-to-Fine Approach

Similar to pyramid approach scaling images at different level.

* Map Optimization

The map, consisting of a set of keyframes and tracked $sim(3)$-constraints, is continuously optimized in the background using pose graph optimization.

$$
\min_{\mathbf{\xi}\_{w1}, \mathbf{\xi}\_{w2}, ..., \mathbf{\xi}\_{wn}} E(\mathbf{\xi}\_{w1}, \mathbf{\xi}\_{w2}, ..., \mathbf{\xi}\_{wn})
:= 
\sum_{\mathbf{\xi}\_{ji}, \Sigma_{ji}}
\big( \mathbf{\xi}\_{ji} \circ \mathbf{\xi}^{-1}\_{wi} \circ \mathbf{\xi}\_{wj} \big)^\top
\Sigma_{ji}
\big( \mathbf{\xi}\_{ji} \circ \mathbf{\xi}^{-1}\_{wi} \circ \mathbf{\xi}\_{wj} \big)
$$

where $\mathbf{\xi}\_{ji}$ denotes the transform from $\mathbf{\xi}\_{j}$ to $\mathbf{\xi}\_{i}$, and the subscript $w$ means in the "world frame". 

