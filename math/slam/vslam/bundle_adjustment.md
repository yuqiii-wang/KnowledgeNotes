# Bundle Adjustment

## Reprojection Error

The reprojection error is a geometric error corresponding to the image distance between a projected point from a real world (a 3d real world point) and a measured one (appeared as a 2D image point). 

The figure below illustrates how a number of cameras indexed by $j=1,2,...,m$ take photos and extract visual feature points (indicated as white dots in the figure) indexed by $i=1,2,...,n$.

Data association/point correspondence refer to a number of image feature points $\bold{x}'_{ij}$ from different cameras describing the same world point $\hat{\bold{X}}_i$

![multi_camera_reprojection_illustration](imgs/multi_camera_reprojection_illustration.png "multi_camera_reprojection_illustration")

Reprojection optimization process can be described as below 

$$
\bold{x}'_{ij} + \hat{\bold{v}}_{\bold{x}'_{ij} } =
\hat{s}_{ij} \hat{\bold{P}}_{j}(\bold{x}_{ij},\bold{p},\bold{q}) \hat{\bold{X}}_i
$$
where
* $\bold{x}'_{ij}$: observed pixel location of the $i$-th visual feature point projected into the $j$-th camera frame. In other words, $i$ is the point index and $j$ is the camera image index.
* $\hat{\bold{v}}_{\bold{x}'_{ij}}$: the estimated correction to $\bold{x}'_{ij}$
* $\hat{s}_{ij}$: the estimated scaling factor
* $\hat{\bold{X}}_i$: the init guess of a world point
* $\hat{\bold{P}}_{j}$: the estimated projection matrix of the $j$-th camera that maps a world point $\hat{\bold{X}}_i$ to the $j$-th camera's frame
* inside $\hat{\bold{P}}_{j}$, $\bold{x}_{ij}$ is the pixel and $\bold{p},\bold{q}$ are projection parameters, camera distortion, respectively

$\hat{\bold{P}}_{j}$ can be decomposed as the product of intrinsics $\hat{\bold{K}}(\bold{x}_{ij}, \hat{\bold{p}},\hat{\bold{q}})$ and extrinsics $[\bold{R}|\bold{t}]_j$ such as $\hat{\bold{P}}_{j}=\hat{\bold{K}}(\bold{x}_{ij}, \hat{\bold{p}},\hat{\bold{q}}) [\bold{R}|\bold{t}]_j$

Here $\bold{x}'_{ij}$ denotes the observed known feature point while $\bold{x}'_{ij}$ represents the unknown feature point.

### Normal Equation/Gauss-Newton Method as A Solution

The solution for calculating $\bold{x}_{ij}$ can be approximated by a Jacobian $\bold{J}$ as a least square problem solver, such as

$$
\bold{e}(\bold{x}+\Delta \bold{x}) \approx
\bold{e}(\bold{x})+\bold{J}^{\text{T}} \Delta \bold{x}
$$

By normalization, there is 

$$
\big(\bold{J}^\text{T}\Sigma^{-1}\bold{J}\big) \Delta\bold{x}_{ij}
=
\big(\bold{J}^\text{T}\Sigma^{-1}\big) \Delta\bold{x}'_{ij}
$$
that yields the estimate $\widehat{\Delta\bold{x}_{ij}}$ such that
$$
\widehat{\Delta\bold{x}_{ij}}
=
\big(\bold{J}^\text{T}\Sigma^{-1}\bold{J}\big)^{-1}
\big(\bold{J}^\text{T}\Sigma^{-1}\big) \Delta\bold{x}'_{ij}
$$

However, this solution suffers from high computation costs since even a small vSLAM problem may contain thousands of images, and each image has hundreds of feature points. This obstacle makes this solution impractical.

## Bundle Adjustment (BA)

*Bundle adjustment* (BA) boils down to minimizing the reprojection error between the image locations of observed and predicted image points, which is expressed as the sum of squares of a large number of nonlinear, real-valued functions. 

BA is statistically optimal under these assumptions
* Gaussian noises