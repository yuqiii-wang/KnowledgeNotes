# Scale-invariant feature transform (SIFT)

## Scale-Space Extrema Detection

Starting with key point detection, the image $I$ is convolved with Gaussian filters at different scales, and then the difference of successive Gaussian-blurred images are taken. 
Keypoints are then taken as maxima/minima of the *Difference of Gaussians* (DoG) that occur at multiple scales. Specifically, a DoG image $D(x,y,\sigma)$ is given by
$$
D(x,y,\sigma) = 
L(x,y,k_i \sigma) - L(x,y,k_j \sigma)
$$
where $L(x,y,k \sigma)$ is the convolution (operation denoted as $*$) of the input image $I(x,y)$ with the Gaussian blur $g(x,y, k\sigma)$ at the scale $k\sigma$ , i.e.,
$$
L(x,y,k \sigma) = 
g(x,y, k\sigma) * I(x,y)
$$

Hence a DoG image between scales $k_i \sigma$  and $k_j \sigma$  is just the difference of the Gaussian-blurred images at the two scales.

Image $I$ is convolved with $g$ at different scales, the derived images are group by octave. DoG images are obtained from adjacent Gaussian-blurred images per octave.

Keypoints are selected as local minima/maxima of the DoG images across scales, by comparing the pixel and its neighbor pixels.

## Keypoint localization

Next is to perform a detailed fit to the nearby data (neighbor pixels of keypoints) for accurate location, scale, and ratio of principal curvatures.

Using the quadratic Taylor expansion of the Difference-of-Gaussian (DoG) scale-space function, DoG can be expressed as
$$
D(\mathbf{x}) = D + 
\frac{\partial D^\text{T}}{\partial \mathbf{x}} \mathbf{x} +
\frac{1}{2} \mathbf{x}^\text{T} \frac{\partial^2 D}{\partial \mathbf{x}^2} \mathbf{x}
$$

where $\mathbf{x}=(x_{\pm}, y_{\pm}, \sigma)^\text{T}$ is the offset from a candidate keypoint.

The location of the extremum, $\hat{\mathbf{x}}$ is determined by taking the derivative of this function $D$ with respect to $\mathbf{x}$ and setting it to zero. Keypoints are adjusted/merged to the computed new local extremum points.

Besides, candidate keypoints with too little differences between $D$ of different scaling factors $k\sigma$ are discarded.

## Edge Removal

The principal curvature across the edge $x_{+}$ would be much larger than the principal curvature along it $x_{-}$. 

<div style="display: flex; justify-content: center;">
      <img src="imgs/eigen_feat_detection.png" width="20%" height="20%" alt="eigen_feat_detection">
</div>
</br>

Finding these principal curvatures amounts to solving for the eigenvalues of the second-order Hessian matrix.

$$
H=\begin{bmatrix}
    D_{xx} & D_{xy} \\\\
    D_{yx} & D_{yy}
\end{bmatrix}
$$

The eigenvalues of $H$ are proportional to the principal curvatures of $D$ for both $x$ and $y$ axes.

Define a ratio $R$ such that

$$
\begin{align*}
R &= \frac{tr(H)^2}{det(H)}
\\\\ &=
\frac{(D_{xx} + D_{yy})^2}{D_{xx}D_{yy}-D_{xy}^2}
\end{align*}
$$

$R$ depends only on the ratio of the eigenvalues rather than $H$'s individual values. $R$ is minimum when the eigenvalues are equal to each other.

Therefore, the higher the absolute difference between the two eigenvalues (greater the discrepancy between $x_{+}$ and $x_{-}$ more likely indicating an edge), which is equivalent to a higher absolute difference between the two principal curvatures of $D$, the higher the value of $R$.

This processing step for suppressing responses at edges is a transfer of a corresponding approach in the Harris operator for corner detection. The difference is that the measure for thresholding is computed from the Hessian matrix instead of a second-moment matrix.

## Orientation Assignment

Compute gradient magnitude and direction of pixels within a window.

$$
\begin{align*}
m(x,y) &= 
\sqrt{\big(I(x+1,y)-I(x-1,y)\big)^2+
\big(I(x,y+1)-I(x,y-1)\big)^2}
\\\\ 
\theta(x,y) &=
atan2\big(I(x,y+1)-I(x,y-1), I(x+1,y)-I(x-1,y)\big)
\end{align*}
$$

The magnitude and direction calculations for the gradient are done for every pixel in a neighboring region around the keypoint in the Gaussian-blurred image $I_{blur}$.

## Keypoint descriptor

Collect pixels' magnitude and direction, and form a circular window ($8 \times 8$ window as in the example below). Then compute the histogram of this window's angles (usually 8 bins/octave).

The $8$ magnitudes and angles of this window can be used for describing this window's feature.


<div style="display: flex; justify-content: center;">
      <img src="imgs/sift.png" width="40%" height="40%" alt="sift">
</div>
</br>


## Fast Library for Approximate Nearest Neighbors (FLANN)

The SIFT is used to find the feature keypoints and descriptors.
A FLANN based matcher with KNN is used to match the descriptors in both images.
In other words, FLANN is a C++ library for approximate nearest neighbor search.