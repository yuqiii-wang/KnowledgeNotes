# SURF (Speeded-Up Robust Features)

## Feature Extraction

The approach for interest point detection uses a very basic Hessian matrix approximation.

### Integral Image

An *integral image* (a.k.a *summed-area table*) is a data structure and algorithm for generating the sum of values in a rectangular subset of a grid by a precomputed summed table.

In computer vision, the integral image is used as a quick and effective way of calculating the sum of values (pixel values) in a given image — or a rectangular subset of a grid (the given image). It can also, or is mainly, used for calculating the average intensity within a given image.

* Definition

The value $I$ at any point $(x, y)$, together denoted as $I_{sum}(x,y)$ in the summed-area table is the sum of all the pixels above and to the left of $i(x, y)$, inclusive, such as
$$
I_{sum}(x,y) = 
\sum_{
    \scriptsize{
    \begin{matrix}
        \forall\space x' \le x \\
        \forall\space y' \le y
    \end{matrix}
    }
}
i (x',y')
$$
where $i(x',y')$ is a pixel value.

Once the summed-area table has been computed, evaluating the sum of intensities over any rectangular area requires exactly four array references regardless of the area size.
$$
\sum_{
    \scriptsize{
    \begin{matrix}
        x_0 \le x \le x_1 \\
        y_0 \le y \le y_1
    \end{matrix}
    }
} i (x,y) =
I_{sum}(x_0, y_0) + I_{sum}(x_1, y_1) 
- I_{sum}(x_0, y_1) - I_{sum}(x_1, y_0)
$$

* Example

In the example below, the $2.$ matrix is the summed-area table of the $1.$ matrix with each entry computed by $I(x,y)$.
The purple rectangle sum from the $1.$ matrix is equal to the four angle points' sum from the $2.$ matrix.

![integral_image](imgs/integral_image.png "integral_image")

### Hessian affine region detector

For selecting the location and the scale (Hessian-Laplace detector), Hessian affine region detector relies on the determinant of the Hessian matrix.

*  Laplacian

The *Laplace operator* or *Laplacian* is a differential operator given by the divergence of the gradient of a scalar function on Euclidean space, denoted as $\nabla \cdot \nabla$, or simply $\nabla^2$, or $\Delta$.

In a Cartesian coordinate system, the Laplacian is given by the sum of second partial derivatives of the function with respect to each independent variable.
$$
\Delta f = \nabla^2 f = 
\sum_{i=1}^{n} \frac{\partial^2 f}{\partial x^2_i}
$$

* Laplacian of Gaussians (LoG)

Given an input image $I(x,y)$, this image is convolved (convolution denoted as $\otimes$) by a Gaussian kernel denoted as $L({x}, {y}, \sigma_I^2)=g({x}, {y}, \sigma_I^2) \otimes I({x}, {y})$ with a scale factor $\sigma_I^2$. The Gaussian kernel is defined as

$$
g(x,y,\sigma_I^2) =
\frac{1}{2\pi \sigma_I^2}
e^{-\frac{x^2+y^2}{2\sigma_I^2}}
$$

Given a pixel, the Hessian of this pixel $i(x,y)$ is defined
$$
\begin{align*}
H\big(i(x,y)\big) &= 
\begin{bmatrix}
    \frac{\partial^2 i}{\partial x^2} & \frac{\partial^2 i}{\partial x \partial y} \\
    \frac{\partial^2 i}{\partial y \partial x} & \frac{\partial^2 i}{\partial y^2}
\end{bmatrix}
\\ &=
\begin{bmatrix}
    L_{xx} & L_{xy} \\
    L_{yx} & L_{yy}
\end{bmatrix}
\end{align*}
$$

At each scale, interest points are those points that simultaneously are local extrema of both the determinant and trace of the Hessian matrix. The trace of Hessian matrix is identical to the Laplacian of Gaussians (LoG)

$$
\begin{align*}
det(H) &= 
\sigma_I^2 \big(
    L_{xx} L_{yy} - L_{xy}^2
\big)
\\
tr(H) &=
\sigma_I (L_{xx}+L_{yy})
\end{align*}$$

By choosing points that maximize the determinant of the Hessian, this measure penalizes longer structures that have small second derivatives (signal changes) in a single direction.

* SURF Implementation with Integral Image

Similar to SIFT 