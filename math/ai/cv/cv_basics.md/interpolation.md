# Interpolation

Interpolation is used to fill "gaps" between each data points, such as construct a continuous function out of discrete data points. 

## Nearest

Interpolation through taking the same value as its nearest neighbour's.

![nearest_interpolation](imgs/nearest_interpolation.png "nearest_interpolation")

## Linear

$$
f(x) = a_1 x + a_0
$$

![linear_interpolation](imgs/linear_interpolation.png "linear_interpolation")

## Cubic

$$
f(x) = \sum^3_{i=0}a_i x^i = a_3x^3 + a_2x^2 + a_1x + a_0
$$

Every curve needs four neighbour data points to determine its coefficients $a_i$

$$
f(-1)=a_3(-1)^3+a_2(-1)^2+a_1(-1)^1+a_0
\\
f(0)=a_3(0)^3+a_2(0)^2+a_1(0)^1+a_0
\\
f(1)=a_3(1)^3+a_2(1)^2+a_1(1)^1+a_0
\\
f(2)=a_3(2)^3+a_2(2)^2+a_1(2)^1+a_0
$$

![cubic_interpolation](imgs/cubic_interpolation.png "cubic_interpolation")

## Bilinear

Given a 2-d matrix with known four corners: $f(0,0), f(0,1), f(1,0), f(1,1)$, interpolation follows

* $f(\frac{1}{2}, 0)$ interpolation by $f(0,0), f(0,1)$
* $f(\frac{1}{2}, 1)$ interpolation by $f(1,0), f(1,1)$
* $f(\frac{1}{2}, \frac{1}{2})$ interpolation by $f(\frac{1}{2},0), f(\frac{1}{2},1)$

![bilinear_interpolation](imgs/bilinear_interpolation.png "bilinear_interpolation")

### Bilinear Interpolation with Weights

The weights control that if four neighbor points $(x_i, y_j)$ for $i,j \in \{0,1\}$ (for example, the upper-left neighbor point is $(x_0, y_1)$) have different distances to the interpolation estimate $(\hat{x}, \hat{y})$, such that $\hat{x} \ne \frac{1}{2} (x_0+x_1)$ or $\hat{y} \ne \frac{1}{2} (y_0+y_1)$, the neighbor points should be linearly assigned different weights to reflect the distance closeness to the estimate point.
This gives larger weights for more adjacent points.

Given the interpolation point estimate $(\hat{x}, \hat{y})$, there should be

$$
\begin{align*}
    & x_0 \le \hat{x} \le x_1 \\
    & y_0 \le \hat{y} \le y_1
\end{align*}
$$

The weights for the interpolation point four neighbors are:

$$
w_{i,j} = \Big(1-\frac{\hat{x}-x_i}{x_0-x_1}\Big) \cdot \Big(1-\frac{\hat{y}-y_j}{y_0-y_1}\Big)
$$

If the gaps $\hat{x}-x_i$ and $\hat{y}-y_j$ are considered normalized already, the formula can be simplified as

$$
w_{i,j} = \big(1-|\hat{x}-x_i|\big) \cdot \big(1-|\hat{y}-y_j|\big)
$$

Finally, the interpolated value at the point $(\hat{x}, \hat{y})$ by the function $f$ given the four neighbor points is

$$
f(\hat{x}, \hat{y}) = \sum_{i,j \in \{0, 1\}} w_{i,j} f(x_i, y_j)
$$

## Bicubic

![bicubic_interpolation](imgs/bicubic_interpolation.png "bicubic_interpolation")
