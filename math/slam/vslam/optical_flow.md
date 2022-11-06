# Optical Flow

* Motivation

Image keypoint localization and descriptor calculation are time-consuming.

By using optical flow to track the motion of the key points, computation on descriptor can be avoided.
The time spent on calculating optical flow itself is less than the descriptor calculation and matching.

Once optical flow successfully tracks a keypoint, there is no more computation on the feature descriptor.

* Definition

Optical flow describes the motion of pixels in the images at different timestamp. 

Given a moving camera, the image coordinates in reference to the object changes accordingly.

The calculation of a part of the pixel’s motion is called *sparse optical flow*, 
and the calculation of all pixels in an image is called *dense optical flow*.

### Sparse vs Dense Optical Flow

* A well-known sparse optical flow method is called *Lucas-Kanade* optical flow

Implementation of sparse optical flow includes finding keypoints of the previous image

The figure below is a result of 

![sparse_optical_flow_example](imgs/sparse_optical_flow_example.png "sparse_optical_flow_example")

* *Gunnar Farneback's algorithm* is an often used dense optical flow

![dense_optical_flow_example](imgs/dense_optical_flow_example.png "dense_optical_flow_example")


## Lucas-Kanade Optical Flow

Define images at consecutive timestamps as $\bold{I}(t)$. For a pixel at $(x,y)$, its gray-scale representation is $\bold{I}(x,y,t)$.

![optical_flow](imgs/optical_flow.png "optical_flow")

* Constant gray scale assumption: 
The pixel’s gray scale is constant in each image.
$$
\bold{I}(x+ dx, y+ dy, t+ dt)
=
\bold{I}(x,y,t)
$$

In a real world scenario, this constant gray scale assumption is false since ambient light has significant impacts on the brightness of an object, even casting a shadow over the object resulting in sharp reduction of pixel value. 

However, in most cases, ambient light environment changes slowly, and approximation can be made by first-order Taylor expansion, such as
$$
\bold{I}(x+ dx, y+ dy, t+ dt)
\approx
\bold{I}(x,y,t)
+\frac{\partial \bold{I}}{\partial x} dx
+\frac{\partial \bold{I}}{\partial y} dy
+\frac{\partial \bold{I}}{\partial t} dt
$$

Consider the constant gray scale condition, here derive
$$
\begin{align*}
&
\frac{\partial \bold{I}}{\partial x} dx
+\frac{\partial \bold{I}}{\partial y} dy
+\frac{\partial \bold{I}}{\partial t} dt
=0
\\ \text{divide by } dt
\Rightarrow \quad &
\frac{\partial \bold{I}}{\partial x} 
\frac{dx}{dt}
+
\frac{\partial \bold{I}}{\partial y} 
\frac{dy}{dt}
=
-\frac{\partial \bold{I}}{\partial t}
\end{align*}
$$
where $\frac{dx}{dt}, \frac{dy}{dt}$ (denoted as $(u,v)$) refer to the speed of the pixel $(x,y)$ motion, 
and $\frac{\partial \bold{I}}{\partial x} , \frac{\partial \bold{I}}{\partial y}$ refer to the gradient of the image with respect to the $x$- and $y$- axis, denoted as $\bold{I}_x, \bold{I}_y$, respectively.
Denote the change rate of image brightness as $\frac{\partial \bold{I}}{\partial t}=\bold{I}_t$.

By the above denotations, the equation can be written in the matrix form
$$
\begin{bmatrix}
    \bold{I}_x &  \bold{I}_y
\end{bmatrix}
\begin{bmatrix}
    u \\
    v
\end{bmatrix}
=
-\bold{I}_t
$$

### Optimal $(u,v)$ Derivation via Least Square Optimization

Consider a window of size $w \times w$ that totally covers $w^2$ pixels. $(u,v)^*$ can be computed via least square optimization over a large window of pixels.

For $k=1,2,...,w^2$, stack each pixel's optical flow, there is
$$
\bold{A} = 
\begin{bmatrix}
    \begin{bmatrix}
        \bold{I}_x &  \bold{I}_y
    \end{bmatrix}_1 \\
    \begin{bmatrix}
        \bold{I}_x &  \bold{I}_y
    \end{bmatrix}_2 \\
    \vdots \\
    \begin{bmatrix}
        \bold{I}_x &  \bold{I}_y
    \end{bmatrix}_{w^2}
\end{bmatrix}
, \quad
\bold{b} = 
\begin{bmatrix}
    \bold{I}_{t_1} \\
    \bold{I}_{t_2} \\
    \vdots \\
    \bold{I}_{t_{w^2}} 
\end{bmatrix}
$$

The whole equation is
$$
\bold{A} \begin{bmatrix}
    u \\
    v
\end{bmatrix}
=
\bold{b}
$$
whose result is
$$
\begin{bmatrix}
    u \\
    v
\end{bmatrix}^*
=
-(\bold{A}^\text{T}\bold{A})^{-1} \bold{A}^\text{T} \bold{b}
$$

The result $(u,v)^*$ illustrates the speed at which the image frame moves relative to the environment.

## Optical Flow for Object Tracking

Given the optimal $(u,v)^*$ representing image frame changing speed, next step is to compute the $(\Delta x, \Delta y)^*$ that best describes the difference in pixel motion between two images.

$$
arg \space \underset{\Delta x, \Delta y}{min} \space
\big|\big|
    \bold{I}_k(x,y) - \bold{I}_{k+1} (x+\Delta x, y+\Delta y)
\big|\big|^2
$$

### Multi-Layer Optical Flow

If a camera moves faster and the difference between the two images is obvious, the single-layer image optical flow method can be easily stuck at a local minimum.