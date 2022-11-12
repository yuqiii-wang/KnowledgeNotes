# Bundle Adjustment

The so-called *Bundle Adjustment* refers to optimizing both camera parameters (intrinsic and extrinsic) and 3D landmarks with images.

Consider the bundles of light rays emitted from 3D points (shown in the example figure below as a table in the image center). 

The purpose of optimization can be explained as to adjust the camera poses and the 3D points (visual feature points/landmarks), to ensure the projected 2D features (bundles) match the detected results.

![multi_camera_reprojection_illustration](imgs/multi_camera_reprojection_illustration.png "multi_camera_reprojection_illustration")

Bundle Adjustment takes the following steps

1. Transform a point $\bold{p}$ from the world coordinates into the camera frame using extrinsics

$$
\begin{align*}
\bold{P}'
&=
[\bold{R}|\bold{t}] \bold{p}
\\ &=
\bold{R} \bold{p} + \bold{t}
\\ &=
[X',Y',Z']^\text{T}
\end{align*}
$$

2. Project $\bold{P}'$ into the normalized plane and get the normalized coordinates $(u,v)$ 
$$
\begin{align*}
\bold{P}_c &= [u_c, v_c, 1]^\text{T}
\\ &=
[\frac{X'}{Z'}, \frac{Y'}{Z'}, 1]^\text{T}
\end{align*}
$$

3. Apply the distortion model.

$$
\begin{align*}
    u_c' &= u_c (1+k_1 r_c^2 +k_2 r_c^4)
    \\
    v_c' &= v_c (1+k_1 r_c^2 +k_2 r_c^4)
\end{align*}
$$

4. Compute the pixel by intrinsic

$$
\begin{align*}
    u_s &= f_x u_c' + c_x
    \\
    v_s &= f_y v_c' + c_y
\end{align*}
$$

5. Denote $\bold{x}=[\bold{x}_1, \bold{x}_2, ..., \bold{x}_k], \bold{y}=[\bold{y}_1, \bold{y}_2, ..., \bold{y}_m]$ as the camera states (usually $6$-d vector, $3$-d position and $3$-d orientation) and landmark states (usually $3$-d position), respectively. Both $\bold{x}$ and $\bold{y}$ are unknown awaiting optimization.

Observation $\bold{z}$ takes $\bold{x}$ and $\bold{y}$ as input arguments
$$
\bold{z} = 
h(\bold{x}, \bold{y})
$$

$\bold{z}\overset{\Delta}{=}(u_s, v_s)$ describes the pixel coordinate. 

6. Optimization

Take $(0,0,0)$ as the origin ($\bold{x}_0=(0,0,0)$), to find $\bold{x}_k$ for the $k$-th camera is same as finding a transformation such that $\bold{x}_k=[\bold{R}|\bold{t}]_k$. In addition, denote $\bold{p}_k=[\bold{p}_{k1},\bold{p}_{k2},...,\bold{p}_{km}]$ as the observed landmarks $\bold{y}$ by the pose $\bold{x}_k=[\bold{R}|\bold{t}]_k$ for the camera $k$. $m$ is the total number of landmarks.

Rewrite indices that $i$ represents the $i$-th camera and $j$ represents the $j$-th landmark. Error $\bold{e}$ can be defined as discrepancy between the computed $h([\bold{R}|\bold{t}]_i, \bold{p}_{j})$ and $\bold{z}_{ij}$. 

Here, landmark $\bold{p}_j$ does not discriminate between landmark estimations at different camera $i$. In other words, $\bold{p} \in \mathbb{R}^{m \times 3}$.
$$
\frac{1}{2} \sum^m_{i=1} \sum^n_{j=1} 
\big|\big| \bold{e}_{ij} \big|\big|^2
=
\frac{1}{2} \sum^m_{i=1} \sum^n_{j=1} 
\big|\big| 
    \bold{z}_{ij} -
    h([\bold{R}|\bold{t}]_i, \bold{p}_{j})
\big|\big|^2
$$

Solving this least-squares is equivalent to adjusting the pose and road signs/landmarks at the same time, which is the so-called Bundle Adjustment.

## Solving Bundle Adjustment

Let $\bold{x}$ represent the whole optimization set such as 
$$
\bold{x}=\big[
    [\bold{R}|\bold{t}]_1, [\bold{R}|\bold{t}]_2, ..., [\bold{R}|\bold{t}]_n, 
    \bold{p}_{1}, \bold{p}_{2}, ..., \bold{p}_{m}
\big]
$$

The error $\bold{e}$ to be minimized can be approximated by first-order derivative.
$$
\frac{1}{2} \big|\big|
    \bold{e}(\bold{x}+\Delta\bold{x})
\big|\big|^2
\approx
\frac{1}{2} \sum^n_{i=1} \sum^m_{j=1} 
\big|\big|
    \bold{e}_{ij} + \bold{F}_{ij}\Delta\bold{\xi}_i + \bold{E}_{ij} \Delta \bold{p}_j
\big|\big|
$$
where $\Delta\bold{x}$ is the correction increment that iteratively sets $\bold{e}$ to minimum. 
$\bold{F}_{ij}$ is the partial derivative of the entire cost function to the $i$-th pose, and $\bold{E}_{ij}$ is the partial derivative of the function to the $j$-th landmark.
$\bold{\xi}_i$ denotes $[\bold{R}|\bold{t}]_i$.

Collectively, represent poses and landmarks as $\bold{x}_\bold{\xi}$ and $\bold{x}_\bold{p}$
$$
\begin{align*}
    \bold{x}_\bold{\xi} &= [
        \bold{\xi}_1, \bold{\xi}_2, ..., \bold{\xi}_n
    ]^\text{T}
    \\
    \bold{x}_\bold{p} &= [
        \bold{p}_1, \bold{p}_2, ..., \bold{p}_m
    ]^\text{T}
\end{align*}
$$

Take away the sum operations, the error approximation can be rewritten as
$$
\frac{1}{2} \big|\big|
    \bold{e}(\bold{x}+\Delta\bold{x})
\big|\big|^2
\approx
\frac{1}{2} \big|\big|
    \bold{e} + \bold{F}\Delta\bold{x}_{\bold{\xi}} + \bold{E}\Delta\bold{x}_{\bold{p}}
\big|\big|^2
$$
where $\bold{F} \in \mathbb{R}^{2 \times 6 \times (n \times m) \times n}$ and $\bold{E} \in \mathbb{R}^{2 \times 3 \times (n \times m) \times m}$ are sparse Jacobian matrices with many non-zero elements crowded along the diagonal line. $2$ represents derivatives with respect to $x$- and $y$- axis.

To employ Gauss-Newton method, this term $(\bold{J}^\text{T} \bold{J})^{-1} \bold{J}^\text{T}$ needs to be computed. here define 
$$
\begin{align*}
 \bold{J} &=
\begin{bmatrix}
    \bold{F} & \bold{E}
\end{bmatrix}
\\ &=
\left[
\begin{array}{cccccccccc}
    \frac{\partial \bold{e}_{11}}{\partial [\bold{R}|\bold{t}]_1} &
    \bold{0}_{2 \times 6} &
    \bold{0}_{2 \times 6} &
    & \bold{0}_{2 \times 6} & 
    \frac{\partial \bold{e}_{11}}{\partial \bold{p}_1} &
    \bold{0}_{2 \times 3} &
    \bold{0}_{2 \times 3} &
    & \bold{0}_{2 \times 3} &
    \\
    \bold{0}_{2 \times 6} &
    \frac{\partial \bold{e}_{21}}{\partial [\bold{R}|\bold{t}]_2} &
    \bold{0}_{2 \times 6} &
    & \bold{0}_{2 \times 6} &
     \frac{\partial \bold{e}_{21}}{\partial \bold{p}_1} &
    \bold{0}_{2 \times 3} &
    \bold{0}_{2 \times 3} &
    & \bold{0}_{2 \times 3} &
    \\
    \bold{0}_{2 \times 6} &
    \bold{0}_{2 \times 6} &
    \frac{\partial \bold{e}_{31}}{\partial [\bold{R}|\bold{t}]_3} &
    & \bold{0}_{2 \times 6} &
    \frac{\partial \bold{e}_{31}}{\partial \bold{p}_1} &
    \bold{0}_{2 \times 3} &
    \bold{0}_{2 \times 3} &
    & \bold{0}_{2 \times 3} &
    \\
    & & & \ddots & &
    & & & \ddots & &
    \\
    \bold{0}_{2 \times 6} & 
    \bold{0}_{2 \times 6} & 
    \bold{0}_{2 \times 6} & &
    \frac{\partial \bold{e}_{n1}}{\partial [\bold{R}|\bold{t}]_n} &
    \frac{\partial \bold{e}_{n1}}{\partial \bold{p}_1} &
    \bold{0}_{2 \times 3} & 
    \bold{0}_{2 \times 3} & 
    & \bold{0}_{2 \times 3} & 

    \\
    \frac{\partial \bold{e}_{12}}{\partial [\bold{R}|\bold{t}]_1} &
    \bold{0}_{2 \times 6} &
    \bold{0}_{2 \times 6} &
    & \bold{0}_{2 \times 6} & 
    \bold{0}_{2 \times 3} &
    \frac{\partial \bold{e}_{12}}{\partial \bold{p}_2} &
    \bold{0}_{2 \times 3} &
    & \bold{0}_{2 \times 3} &
    \\
    \bold{0}_{2 \times 6} &
    \frac{\partial \bold{e}_{22}}{\partial [\bold{R}|\bold{t}]_2} &
    \bold{0}_{2 \times 6} &
    & \bold{0}_{2 \times 6} &
    \bold{0}_{2 \times 3} &
    \frac{\partial \bold{e}_{22}}{\partial \bold{p}_2} &
    \bold{0}_{2 \times 3} &
    & \bold{0}_{2 \times 3} &
    \\
    \bold{0}_{2 \times 6} &
    \bold{0}_{2 \times 6} &
    \frac{\partial \bold{e}_{32}}{\partial [\bold{R}|\bold{t}]_3} &
    & \bold{0}_{2 \times 6} &
    \bold{0}_{2 \times 3} &
    \frac{\partial \bold{e}_{32}}{\partial \bold{p}_2} &
    \bold{0}_{2 \times 3} &
    & \bold{0}_{2 \times 3} &
    \\
    & & & \ddots & &
    & & & \ddots & &
    \\
    \bold{0}_{2 \times 6} & 
    \bold{0}_{2 \times 6} & 
    \bold{0}_{2 \times 6} & &
    \frac{\partial \bold{e}_{n2}}{\partial [\bold{R}|\bold{t}]_n} &
    \bold{0}_{2 \times 3} & 
    \frac{\partial \bold{e}_{n2}}{\partial \bold{p}_2} &
    \bold{0}_{2 \times 3} & 
    & \bold{0}_{2 \times 3} & 

    \\
    \frac{\partial \bold{e}_{13}}{\partial [\bold{R}|\bold{t}]_1} &
    \bold{0}_{2 \times 6} &
    \bold{0}_{2 \times 6} &
    & \bold{0}_{2 \times 6} & 
    \bold{0}_{2 \times 3} &
    \bold{0}_{2 \times 3} &
    \frac{\partial \bold{e}_{13}}{\partial \bold{p}_3} &
    & \bold{0}_{2 \times 3} &
    \\
    \bold{0}_{2 \times 6} &
    \frac{\partial \bold{e}_{23}}{\partial [\bold{R}|\bold{t}]_2} &
    \bold{0}_{2 \times 6} &
    & \bold{0}_{2 \times 6} &
    \bold{0}_{2 \times 3} &
    \bold{0}_{2 \times 3} &
    \frac{\partial \bold{e}_{23}}{\partial \bold{p}_3} &
    & \bold{0}_{2 \times 3} &
    \\
    \bold{0}_{2 \times 6} &
    \bold{0}_{2 \times 6} &
    \frac{\partial \bold{e}_{33}}{\partial [\bold{R}|\bold{t}]_3} &
    & \bold{0}_{2 \times 6} &
    \bold{0}_{2 \times 3} &
    \bold{0}_{2 \times 3} &
    \frac{\partial \bold{e}_{33}}{\partial \bold{p}_3} &
    & \bold{0}_{2 \times 3} &
    \\
    & & & \ddots & &
    & & & \ddots & &
    \\
    \bold{0}_{2 \times 6} & 
    \bold{0}_{2 \times 6} & 
    \bold{0}_{2 \times 6} & &
    \frac{\partial \bold{e}_{n3}}{\partial [\bold{R}|\bold{t}]_n} &
    \bold{0}_{2 \times 3} & 
    \bold{0}_{2 \times 3} & 
    \frac{\partial \bold{e}_{n3}}{\partial \bold{p}_3} &
    & \bold{0}_{2 \times 3} & 

    \\
    & & \vdots & &
    & & \vdots & &

    \\
    \frac{\partial \bold{e}_{1m}}{\partial [\bold{R}|\bold{t}]_1} &
    \bold{0}_{2 \times 6} &
    \bold{0}_{2 \times 6} &
    & \bold{0}_{2 \times 6} & 
    \bold{0}_{2 \times 3} &
    \bold{0}_{2 \times 3} &
    \bold{0}_{2 \times 3} &
    & \frac{\partial \bold{e}_{1m}}{\partial \bold{p}_m} &
    \\
    \bold{0}_{2 \times 6} &
    \frac{\partial \bold{e}_{2m}}{\partial [\bold{R}|\bold{t}]_2} &
    \bold{0}_{2 \times 6} &
    & \bold{0}_{2 \times 6} &
    \bold{0}_{2 \times 3} &
    \bold{0}_{2 \times 3} &
    \bold{0}_{2 \times 3} &
    & \frac{\partial \bold{e}_{2m}}{\partial \bold{p}_m} &
    \\
    \bold{0}_{2 \times 6} &
    \bold{0}_{2 \times 6} &
    \frac{\partial \bold{e}_{3m}}{\partial [\bold{R}|\bold{t}]_3} &
    & \bold{0}_{2 \times 6} &
    \bold{0}_{2 \times 3} &
    \bold{0}_{2 \times 3} &
    \bold{0}_{2 \times 3} &
    & \frac{\partial \bold{e}_{3m}}{\partial \bold{p}_m} &
    \\
    & & & \ddots & &
    & & & \ddots & &
    \\
    \bold{0}_{2 \times 6} & 
    \bold{0}_{2 \times 6} & 
    \bold{0}_{2 \times 6} & &
    \frac{\partial \bold{e}_{nm}}{\partial [\bold{R}|\bold{t}]_n} &
    \bold{0}_{2 \times 3} & 
    \bold{0}_{2 \times 3} & 
    \bold{0}_{2 \times 3} & 
    & \frac{\partial \bold{e}_{nm}}{\partial \bold{p}_m} &
\end{array}
\right]
\end{align*}
$$
where each row $\bold{J}_{ij}$ can be expressed as below. Inside, $\frac{\partial \bold{e}_{ij}}{\partial [\bold{R}|\bold{t}]_i}$ is of a size $2 \times 6$ and $\frac{\partial \bold{e}_{ij}}{\partial \bold{p}_j}$ is of a size $2 \times 3$.

$$
\bold{J}_{ij} =
\begin{bmatrix}
    \bold{0}_{2 \times 6} &
    \bold{0}_{2 \times 6} &
    \frac{\partial \bold{e}_{ij}}{\partial [\bold{R}|\bold{t}]_i}
    & \bold{0}_{2 \times 6} &
    ... &
    \bold{0}_{2 \times 3} &
    \bold{0}_{2 \times 3} &
    \frac{\partial \bold{e}_{ij}}{\partial \bold{p}_j}
    & \bold{0}_{2 \times 3} &
    ... &
\end{bmatrix}
$$

However, computation of $(\bold{J}^\text{T} \bold{J})^{-1}$ remains impractical since it is of $O(n^3)$ computation complexity. The rescue is to take advantage of matrix $\bold{J}$'s sparsity.

### Sparsity Exploitation

The quadratic form for $\bold{J}$ can be computed by
$$
\begin{align*}
\bold{J}^\text{T} \bold{J} &= 
\begin{bmatrix}
    \bold{F}^\text{T} \bold{F} & \bold{F}^\text{T} \bold{E} \\
    \bold{E}^\text{T} \bold{F} & \bold{E}^\text{T} \bold{E} \\
\end{bmatrix}
\\ &=
\sum^n_{i=1} \sum^m_{j=1} 
\bold{J}^\text{T}_{ij} \bold{J}_{ij}
\end{align*}
$$

$\bold{F}^\text{T} \bold{F}$ and $\bold{E}^\text{T} \bold{E}$ only relate to the derivatives with respect to camera poses and landmarks, respectively. They are block-diagonal matrices.

Depending on the specific observation data, $\bold{F}^\text{T} \bold{E}$ and $\bold{E}^\text{T} \bold{F}$ might be dense or sparse.

Rewrite the denotations of four sub matrices of $\bold{J}^\text{T} \bold{J}$, such as
$$
\begin{align*}
\bold{J}^\text{T} \bold{J}
&=
\begin{bmatrix}
    \bold{F}^\text{T} \bold{F} & \bold{F}^\text{T} \bold{E} \\
    \bold{E}^\text{T} \bold{F} & \bold{E}^\text{T} \bold{E} \\
\end{bmatrix}
\\ &=
\begin{bmatrix}
    \bold{B} & \bold{E} \\
    \bold{E}^\text{T} & \bold{C}
\end{bmatrix}
\end{align*}
$$
where, as clearly observed in the figure below, $\bold{B}$ and $\bold{C}$ exhibit obvious diagonal formats. $\bold{E}$ and $\bold{E}^\text{T}$ are rectangular. 

In real world scenarios, for example, an autonomous vehicle carrying $6$ cameras touring a street, should see thousands of visual features/landmarks just in one camera shot. Therefore, there should be $n \ll m$ that renders the shape of $\bold{E}$ being a long rectangle, and $\bold{C}$'s size being much larger than $\bold{B}$'s.

![JTJ_mat](imgs/JTJ_mat.png "JTJ_mat")


### Schur Trick

Take into consideration the Gaussian noises $\bold{g}=[\bold{v} \quad \bold{w}]^\text{T}$ for the $x$-th and $y$-th dimensions.

Perform Gaussian elimination to triangulate the matrix for solution.

$$
\begin{align*}
   \bold{J}^\text{T} \bold{J} \bold{x} &= \bold{g}
   \\
    \begin{bmatrix}
        \bold{B} & \bold{E} \\
        \bold{E}^\text{T} & \bold{C}
    \end{bmatrix}
    \begin{bmatrix}
        \Delta \bold{x}_{\bold{\xi}} \\
        \Delta \bold{x}_{\bold{p}}
    \end{bmatrix}
    &=
    \begin{bmatrix}
        \bold{v} \\
        \bold{w}
    \end{bmatrix}
    \\
    \begin{bmatrix}
        \bold{I} & -\bold{E}\bold{C}^{-1} \\
        \bold{0} & \bold{I}
    \end{bmatrix}
    \begin{bmatrix}
        \bold{B} & \bold{E} \\
        \bold{E}^\text{T} & \bold{C}
    \end{bmatrix}
    \begin{bmatrix}
        \Delta \bold{x}_{\bold{\xi}} \\
        \Delta \bold{x}_{\bold{p}}
    \end{bmatrix}
    &=
    \begin{bmatrix}
        \bold{I} & -\bold{E}\bold{C}^{-1} \\
        \bold{0} & \bold{I}
    \end{bmatrix}
    \begin{bmatrix}
        \bold{v} \\
        \bold{w}
    \end{bmatrix}
    \\
    \begin{bmatrix}
        \bold{B}-\bold{E}\bold{C}^{-1}\bold{E}^\text{T} & \bold{0}
        \\
        \bold{E}^\text{T} & \bold{C}
    \end{bmatrix}
    \begin{bmatrix}
        \Delta \bold{x}_{\bold{\xi}} \\
        \Delta \bold{x}_{\bold{p}}
    \end{bmatrix}
    &=
    \begin{bmatrix}
        \bold{v} - \bold{E}\bold{C}^{-1} \bold{w} \\
        \bold{w}
    \end{bmatrix}
\end{align*}
$$

*Schur elimination* (Schur trick) is a particular method dedicated to solve the above equation.

First, it looks at this equation. 

$$
(\bold{B}-\bold{E}\bold{C}^{-1}\bold{E}^\text{T})
\Delta \bold{x}_{\bold{\xi}}
=
\bold{v} - \bold{E}\bold{C}^{-1} \bold{w}
$$

* The linear equation is of the size of $\bold{B}$
* $\bold{C}$ is a diagonal matrix, hence $\bold{C}^{-1}$ is easy to compute

Second, with the derived $\Delta\bold{x}_{\bold{\xi}}$,
compute $\Delta \bold{x}_{\bold{p}}=\bold{C}^{-1}(\bold{w}-\bold{E}^\text{T}\Delta\bold{x}_{\bold{\xi}})$. This should be easy since $\bold{C}^{-1}$ and $\bold{E}^\text{T}\Delta\bold{x}_{\bold{\xi}}$ are known.

### Robust Kernels

The assumed total $\mathcal{L}_2$ norm error $||\bold{e}||^2$ in the above equations can grow fast if any particular error term $||\bold{e}_{ij}||^2 = \big|\big|    \bold{z}_{ij} - h([\bold{R}|\bold{t}]_i, \bold{p}_{j})\big|\big|^2$ is absurdly wrong. 
This is attributed to optimization attempting to reduce overall $||\bold{e}||^2$ and the large error term $||\bold{e}_{ij}||^2$ has a significant weight that causes optimization focusing too much on this particular error, rather than taking care of all error terms $||\bold{e}_{ij}||^2, \forall i \in [1,n], \forall j \in [1,m]$.

Solution to address this issue is by employing a robust kernel error, such as Huber loss, that constraints $\mathcal{L}_2$-norm error when error is small $|e|\le \delta$, and linear otherwise.
$$
L_{\delta}(e)=
\left\{
    \begin{array}{c}
        \frac{1}{2}e^2 &\quad \text{for} |e|\le \delta
        \\
        \delta \cdot (|e|-\frac{1}{2}\delta) &\quad \text{otherwise}
    \end{array}
\right.
$$