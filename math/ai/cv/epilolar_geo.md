# Epipolar geometry

Epipolar geometry is the geometry of stereo vision. When two cameras view a 3D scene from two distinct positions, there are a number of geometric relations between the 3D points and their projections onto the 2D images that lead to constraints between the image points.

## Image Formulation

![epipolar_geo](imgs/epipolar_geo.png "epipolar_geo")


### Epipolar Geo

* Epipole/epipolar point

$e_L$ and $e_R$ are called epipoles, defined as the two points on a *base line* $O_b = O_L - O_R$ penetrating through left and right camera views. $O_L$ and $O_R$ are called *optical centers*.

* Epipolar line

The line $O_L - X$ is seen by the left camera view as a single point $X_L$ projected on the left view plane. Other points $[x_1, x_2, x_3, ...]$ on this line $O_L - X$ are seen by the right camera view projected on the line $e_R - x_R$. The red line $e_R - x_R$ is called epilolar line.

* Epipolar plane

The triangular plane $x - O_L - O_R$ is called epipolar plane.

* Epipolar constraints

The projection of a real world point $x$ on the right camera plane as $x_R$ must be contained in the $e_R - x_R$ epipolar line. 

## Essential Matrix

Essential matrix $E$ describes a camera motion (extrinsics $M_{ex}$: rotation and translation) from $O_L$ to $O_R$.

If the right camera's intrinsics $K$ is same as the left's, the right camera can be said a rotation and translation result of the left camera. 

### Extrinsic calibration
Extrinsic calibration provides a 3d rigid coordinate transformation $x_{\tiny{C,L}}=M_{ex}[x^\text{T}_W, 1]^\text{T}$ from real world coordinates to left camera's coordinates, in which 
$$
M_{ex}=[R_L \space -R_L \overrightarrow{O}_{\tiny{W,L}}]_{3 \times 4}
$$
where $\overrightarrow{O}_{\tiny{W,L}}$ denotes left camera optical center in the real world coordinates; $R_L$ is a $3 \times 3$ rotation matrix. Intuitively, $M_{ex}$ can be interpreted as transformation by rotation $R_L$ then translation by $R_L\overrightarrow{O}_{\tiny{W,L}}$.

### Epipolar constraint 

Let $\times$ represent a cross product, hence $(O_{\tiny{W,L}}-O_{\tiny{W,R}})\times(x_{\tiny{W,R}}-O_{\tiny{W,R}})$ represents the normal to epilolar plane. Since $x_{\tiny{W,L}}, O_{\tiny{W,L}}, x_{\tiny{W,R}}, O_{\tiny{W,R}}$ are coplanar, the scalar product by $(x_{\tiny{W,L}}-O_{\tiny{W,L}})^\text{T}$ and the epipolar plane normal should be zero.
$$
(x_{\tiny{W,L}}-O_{\tiny{W,L}})^\text{T}[(O_{\tiny{W,L}}-O_{\tiny{W,R}})
\times
(x_{\tiny{W,R}}-O_{\tiny{W,R}})]=0
$$

### Essential matrix

Since $(O_{\tiny{W,L}}-O_{\tiny{W,R}})\times$ is a vector cross product operation, it can be transformed into an anti-symmetric/skew matrix operation:
$$
(O_{\tiny{W,L}}-O_{\tiny{W,R}})\times=
[O_{b}]_{\times}=
\begin{bmatrix}
    0 & -o_3 & o_2 \\
    o_3 & 0 & -o_1 \\
    -o_2 & o_1 & 0
\end{bmatrix}
$$
This transforms a vector multiplication into a matrix multiplication. Some articles denote this vector to matrix transformation as $O_{b}^{\vee}=[O_{b}]_{\times}$. 

The vector representation of a real world point in the left camera frame $x_{\tiny{W,L}}-O_{\tiny{W,L}}$ can be computed with its pixel vector $x_{\tiny{C,L}}$ multiplied by the orientation $R^\text{T}_{L}$ of the left camera
$$
x_{\tiny{W,L}}-O_{\tiny{W,L}} = R^\text{T}_{L}x_{\tiny{C,L}}
$$
Similarly for the right camera view:
$$
x_{\tiny{W,R}}-O_{\tiny{W,R}} = R^\text{T}_{R}x_{\tiny{C,R}}
$$

So that the epilolar constraint can be written as
$$
\begin{align*}
& \space \space \space \space \space 
R^\text{T}_{L}x_{\tiny{C,L}}
\big((O_{\tiny{W,L}}-O_{\tiny{W,R}})
\times R^\text{T}_{R}x_{\tiny{C,R}}\big)
\\ &=
x_{\tiny{C,L}}^\text{T}R_{L}
\big((O_{\tiny{W,L}}-O_{\tiny{W,R}})
\times R^\text{T}_{R}x_{\tiny{C,R}}\big)
\\ &=
x_{\tiny{C,L}}^\text{T}R_{L}
[O_{b}]_{\times}
R^\text{T}_{R}x_{\tiny{C,R}}
\\ &=
x_{\tiny{C,L}}^\text{T}
E
x_{\tiny{C,R}}
\\ &= 
0
\end{align*}
$$
where $E=R_{L}\big((O_{\tiny{W,L}}-O_{\tiny{W,R}})\times R^\text{T}_{R}\big)$ is the essential matrix.

### Essential Matrix Degree of Freedom (DoF)

$E$ has $rank(E)=2$ for its homogeneous coordinates' representation scaling by $f$.

Since translation and rotation each have 3 degrees of freedom, $E$ has $6$ degrees of freedom. 
But due to the equivalence of scales $f$, $E$ actually has 5 degrees of freedom.

### Essential Matrix Computation

For $x_{\tiny{L}}$ and $x_{\tiny{R}}$ such as
$$
x_{\tiny{L}}
=
\begin{bmatrix}
    u \\
    v \\
    1
\end{bmatrix}
\text{, }
x_{\tiny{R}}
=
\begin{bmatrix}
    u' \\
    v' \\
    1
\end{bmatrix}
$$

given $x_{\tiny{C,L}}^\text{T} E x_{\tiny{C,R}} = 0$, that gives
$$
[u, v, 1]
\begin{bmatrix}
    e_1 & e_2 & e_3 \\
    e_4 & e_5 & e_6 \\
    e_7 & e_8 & e_9 \\
\end{bmatrix}
\begin{bmatrix}
    u' \\
    v' \\
    1
\end{bmatrix}
=
0
$$

*Eight-Point Algorithm* requires $8$ point pairs $x_{\tiny{L}}$ and $x_{\tiny{R}}$ (the scaling factor $f$ is considered equivalent to setting $f=1$), rearrange the equation above, there is
$$
\begin{bmatrix}
    u_1'u_1 & u_1'v_1 & u_1' & v_1'v_1 & v_1'u_1 & v_1' & u_1 & v_1 & 1 \\
    u_2'u_2 & u_2'v_2 & u_2' & v_2'v_2 & v_2'u_2 & v_2' & u_2 & v_2 & 1 \\
    \vdots & \vdots &\vdots &\vdots &\vdots &\vdots &\vdots &\vdots &\vdots \\
    u_8'u_8 & u_8'v_8 & u_8' & v_8'v_8 & v_8'u_8 & v_8' & u_8 & v_8 & 1 \\
\end{bmatrix}
\begin{bmatrix}
    e_1 \\
    e_2 \\
    e_3 \\
    e_4 \\
    e_5 \\
    e_6 \\
    e_7 \\
    e_8 \\
    e_9 \\
\end{bmatrix}
=0
$$

### Essential Matrix to Camera Motion Recovery

Since the essential matrix $E$ describes a camera motion (extrinsics $M_{ex}$: rotation and translation) from $O_L$ to $O_R$, 
there should be a mapping decomposing $E$ into rotation and translation (here defines $M_{ex}$ composed of rotation $R$ and translation $T$.) such that $E \rightarrow M_{ex}$. SVD can help in this mapping.
$$
E = U \Sigma V^\text{T}
$$




## Fundamental matrix

An instrinsic camera calibration matrix $M_{in}$ defines the transformation from a camera coordinate point $x_{\tiny{W,C}}$ to its homogeneous coorsdinate point $x_{\tiny{W,h}}$.
$$
x_{\tiny{W, h}} = M_{in} x_{\tiny{W,C}}
$$

For a camera with rectangular pixels of size $1/s_x \times 1/s_y$, optical center $(o_x, o_y)$ and focus length $f$, there is
$$
M_{in}
=
\begin{bmatrix}
    s_x & 0 & o_x/f \\
    0 & s_y & o_y/f \\
    0 & 0 & 1/f \\
\end{bmatrix}
$$

Given an essential matrix $E$, the left and right camera instrinsics are $M_{L,in}, M_{R,in}$, here defines a fundamental matrix $F$
$$
F=
(M_{L,in}^{-1})^\text{T} E M_{R,in}^{-1}
$$

The epipolar constraint can be rewritten in homogeneous coordinates.
$$
x_{\tiny{L, h}}^\text{T}
F
x_{\tiny{R, h}}
=
0
$$

### Fundamental matrix use case

$F x_{\tiny{R, h}}$ is the epipolar line $x_{\tiny{L, h}}-e_{\tiny{L}}$ associated with $x_{\tiny{R, h}}$.

$F^\text{T} x_{\tiny{L, h}}$ is the epipolar line $x_{\tiny{R, h}}-e_{\tiny{R}}$ associated with $x_{\tiny{L, h}}$ 

For epipoles, there are $F e_{\tiny{L}} = 0$ and $F^\text{T} e_{\tiny{R}} = 0$

In a common scenario, camera views start as the grey image planes, we can use two homographies to transform them into parallel camera views such as yellow image planes.

![stereo_img_rectification](imgs/stereo_img_rectification.png "stereo_img_rectification")

The below figure shows an example of such a rectification result.

![homography_rectification](imgs/homography_rectification.png "homography_rectification")

A pair of parallel camera views give simple essential matrix performing one-dimension translation such that
$$
x_{\tiny{R}}^\text{T} E x_{\tiny{L}} = 0
$$
where
$$
E=
T \times R
=
\begin{bmatrix}
    0 & 0 & 0 \\
    0 & 0 & -t \\
    0 & t & 0
\end{bmatrix}
$$

Define $x_{\tiny{R}}$ and $x_{\tiny{L}}$ as unit homogeneous vectors
$$
x_{\tiny{L}}
=
\begin{bmatrix}
    u \\
    v \\
    1
\end{bmatrix}
\text{, }
x_{\tiny{R}}
=
\begin{bmatrix}
    u' \\
    v' \\
    1
\end{bmatrix}
$$

Hence, we can prove that $x_{\tiny{R}}$ and $x_{\tiny{L}}$ are on the same epipolar line that at the height of $v=v'$.
$$
\begin{align*}
\begin{bmatrix}
    u & v & 1
\end{bmatrix}
\begin{bmatrix}
    0 & 0 & 0 \\
    0 & 0 & -t \\
    0 & t & 0
\end{bmatrix}
\begin{bmatrix}
    u' \\
    v' \\
    1
\end{bmatrix}
&= 0
\\
\begin{bmatrix}
    u & v & 1
\end{bmatrix}
\begin{bmatrix}
    0 \\
    -t \\
    tv'
\end{bmatrix}
&= 0
\\
tv &= tv'
\end{align*}
$$

![parallel_img_epi](imgs/parallel_img_epi.png "parallel_img_epi")

## Correspondence search

By Fundamental matrix, a point $x_{\tiny{L}}$ on the left camera view should exist on its corresponding epipolar line $x_{\tiny{R}}-e_{\tiny{R}}$. Since having implemented homographical transformation that two camera views are now in parallel, the point $x_{\tiny{L}}$'s correspondant point $x_{\tiny{R}}$ must be on this horizontal scanline.

Consider a shifting window $\bold{W}$ of a size of $m \times n$, window moving step of $(u,v)$ on an image $I$, and define an error *sum of squared differences* (SSD) which is the squared differences of all pixels in a window before and after window's shifting.
$$
E_{ssd}(u,v)=\sum_{(x,y)\in\bold{W}_{m \times n}} 
\big[
    I(x+u, y+v)-I(x,y)    
\big]^2
$$

![scanline_match_epi](imgs/scanline_match_epi.png "scanline_match_epi")

### Correspondance priors and constaints

There are priors that can help windows fast locate feature points.

* Uniqueness

There should be only one $x_{\tiny{R}}$ on $x_{\tiny{L}}$'s epipolar line.

* Ordering

If there are more than one feature points on the same correspondant epipolar lines $x_{\tiny{R}} - e_{\tiny{R}}$ and $x_{\tiny{L}} - e_{\tiny{L}}$, their corresponding points should have the same order on epipolar lines.

* Smoothness

If features changes slowly, the drastic changing feature points are discarded.