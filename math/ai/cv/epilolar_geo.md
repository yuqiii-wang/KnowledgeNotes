# Epipolar geometry

Epipolar geometry is the geometry of stereo vision. When two cameras view a 3D scene from two distinct positions, there are a number of geometric relations between the 3D points and their projections onto the 2D images that lead to constraints between the image points.

## Image formulation

![epipolar_geo](imgs/epipolar_geo.png "epipolar_geo")

Every camera image point $x_c$ follows homogeneous coordinate's representation that
$$
x_c=
f \begin{bmatrix}
    x_{x, h} \\
    x_{y,h} \\
    1 \\
\end{bmatrix}
=
\begin{bmatrix}
    x_{x,c} \\
    x_{y,c} \\
    f \\
\end{bmatrix}
$$
where $f$ is the distance between the camera image plane and optical center $O$, that 

$f_L=||x_{\tiny{L,C}}-O_L||
\\
f_R=||x_{\tiny{R,C}}-O_R||$

* Epipole/epipolar point

$e_L$ and $e_R$ are called epipoles, defined as the two points on a *base line* $O_b = O_L - O_R$ penetrating through left and right camera views. $O_L$ and $O_R$ are called *optical centers*.

* Epipolar line

The line $O_L - X$ is seen by the left camera view as a single point $X_L$ projected on the left view plane. Other points $[x_1, x_2, x_3, ...]$ on this line $O_L - X$ are seen by the right camera view projected on the line $e_R - x_R$. The red line $e_R - x_R$ is called epilolar line.

* Epipolar plane

The triangular plane $x - O_L - O_R$ is called epipolar plane.

* Epipolar constraints

The projection of a real world point $x$ on the right camera plane as $x_R$ must be contained in the $e_R - x_R$ epipolar line. 

## Essential matrix and fundamental matrix

### Extrinsic calibration
Extrinsic calibration provides a 3d rigid coordinate transformation $x_{\tiny{C,L}}=M_{ex}[x^\text{T}_W, 1]^\text{T}$ from real world coordinates to left camera's coordinates, in which 
$$
M_{ex}=[R_L \space -R_L \overrightarrow{O}_{\tiny{W,L}}]_{3 \times 4}
$$
where $\overrightarrow{O}_{\tiny{W,L}}$ denotes left camera optical center in the real world coordinates; $R_L$ is a $3 \times 3$ rotation matrix. Intuitively, $M_{ex}$ can be interpreted as transoformation by rotation $R_L$ then translation by $R_L\overrightarrow{O}_{\tiny{W,L}}$.

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
This transforms a vector multiplication into a matrix multiplication.

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

$E$ has $rank(E)=2$ for its homogeneous coordinates' representation scaling by $f$.

### Fundamental matrix

An instrinsic camera calibration matrix $M_{in}$ defines the transformation from a camera coordinate point $x_{\tiny{W,C}}$ to its homogeneous coorsdinate point $x_{\tiny{W}}$.
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

## Compute fundamental matrix 

## Correspondance search