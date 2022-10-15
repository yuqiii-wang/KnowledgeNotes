# 6 DOF Transformation

6 DOF (Dimensions of Freedom) refers to three dimensional translation and three dimensional rotation $T = [t_x, t_y, t_z, \theta_x, \theta_y, \theta_z]^T$

## Homogeneous Transformation

For rigid motion, there are rotations and translation.

Given a three-dimensional space, there is unit orthogonal basis vector for a global coordinate $\bold{e}$
$$
\begin{bmatrix}
e_1 , e_2, e_3
\end{bmatrix}
$$

Given robot as origin, there exists another robot-based coordinate $\bold{e^{'}}$
$$
\begin{bmatrix}
e^{'}_1, e^{'}_2, e^{'}_3
\end{bmatrix}
$$

To represent an object in the two aforementioned coordinates, there are $\bold{a}$ and $\bold{a^{'}}$
$$
\begin{bmatrix}
a_1 \\
a_2 \\
a_3
\end{bmatrix}
and
\begin{bmatrix}
a^{'}_1 \\
a^{'}_2 \\
a^{'}_3
\end{bmatrix}
$$

In practice, when a robot starts, its origin overlaps with the origin of the global coordinate
$$
\begin{bmatrix}
e_1 , e_2, e_3
\end{bmatrix}
\begin{bmatrix}
a_1 \\
a_2 \\
a_3
\end{bmatrix}
=
\begin{bmatrix}
e^{'}_1, e^{'}_2, e^{'}_3
\end{bmatrix}
\begin{bmatrix}
a^{'}_1 \\
a^{'}_2 \\
a^{'}_3
\end{bmatrix}
$$

When there is a robot movement
$$
\begin{bmatrix}
a_1 \\
a_2 \\
a_3
\end{bmatrix}
=
\begin{bmatrix}
e^T_1 e^{'}_1 & e^T_1 e^{'}_2 & e^T_1 e^{'}_3 \\
e^T_2 e^{'}_1 & e^T_2 e^{'}_2 & e^T_2 e^{'}_2 \\
e^T_3 e^{'}_1 & e^T_3 e^{'}_2 & e^T_3 e^{'}_3
\end{bmatrix}
\begin{bmatrix}
a^{'}_1 \\
a^{'}_2 \\
a^{'}_3
\end{bmatrix}
$$

To put it in a simple way
$$
\bold{a} = R \bold{a^{'}}
$$

Remember, $R$'s determinant is $1$: $det(R)=1$, and $R^{-1} = R^T$, hence $R^T R = I$

It is coined *Special Orthogonal Group*. Given $3$ as the dimensionality, it is denoted as $SO(3)$.

Consider translation:

$$
\begin{bmatrix}
a_1 \\
a_2 \\
a_3
\end{bmatrix}
=
\begin{bmatrix}
e^T_1 e^{'}_1 & e^T_1 e^{'}_2 & e^T_1 e^{'}_3 \\
e^T_2 e^{'}_1 & e^T_2 e^{'}_2 & e^T_2 e^{'}_2 \\
e^T_3 e^{'}_1 & e^T_3 e^{'}_2 & e^T_3 e^{'}_3
\end{bmatrix}
\begin{bmatrix}
a^{'}_1 \\
a^{'}_2 \\
a^{'}_3
\end{bmatrix}
+
\begin{bmatrix}
t_1 \\
t_2 \\
t_3
\end{bmatrix}
$$

To put it in a simple way
$$
\bold{a} = T \bold{a^{'}}
$$

Here introduces *Homogeneous Transformation* that takes translation and rotation merged as one: 

$$
\begin{bmatrix}
a_1 \\
a_2 \\
a_3 \\
1
\end{bmatrix}
=
\begin{bmatrix}
e^T_1 e^{'}_1 & e^T_1 e^{'}_2 & e^T_1 e^{'}_3 & t_1 \\
e^T_2 e^{'}_1 & e^T_2 e^{'}_2 & e^T_2 e^{'}_2 & t_2 \\
e^T_3 e^{'}_1 & e^T_3 e^{'}_2 & e^T_3 e^{'}_3 & t_3 \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
a^{'}_1 \\
a^{'}_2 \\
a^{'}_3 \\
1
\end{bmatrix}
$$

This set of transform matrix is also known
as the *special Euclidean group*:

$$
SE(3) = 
\bigg\{
    T = 
    \begin{bmatrix}
        R & T \\
        0 & 1
    \end{bmatrix}
    \in \mathbb{R}^{4 \times 4}
    |
    R \in SO(3), T \in \mathbb{R}^3
\bigg\}
$$

## Quaternion

### Euler Angle and Singularity

Euler angle provides an intuitive perspective into rotation that 
it uses primal axes to decompose a rotation into three rotations around different axes.

* Rotate around the $Z$ axis of the object to get the yaw angle $\theta_{yaw}=y$;
* Rotate around the $Y$ axis of the object to get the pitch angle $\theta_{pitch}=p$;
* Rotate around the $X$ axis of the object to get the roll angle $\theta_{roll}=r$;

However, it suffers from *Gimbal lock* (singularity problem), that results in loss of one degree of freedom. 
Shown as below, when the pitch (green) and yaw (magenta) gimbals become aligned, changes to roll (blue) and yaw apply the same rotation to the airplane.

![Gimbal_Lock_Plane](imgs/Gimbal_Lock_Plane.gif "Gimbal_Lock_Plane")

Euler angle's $\theta$ operates on a 2-d plane at a time. If all three axes's $\theta$ s are aligned to one 2-d plane, gimbal lock happens.

Quaternion uses four elements to represent rotation that avoids Gimbal lock issues. 

### Quaternion Definition

A rotation of angle $\theta$ in the three dimensional space given three bases $u_x\overrightarrow{i}, u_y\overrightarrow{j}, u_z\overrightarrow{k}$ are defined by the unit vector
$$
\overrightarrow{u}
=(u_x, u_y, u_z)
=u_x \overrightarrow{i} + u_y \overrightarrow{j} + u_z \overrightarrow{k}
$$
can be represented by a quaternion using an extension of Euler's formula:
$$
\begin{align*}
\bold{q}
&=
e^{\frac{\theta}{2}(u_x \overrightarrow{i} + u_y \overrightarrow{j} + u_z \overrightarrow{k})}
\\
&=
cos\frac{\theta}{2} + (u_x \overrightarrow{i} + u_y \overrightarrow{j} + u_z \overrightarrow{k})sin\frac{\theta}{2}
\end{align*}
$$

### Quaternion Operations

Quaternion given the above can be expressed in the below general form

$$
\bold{q} = 
[s, \bold{v}]^\text{T},
\quad s=q_0 \in \mathbb{R},
\quad \bold{v}=[x \overrightarrow{i}, y \overrightarrow{j}, z \overrightarrow{k}]^\text{T} \in \mathbb{R}^3
$$
where $\overrightarrow{i},\overrightarrow{j},\overrightarrow{k}$ represent imaginary parts in respect to the three dimensions.

* Addition/Subtraction

$$
\bold{q}_a + \bold{q}_b =
[s_a \pm s_b, \bold{v}_a \pm \bold{v}_b ]^\text{T}
$$

* Multiplication: 

Given the relationship of the respective imaginary parts (derived from cross product that sees $a \times b$ being perpendicular to the $<a , b>$ plane),
$$
\left\{\begin{array}{cc}
    \overrightarrow{i}^2 = \overrightarrow{j}^2 = \overrightarrow{k}^2 = -1 \\
    \overrightarrow{i}\overrightarrow{j} = \overrightarrow{k}, 
    \quad \overrightarrow{j}\overrightarrow{i} = -\overrightarrow{k} \\
    \overrightarrow{j}\overrightarrow{k} = \overrightarrow{i}, 
    \quad \overrightarrow{k}\overrightarrow{j} = -\overrightarrow{i} \\
    \overrightarrow{k}\overrightarrow{i} = \overrightarrow{j}, 
    \quad \overrightarrow{i}\overrightarrow{k} = -\overrightarrow{i} \\
\end{array}\right.
$$

define multiplication $\mathbb{R}^3 \rightarrow \mathbb{R}^3$:
$$
\begin{align*}
\bold{q}_a  \bold{q}_b &=
s_a s_b - x_a x_b - y_a y_b - z_a z_b
\\ & \quad +
(s_a x_b + x_a s_b + y_a z_b - z_a y_b)\overrightarrow{i}
\\ & \quad +
(s_b y_b - x_a z_b + y_a z_b - z_a x_b)\overrightarrow{j}
\\ & \quad +
(s_c z_b + x_a y_b + y_a z_b - z_a s_b)\overrightarrow{k}
\end{align*}
$$

### Quaternion Example

Consider a rotation around $\overrightarrow{v}=\overrightarrow{i} + \overrightarrow{j} + \overrightarrow{k}$ with a rotation angle of $\theta=\frac{2\pi}{3}$. The length of $\overrightarrow{v}$ is $\sqrt{3}$.

Hence
$$
\begin{align*}
\overrightarrow{u}
&=
cos \frac{\theta}{2} + sin \frac{\theta}{2} \cdot \frac{\overrightarrow{v}}{||\overrightarrow{v}||}
\\ &=
cos\frac{\pi}{3} + sin\frac{\pi}{3} \cdot \frac{\overrightarrow{v}}{||\sqrt{3}||}
\\ &=
cos\frac{\pi}{3} + sin\frac{\pi}{3} \cdot \frac{\overrightarrow{i} + \overrightarrow{j} + \overrightarrow{k}}{||\sqrt{3}||}
\\ &=
\frac{1+\overrightarrow{i} + \overrightarrow{j} + \overrightarrow{k}}{2}
\end{align*}
$$

### Quaternion Derivation

Define $\bold{q}^+$ and $\bold{q}^{\oplus}$ as the matrix representation of quaternion.

$$
\bold{q}^+ =
\begin{bmatrix}
    s & - \bold{v}^\text{T} \\
    \bold{v} & sI+\bold{v}^\wedge
\end{bmatrix}
, \quad
\bold{q}^{\oplus} =
\begin{bmatrix}
    s & - \bold{v}^\text{T} \\
    \bold{v} & sI-\bold{v}^\wedge
\end{bmatrix}
$$
where $\wedge$ denotes the skew-symmetric matrix representation of the vector and $I$ is the identity matrix.

Derivation shows as below.
$$
\bold{q}_a^+ \bold{q}_b
=
\begin{bmatrix}
    s_a & - \bold{v}_a^\text{T} \\
    \bold{v}_a & s_a I+\bold{v}_a^\wedge
\end{bmatrix}
\begin{bmatrix}
    s_b \\
    \bold{v}_b
\end{bmatrix}
=
\begin{bmatrix}
    -\bold{v}_a^\text{T} \bold{v}_b + s_a s_b \\
    s_b \bold{a} + s_b \bold{v}_b + \bold{v}^{\wedge}_a \bold{v}_b
\end{bmatrix}
=
\bold{q}_a \bold{q}_b
$$

Similarly, there is
$$
\bold{q}_a \bold{q}_b
=
\bold{q}_a^+ \bold{q}_b
=
\bold{q}_a \bold{q}_b^{\oplus}
$$

Define a spacial point represented in quaternion $\bold{p}=[0,\bold{v}_p] \in \mathbb{R}^3$ whose rotation is $\bold{p}'=\bold{q}\bold{p}\bold{q}^{-1}$, where $\bold{p}^{-1}$ is the matrix normalization term, there is
$$
\begin{align*}
    \bold{p}'&=\bold{q}\bold{p}\bold{q}^{-1}
    \\ &=
    \bold{q}^+\bold{p}^+\bold{q}^{-1}
    \\ &=
    \bold{q}^+\bold{q}^{-1^\oplus}\bold{p}
\end{align*}
$$

Here computes $\bold{q}^+\bold{q}^{-1^\oplus}$:
$$
\begin{align*}
    
\bold{q}^+\bold{q}^{-1^\oplus}
&=
\begin{bmatrix}
    s_a & - \bold{v}_a^\text{T} \\
    \bold{v}_a & s_a I+\bold{v}_a^\wedge
\end{bmatrix}
\begin{bmatrix}
    s & - \bold{v}^\text{T} \\
    \bold{v} & sI-\bold{v}^\wedge
\end{bmatrix}

\\ &=
\begin{bmatrix}
    1 & 0 \\
    \bold{0} & \bold{v}\bold{v}^\text{T}+s^2I+ 2s\bold{v}^{\wedge}+(\bold{v}^{\wedge})^2
\end{bmatrix}
\end{align*}
$$

Since $\bold{q}$ is defined as purely imaginary, so that the *quaternion-to-rotation matrix* can be defined as
$$
R =
\bold{v}\bold{v}^\text{T}+s^2I+ 2s\bold{v}^{\wedge}+(\bold{v}^{\wedge})^2
$$

In order to compute $\theta$, trace operation is performed as below
$$
\begin{align*}
    tr(R) &=
    tr \big(
        \bold{v}\bold{v}^\text{T}+s^2I+ 2s\bold{v}^{\wedge}+(\bold{v}^{\wedge})^2
    \big)

    \\ &=
    v_x^2 + v_y^2 + v_z^2 + 3s^2 + 0
    -2(v_x^2 + v_y^2 + v_z^2)

    \\ &=
    (1-s^2) + 3s^2 -2(1-s^2)

    \\ &=
    4s^2-1
\end{align*}
$$

According to *Rodrigues' rotation formula*, $\theta$ can be computed as 
$$
\begin{align*}
    
\theta &= arccos \bigg(
      \frac{tr(R)-1}{2}
\bigg)
\\ &=
arccos(2s^2-1)
\\ &=
2 \space arccos \space s

\end{align*}
$$

