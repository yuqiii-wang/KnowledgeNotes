# Skew Symmetric

A skew-symmetric (or antisymmetric or antimetric) matrix is a square matrix whose transpose equals its negative.
$$
A^T=-A
$$

$$
a_{j,i} = -a_{i,j}
$$

For example
$$
A=
\begin{bmatrix}
      0 & a_1 & a_2 \\
      -a_1 & 0 & a_3 \\
      -a_2 & -a_3 & 0
\end{bmatrix}
$$

There is
$$
-A=
\begin{bmatrix}
      0 & -a_1 & -a_2 \\
      a_1 & 0 & -a_3 \\
      a_2 & a_3 & 0
\end{bmatrix}=
A^T
$$

## Vector Space 

* The sum of two skew-symmetric matrices is skew-symmetric.
* A scalar multiple of a skew-symmetric matrix is skew-symmetric.

The space of $n \times n$ skew-symmetric matrices has dimensionality $\frac{1}{2} n (n - 1)$.

## Cross Product

Given $\bold{a}=(a_1, a_2, a_3)^\text{T}$ and $\bold{b}=(b_1, b_2, b_3)^\text{T}$

Define $\bold{a}$'s skew matrix representation
$$
[\bold{a}]_{\times}=
\begin{bmatrix}
      0 & a_1 & a_2 \\
      -a_1 & 0 & a_3 \\
      -a_2 & -a_3 & 0
\end{bmatrix}
$$

Cross product can be computed by its matrix multiplication 
$$
\bold{a} \times \bold{b}=
[\bold{a}]_{\times} \bold{b}
$$

## Use Case in Geometry

Given a cross operation of two vector $\bold{a} \in \mathbb{R}^3$ and $\bold{b} \in \mathbb{R}^3$, there is
$$
\begin{align*}
\bold{a} \times \bold{b}&=
\bigg|\bigg|
\begin{array}{ccc}
      \bold{e}_1 & \bold{e}_2 & \bold{e}_3 \\
      a_1 & a_2 & a_3 \\
      b_1 & b_2 & b_3
\end{array}
\bigg|\bigg|\\ &=
\begin{bmatrix}
      a_2 b_3 - a_3 b_2 \\
      a_3 b_1 - a_1 b_3 \\
      a_1 b_2 - a_2 b_1 \\
\end{bmatrix}\\ &=
\begin{bmatrix}
      0 & -a_3 & a_2 \\
      a_3 & 0 & -a_1 \\
      -a_2 & a_1 & 0 \\
\end{bmatrix}
\begin{bmatrix}
      b_1 \\
      b_2 \\
      b_3 \\
\end{bmatrix}\\ &=
\begin{bmatrix}
      0 & -a_3 & a_2 \\
      a_3 & 0 & -a_1 \\
      -a_2 & a_1 & 0 \\
\end{bmatrix}
\bold{b}\\ &=
\bold{a}^{\wedge} \bold{b}
\end{align*}
$$
where $\bold{a}^{\wedge}$ denotes the skew-symmetric matrix representation of the vector $\bold{a}$.

The length of the cross product result is $|\bold{a}||\bold{b}|sin\angle\bold{a}, \bold{b}$, where $\angle\bold{a}, \bold{b}$ represents the angle between the two vectors.