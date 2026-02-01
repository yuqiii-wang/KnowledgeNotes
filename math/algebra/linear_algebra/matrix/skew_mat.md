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
      0 & a_1 & a_2 \\\\
      -a_1 & 0 & a_3 \\\\
      -a_2 & -a_3 & 0
\end{bmatrix}
$$

There is
$$
-A=
\begin{bmatrix}
      0 & -a_1 & -a_2 \\\\
      a_1 & 0 & -a_3 \\\\
      a_2 & a_3 & 0
\end{bmatrix}=
A^T
$$

## Vector Space 

* The sum of two skew-symmetric matrices is skew-symmetric.
* A scalar multiple of a skew-symmetric matrix is skew-symmetric.

The space of $n \times n$ skew-symmetric matrices has dimensionality $\frac{1}{2} n (n - 1)$.

## Cross Product

Given $\mathbf{a}=(a_1, a_2, a_3)^\text{T}$ and $\mathbf{b}=(b_1, b_2, b_3)^\text{T}$

Define $\mathbf{a}$'s skew matrix representation
$$
[\mathbf{a}]_{\times}=
\begin{bmatrix}
      0 & a_1 & a_2 \\\\
      -a_1 & 0 & a_3 \\\\
      -a_2 & -a_3 & 0
\end{bmatrix}
$$

Cross product can be computed by its matrix multiplication 
$$
\mathbf{a} \times \mathbf{b}=
[\mathbf{a}]_{\times} \mathbf{b}
$$

## Use Case in Geometry

Given a cross operation of two vector $\mathbf{a} \in \mathbb{R}^3$ and $\mathbf{b} \in \mathbb{R}^3$, there is

$$
\begin{align*}
\mathbf{a} \times \mathbf{b}&=
\bigg|\bigg|
\begin{array}{ccc}
      \mathbf{e}_1 & \mathbf{e}_2 & \mathbf{e}_3 \\\\
      a_1 & a_2 & a_3 \\\\
      b_1 & b_2 & b_3
\end{array}
\bigg|\bigg|\\\\ &=
\begin{bmatrix}
      a_2 b_3 - a_3 b_2 \\\\
      a_3 b_1 - a_1 b_3 \\\\
      a_1 b_2 - a_2 b_1 \\\\
\end{bmatrix}\\\\ &=
\begin{bmatrix}
      0 & -a_3 & a_2 \\\\
      a_3 & 0 & -a_1 \\\\
      -a_2 & a_1 & 0 \\\\
\end{bmatrix}
\begin{bmatrix}
      b_1 \\\\
      b_2 \\\\
      b_3 \\\\
\end{bmatrix}\\\\ &=
\begin{bmatrix}
      0 & -a_3 & a_2 \\\\
      a_3 & 0 & -a_1 \\\\
      -a_2 & a_1 & 0 \\\\
\end{bmatrix}
\mathbf{b}\\\\ &=
\mathbf{a}^{\wedge} \mathbf{b}
\end{align*}
$$

where $\mathbf{a}^{\wedge}$ denotes the skew-symmetric matrix representation of the vector $\mathbf{a}$.

The length of the cross product result is $|\mathbf{a}||\mathbf{b}|sin\angle\mathbf{a}, \mathbf{b}$, where $\angle\mathbf{a}, \mathbf{b}$ represents the angle between the two vectors.