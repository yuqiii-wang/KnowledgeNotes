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
\end{bmatrix}
=
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
\bold{a} \times \bold{b}
=
[\bold{a}]_{\times} \bold{b}
$$
