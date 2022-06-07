## Jacobian

In vector calculus, the Jacobian matrix of a vector-valued function of several variables is the matrix of all its first-order partial derivatives. 

Given a mapping: $f : R_n \rightarrow R_m$ is a function such that each of its first-order partial derivatives exist on $R_n$, with input $x \in R^n$ ($n$ dimensions for input) and output $f(x) \in R^m$ ($m$ dimensions for output), define $J_{n \times m}$

$$
J_{n \times m} = \bigg[ \frac{\partial f}{\partial x_1} ... \frac{\partial f}{\partial x_n} \bigg]
=
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & ... & \frac{\partial f_1}{\partial x_n} 
\\
... & ... & ...
\\
\frac{\partial f_m}{\partial x_1} & ... & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

### Determinant

When $m = n$, the Jacobian matrix is square, so its determinant is a well-defined function of $x$, known as the Jacobian determinant of $f$.


## Isomorphism

In mathematics, an isomorphism is a structure-preserving mapping between two structures of the same type that can be reversed by an inverse mapping.

Rotation and translation are two typical isomorphic mapping.

## Tangent Space

Given a manifold $M$, a tangent space at $x$ on $M$ is $\Tau_x M$.

![tangent_space](imgs/tangent_space.png "tangent_space")

## Skew Symmetric

a skew-symmetric (or antisymmetric or antimetric) matrix is a square matrix whose transpose equals its negative.
$$
A^T=-A
$$

$$
a_{j,i} = -a_{i,j}
$$