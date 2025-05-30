# More Matrix Knowledge

## Complex conjugate

The complex conjugate of a complex number is the number with an equal real part and an imaginary part equal in magnitude but opposite in sign, such as the complex conjugate of $a + b i$ is equal to $a − b i$.

## Unitary Matrix

A complex square matrix $U$ is unitary if its conjugate transpose $U^{\dagger}$ is also its inverse, that is

$$
U^{\dagger}U = UU^{\dagger} = UU^{-1} = I
$$

## Isometry

An isometry (or congruence, or congruent transformation) is a distance-preserving transformation between metric spaces.

Geometrically speaking, it refers to rigid translation and rotation of geometrical figures.

### Isomorphism

In mathematics, an isomorphism is a structure-preserving mapping between two structures of the same type that can be reversed by an inverse mapping.

Rotation and translation are two typical isomorphic mapping.

## Similarity Transform And Diagonalizable Matrix

For $A$ and $B$ are said to be similar if there exists an invertible if there is an invertible $P \in \mathbb{C}^{r \times r}$ and satisfies

$$
B = P^{-1}A P
$$

If two matrices are similar, then they have the same rank, trace, determinant and eigenvalues.

### Diagonalizable Matrix

Define $A \in \mathbb{C}^{r \times r}$, that $A$ is diagonalizable if and only if it is similar to a diagonal matrix $\Lambda$.

$$
\Lambda = Q^{-1}A Q \qquad\text{or}\qquad  A=Q\Lambda Q^{-1}
$$

To compute/diagonalize a matrix $A$, first check $\text{rank}(A)=r$ if it is full rank, then solve $\text{det}(\lambda I - A)=0$.
The result $\Lambda$ is an eigenvalue-composed diagonal matrix, and $Q$ is an eigenvector-composed matrix.

This $A=Q\Lambda Q^{-1}$ expression means that $A$ can be eigen-decomposed to reveal its eigenvalues and eigenvectors.

## Hermitian matrix

A Hermitian matrix (or self-adjoint matrix) is a complex square matrix that is equal to its own conjugate transpose.

For example,
$$
\begin{bmatrix}
      0 & a-ib & c-id \\
      a+ib & 1 & 0 \\
      c+id & 0 & 2
\end{bmatrix}
$$

## Conjugate transpose (Hermitian transpose)

The conjugate transpose (or Hermitian transpose) of $A_{n \times m}$ is

1. take the transpose of $A_{n \times m}$
2. replace each entry $a_{i,j}$ with its complex conjugate

Denotation:

$$
(A^H)_{ij}=\overline{A}_{ji}
$$

This definition can also be written as

$$
A^H=\Big(\overline{A}\Big)^{\top}=\overline{\big(A^{\top}\big)}
$$

where $\overline{A}$ denotes the matrix with complex conjugated entries.

For example, given $A=\begin{bmatrix}1 & -2-i & 5 \\ 1+i & i & 4-2i \end{bmatrix}$,
the transpose is

$$
A^{\top} = \begin{bmatrix}
    1 & 1+i \\
    -2-i & i \\
    5 & 4-2i \\
\end{bmatrix}
$$

The conjugate transpose is

$$
A^{H} = \begin{bmatrix}
    1 & 1-i \\
    -2+i & -i \\
    5 & 4+2i \\
\end{bmatrix}
$$

## Permutation matrix

A permutation matrix is a square binary matrix that has exactly one entry of 1 in each row and each column and 0s elsewhere.

## Jacobian

In vector calculus, the Jacobian matrix of a vector-valued function of several variables is the matrix of all its first-order partial derivatives.

Given a mapping: $f : R_n \rightarrow R_m$ is a function such that each of its first-order partial derivatives exist on $R_n$, with input $x \in R^n$ ($n$ dimensions for input) and output $f(x) \in R^m$ ($m$ dimensions for output), define $J_{n \times m}$

$$
J_{n \times m} = \bigg[ \frac{\partial f}{\partial x_1} ... \frac{\partial f}{\partial x_n} \bigg] =
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

## Hessian

Hessian is a square matrix of second-order partial derivatives of a scalar-valued function, or scalar field. It describes the local curvature of a function of many variables.

Define $f:\mathbb{R}^n \rightarrow \mathbb{R}$ whose input is a vector $\bold{x} \in \mathbb{R}^n$ with a scalar output $f(\bold{x}) \in \mathbb{R}$. $\bold{H}$ of $f$ is an $n \times n$ matrix such as
$$
(\bold{H}_f)_{i,j}=\frac{\partial^2 f}{\partial x_i \partial x_j}
$$
or
$$
\bold{H}_f = 
\begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & ... & \frac{\partial^2 f}{\partial x_1 \partial x_n} 
\\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & ... & \frac{\partial^2 f}{\partial x_2 \partial x_n} 
\\
... & ... & ... & ...
\\
\frac{\partial^2 f}{\partial x_n \partial x_1}  & \frac{\partial^2 f}{\partial x_n \partial x_2}  & ... & \frac{\partial f}{\partial x_n^2}R
\end{bmatrix}
$$

## Tangent Space

Given a manifold $M$, a tangent space at $x$ on $M$ is $\Tau_x M$.

![tangent_space](imgs/tangent_space.png "tangent_space")

## Trace

The trace of a square matrix $A\in\mathbb{C}^{n \times n}$, denoted $tr(A)$,is defined to be the sum of elements on the main diagonal (from the upper left to the lower right) of $A$.
The trace is only defined for a square matrix ($n × n$).

$$
\text{tr}(A) = \sum^n_{i=1} a_{ii}
$$

For example, given $A$

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{bmatrix}
$$

its trace is

$$
\text{tr}(A) =
\sum^3_i a_{ii} =
a_{11} + a_{22} + a_{33}
$$

### Derivative

$$
\frac{
      \partial \space tr(ABC)
}{
      \partial \space B
} =
A^{\top} C^{\top}
$$

### Proof: A Matrix Trace Equals the Sum of Its Diagonal Entries and the Sum of Its Eigenvalues

Let $\lambda_1, \lambda_2, ..., \lambda_n$ be the roots of the characteristic polynomial for $A\in\mathbb{C}^{n \times n}$, here expands it:

$$
\text{det}(A-\bold{\lambda}I)=
(-1)^n\lambda_n+(-1)^{n-1}\text{tr}(A)\lambda_{n-1}+...+\text{det}(A)
$$

By Vieta's formulas, the sum of the roots (eigenvalues) is

$$
\begin{align*}
\sum^n_{i=1}\lambda_i&=
   \frac{\text{Coefficient of }\lambda^{n-1}}{\text{Coefficient of }\lambda^{n}}(-1)^{n-1} \\
   &= -\frac{(-1)^{n-1}\text{tr}(A)}{(-1)^n} \\
   &= \text{tr}(A)
\end{align*}
$$

## Kronecker Product

Given $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{p \times q}$, the *Kronecker product* is defined as $C=A \otimes B$ whose result is $C \in \mathbb{R}^{pm \times qn}$, suhc as
$$
A \otimes B = 
\begin{bmatrix}
    a_{11} B & a_{12} B &  & a_{1n} B \\
    a_{21} B & a_{22} B &  & a_{2n} B \\
    & & \ddots & \\
    a_{m1} B & a_{m2} B &  & a_{mn} B \\
\end{bmatrix}
$$

For example,

$$
\begin{align*}
\begin{bmatrix}
    1 & 2 \\
    3 & 4
\end{bmatrix}
\otimes
\begin{bmatrix}
    0 & 5 \\
    6 & 7
\end{bmatrix}&=
\begin{bmatrix}
    1 \begin{bmatrix}
        0 & 5 \\
        6 & 7
    \end{bmatrix}
    &
    2 \begin{bmatrix}
        0 & 5 \\
        6 & 7
    \end{bmatrix} 
    \\
    3 \begin{bmatrix}
        0 & 5 \\
        6 & 7
    \end{bmatrix}
    & 
    4 \begin{bmatrix}
        0 & 5 \\
        6 & 7
    \end{bmatrix}
\end{bmatrix}
\\ &=
\begin{bmatrix}
    1 \times 0 & 1 \times 5 & 2 \times 0 & 2 \times 5 \\
    1 \times 6 & 1 \times 7 & 2 \times 6 & 2 \times 7 \\
    3 \times 0 & 3 \times 5 & 4 \times 0 & 4 \times 5 \\
    3 \times 6 & 3 \times 7 & 4 \times 6 & 4 \times 7 \\
\end{bmatrix}
\\ &=
\begin{bmatrix}
    0 & 5 & 0 & 10 \\
    6 & 7 & 12 & 14 \\
    0 & 15 & 0 & 20 \\
    18 & 21 & 24 & 28 \\
\end{bmatrix}
\end{align*}
$$
