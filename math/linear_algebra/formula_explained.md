# Formula explained

## Orthogonal matrix proof: $A^T$ = $A^{-1}$

Matrix $A$ is orthogonal if the column and row vectors are orthonormal vectors. Define $v_i$ as column vector of $A$, given orthogonality here define:
$$
v_i^T v_j =
\begin{array}{cc}
  \Big \{ & 
    \begin{array}{cc}
      1 & i = j \\
      0 & i \neq j
    \end{array}
\end{array}
$$

hence
$$
A^T A = I
$$

and derived the conclusion:
$$
A^T = A^{-1}
$$

## Determinant

Determinant is a scalar value that is a function of the entries of a square matrix.

Geometrically speaking, determinant is area of the $n \times n$ squared matrix, for example, for a $2 \times 2$ matrix, the area of parallellogram is
$$
|u||v|sin\theta = 
\begin{array}{c}
    \bigg (
    \begin{array}{c}
      -b \\
      a
    \end{array}
    \bigg )
\end{array}
\begin{array}{c}
    \bigg (
    \begin{array}{c}
      c \\
      d
    \end{array}
    \bigg )
\end{array}
=
ad-bc
$$

![alt text](imgs/parallellogram_as_determinant.svg.png "parallellogram_as_determinant")

The following shows the calculation of a $3 \times 3$ matrix's determinant:
$$
\bigg |
\begin{array}{ccc}
    \begin{array}{ccc}
      a & b & c \\
      d & e & f \\
      g & h & i
    \end{array}
\end{array}
\bigg |
=
a
\big |
\begin{array}{cc}
    \begin{array}{cc}
      e & f \\
      h & i 
    \end{array}
\end{array}
\big |
-
d
\big |
\begin{array}{cc}
    \begin{array}{cc}
      b & c \\
      h & i 
    \end{array}
\end{array}
\big |
+
g
\big |
\begin{array}{cc}
    \begin{array}{cc}
      b & c \\
      e & f 
    \end{array}
\end{array}
\big |
$$

further,
$$
\bigg |
\begin{array}{ccc}
    \begin{array}{ccc}
      a & b & c \\
      d & e & f \\
      g & h & i
    \end{array}
\end{array}
\bigg |
=
a(ei-fh)-d(bi-hc)+g(bf-ec)
$$

## Eigenvalue and eigenvector

Provided $v$ as basis coordinate vector and $A$ as a transformation $n\times n$ matrix, eigenvalues can be defined as
$$
A v = \lambda v
$$
where $\lambda$ is eigenvalue and $v$ is eigenvector. Geometrically speaking, applied transformation $A$ to $v$ is same as scaling $v$ by $\lambda$.

This can be written as
$$
(A - \lambda I) v = 0
$$

When the determinant of $(A - \lambda I)$ is non-zero, $(A - \lambda I)$ has non-zero solutions of $v$ that satisfies
$$
|A - \lambda I| = 0
$$

## Diagonalization and the eigendecomposition

Given $Q$ as a matrix of eigenvectors $v_i$

$$
Q = [v_1, v_2, v_3, ..., v_n]
$$

then, take scalar $\lambda_i$ into consideration
$$
A Q = [\lambda_1 v_1, \lambda_2 v_2, \lambda_3 v_3, ..., \lambda_n v_n]
$$

$$
A Q = Q \Lambda
$$

$$
A = Q \Lambda Q^{-1}
$$

$A$ can therefore be decomposed into a matrix composed of its eigenvectors, a diagonal matrix with its eigenvalues along the diagonal, and the inverse of the matrix of eigenvectors.

## Singular Value Decomposition

Given an $m \times n$ matrix $A$, define two orthonormal (orthogonal, or perpendicular along a line) bases $v$ ($n \times 1$) and $u$ ($m \times 1$), and derived diagonalized $A$:

$$
A v_1 = \sigma_1 u_1, A v_2 = \sigma_2 u_2, ..., A v_r = \sigma_r u_r
$$

Geometrically speaking, $A v_i = \sigma_i u_i$ indicates transformation of $v_i$ by $A$ is same as scaling an orthonormal vector $u_i$ by singular value $\sigma_i$.

Define $\Sigma$ as a diagnal matrix of singular values $\sigma_i$, and $r=min(m,n)$:
$$
\Sigma = 
\begin{bmatrix}
\sigma_1 & 0 & 0& ... & 0\\
0 & \sigma_2 & 0 & ... & 0 \\
0 & 0 & \sigma_3 & ... & 0 \\
... & ... & ... & ... & ... \\
0 & 0 & 0 & ... & \sigma_r
\end{bmatrix}
$$

Define $V_{n \times r}$ and $U_r$ as
$$
V_r = [v_1, v_2, v_3, ..., v_r]
$$
$$
U_r = [u_1, u_2, u_3, ..., u_r]
$$

Here derived:
$$
A_{m \times n} V_{n \times r} = U_{m \times r} \Sigma_{r \times r}
$$

Since $V_{n \times r}$ is orthogonal, $V_{n \times r}^{-1} = V_{n \times r}^T$, $A_{m \times n}$ can be expressed as
$$
A_{m \times n} = U_{m \times r} \Sigma_{r \times r} V_{n \times r}^T
$$
$$
A_{m \times n} = u_1 \sigma_1 v_1^T + u_2 \sigma_2 v_2^T + u_3 \sigma_3 v_3^T + ... + u_r \sigma_r v_r^T
$$

In addition, 
$$
A A^T = (U \Lambda V^T)^T (U \Lambda V^T) = V \Lambda^T U^T U \Lambda V^T = V \Lambda^T \Lambda V ^T
$$

Take into consideration eigen-decomposition, $V$ is eigenvector matrix of $A A^T$ and each $\sigma^2$ in $\Lambda^T \Lambda$ (remember $\Lambda$ is diagnal) is eigenvalue for $A A^T$. Vice Vesa, $U$ is eigenvector matrix and each $\sigma^2$ is eigenvalue for $A^T A$.

### Intuition

$A_{m \times n}$ can be filled with null ($0$) values to be $A_{n \times n}$ (assumed $n \gt m$)
$$
A_{n \times n} =
\begin{bmatrix}
a_{0,0} & a_{0,1} & a_{0,2} & ... & a_{0,n} \\
a_{1,0} & a_{1,1} & a_{1,2} & ... & a_{1,n} \\
a_{2,0} & a_{2,1} & a_{1,2} & ... & a_{2,n} \\
... & ... & ... & ... & ... \\
a_{m,0} & a_{m,1} & a_{m,2} & ... & a_{2,n} \\
0 & 0 & 0 & ... & 0 \\
... & ... & ... & ... & ... \\
0 & 0 & 0 & ... & 0
\end{bmatrix}
$$

Eigenvalues are zeros for the added null elements in $A_{n \times n}$, so that only $\sigma_i \in [1, m]$ have positive values. Such $\sigma_i$ represent how scaled the transformation matrix $A$ is on a basis vector of size $(1 \times n)$, hence representing $A_{m \times n}$ on the basis vector associated with $\sigma_{max}$ has the greatest deviation and most information.