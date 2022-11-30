# Positive-definite matrix

Given an $n \times n$ symmetric matrix $A$:
$$
A=
\begin{bmatrix}
a_{1,1} & a_{1,2} & ... & a_{1,n} \\
a_{2,1} & a_{2,2} & ... & a_{2,n} \\
... & ... & ... & ... \\
a_{n,1} & a_{n,2} & ... & a_{n,n} \\
\end{bmatrix}
$$ 

$A$ is positive-definite if the real number $\bold{x}^\text{T} A \bold{x}$ is positive for every nonzero real column vector $x$ (here $\setminus$ denotes *set minus* operator):
$$
\bold{x}^\text{T} A \bold{x}>0 \quad \forall \bold{x} \in \mathbb{R}^n \setminus \{0\}
$$

$A$ is positive-semi-definite if if the real number $x^\text{T} A x$ is positive or zero:
$$
\bold{x}^\text{T} A \bold{x} \ge 0 \quad \forall \bold{x} \in \mathbb{R}^n 
$$

$A$ is positive-definite if it satisfies any of the following equivalent conditions.

* $A$ is congruent (exists an invertible matrix $P$ that $P^\text{T}AP=B$) with a diagonal matrix with positive real entries.
* $A$ is symmetric or Hermitian, and all its eigenvalues are real and positive.
* $A$ is symmetric or Hermitian, and all its leading principal minors are positive.
* There exists an invertible matrix $B$ with conjugate transpose $B^*$ such that $A=BB^*$

## Proof of Positive-Definite Matrix

### Eigenvalues of A Real Symmetric Matrix Are Real

Denote the complex number inner product operator as $\langle \bold{x}, \bold{y} \rangle=\sum_i \overline{x}_i y_i$; denote $A^*$ as the conjugate transpose of $A$, there is $A=A^*=A^\text{T}$. Besides, $\lambda$ is the corresponding eigenvalue.

By $A\bold{x}=\lambda\bold{x}$, there is $A^2 \bold{x}=A(A\bold{x})=A\lambda \bold{x}=\lambda A\bold{x}=\lambda^2\bold{x}$.

$$
\langle A\bold{x}, A\bold{x} \rangle = 
\bold{x}^*A^* A\bold{x} = 
\bold{x}^* (A^2 \bold{x}) = 
\lambda^2 ||\bold{x}||^2
$$

So that $\lambda^2=\frac{\langle A\bold{x}, A\bold{x} \rangle}{||\bold{x}||^2} > 0$ is a real positive number.

Assume $\lambda=a+bi$, so that $\lambda^2=a^2-b^2+2abi$. Since $\lambda^2$ is a real positive number, $2abi=0$. If $a\ne 0$, by $2abi=0$, there is $b=0$, prove that $\lambda=a$ is a real number; if  $b\ne 0$, by $2abi=0$, there is $a=0$, so that $\lambda^2 = -b^2 < 0$ which contradicts the above derivation $\lambda^2 > 0$. Hence, $\lambda$ must be a real number.

### Eigenvectors of Real Symmetric Matrices Are Orthogonal

Define a real symmetric matrix $A$, whose eigenvectors are $\bold{x}_i$ and the corresponding eigenvalues are $\lambda_i$. By inner product operation, there is $\langle A\bold{x}_i, \bold{y}_j \rangle=\langle \bold{x}_i, A^\text{T}\bold{y}_j \rangle$ for $i \ne j$.

So that,
$$
\begin{align*}
\lambda_i\langle \bold{x}_i, \bold{x}_j \rangle
&=
\langle \lambda_i\bold{x}_i, \bold{x}_j \rangle
\\ &=
\langle A\bold{x}_i, \bold{x}_j \rangle
\\ &=
\langle \bold{x}_i, A^\text{T}\bold{x}_j \rangle
\\ &=
\langle \bold{x}_i, A\bold{x}_j \rangle
\\ &=
\langle \bold{x}_i, \lambda_j\bold{x}_j \rangle
\\ &=
\lambda_j\langle \bold{x}_i, \bold{x}_j \rangle
\end{align*}
$$

Therefore, there is $(\lambda_i-\lambda_j)\langle \bold{x}_i, \bold{x}_j \rangle=0$. Since $\lambda_i-\lambda_j \ne 0$, the other term must be zero $\langle \bold{x}_i, \bold{x}_j \rangle=0$, hereby proved $\bold{x}_i \perp \bold{x}_j$.

### Principal Axis Theorem

The *principal axis theorem* concerns quadratic forms in $\mathbb{R}^n$, which are homogeneous polynomials of degree $2$. Principal axis theorem provides a solution to represent the quadratic term by eigenvectors and eigenvalues:
$$
\begin{align*}
    Q(\bold{x}) &= \bold{x}^\text{T} A \bold{x}
    \\ &=
    \lambda_1 {c}_1^2 + \lambda_2 {c}_2^2 + ... + \lambda_n {c}_n^2
\end{align*}
$$
where $A$ is a symmetric matrix, and $\lambda_i$ are eigenvalues of $A$. Define $\bold{v}_i$ as the corresponding eigenvectors, the $c_i$ is defined by $c_i=\bold{v}_i\bold{x}$. $Q(\bold{x})$ which is a scalar number.

$A$ has the below properties:
* The eigenvalues of $A$ are real.
* $A$ is diagonalizable, and the eigenspaces of $A$ are mutually orthogonal.

Example:
$$
5x^2 + 8xy + 5y^2 
=
\begin{bmatrix}
    x & y
\end{bmatrix}
\begin{bmatrix}
    5 & 4 \\
    4 & 5
\end{bmatrix}
\begin{bmatrix}
    x \\ y
\end{bmatrix}
= \bold{x}^\text{T} A \bold{x}
$$

To orthogonally diagonalize $A$, there are
$$
\lambda_1 = 1, \quad \lambda_2 = 9
$$
whose corresponding eigenvectors are
$$
\bold{v}_1 =
\begin{bmatrix}
    1 \\ -1
\end{bmatrix}, \quad
\bold{v}_2 =
\begin{bmatrix}
    1 \\ 1
\end{bmatrix}
$$
whose orthonormal eigenbasis vectors are (Dividing these by their respective lengths)
$$
\bold{u}_1 =
\begin{bmatrix}
    \frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}}
\end{bmatrix}, \quad
\bold{u}_2 =
\begin{bmatrix}
    \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}}
\end{bmatrix}
$$

Set $U=[\bold{u}_1 \quad \bold{u}_2]$, decompose $A$, there is
$$
\begin{align*}
  A &= U\Sigma U^{-1} = U\Sigma U^{\text{T}}
  \\ &=
  \begin{bmatrix}
    \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ 
    -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}
  \end{bmatrix}
  \begin{bmatrix}
    1 & 0 \\ 
    0 & 9
  \end{bmatrix}
  \begin{bmatrix}
    \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ 
    \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}
  \end{bmatrix}
\end{align*}
$$

So that, 
$$
\begin{align*}
    5x^2 + 8xy + 5y^2 &=
    \bold{x}^\text{T} A \bold{x}
    \\ &=
    \bold{x}^\text{T} (U\Sigma U^{\text{T}}) \bold{x}
    \\ &=
    (U^\text{T}\bold{x})^\text{T} \Sigma (U^\text{T}\bold{x})
    \\ &=
    \begin{bmatrix}
      \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ 
      \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}
  \end{bmatrix}^\text{T}
  \begin{bmatrix}
    x \\ y
  \end{bmatrix}
  \begin{bmatrix}
    1 & 0 \\ 
    0 & 9
  \end{bmatrix}
  \begin{bmatrix}
      \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ 
      \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}
  \end{bmatrix}
  \begin{bmatrix}
    x \\ y
  \end{bmatrix}
  \\ &=
  \begin{bmatrix}
      \frac{x-y}{\sqrt{2}} \\ 
      \frac{x+y}{\sqrt{2}} 
  \end{bmatrix}^\text{T}
  \begin{bmatrix}
    1 & 0 \\ 
    0 & 9
  \end{bmatrix}
  \begin{bmatrix}
      \frac{x-y}{\sqrt{2}} \\ 
      \frac{x+y}{\sqrt{2}} 
  \end{bmatrix}
  \\ &=
  \begin{bmatrix}
      1 \cdot \frac{x-y}{\sqrt{2}} &
      9 \cdot \frac{x+y}{\sqrt{2}} 
  \end{bmatrix}
  \begin{bmatrix}
      \frac{x-y}{\sqrt{2}} \\ 
      \frac{x+y}{\sqrt{2}} 
  \end{bmatrix}
  \\ &=
  1 \cdot \big(\underbrace{\frac{x-y}{\sqrt{2}}}_{c_1} \big)^2 + 
  9 \cdot \big(\underbrace{\frac{x+y}{\sqrt{2}}}_{c_2} \big)^2
  \\ &=
  \lambda_1 c_1^2 + \lambda_2 c_2^2
\end{align*}
$$

By principal axis theorem, here to prove

* A symmetric matrix $A$ is positive definite if and only if $\bold{x}^\text{T} A \bold{x} > 0$ for every column $\bold{x}\ne\bold{0}$.

Given $\bold{x}^\text{T} A \bold{x} = \lambda_1 {c}_1^2 + \lambda_2 {c}_2^2 + ... + \lambda_n {c}_n^2$, for some $c_i>0$ and $\forall \lambda_i>0$, there must be $\bold{x}^\text{T} A \bold{x} > 0$.

### Orthogonal Diagonalization Proof

A real square matrix $A$ is orthogonally diagonalizable if there exist an orthogonal matrix $U$ and a diagonal matrix $\Sigma$ such that $A=U^\text{T}\Sigma U$

Directly from definition, since $U=[\bold{u}_1,\bold{u}_2,...,\bold{u}_n]$ is orthogonal, and $\Sigma=diag(\lambda_1, \lambda_2, ..., \lambda_n)$ is diagonal, $A$ is orthogonally diagonalizable.

### Principal Sub-Matrices are Positive Definite

If $A$ is positive definite, so is each principal submatrix ${}^{(r)}A$ for $r = 1, 2, . . . , n$.

Principal sub-matrices are defined such as illustrated by the example below:
$$
A = \begin{bmatrix}
    a_{11} & a_{12} & a_{13} \\
    a_{21} & a_{22} & a_{23} \\
    a_{31} & a_{32} & a_{33} \\
\end{bmatrix}
\\
{}^{(1)}A = [a_{11}], \quad
{}^{(2)}A = \begin{bmatrix}
    a_{11} & a_{12} \\
    a_{21} & a_{22}
\end{bmatrix}, \quad
{}^{(3)}A = A = \begin{bmatrix}
    a_{11} & a_{12} & a_{13} \\
    a_{21} & a_{22} & a_{23} \\
    a_{31} & a_{32} & a_{33} \\
\end{bmatrix}
$$

To prove principal sub-matrices are positive definite, write $A$ in block form
$$
A = \begin{bmatrix}
    {}^{(r)}A & P \\
    P^\text{T} & B \\
\end{bmatrix}
$$

Set $\bold{x}=\begin{bmatrix}\bold{y} \\ \bold{0}\end{bmatrix}$, where $\bold{y} \in \mathbb{R}^r$ has the matching dimension number as that of ${}^{(r)}A$.

$$
\begin{align*}
\bold{x}^\text{T} A \bold{x} &= 
\begin{bmatrix}
    \bold{y}^\text{T} & \bold{0}^\text{T}
\end{bmatrix}
\begin{bmatrix}
    {}^{(r)}A & P \\
    P^\text{T} & B \\
\end{bmatrix}
\begin{bmatrix}
    \bold{y} \\ \bold{0}
\end{bmatrix}
\\ &=
\bold{y}^\text{T} {}^{(r)}A \bold{y}
\\ &> 0
\end{align*}
$$

## Use in Optimization

If $A$ is positive-definite, $x^\text{T}Ax$ has global minima solution $x^*$ (a convex).

If $A$ neither satisfies $x^\text{T}Ax>0$ nor $x^\text{T}Ax<0$, there exist saddle points.