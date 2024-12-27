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

$A$ is positive-definite if the real number $\bold{x}^\text{T} A \bold{x}$ is positive for every nonzero real column vector $\bold{x}$ (here $\setminus$ denotes *set minus* operator):

$$
\bold{x}^\text{T} A \bold{x}>0 \quad \forall \bold{x} \in \mathbb{R}^n \setminus \{0\}
$$

$A$ is positive-semi-definite if if the real number $\bold{x}^\text{T} A \bold{x}$ is positive or zero:

$$
\bold{x}^\text{T} A \bold{x} \ge 0 \quad \forall \bold{x} \in \mathbb{R}^n
$$

$A$ is positive-definite if it satisfies any of the following equivalent conditions.

* $A$ is congruent (exists an invertible matrix $P$ that $P^\text{T}AP=B$) with a diagonal matrix with positive real entries.
* $A$ is symmetric or Hermitian, and all its eigenvalues are real and positive.
* $A$ is symmetric or Hermitian, and all its leading principal minors are positive.
* There exists an invertible matrix $B$ with conjugate transpose $B^*$ such that $A=BB^*$

## Use Examples

### Use Example in Optimization

Given a quadratic function to

$$
f(\bold{x}) = \frac{1}{2} \bold{x}^{\top} Q \bold{x} + b^{\top}\bold{x} + c
$$

For $f(\bold{x})$ be convex (indicating it has a unique global minimum.), $Q$ must be positive semi-definite (all eigenvalues are non-negative).

Definition of Convexity, given $\forall \theta \in [0,1]$:

$$
f(\theta \bold{x}_1+(1-\theta)\bold{x}_2) \le \theta f(\bold{x}_1) + (1-\theta) f(\bold{x}_2)
$$

Newton's Method updates the parameter $\bold{x}$ iteratively to find the local minimum.
Each step is defined by the Hessian matrix of $f(\bold{x})$.

$$
\bold{x}_{k+1} = \bold{x}_k - H^{-1} \nabla f(\bold{x}_k)
$$

where

* $H=Q$ is the Hessian matrix of $f(\bold{x})$
* $\nabla f(\bold{x}) = Q\bold{x} - \bold{b}$

Proof of $H=Q$ is the Hessian matrix of $f(\bold{x})$: if $f(\bold{x})$ has an extreme/local minimum, it should have derivative zero at that point $\nabla f(\bold{x}) = \bold{0}$, such that

$$
\begin{align*}
    && \bold{x}^{\top} Q + \bold{b} &= \bold{0} \\
    \Rightarrow && \bold{x} &= -Q^{-1}\bold{b}
\end{align*}
$$

Also for the second derivative, there is $\nabla^2 f(\bold{x}) = H = Q$.

As a result, the iterative update can be written as

$$
\bold{x}_{k+1} = \bold{x}_k - Q^{-1} \nabla f(\bold{x}_k)
$$

### Use Example in PCA (Principle Component Analysis)

For sample data matrix $X \in \mathbb{R}^{n \times m}$ ($n$ samples each of which has $m$ features), define covariance matrix $\Sigma=\frac{1}{n-1}X^{\top}X$, and $\Sigma \in \mathbb{R}^{m \times m}$ is about the covariance between each of two feature pairs.

* The covariance matrix is symmetric because:

$$
\Sigma_{i,j} = \text{Cov}(X_i, X_j) = \Sigma_{j,i}
$$

* It is positive semi-definite because for any vector $\bold{z}\in\mathbb{R}^{m}$:

$$
\bold{z}^{\top}\Sigma\bold{z} = \text{Var}(\bold{z}^{\top}X) \ge \bold{0}
$$

Proof:

$$
\begin{align*}
    \bold{z}^{\top}\Sigma\bold{z} &=
    \bold{z}^{\top} \Big(\frac{1}{n-1}X^{\top}X\Big) \bold{z} \\ &=
    \frac{1}{n-1} \bold{z}^{\top} \Big(X^{\top}X\Big) \bold{z} \\ &=
    \frac{1}{n-1} (X\bold{z})^{\top} (X\bold{z}) \\ &=
    \frac{1}{n-1} \big|\big| X\bold{z} \big|\big|^2_2 \ge \bold{0}
\end{align*}
$$

This indicates that all eigenvalues of $\Sigma$ are non-negative.

PCA identifies directions (principal components) in the data where variance is maximized.
Large variance features contain richer info than small ones.

## Proof of Positive-Definite Matrix

### Eigenvalues of A Real Symmetric Matrix Are Real

Denote the complex number inner product operator as $\langle \bold{x}, \bold{y} \rangle=\sum_i \overline{x}_i y_i$; denote $A^{\dagger}$ as the conjugate transpose of $A$, there is $A=A^{\dagger}=A^\text{T}$. Besides, $\lambda$ is the corresponding eigenvalue.

By $A\bold{x}=\lambda\bold{x}$, there is $A^2 \bold{x}=A(A\bold{x})=A\lambda \bold{x}=\lambda A\bold{x}=\lambda^2\bold{x}$.

$$
\langle A\bold{x}, A\bold{x} \rangle =
\bold{x}^*A^{\dagger} A\bold{x} =
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
\lambda_i\langle \bold{x}_i, \bold{x}_j \rangle&=
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
5x^2 + 8xy + 5y^2 =
\begin{bmatrix}
    x & y
\end{bmatrix}
\begin{bmatrix}
    5 & 4 \\
    4 & 5
\end{bmatrix}
\begin{bmatrix}
    x \\ y
\end{bmatrix}= \bold{x}^\text{T} A \bold{x}
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
