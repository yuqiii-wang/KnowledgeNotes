# Positive-definite matrix

Given an $n \times n$ symmetric matrix $A$:

$$
A=
\begin{bmatrix}
a_{1,1} & a_{1,2} & ... & a_{1,n} \\\\
a_{2,1} & a_{2,2} & ... & a_{2,n} \\\\
... & ... & ... & ... \\\\
a_{n,1} & a_{n,2} & ... & a_{n,n} \\\\
\end{bmatrix}
$$

$A$ is positive-definite if the real number $\mathbf{x}^\text{T} A \mathbf{x}$ is positive for every nonzero real column vector $\mathbf{x}$ (here $\setminus$ denotes *set minus* operator):

$$
\mathbf{x}^\text{T} A \mathbf{x}>0 \quad \forall \mathbf{x} \in \mathbb{R}^n \setminus \{0\}
$$

$A$ is positive-semi-definite if if the real number $\mathbf{x}^\text{T} A \mathbf{x}$ is positive or zero:

$$
\mathbf{x}^\text{T} A \mathbf{x} \ge 0 \quad \forall \mathbf{x} \in \mathbb{R}^n
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
f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^{\top} Q \mathbf{x} + b^{\top}\mathbf{x} + c
$$

For $f(\mathbf{x})$ be convex (indicating it has a unique global minimum.), $Q$ must be positive semi-definite (all eigenvalues are non-negative).

Definition of Convexity, given $\forall \theta \in [0,1]$:

$$
f(\theta \mathbf{x}_1+(1-\theta)\mathbf{x}_2) \le \theta f(\mathbf{x}_1) + (1-\theta) f(\mathbf{x}_2)
$$

Newton's Method updates the parameter $\mathbf{x}$ iteratively to find the local minimum.
Each step is defined by the Hessian matrix of $f(\mathbf{x})$.

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - H^{-1} \nabla f(\mathbf{x}_k)
$$

where

* $H=Q$ is the Hessian matrix of $f(\mathbf{x})$
* $\nabla f(\mathbf{x}) = Q\mathbf{x} - \mathbf{b}$

Proof of $H=Q$ is the Hessian matrix of $f(\mathbf{x})$: if $f(\mathbf{x})$ has an extreme/local minimum, it should have derivative zero at that point $\nabla f(\mathbf{x}) = \mathbf{0}$, such that

$$
\begin{align*}
    && \mathbf{x}^{\top} Q + \mathbf{b} &= \mathbf{0} \\\\
    \Rightarrow && \mathbf{x} &= -Q^{-1}\mathbf{b}
\end{align*}
$$

Also for the second derivative, there is $\nabla^2 f(\mathbf{x}) = H = Q$.

As a result, the iterative update can be written as

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - Q^{-1} \nabla f(\mathbf{x}_k)
$$

### Use Example in PCA (Principle Component Analysis)

For sample data matrix $X \in \mathbb{R}^{n \times m}$ ($n$ samples each of which has $m$ features), define covariance matrix $\Sigma=\frac{1}{n-1}X^{\top}X$, and $\Sigma \in \mathbb{R}^{m \times m}$ is about the covariance between each of two feature pairs.

* The covariance matrix is symmetric because:

$$
\Sigma_{i,j} = \text{Cov}(X_i, X_j) = \Sigma_{j,i}
$$

* It is positive semi-definite because for any vector $\mathbf{z}\in\mathbb{R}^{m}$:

$$
\mathbf{z}^{\top}\Sigma\mathbf{z} = \text{Var}(\mathbf{z}^{\top}X) \ge \mathbf{0}
$$

Proof:

$$
\begin{align*}
    \mathbf{z}^{\top}\Sigma\mathbf{z} &=
    \mathbf{z}^{\top} \Big(\frac{1}{n-1}X^{\top}X\Big) \mathbf{z} \\\\ &=
    \frac{1}{n-1} \mathbf{z}^{\top} \Big(X^{\top}X\Big) \mathbf{z} \\\\ &=
    \frac{1}{n-1} (X\mathbf{z})^{\top} (X\mathbf{z}) \\\\ &=
    \frac{1}{n-1} \big|\big| X\mathbf{z} \big|\big|^2_2 \ge \mathbf{0}
\end{align*}
$$

This indicates that all eigenvalues of $\Sigma$ are non-negative.

PCA identifies directions (principal components) in the data where variance is maximized.
Large variance features contain richer info than small ones.

## Proof of Positive-Definite Matrix

### Eigenvalues of A Real Symmetric Matrix Are Real

Denote the complex number inner product operator as $\langle \mathbf{x}, \mathbf{y} \rangle=\sum_i \overline{x}\_i y_i$; denote $A^{\dagger}$ as the conjugate transpose of $A$, there is $A=A^{\dagger}=A^\text{T}$. Besides, $\lambda$ is the corresponding eigenvalue.

By $A\mathbf{x}=\lambda\mathbf{x}$, there is $A^2 \mathbf{x}=A(A\mathbf{x})=A\lambda \mathbf{x}=\lambda A\mathbf{x}=\lambda^2\mathbf{x}$.

$$
\langle A\mathbf{x}, A\mathbf{x} \rangle =
\mathbf{x}^*A^{\dagger} A\mathbf{x} =
\mathbf{x}^* (A^2 \mathbf{x}) =
\lambda^2 ||\mathbf{x}||^2
$$

So that $\lambda^2=\frac{\langle A\mathbf{x}, A\mathbf{x} \rangle}{||\mathbf{x}||^2} > 0$ is a real positive number.

Assume $\lambda=a+bi$, so that $\lambda^2=a^2-b^2+2abi$. Since $\lambda^2$ is a real positive number, $2abi=0$. If $a\ne 0$, by $2abi=0$, there is $b=0$, prove that $\lambda=a$ is a real number; if  $b\ne 0$, by $2abi=0$, there is $a=0$, so that $\lambda^2 = -b^2 < 0$ which contradicts the above derivation $\lambda^2 > 0$. Hence, $\lambda$ must be a real number.

### Eigenvectors of Real Symmetric Matrices Are Orthogonal

Define a real symmetric matrix $A$, whose eigenvectors are $\mathbf{x}\_i$ and the corresponding eigenvalues are $\lambda_i$. By inner product operation, there is $\langle A\mathbf{x}\_i, \mathbf{y}_j \rangle=\langle \mathbf{x}\_i, A^\text{T}\mathbf{y}_j \rangle$ for $i \ne j$.

So that,
$$
\begin{align*}
\lambda_i\langle \mathbf{x}\_i, \mathbf{x}_j \rangle&=
\langle \lambda_i\mathbf{x}\_i, \mathbf{x}_j \rangle
\\\\ &=
\langle A\mathbf{x}\_i, \mathbf{x}_j \rangle
\\\\ &=
\langle \mathbf{x}\_i, A^\text{T}\mathbf{x}_j \rangle
\\\\ &=
\langle \mathbf{x}\_i, A\mathbf{x}_j \rangle
\\\\ &=
\langle \mathbf{x}\_i, \lambda_j\mathbf{x}_j \rangle
\\\\ &=
\lambda_j\langle \mathbf{x}\_i, \mathbf{x}_j \rangle
\end{align*}
$$

Therefore, there is $(\lambda_i-\lambda_j)\langle \mathbf{x}\_i, \mathbf{x}_j \rangle=0$. Since $\lambda_i-\lambda_j \ne 0$, the other term must be zero $\langle \mathbf{x}\_i, \mathbf{x}_j \rangle=0$, hereby proved $\mathbf{x}\_i \perp \mathbf{x}_j$.

### Principal Axis Theorem

The *principal axis theorem* concerns quadratic forms in $\mathbb{R}^n$, which are homogeneous polynomials of degree $2$. Principal axis theorem provides a solution to represent the quadratic term by eigenvectors and eigenvalues:
$$
\begin{align*}
    Q(\mathbf{x}) &= \mathbf{x}^\text{T} A \mathbf{x}
    \\\\ &=
    \lambda_1 {c}_1^2 + \lambda_2 {c}_2^2 + ... + \lambda_n {c}_n^2
\end{align*}
$$

where $A$ is a symmetric matrix, and $\lambda_i$ are eigenvalues of $A$. Define $\mathbf{v}\_i$ as the corresponding eigenvectors, the $c_i$ is defined by $c_i=\mathbf{v}\_i\mathbf{x}$. $Q(\mathbf{x})$ which is a scalar number.

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
    5 & 4 \\\\
    4 & 5
\end{bmatrix}
\begin{bmatrix}
    x \\\\ y
\end{bmatrix}= \mathbf{x}^\text{T} A \mathbf{x}
$$

To orthogonally diagonalize $A$, there are
$$
\lambda_1 = 1, \quad \lambda_2 = 9
$$
whose corresponding eigenvectors are
$$
\mathbf{v}_1 =
\begin{bmatrix}
    1 \\\\ -1
\end{bmatrix}, \quad
\mathbf{v}_2 =
\begin{bmatrix}
    1 \\\\ 1
\end{bmatrix}
$$

whose orthonormal eigenbasis vectors are (Dividing these by their respective lengths)
$$
\mathbf{u}_1 =
\begin{bmatrix}
    \frac{1}{\sqrt{2}} \\\\ -\frac{1}{\sqrt{2}}
\end{bmatrix}, \quad
\mathbf{u}_2 =
\begin{bmatrix}
    \frac{1}{\sqrt{2}} \\\\ \frac{1}{\sqrt{2}}
\end{bmatrix}
$$

Set $U=[\mathbf{u}_1 \quad \mathbf{u}_2]$, decompose $A$, there is

$$
\begin{align*}
  A &= U\Sigma U^{-1} = U\Sigma U^{\text{T}}
  \\\\ &=
  \begin{bmatrix}
    \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\\\ 
    -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}
  \end{bmatrix}
  \begin{bmatrix}
    1 & 0 \\\\ 
    0 & 9
  \end{bmatrix}
  \begin{bmatrix}
    \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\\\ 
    \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}
  \end{bmatrix}
\end{align*}
$$

So that, 
$$
\begin{align*}
    5x^2 + 8xy + 5y^2 &=
    \mathbf{x}^\text{T} A \mathbf{x}
    \\\\ &=
    \mathbf{x}^\text{T} (U\Sigma U^{\text{T}}) \mathbf{x}
    \\\\ &=
    (U^\text{T}\mathbf{x})^\text{T} \Sigma (U^\text{T}\mathbf{x})
    \\\\ &=
    \begin{bmatrix}
      \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\\\ 
      \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}
  \end{bmatrix}^\text{T}
  \begin{bmatrix}
    x \\\\ y
  \end{bmatrix}
  \begin{bmatrix}
    1 & 0 \\\\ 
    0 & 9
  \end{bmatrix}
  \begin{bmatrix}
      \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\\\ 
      \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}
  \end{bmatrix}
  \begin{bmatrix}
    x \\\\ y
  \end{bmatrix}
  \\\\ &=
  \begin{bmatrix}
      \frac{x-y}{\sqrt{2}} \\\\ 
      \frac{x+y}{\sqrt{2}} 
  \end{bmatrix}^\text{T}
  \begin{bmatrix}
    1 & 0 \\\\ 
    0 & 9
  \end{bmatrix}
  \begin{bmatrix}
      \frac{x-y}{\sqrt{2}} \\\\ 
      \frac{x+y}{\sqrt{2}} 
  \end{bmatrix}
  \\\\ &=
  \begin{bmatrix}
      1 \cdot \frac{x-y}{\sqrt{2}} &
      9 \cdot \frac{x+y}{\sqrt{2}} 
  \end{bmatrix}
  \begin{bmatrix}
      \frac{x-y}{\sqrt{2}} \\\\ 
      \frac{x+y}{\sqrt{2}} 
  \end{bmatrix}
  \\\\ &=
  1 \cdot \big(\underbrace{\frac{x-y}{\sqrt{2}}}_{c_1} \big)^2 + 
  9 \cdot \big(\underbrace{\frac{x+y}{\sqrt{2}}}_{c_2} \big)^2
  \\\\ &=
  \lambda_1 c_1^2 + \lambda_2 c_2^2
\end{align*}
$$

By principal axis theorem, here to prove

* A symmetric matrix $A$ is positive definite if and only if $\mathbf{x}^\text{T} A \mathbf{x} > 0$ for every column $\mathbf{x}\ne\mathbf{0}$.

Given $\mathbf{x}^\text{T} A \mathbf{x} = \lambda_1 {c}_1^2 + \lambda_2 {c}_2^2 + ... + \lambda_n {c}_n^2$, for some $c_i>0$ and $\forall \lambda_i>0$, there must be $\mathbf{x}^\text{T} A \mathbf{x} > 0$.

### Orthogonal Diagonalization Proof

A real square matrix $A$ is orthogonally diagonalizable if there exist an orthogonal matrix $U$ and a diagonal matrix $\Sigma$ such that $A=U^\text{T}\Sigma U$

Directly from definition, since $U=[\mathbf{u}_1,\mathbf{u}_2,...,\mathbf{u}_n]$ is orthogonal, and $\Sigma=diag(\lambda_1, \lambda_2, ..., \lambda_n)$ is diagonal, $A$ is orthogonally diagonalizable.

### Principal Sub-Matrices are Positive Definite

If $A$ is positive definite, so is each principal submatrix ${}^{(r)}A$ for $r = 1, 2, . . . , n$.

Principal sub-matrices are defined such as illustrated by the example below:
$$
A = \begin{bmatrix}
    a_{11} & a_{12} & a_{13} \\\\
    a_{21} & a_{22} & a_{23} \\\\
    a_{31} & a_{32} & a_{33} \\\\
\end{bmatrix} \\\\
{}^{(1)}A = [a_{11}], \quad
{}^{(2)}A = \begin{bmatrix}
    a_{11} & a_{12} \\\\
    a_{21} & a_{22}
\end{bmatrix}, \quad
{}^{(3)}A = A = \begin{bmatrix}
    a_{11} & a_{12} & a_{13} \\\\
    a_{21} & a_{22} & a_{23} \\\\
    a_{31} & a_{32} & a_{33} \\\\
\end{bmatrix}
$$

To prove principal sub-matrices are positive definite, write $A$ in block form
$$
A = \begin{bmatrix}
    {}^{(r)}A & P \\\\
    P^\text{T} & B \\\\
\end{bmatrix}
$$

Set $\mathbf{x}=\begin{bmatrix}\mathbf{y} \\\\ \mathbf{0}\end{bmatrix}$, where $\mathbf{y} \in \mathbb{R}^r$ has the matching dimension number as that of ${}^{(r)}A$.

$$
\begin{align*}
\mathbf{x}^\text{T} A \mathbf{x} &= 
\begin{bmatrix}
    \mathbf{y}^\text{T} & \mathbf{0}^\text{T}
\end{bmatrix}
\begin{bmatrix}
    {}^{(r)}A & P \\\\
    P^\text{T} & B \\\\
\end{bmatrix}
\begin{bmatrix}
    \mathbf{y} \\\\ \mathbf{0}
\end{bmatrix}
\\\\ &=
\mathbf{y}^\text{T} {}^{(r)}A \mathbf{y}
\\\\ &> 0
\end{align*}
$$
