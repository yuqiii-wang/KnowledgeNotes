
# Matrix Norms

## p-norm

Let $p \ge 1$ be a real number.
The p-norm (a.k.a. $\mathcal{L}^p$ norm) of a vector $\mathbf{x} \in \mathbb{R}^n$ is

$$
||\mathbf{x}||_p := \Bigg( \sum^n\_{i=1} |x\_i|^p \Bigg)^{{\frac{1}{p}}}
$$

## Matrix p-norm

Given a field $K$ of either real or complex numbers, and let $K^{m \times n}$ be the $K$'s vector space,
a matrix norm is a norm on $K^{m \times n}$ denoted as $||A||$ that leads to $|| \space \cdot \space || : K^{m \times n} \rightarrow \mathbb{R}^1$.

Suppose $A,B \in K^{m \times n}$, and vector norm $|| \space \cdot \space ||_{\alpha}$ on $K^n$ and $|| \space \cdot \space ||_{\beta}$ on $K^m$ are known.
Any $m \times n$ matrix $A$ induces a linear operator transforming $\mathbf{x} \in K^n$ from $K^n$ to $K^m$.

$$
\begin{align*}
    ||A||_{\alpha, \beta} &=
    \sup \{ ||A\mathbf{x}||_{\beta} : \mathbf{x} \in K^n \text{ with } ||\mathbf{x}||_{\alpha} = 1 \}
\\ &=
    \sup \{ \frac{||A\mathbf{x}||_{\beta}}{||\mathbf{x}||_{\alpha}} : \mathbf{x} \in K^n \text{ with } \mathbf{x} \ne \mathbf{0} \}
\end{align*}
$$

where $\sup$ means supremum of the set. 

For matrix norms induced by vector *p-norms* ($1 \le p +\infty$) that sees $\alpha=\beta=p$, there is

$$
||A||_p = \sup_{\mathbf{x} \ne \mathbf{0}}  \frac{||A\mathbf{x}||_{p}}{||\mathbf{x}||_{p}}
$$

$||A||_p$ is interesting for it can be considered the "degree" of how much $\mathbf{x}$ is stretched by $A$.

* $||A||_p > 1$, the input vector $\mathbf{x}$ is increased in length
* $||A||_p < 1$, the input vector $\mathbf{x}$ is shrunken in length
* $||A||_p = 1$, the input vector $\mathbf{x}$ does not change in length

### p=0 Matrix Norm

The $\mathcal{L}_0$  measures how many zero-elements are in a tensor $\mathbf{x}$, or the element is either zero or one $x\_i \in \{ 0, 1 \}$:

$$
||\mathbf{x}||_0 = |x_1|^0 + |x_2|^0 + ... + |x_n|^0
$$

It is useful in sparsity vs density for neural network:
the number of non-zero elements and sparsity's complement relationship is as below:

$$
\text{density} = 1 - \text{sparsity}
$$

## p=1 Matrix Norm

p=1 matrix norm is simply the maximum absolute column sum of the matrix.

$$
||A||_1 = \max_{1 \le j \le n} \sum^{m}\_{i=1} |a_{ij}|
$$

## p=2 Matrix (Frobenius) Norm (Spectral Radius)

In vectors, the $p=2$'s norm is named Euclidean norm ($\mathcal{l}_2$).
In matrix, $||A||_2$ is named spectral norm ($\mathcal{L}_2$).

The $p=2$'s matrix norm is coined *Frobenius norm* (equivalent of Euclidean norm $\mathcal{l}_2$ but for matrix)

$$
||A||_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n |a_{ij}|^2} =
\sqrt{\text{trace} (A^{\dagger} A) }
$$

where $A^{\dagger}$ is $A$'s conjugate transpose.

Let $\Sigma(\space \cdot \space)$ be the eigenvalue matrix of $\space \cdot \space$ (by diagnalization), and $\max (\space \cdot \space)$ takes the maximum out of the matrix/vector argument, there is

$$
||A||_2 = \sqrt{\max \Big( \Sigma\big( A^{\dagger} A \big) \Big)} =
\sigma_{max}(A) \le ||A||_F
$$

where $\sigma_{max}(A)$ represents the largest singular value of matrix $A$.

Equality of $\sigma_{max}(A) \le ||A||_F$ holds if and only if the matrix $A$ is a rank-one matrix or a zero matrix. 

This inequality can be derived from the fact that the trace of a matrix is equal to the sum of its eigenvalues $\text{trace}(A) = \sum_{i=1}^n \lambda_i$, where $\lambda_i$ is the eigenvalue of $A$.

### Proof: A Matrix Trace Equals the Sum of Its Diagonal Entries and the Sum of Its Eigenvalues

Let $\lambda_1, \lambda_2, ..., \lambda_n$ be the roots of the characteristic polynomial for $A\in\mathbb{C}^{n \times n}$, here expands it:

$$
\text{det}(A-\mathbf{\lambda}I)=
(-1)^n\lambda_n+(-1)^{n-1}\text{tr}(A)\lambda_{n-1}+...+\text{det}(A)
$$

By Vieta's formulas, the sum of the roots (eigenvalues) is

$$
\begin{align*}
\sum^n\_{i=1}\lambda_i&=
   \frac{\text{Coefficient of }\lambda^{n-1}}{\text{Coefficient of }\lambda^{n}}(-1)^{n-1} \\
   &= -\frac{(-1)^{n-1}\text{tr}(A)}{(-1)^n} \\
   &= \text{tr}(A)
\end{align*}
$$

### Spectral Radius of A Square Matrix

$$
||A^{\dagger} A||_2 = ||A A^{\dagger}||_2 = ||A||^2_2
$$

The *spectral radius* of a square matrix $A$ of $\text{rank}(A)=r$ is the maximum of the absolute values of its eigenvalues $||A||_2 = \sigma_{max}(A)=\max \{ |\lambda_1|, |\lambda_2|, ..., |\lambda_r| \}$.

Spectral radius describes to what max length $\mathbf{x}$ can be stretched by a square matrix $A$, that the max length happens to be the max eigenvalue of $A$.

$$
||A\mathbf{x}|| \le \lambda_{max}||\mathbf{x}||
$$

Nuclear Norm $||A||_*$:

$||A||_*$ is defined as the sum of its singular values:

$$
||A||_* = \sum_{i} \sigma_i(A)
$$

where $\sigma_i(\space . \space)$ is the $i$-th singular value of $A$.

## $p=+\infty$ Max Norm

It is defined as when $p = q$ goes to infinity:

$$
||A||_{max} = \max_{i,j} |a_{ij}|
$$
