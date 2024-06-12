
# Matrix Norms

## p-norm

Let $p \ge 1$ be a real number.
The p-norm (a.k.a. $\mathcal{L}^p$ norm) of a vector $\bold{x} \in \mathbb{R}^n$ is

$$
||\bold{x}||_p := \Bigg( \sum^n_{i=1} |x_i|^p \Bigg)^{{\frac{1}{p}}}
$$

## Matrix p-norm

Given a field $K$ of either real or complex numbers, and let $K^{m \times n}$ be the $K$'s vector space,
a matrix norm is a norm on $K^{m \times n}$ denoted as $||A||$ that leads to $|| \space \cdot \space || : K^{m \times n} \rightarrow \mathbb{R}^1$.

Suppose $A,B \in K^{m \times n}$, and vector norm $|| \space \cdot \space ||_{\alpha}$ on $K^n$ and $|| \space \cdot \space ||_{\beta}$ on $K^m$ are known.
Any $m \times n$ matrix $A$ induces a linear operator transforming $\bold{x} \in K^n$ from $K^n$ to $K^m$.

$$
\begin{align*}
    ||A||_{\alpha, \beta} &=
    \sup \{ ||A\bold{x}||_{\beta} : \bold{x} \in K^n \text{ with } ||\bold{x}||_{\alpha} = 1 \}
\\ &=
    \sup \{ \frac{||A\bold{x}||_{\beta}}{||\bold{x}||_{\alpha}} : \bold{x} \in K^n \text{ with } \bold{x} \ne \bold{0} \}
\end{align*}
$$

where $\sup$ means supremum of the set. 

For matrix norms induced by vector *p-norms* ($1 \le p +\infty$) that sees $\alpha=\beta=p$, there is

$$
||A||_p = \sup_{\bold{x} \ne \bold{0}}  \frac{||A\bold{x}||_{p}}{||\bold{x}||_{p}}
$$

$||A||_p$ is interesting for it can be considered the "degree" of how much $\bold{x}$ is stretched by $A$.

* $||A||_p > 1$, the input vector $\bold{x}$ is increased in length
* $||A||_p < 1$, the input vector $\bold{x}$ is shrunken in length
* $||A||_p = 1$, the input vector $\bold{x}$ does not change in length

### p=0 Matrix Norm

The $\mathcal{L}_0$  measures how many zero-elements are in a tensor $\bold{x}$, or the element is either zero or one $x_i \in \{ 0, 1 \}$:

$$
||\bold{x}||_0 = |x_1|^0 + |x_2|^0 + ... + |x_n|^0
$$

It is useful in sparsity vs density for neural network:
the number of non-zero elements and sparsity's complement relationship is as below:

$$
\text{density} = 1 - \text{sparsity}
$$

## p=1 Matrix Norm

p=1 matrix norm is simply the maximum absolute column sum of the matrix.

$$
||A||_1 = \max_{1 \le j \le n} \sum^{m}_{i=1} |a_{ij}|
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

$$
\begin{align*}
\text{det}(\underbrace{A-tI}_{W}) &=
\begin{vmatrix}
    (a_{11}-t) \text{det}(W_{2:n, 2:n}) &
    a_{12} \text{det}(W_{2:n, (1, 3:n)}) & &
    a_{1n} \text{det}(W_{2:n, 1:n-1}) \\
    a_{21} \text{det}(W_{(1, 3:n), 2:n}) &
    (a_{22}-t) \text{det}(W_{(1, 3:n), (1, 3:n)}) & &
    a_{2n} \text{det}(W_{(1, 3:n), 1:n-1}) \\
    & & \ddots & \\
    a_{n1} \text{det}(W_{1:n-1, 2:n}) &
    a_{n2} \text{det}(W_{1:n-1, (1, 3:n)}) & &
    (a_{nn}-t) \text{det}(W_{1:n-1, 1:n-1}) \\
\end{vmatrix} 
\\ &=
\begin{vmatrix}
    \underbrace{
    (a_{11}-t) \begin{vmatrix}
        (a_{22}-t)\text{det}(W_{3:n, 3:n}) &
        a_{23} \text{det}(W_{3:n, (2, 4:n)}) & ... \\
        (a_{23}-t)\text{det}(W_{3:n, (2, 4:n)}) &
        a_{33} \text{det}(W_{(2, 4:n), (2, 4:n)}) & ... \\
        & \vdots & \\
        (a_{2n}-t)\text{det}(W_{3:n, 3:n-1}) &
        a_{3n} \text{det}(W_{(2, 4:n), 3:n-1}) & ... 
    \end{vmatrix}}_{(a_{11}-t) \text{det}(W_{2:n, 2:n})}
    &
    a_{12} \text{det}(W_{2:n, (1, 3:n)}) & &
    a_{1n} \text{det}(W_{2:n, 1:n-1}) \\
    a_{21} \text{det}(W_{(1, 3:n), 2:n}) &
    (a_{22}-t) \text{det}(W_{(1, 3:n), (1, 3:n)}) & &
    a_{2n} \text{det}(W_{(1, 3:n), 1:n-1}) \\
    & & \ddots & \\
    a_{n1} \text{det}(W_{1:n-1, 2:n}) &
    a_{n2} \text{det}(W_{1:n-1, (1, 3:n)}) & &
    (a_{nn}-t) \text{det}(W_{1:n-1, 1:n-1}) \\
\end{vmatrix} 
\\ &=
(-1)^n \big(t^n - (t)^{n-1} \text{trace}(A) + ... + \text{det}(A) \big)
\end{align*}
$$

For any polynomial should see the solution in this format.

$$
\begin{align*}
(-1)^n (t-\lambda_1)(t-\lambda_2) ... (t-\lambda_n) &=
(-1)^n \big(t^2-(\lambda_1+\lambda_2)t + 2\lambda_1\lambda_2 \big)(t-\lambda_3) ... (t-\lambda_n)
\\ &=
(-1)^n \big(t^3-(\lambda_1+\lambda_2+\lambda_3)t^2 +2\lambda_1\lambda_2 t - 2\lambda_1\lambda_2\lambda_3 + (\lambda_1+\lambda_2)\lambda_3 t  \big)(t-\lambda_4) ... (t-\lambda_n)
\\ &=
(-1)^n \big(t^4-(\lambda_1+\lambda_2+\lambda_3+\lambda_4)t^3 -(\lambda_1+\lambda_2+\lambda_3)\lambda_4 t^2 + ...  \big)(t-\lambda_4) ... (t-\lambda_n)
\\ &=
(-1)^n \big(t^n + (t)^{n-1} \underbrace{\sum_{i=1}^n \lambda_i}_{\text{trace}(A)} + (t)^{n-2}\lambda_n \sum_{i=1}^{n-1} \lambda_i + ... \big)
\end{align*}
$$

By comparing against the above $\text{det}(A-tI)$, there is

$$
\text{trace}(A) = \sum_{i=1}^n \lambda_i
$$

Proof: https://math.stackexchange.com/questions/586663/why-does-the-spectral-norm-equal-the-largest-singular-value/586835#586835

Also,

$$
||A^{\dagger} A||_2 = ||A A^{\dagger}||_2 = ||A||^2_2
$$

The *spectral radius* of a square matrix $A$ of $\text{rank}(A)=r$ is the maximum of the absolute values of its eigenvalues $||A||_2 = \sigma_{max}(A)=\max \{ |\lambda_1|, |\lambda_2|, ..., |\lambda_r| \}$.

Spectral radius describes to what max length $\bold{x}$ can be stretched by a square matrix $A$, that the max length happens to be the max eigenvalue of $A$.

$$
||A\bold{x}|| \le \lambda_{max}||\bold{x}||
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
