# Singular Value Decomposition (SVD)

SVD is a factorization that decompose an $m \times n$ matrix A into

$$
A=U \Sigma V^{\top}
$$

SVD generalizes the eigen-decomposition of a square normal matrix with an orthonormal eigen basis to any $n \times m$ matrix $A$.

## SVD Derivation

Compute the eigenvalues and eigenvectors of $A^{\top}A$
$$
A^{\top}A \bold{v}_1 = \sigma_1^2 \bold{v}_1
\\
A^{\top}A \bold{v}_2 = \sigma_2^2 \bold{v}_2
\\
...
\\
A^{\top}A \bold{v}_n = \sigma_n^2 \bold{v}_n
$$

* So that $V$ is

$$
V=
\begin{bmatrix}
    \vdots & & \vdots \\
    \bold{v}_1 & ... & \bold{v}_n \\
    \vdots & & \vdots
\end{bmatrix}
$$

* So that $\Sigma$ is

$$
\Sigma=
\begin{bmatrix}
    \sigma_1 & & & \\
     & \ddots & & \bold{0} \\
     & & \sigma_n & \\
\end{bmatrix}
$$

or (depending on the relative matrix sizes of $m$ vs $n$)

$$
\Sigma=
\begin{bmatrix}
    \sigma_1 & & \\
     & \ddots & \\
     & & \sigma_n \\
     & \bold{0} &
\end{bmatrix}
$$

* So that $U$ is

$$
U=A V \Sigma^{-1}
$$

### Proof of Orthogonality

#### $A^{\top}A$ is Symmetric Hence $V$ Is Orthogonal

Define follows for $i\ne j$

$$
A^{\top}A\bold{v}_i=\sigma_i^2\bold{v}_i \qquad
A^{\top}A\bold{v}_j=\sigma_j^2\bold{v}_j
$$

then multiply by $\bold{v}_j^{\top}$, there is

$$
\begin{align*}
    && \bold{v}_j^{\top}A^{\top}A\bold{v}_i &=
  \bold{v}_j^{\top}\sigma_i^2\bold{v}_i \\
  \Rightarrow && (A^{\top}A\bold{v}_j)^{\top}\bold{v}_i &=
  \bold{v}_j^{\top}\sigma_i^2\bold{v}_i \qquad\text{ for symmetry } \big(A^{\top}A\big)^{\top}=A^{\top}A  \\
  \Rightarrow && \sigma_j^2\bold{v}_j^{\top}\bold{v}_i &=
  \sigma_i^2\bold{v}_j^{\top}\bold{v}_i \\
\end{align*}
$$

For by eigen-decomposition, there is $\sigma_i^2 \ne \sigma_j^2 \ne 0$, there could be only $\bold{v}_j^{\top}\bold{v}_i=0$, hence orthogonal.

For $\{\bold{v}_1, \bold{v}_2, ..., \bold{v}_n\}$ are defined as eigenvectors of $A$ such that $V=\begin{bmatrix} \vdots & & \vdots \\ \bold{v}_1 & ... & \bold{v}_n \\ \vdots & & \vdots \end{bmatrix}$, $V$ is orthogonal.

#### $U$ Is Orthogonal For It Is Equivalent of $V$ For $AA^{\top}$

For $A^{\top}A=Q\Sigma Q^{\top}$ established by spectral theorem that for symmetric matrix eigen-decomposition, $Q$ is orthogonal.
The difference is that the spectral theorem only works for square matrix, while SVD works for non-square matrix as well.

The non-square matrix $A\in\mathbb{R}^{n\times m}$ has diff sizes for $A^{\top}A\in\mathbb{R}^{m\times m}$ vs $AA^{\top}\in\mathbb{R}^{n\times n}$, as a result, for $A^{\top}A=V\Sigma V^{\top}$, there is equivalent $AA^{\top}=U\Sigma U^{\top}$.

In conclusion, $U$ is orthogonal for it is equivalent of $V$ for $AA^{\top}$ and is of different size.

### Intuition of SVD

In geometry intuition in $A=U \Sigma V^{\top}$, the $U$ and $V^{\top}$ are considered rotation, and $\Sigma$ is a scaling matrix.

#### $V$ and $U$ As Rotation

Given two input vectors $\bold{x}_i$ and $\bold{x}_j$, the angle between them satisfies $\cos(\theta)=\frac{\bold{x}_i^{\top}\bold{x}_j}{||\bold{x}_i||\space||\bold{x}_i||}$. Apply the orthonormal matrix $V$, there is

$$
\cos(\theta')=\frac{(V\bold{x}_i)^{\top}V\bold{x}_j}{||V\bold{x}_i||\space||V\bold{x}_i||}=
\frac{\bold{x}_i^{\top}\bold{x}_j}{||\bold{x}_i||\space||\bold{x}_i||}=
\cos(\theta)
$$

Thus, $\theta'=\theta$, the angle is preserved.

This holds true for $U$ as well.

If $V$ and $U$ are chosen as normalized, i.e., they are orthonormal, $V$ and $U$ are pure rotation.

#### $\Sigma$ As Scaling

$\Sigma$ is a diagonal matrix, hence its non-zero entries act as pure scaling.

## SVD in Machine Learning

Typically, for a population of samples $A$, the covariance ${\Omega}$ of $A$ (typically use ${\Sigma}$ as covariance matrix notation, but here use ${\Omega}$ to avoid duplicate notations as ${\Sigma}$ means singular value matrix in this article) of the samples' features describes how rich information they are.
Larger the variance of a feature, likely richer the information.

Take SVD on the covariance matrix such that ${\Omega}=U \Sigma V^\top$, and obtain singular value matrix ${\Sigma}$ and new orthogonal basis space $V$.
Intuitively speaking, ${\Sigma}$ describes how significant is for each corresponding orthogonal basis vector in $V$.

The transformed new orthogonal space $V$ can help recover the source sample data by $A=AV$.

### SVD for PCA

PCA (Principal Component Analysis) simply takes the first few most significant components out of the result of SVD (Singular Value Decomposition).

## SVD for least squares problem

Given a least squares problem:
for a residual $\bold{r} = A \bold{x} - \bold{b}$, where $A \in \mathbb{R}^{m \times n}$ (assumed $A$ is full rank that $n = \text{rank}(A)$), and there is $m > n$, here attempts to minimize

$$
\space \underset{\bold{x}}{\text{min}} \space
||A \bold{x} - \bold{b}||^2=
r_1^2 + r_2^2 + ... + r^2_m
$$

Process:

$$
\begin{align*}
& ||A \bold{x} - \bold{b}||^2 \\ =& 
||U \Sigma V^{\top} \bold{x} - \bold{b}||^2 \\ =&
||U^{\top}(U \Sigma V^{\top} \bold{x} - \bold{b})||^2 \\ =& 
||U^{\top}U \Sigma V^{\top} \bold{x} - U^{\top}\bold{b}||^2
\quad U\text{ is orthoganal that } U^{\top}U=I\\ =&
||\Sigma V^{\top} \bold{x} - U^{\top}\bold{b}||^2\\ =&
||\Sigma \bold{y} - U^{\top}\bold{b}||^2
\quad \text{denote } \bold{y}=V^\top\bold{x}
\text{ and } \bold{z}=U^\top\bold{b} \\ =&
\Bigg|\Bigg|
\begin{bmatrix}
    \sigma_1 & & & \\
     & \ddots & & \\
    & & \sigma_n & \\
    & & & \bold{0}
\end{bmatrix}
\bold{y} - \bold{z}
\Bigg|\Bigg|^2\\ =&
\sum^{n}_{i=1} \big( \sigma_i {y}_i - \bold{u}^{\top}_i \bold{b} \big)^2+\sum^{m}_{i=n+1} \big( \bold{u}^{\top}_i \bold{b} \big)^2
\end{align*}
$$

$\bold{y}$ is determined as

$$
y_i=
\left\{
    \begin{array}{cc}
        \frac{\bold{u}^{\top}_i \bold{b}}{\sigma_i} &\quad \sigma_i \ne 0 \text{ same as } i \le n
        \\
        \text{any value} &\quad \sigma_i = 0 \text{ same as } i > n
    \end{array}
\right.
$$

Then, it is easy to find $\bold{x}$ by $\bold{x} = V\bold{y}$.

The residual is $\sum^{m}_{i=n+1} \big( \bold{u}^{\top}_i \bold{b} \big)^2$.

### Proof

## SVD vs Eigen Decomposition

* SVD generalizes the eigen decomposition of a square normal matrix with an orthonormal eigen basis to any $m \times n$ matrix.

* Eigen decomposition: not necessarily orthonormal vs SVD: orthonormal

Here defines a typical linear system $A\bold{x}=\bold{b}$.
Consider the eigen decomposition $A = P\Lambda P^{-1}$ and $A=U\Sigma V^{\top}$.

Eigen decomposition only takes one basis $P$ in contrast to SVD using two bases $U$ and $V$. Besides, $P$ might not be orthogonal but $U$ and $V$ are orthonormal (orthogonal + unitary).
