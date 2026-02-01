# Some Matrix Types

## Rank

The rank of a matrix is the maximum number of linearly independent rows (or columns) in the matrix.

For example, below $A$'s rank is $\text{rank}(A)=2$ for

$$
A =
\begin{bmatrix}
      1 & 2 & 3 \\\\
      4 & 5 & 6 \\\\
      2 & 4 & 6
\end{bmatrix} \underrightarrow{-2 R_1 + R_3}
\begin{bmatrix}
      1 & 2 & 3 \\\\
      4 & 5 & 6 \\\\
      0 & 0 & 0
\end{bmatrix}
$$

### Gaussian Elimination

Gaussian elimination, also known as row reduction, does not change column linearity. The above matrix $A$ can be reduced to

$$
\begin{align*}
A = 
\begin{bmatrix}
      1 & 4 & 5 \\\\
      2 & 5 & 7 \\\\
      3 & 6 & 9
\end{bmatrix}
\underrightarrow{-3 R_1 + R_3}
\begin{bmatrix}
      1 & 4 & 5 \\\\
      2 & 5 & 7 \\\\
      0 & -6 & -6
\end{bmatrix}
& \underrightarrow{-2 R_1 + R_2} \\\\
\begin{bmatrix}
      1 & 4 & 5 \\\\
      0 & -3 & -3 \\\\
      0 & -6 & -6
\end{bmatrix}
\underrightarrow{-2 R_2 + R_3} 
\begin{bmatrix}
      1 & 4 & 5 \\\\
      0 & -3 & -3 \\\\
      0 & 0 & 0
\end{bmatrix}
& \underrightarrow{-3 R_2} \\\\
\begin{bmatrix}
      1 & 4 & 5 \\\\
      0 & 1 & 1 \\\\
      0 & 0 & 0
\end{bmatrix}
\underrightarrow{-4 R_2 + R_1}
\begin{bmatrix}
      1 & 0 & 1 \\\\
      0 & 1 & 1 \\\\
      0 & 0 & 0
\end{bmatrix}
\end{align*}
$$

hence proving rank $2$ for matrix $A$.

## Transpose

$A\in\mathbb{R}^{n \times m}$'s transpose is denoted as $A^{\top}\in\mathbb{R}^{m \times n}$.

### Reverse Order Property

$$
\big(A_1A_2...A_n\big)^{\top}=A^{\top}_n...A^{\top}_2A^{\top}_1
$$

### Gram Matrix

Given a set of vectors $\{\mathbf{a}_1, \mathbf{a}_2, ... \mathbf{a}_n\}$, the Gram matrix $G\in\mathbb{R}^{n \times n}$ has its entries as the inner products of these vectors.

$$
G_{ij}=\langle \mathbf{a}\_i, \mathbf{a}_j \rangle
$$

If $\{\mathbf{a}_1, \mathbf{a}_2, ... \mathbf{a}_n\}$ are arranged as the columns of a matrix $A$, then

$$
G=A^{\top}A
$$

#### Proof of $A^{\top}A$ Be Symmetric

For

$$
\big(A^{\top}A\big)^{\top}=A^{\top}(A^{\top})^{\top}=A^{\top}A
$$

$A^{\top}A$ is symmetric.

#### Proof of $A^{\top}A$ Be Positive Semi-Definite

For a matrix be positive semi-definite, its quadratic form $\mathbf{x}\big(A^{\top}A\big)\mathbf{x} \ge 0$ is non-negative.

$$
\mathbf{x}\big(A^{\top}A\big)\mathbf{x}=\big(A\mathbf{x}\big)^{\top}A\mathbf{x}=\big|\big|A\mathbf{x}\big|\big|^2
$$

#### $A^{\top}A$ is Symmetric Hence Orthogonal

Define follows for $i\ne j$

$$
A^{\top}A\mathbf{v}\_i=\sigma_i^2\mathbf{v}\_i \qquad
A^{\top}A\mathbf{v}_j=\sigma_j^2\mathbf{v}_j
$$

then multiply by $\mathbf{v}_j^{\top}$, there is

$$
\begin{align*}
    && \mathbf{v}_j^{\top}A^{\top}A\mathbf{v}\_i &=
  \mathbf{v}_j^{\top}\sigma_i^2\mathbf{v}\_i \\\\
  \Rightarrow && (A^{\top}A\mathbf{v}_j)^{\top}\mathbf{v}\_i &=
  \mathbf{v}_j^{\top}\sigma_i^2\mathbf{v}\_i \qquad\text{ for symmetry } \big(A^{\top}A\big)^{\top}=A^{\top}A  \\\\
  \Rightarrow && \sigma_j^2\mathbf{v}_j^{\top}\mathbf{v}\_i &=
  \sigma_i^2\mathbf{v}_j^{\top}\mathbf{v}\_i \\\\
\end{align*}
$$

For by eigen-decomposition, there is $\sigma_i^2 \ne \sigma_j^2 \ne 0$, there could be only $\mathbf{v}_j^{\top}\mathbf{v}\_i=0$, hence orthogonal.

##### Spectral Theorem

If $M$ is a real symmetric matrix, then:

* All eigenvalues of $M$ are **real**
* This means the eigenvectors of $M$ can be chosen to **be orthogonal and normalized**.
* $M$ can be can be orthogonally diagonalized $M=Q\Lambda Q^{\top}$, where 1) $\Lambda$ is a diagonal matrix containing the eigenvalues of $M$, 2) the columns of $Q$ are the orthonormal eigenvectors of $M$.

## Triangular Matrix

Either upper or lower area relative the diagonal of a matrix is all zeros:

* Upper Triangular Matrix

$$
A_{\text{upper}} = \begin{bmatrix}
    1 & 2 & 3 \\\\
    0 & 5 & 6 \\\\
    0 & 0 & 9
\end{bmatrix}
$$

* Lower Triangular Matrix

$$
A_{\text{lower}} = \begin{bmatrix}
    1 & 0 & 0 \\\\
    4 & 5 & 0 \\\\
    7 & 8 & 9
\end{bmatrix}
$$

## Orthogonal matrix

An orthogonal matrix, or orthonormal matrix, is a real square matrix whose columns and rows are orthonormal vectors.

It has the below properties:

$$
Q^{\top}Q=QQ^{\top}=I
$$

This leads to
$$
Q^{\top}=Q^{-1}
$$

The determinant of any orthogonal matrix is either $+1$ or $−1$, i.e., $\text{det}(Q)=\pm 1$.

### Proof of Transpose vs Inverse Given Orthogonality

$Q^{\top}=Q^{-1}$ if and only if $Q$ is orthogonal.

From the definition of orthogonality ($Q^{\top}Q=QQ^{\top}=I$) and the inverse property ($Q^{-1}Q=I$),
it can be derived $Q^{\top}=Q^{-1}$.

### Proof of Product Orthogonality

Assume $A$ and $B$ are orthogonal, to prove $AB$ also orthogonal (The product of two orthogonal matrices retains orthogonality, with determinant $\text{det}(Q)=\pm 1$.)

First, there is $(AB)^{\top}=B^{\top}A^{\top}$.
Then, multiply $(AB)^{\top}$ by $AB$, there is

$$
(AB)^{\top}(AB)=B^{\top}A^{\top}AB
$$

For $A$ and $B$ are orthogonal, there are $A^{\top}A=I$ and $B^{\top}B=I$,
finally, there is

$$
B^{\top}A^{\top}AB=B^{\top}IB=I
$$

### Disjoint Linear Projection

In vector space $V$ define two linear projection $P$ and $Q$, if they satisfy $PQ=QP=\mathbf{0}$, it is termed *disjoint linear projection*.

This means that the projections are orthogonal to each other, and their ranges (the subspaces they project onto) do not overlap.

### Orthogonal Group

The set of $n \times n$ orthogonal matrices forms a group, $O(n)$, known as the orthogonal group. The subgroup $SO(n)$ consisting of orthogonal matrices with determinant $+1$ is called the *special orthogonal group*, and each of its elements is a *special orthogonal matrix*.

## Determinant

Determinant is a scalar value that is a function of the entries of a square matrix.

Geometrically speaking, determinant is area of the $n \times n$ squared matrix, for example, for a $2 \times 2$ matrix, the area of parallellogram is

$$
|u||v|\sin\theta =
\begin{array}{c}
    \bigg (
    \begin{array}{c}
      -b \\\\
      a
    \end{array}
    \bigg )
\end{array}
\begin{array}{c}
    \bigg (
    \begin{array}{c}
      c \\\\
      d
    \end{array}
    \bigg )
\end{array} =
ad-bc
$$

![parallellogram_as_determinant](imgs/parallellogram_as_determinant.svg.png "parallellogram_as_determinant")

The following shows the calculation of a $3 \times 3$ matrix's determinant:

$$
\bigg |
\begin{array}{ccc}
    \begin{array}{ccc}
      a & b & c \\\\
      d & e & f \\\\
      g & h & i
    \end{array}
\end{array}
\bigg | =
a
\big |
\begin{array}{cc}
    \begin{array}{cc}
      e & f \\\\
      h & i 
    \end{array}
\end{array}
\big | -
d
\big |
\begin{array}{cc}
    \begin{array}{cc}
      b & c \\\\
      h & i 
    \end{array}
\end{array}
\big | +
g
\big |
\begin{array}{cc}
    \begin{array}{cc}
      b & c \\\\
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
      a & b & c \\\\
      d & e & f \\\\
      g & h & i
    \end{array}
\end{array}
\bigg | =
a(ei-fh)-d(bi-hc)+g(bf-ec)
$$

which give the volume of a parallelotope.

### Input Vector Updates With Different Determinants

Given an input vector $\mathbf{x}$ and linear transform $A$, the result vector $A\mathbf{x}$ is a combination of scaling, rotation, and shearing, depending on the structure of $A$.

#### $\text{det}(A)>1$ Expansion

$$
A\mathbf{x}=\begin{bmatrix}
    1 & 2 \\\\
    3 & 4
\end{bmatrix} \begin{bmatrix}
    1 \\\\ 1
\end{bmatrix} = \begin{bmatrix}
    3 \\\\ 7
\end{bmatrix}
$$

#### $\text{det}(A)<1$ Contraction

$$
A\mathbf{x}=\begin{bmatrix}
    0.1 & 0.2 \\\\
    0.3 & 0.4
\end{bmatrix} \begin{bmatrix}
    1 \\\\ 1
\end{bmatrix} = \begin{bmatrix}
    0.3 \\\\ 0.7
\end{bmatrix}
$$

#### $\text{det}(A)=1$ Volume Preservation/Pure Rotation

$$
A\mathbf{x}=\begin{bmatrix}
    0 & -1 \\\\
    1 & 0
\end{bmatrix} \begin{bmatrix}
    1 \\\\ 1
\end{bmatrix} = \begin{bmatrix}
    -1 \\\\ 1
\end{bmatrix}
$$

The vector $\mathbf{x}$ is rotated by $90$ degrees counterclockwise.

#### $\text{det}(A)=0$ Collapse

$\text{det}(A)=0$ happens when $\text{rank}(A)$ is not full.

$$
A\mathbf{x}_1=\begin{bmatrix}
    1 & 1 \\\\
    1 & 1
\end{bmatrix} \begin{bmatrix}
    1 \\\\ 1
\end{bmatrix} = \begin{bmatrix}
    2 \\\\ 2
\end{bmatrix} \\\\
A\mathbf{x}_2=\begin{bmatrix}
    1 & 1 \\\\
    1 & 1
\end{bmatrix} \begin{bmatrix}
    1 \\\\ 2
\end{bmatrix} = \begin{bmatrix}
    3 \\\\ 3
\end{bmatrix} \\\\
A\mathbf{x}_3=\begin{bmatrix}
    1 & 1 \\\\
    1 & 1
\end{bmatrix} \begin{bmatrix}
    2 \\\\ 1
\end{bmatrix} = \begin{bmatrix}
    3 \\\\ 3
\end{bmatrix}
$$

All $\mathbf{x}\_i$ are collapsed into the line $0=x_2-x_1$.

### Derivation

#### Adjugate Matrix

Adjugate, adjunct or classical adjoint of a square matrix $Adj(A)$ is the transpose of its cofactor matrix $C$.

$$
Adj(A) = C^\text{T}
$$

where

$$
C = \big( (-1)^{i+j} M_{i,j} \big)_{1\leq i,j \leq n}
$$

where $M_{i,j}$ is the determinant of the $(i,j)$-th element of a square matrix $A$ .

For example, given

$$
A =
\begin{bmatrix}
      a & b & c \\\\
      d & e & f \\\\
      g & h & i
\end{bmatrix}
$$

the $(1,2)$-th element is $b$, whose determinant can be expressed as

$$
M_{1,2} = -
\bigg |
\begin{array}{cc}
    \begin{array}{cc}
      d & f \\\\
      g & i
    \end{array}
\end{array}
\bigg |= -(di-fg)
$$

Cofactor matirx $C$:

$$
C =
\begin{bmatrix}
      M_{1,1} & M_{1,2} & M_{1,3} \\\\
      M_{1,2} & M_{2,2} & M_{2,3} \\\\
      M_{1,3} & M_{3,2} & M_{3,3}
\end{bmatrix}
$$

Finding classical adjoint of a matrix is same as applying a linear transformation which brings the coordinates of $i$ and $j$ to a square of area equal to the determinant of that matrix.

#### Laplace Expansion (Cofactor Expansion)


### Proof of Determinant

Take an orthonormal basis $\mathbf{e}_1,…,\mathbf{e}_n$ and let columns of $A$ be $a_1,…,a_n$, where $∧$ represents exterior product operator,

$$
 \mathbf{a}_1 ∧ ... ∧\mathbf{a}_n=\text{det}(A) (\mathbf{e}_1 ∧ ⋯ ∧ \mathbf{e}_n)
$$

hence

$$
\text{det}(A)=(\mathbf{e}_1 ∧ ⋯ ∧ \mathbf{e}_n)^{−1}(a_1 ∧ ⋯ ∧ \mathbf{a}_n)
$$

given orthogonality ($E^{-1}=E^T$):

$$
(\mathbf{e}_1 ∧ ⋯ ∧ \mathbf{e}_n)^{−1} = (\mathbf{e}_1 ⋯ \mathbf{e}_n)^{−1} = \mathbf{e}_n ⋯ \mathbf{e}_1 = \mathbf{e}_n ∧ ⋯ ∧ \mathbf{e}_1
$$

Note that $\mathbf{e}_n ∧ ⋯ ∧ \mathbf{e}_1$ is a subspace of $a_1 ∧ ⋯ ∧ \mathbf{a}_n$, we can further write

$$
\begin{align*}
\text{det}(A)
& =(\mathbf{e}_n ∧ ⋯ ∧ \mathbf{e}_1)⋅(a_1 ∧ ⋯ ∧ \mathbf{a}_n) \\\\
& =(\mathbf{e}_n ∧ ⋯ ∧ \mathbf{e}_2)⋅\big(\mathbf{e}_1⋅(a_1 ∧ ⋯ ∧ \mathbf{a}_n)\big) \\\\
& =(\mathbf{e}_n ∧ ⋯ ∧ \mathbf{e}_2)⋅\bigg(a_{1,1}(a_2 ∧ ⋯ ∧ \mathbf{a}_n)−\sum_{i=2}^n (-1)^i \mathbf{a}\_{1,i}(a_1 ∧ ⋯ ∧ \hat{\mathbf{a}}\_i ∧ ⋯ ∧ \mathbf{a}_n) \bigg)
\end{align*}
$$

## Adjoint of A Matrix (Hermitian Adjoint)

The adjoint of a matrix (Hermitian Adjoint) and classical adjoint are two different things, do not confuse.

Consider a linear map $A: H_1 \rightarrow H_2$ between Hilbert spaces, the adjoint operator is the linear operator $A^\dag: H_2 \rightarrow H_1$ satisfying

$$
\langle A h_1, h_2 \rangle_{H_2} =
\langle h_1, A^\dag h_2 \rangle_{H_1}
$$

where $\langle \space . \space, \space . \space \rangle_{H_i}$ is the inner product in the Hilbert space $H_i$, and $\space^\dag$ is the notation for Hermitian/conjugate transpose.

### Self-Adjoint Matrix

A matrix $A$ is self-adjoint if it equals its adjoint $A = A^\dag$.
For real matrices, this means that the matrix is symmetric: it equals its transpose $A = A^\top$.

Eigen-decomposition of a real self-adjoint matrix is $A = V\Sigma V^{-1}$ where $\Sigma$ is a diagonal matrix whose diagonal elements are real eigenvalues., and $V$ is composed of eigenvectors by columns.
Hurthermore, $V$ is unitary, meaning that its invverse is equal to its adjoint $V^{-1}=V^{\dag}$.

## Inverse Matrix

A square matrix $A$ (non-square matrix has no inverse) has its inverse when its determinant is not zero.

$$
AA^{-1} = I
$$

and,

$$
A^{-1} = \frac{1}{|A|}Adj(A)
$$

where

$|A|$ is determiant of $A$ (denoted as $det(A)$) and $Adj(A)$ is an adjugate matrix of $A$.

Geometrically speaking, an inverse matrix $A^{-1}$ takes a transformation $A$ back to its origin (same as reseting basis vectors).

### Pseudo Inverse

Pseudo inverse (aka Moore-Penrose inverse) denoted as $A^{\dagger}$, satisfying the below conditions:

* $AA^{\dagger}$ does not neccessarily give to identity matrix $I$, but mapping to itself

$$
AA^{\dagger}A=A \\\\
A^{\dagger}AA^{\dagger}=A^{\dagger}
$$

* $AA^{\dagger}$ is Hermitian, and vice versa

$$
(AA^{\dagger})^*=AA^{\dagger} \\\\
(A^{\dagger}A)^*=A^{\dagger}A
$$

* If $A$ is invertible, its pseudoinverse is its inverse

$$
A^{\dagger}=A^{-1}
$$

#### Pseudo Inverse for Non-Square Matrix Inverse

Given a non-square matrix $A \in \mathbb{R}^{n \times m}$ for $m \ne n$, the "best approximation" of the inverse is defined as $A^{\dagger}$ that satisfies the above pseudo inverse definition $AA^{\dagger}A=A$.
By strict definition, non-square matrix has no inverse.

The motivation is that, consider a linear system $A\mathbf{x} = \mathbf{b}$, if $A$ is a square matrix ($m=n$), there is an exact solution for the system $\mathbf{x}=A^{-1}\mathbf{b}$, if not ($m \ne n$), there is $AA^{\dagger}A=A$.

To approximate $\mathbf{x}=A^{-1}\mathbf{b}$ for non-square matrix $A$, set $\mathbf{x}=A^{\dagger}\mathbf{b}$ as the pseudo solution, so that there is $A\mathbf{x}=AA^{\dagger}\mathbf{b}=\mathbf{b}$, where $A^{\dagger} \in \mathbb{R}^{m \times n}$.

OpenCV has builtin API for $\mathbf{x}=A^{\dagger}\mathbf{b}$.
In the below code, first construct the linear system by pushing back rows (such as robot states) to `A` and `b`.
Then, find the pseudo inverse of `A` denoted as `pinA`, by which the solution can be constructed as `x = pinA * b;`.

In least squares problem, solution $\mathbf{x} \in \mathbb{R}^m$ should be derived from an over-determined system where $n > m$.

```cpp
double cv::invert(InputArray src,
                        OutputArray dst,
                        int flags = DECOMP_LU 
                    );

cv::Mat A;
cv::Mat b;
cv::Mat pinA;
 
for (int i = 0; i < nRows; i++) {
    A.push_back(tempA);
    b.push_back(tempb);
}

cv::invert(A, pinA, DECOMP_SVD);

cv::Mat x = pinA * b;
```
