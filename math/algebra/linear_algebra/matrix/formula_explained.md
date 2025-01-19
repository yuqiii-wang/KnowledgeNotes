# Formula explained

## Proof of Determinant

Take an orthonormal basis $\bold{e}_1,…,\bold{e}_n$ and let columns of $A$ be $a_1,…,a_n$, where $∧$ represents exterior product operator,
$$
 \bold{a}_1 ∧ ⋯ ∧\bold{a}_n=det(A) (\bold{e}_1 ∧ ⋯ ∧ \bold{e}_n)
$$

hence

$$
det(A)=(\bold{e}_1 ∧ ⋯ ∧ \bold{e}_n)^{−1}(a_1 ∧ ⋯ ∧ \bold{a}_n)
$$

given orthogonality ($E^{-1}=E^T$):

$$
(\bold{e}_1 ∧ ⋯ ∧ \bold{e}_n)^{−1} = (\bold{e}_1 ⋯ \bold{e}_n)^{−1} = \bold{e}_n ⋯ \bold{e}_1 = \bold{e}_n ∧ ⋯ ∧ \bold{e}_1
$$

Note that $\bold{e}_n ∧ ⋯ ∧ \bold{e}_1$ is a subspace of $a_1 ∧ ⋯ ∧ \bold{a}_n$, we can further write

$$\begin{align*}
det(A)
& =(\bold{e}_n ∧ ⋯ ∧ \bold{e}_1)⋅(a_1 ∧ ⋯ ∧ \bold{a}_n) \\
& =(\bold{e}_n ∧ ⋯ ∧ \bold{e}_2)⋅\big(\bold{e}_1⋅(a_1 ∧ ⋯ ∧ \bold{a}_n)\big) \\
& =(\bold{e}_n ∧ ⋯ ∧ \bold{e}_2)⋅\bigg(a_{1,1}(a_2 ∧ ⋯ ∧ \bold{a}_n)−\sum_{i=2}^n (-1)^i \bold{a}_{1,i}(a_1 ∧ ⋯ ∧ \hat{\bold{a}}_i ∧ ⋯ ∧ \bold{a}_n) \bigg)
\end{align*}
$$

## Orthogonal matrix proof: $A^T$ = $A^{-1}$

Matrix $A$ is orthogonal if the column and row vectors are orthonormal vectors. Define $a_i$ as column vector of $A$, given orthogonality here define:
$$
 \bold{a}_i^T \bold{a}_j =
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

## Eigenvalue and eigenvector

Provided $v$ as basis coordinate vector and $A$ as a transformation $n\times n$ matrix, eigenvalues can be defined as
$$
A\bold{v}= \lambda \bold{v}
$$
where $\lambda$ is eigenvalue and $v$ is eigenvector. Geometrically speaking, applied transformation $A$ to $v$ is same as scaling $v$ by $\lambda$.

This can be written as
$$
(A - \lambda I)\bold{v}= 0
$$

When the determinant of $(A - \lambda I)$ is non-zero, $(A - \lambda I)$ has non-zero solutions of $v$ that satisfies
$$
|A - \lambda I| = 0
$$

## Diagonalization and the eigendecomposition

Given $Q$ as a matrix of eigenvectors $\bold{v}_i$

$$
Q = [\bold{v}_1, \bold{v}_2, \bold{v}_3, ..., \bold{v}_n]
$$

then, take scalar $\lambda_i$ into consideration constructing a diagnal matrix $\Lambda$, and 
$$
A Q = [\lambda_1 \bold{v}_1, \lambda_2 \bold{v}_2, \lambda_3 \bold{v}_3, ..., \lambda_n \bold{v}_n]
$$

$$
A Q = Q \Lambda
$$

$$
A = Q \Lambda Q^{-1}
$$

$A$ can therefore be decomposed into a matrix composed of its eigenvectors, a diagonal matrix with its eigenvalues along the diagonal, and the inverse of the matrix of eigenvectors.

If $Q$ is orthogonal, there is
$$
A = Q \Lambda Q^T
$$
