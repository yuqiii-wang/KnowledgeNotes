# Positive-definite matrix

Given an $n \times n$ matrix $A$:
$$
A=
\begin{bmatrix}
a_{1,1} & a_{1,2} & ... & a_{1,n} \\
a_{2,1} & a_{2,2} & ... & a_{2,n} \\
... & ... & ... & ... \\
a_{n,1} & a_{n,2} & ... & a_{n,n} \\
\end{bmatrix}
$$ 

$A$ is positive-definite if the real number $x^\text{T} A x$ is positive for every nonzero real column vector $x$:
$$
x^\text{T}Ax>0 \quad \forall x \in \mathbb{R}^n \setminus \{0\}
$$

$A$ is positive-semi-definite if if the real number $x^\text{T} A x$ is positive or zero:
$$
x^\text{T}Ax \ge 0 \quad \forall x \in \mathbb{R}^n 
$$

$A$ is positive-definite if it satisfies any of the following equivalent conditions.

* $A$ is congruent (exists an invertible matrix $P$ that $P^\text{T}AP=B$) with a diagonal matrix with positive real entries.
* $A$ is symmetric or Hermitian, and all its eigenvalues are real and positive.
* $A$ is symmetric or Hermitian, and all its leading principal minors are positive.
* There exists an invertible matrix $B$ with conjugate transpose $B^*$ such that $A=BB^*$

## Use in optimization

If $A$ is positive-definite, $x^\text{T}Ax$ has global minima solution $x^*$ (a convex).

If $A$ neither satisfies $x^\text{T}Ax>0$ nor $x^\text{T}Ax<0$, there exist saddle points.