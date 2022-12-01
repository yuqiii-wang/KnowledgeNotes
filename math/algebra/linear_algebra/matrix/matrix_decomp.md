# Matrix Decomposition

## LU Decomposition/Factorization by Gaussian Elimination

Lower–upper (LU) decomposition or factorization factors a matrix as the product of a lower triangular matrix and an upper triangular matrix.

*Gaussian elimination* can be used for decomposition, since just some proper row and/or column orderings or permutations can achieve LU decomposition.

Let $A$ be a $3 \times 3$ square matrix, LU can be
$$
\begin{bmatrix}
      a_{1,1} & a_{1,2} & a_{1,3} \\
      a_{2,1} & a_{2,2} & a_{2,3} \\
      a_{3,1} & a_{3,2} & a_{3,3}
\end{bmatrix}
=
\begin{bmatrix}
      l_{1,1} & 0 & 0 \\
      l_{2,1} & l_{2,2} & 0 \\
      l_{3,1} & l_{3,2} & l_{3,3}
\end{bmatrix}
\begin{bmatrix}
      u_{1,1} & u_{1,2} & u_{1,3} \\
      0 & u_{2,2} & u_{2,3} \\
      0 & 0 & u_{3,3}
\end{bmatrix}
$$

Define $P$ as a permutation matrix, there is 
$$
PA=LU
$$

### Application

Given a system of linear equations in matrix form $A$ and we want to solve the linear system (find a solution for $\bold{x}$)
$$
A\bold{x}=\bold{b}
$$

By LU decomposition, there is 
$$
PA=LU
$$

So that
$$
LU\bold{x}=P\bold{b}
$$

Given the nature of triangular matrix, by forward and backward substitution, define a temp vector $\bold{y}$, there is
$$
L\bold{y}=P\bold{b}
$$
Then, solve $\bold{y}$, and use $\bold{y}$ to solve $\bold{x}$ by
$$
U\bold{x} = \bold{y}
$$

## Cholesky Decomposition/Factorization

Cholesky decomposition decomposes a Hermitian, positive-definite matrix $A$ into the product of a lower triangular matrix $L$ and its conjugate transpose $L^*$, such as $A=LL^*$ or in real number $A=LL^\text{T}$. 

Every Hermitian positive-definite matrix (and thus also every real-valued symmetric positive-definite matrix) has a unique Cholesky decomposition.

It is useful to solve Linear least squares (LLS) approximation problems.

### Derivation Proof

Define a positive definite matrix $A \in \mathbb{R}^{n \times n}$. It is trivial when $n=1$, and then only consider $n>1$ cases. Partition $A$ as below, where $\~{A} \in \mathbb{R}^{(n-1) \times (n-1)}$, and $\bold{a} \in \mathbb{R}^{n-1}$ is a vector. $\alpha$ is a scalar number.
$$
A = 
\begin{bmatrix}
      \~{A} & \bold{a} \\
      \bold{a}^\text{T} & \alpha
\end{bmatrix}
$$

Since a principal submatrix of a positive definite matrix is positive definite, so that $\~{A}$ is positive definite as well, and $\~{A}$ has Cholesky factorization $\~{A}=\~{L}\~{L}^\text{T}$.

Define $L_1^{-1}:=\begin{bmatrix}      \~{L}^{-1} & 0 \\      0 & 1\end{bmatrix}$, then consider

$$
\begin{align*}
L_1^{-1} A L_1^{-\text{T}} &= 
\begin{bmatrix}
      \~{L}^{-1} & 0 \\
      0 & 1
\end{bmatrix}
\begin{bmatrix}
      \~{A} & \bold{a} \\
      \bold{a}^\text{T} & \alpha
\end{bmatrix}
\begin{bmatrix}
      \~{L}^{-\text{T}} & 0 \\
      0 & 1
\end{bmatrix}
\\ &=
\begin{bmatrix}
      I & \bold{b} \\
      \bold{b}^\text{T} & \alpha
\end{bmatrix}
\\ : &=
B
\end{align*}
$$
where $\bold{b}:=\~{L}^{-1}\bold{a}$.

Then, to eliminate $\bold{b}$, define $L_2^{-1}:=\begin{bmatrix}      I & 0 \\      -\bold{b}^\text{T} & 1\end{bmatrix}$, then consider

$$
\begin{align*}
L_2^{-1} B L_2^{-\text{T}} &= 
\begin{bmatrix}
      I & 0 \\
      -\bold{b}^\text{T} & 1
\end{bmatrix}
\begin{bmatrix}
      I & \bold{b} \\
      \bold{b}^\text{T} & \alpha
\end{bmatrix}
\begin{bmatrix}
      I & -\bold{b} \\
      0 & 1
\end{bmatrix}
\\ &=
\begin{bmatrix}
      I & 0 \\
      0 & \alpha-\bold{b}^\text{T}\bold{b}
\end{bmatrix}
\\ &=
\begin{bmatrix}
      I & 0 \\
      0 & \alpha-\bold{a}^\text{T} \~{L}_1^{-\text{T}}\~{L}_1^{-1} \bold{a}
\end{bmatrix}
\\ &=
\begin{bmatrix}
      I & 0 \\
      0 & \alpha-\bold{a}^\text{T} \~{A}^{-1} \bold{a}
\end{bmatrix}
\end{align*}
$$

Summarize the above equations, there is
$$
L_2^{-1} B L_2^{-\text{T}} =
(L_2^{-1}L_1^{-1}) A (L_1^{-\text{T}}L_2^{-\text{T}})
$$

This is a congruent transformation on a positive definite matrix $A$, so that the diagonal matrix has positive real entries including $\alpha-\bold{a}^\text{T} \~{A}^{-1} \bold{a} > 0$.

Define $\lambda^2 = \alpha-\bold{a}^\text{T} \~{A}^{-1} \bold{a}$, and $L_3:=\begin{bmatrix}      I & 0 \\     0 & \lambda\end{bmatrix}$, so that

$$
\begin{align*}
&&
(L_2^{-1}L_1^{-1}) A (L_1^{-\text{T}}L_2^{-\text{T}}) 
&=
L_3 L_3^\text{T} 
\\ \Rightarrow && 
A
&=
\underbrace{L_1 L_2 L_3}_{:= L} 
\underbrace{L_3^\text{T} L_2^\text{T} L_1^\text{T}}_{:= L^\text{T}}
\\ && &=
L L^\text{T}
\end{align*}
$$
where
$$
\begin{align*}
L=L_1 L_2 L_3 &= 
\begin{bmatrix}
      \~{L} & 0 \\
      0 & 1
\end{bmatrix}
\begin{bmatrix}
      I & 0 \\
      -\bold{b}^\text{T} & 1
\end{bmatrix}
\begin{bmatrix}
      I & 0 \\
      0 & \lambda
\end{bmatrix}
\\ &=
\begin{bmatrix}
      \~{L} & 0 \\
      \bold{b}^\text{T} & \lambda
\end{bmatrix}
\\ &=
\begin{bmatrix}
      \~{L} & 0 \\
      \bold{a}^\text{T}\~{L}^{-\text{T}} & \lambda
\end{bmatrix}
\end{align*}
$$

Hence, $A=LL^\text{T}$ is Cholesky decomposition. Recall that $\~{A} \in \mathbb{R}^{(n-1) \times (n-1)}$, and $n=1$ is trivial proved for $A \in \mathbb{R}^{2 \times 2}$, then $\~{A} \in \mathbb{R}^{2 \times 2}$ can prove for ${A} \in \mathbb{R}^{3 \times 3}$, and recursively onwards so that $A \in \mathbb{R}^{n \times n}$ has Cholesky decomposition.

### Computation by LDLT

Given that $A$ is positive definite, $A$ can be decomposed as below
$$
\begin{align*}
A &= LD L^{-1} = LD L^{\text{T}}
\\ &=
\begin{bmatrix}
      1 & 0 & 0 \\
      L_{21} & 1 & 0 \\
      L_{31} & L_{32} & 1
\end{bmatrix}
\begin{bmatrix}
      D_1 & 0 & 0 \\
      0 & D_2 & 0 \\
      0 & 0 & D_3
\end{bmatrix}
\begin{bmatrix}
      1 & L_{21} & L_{31} \\
      0 & 1 & L_{32} \\
      0 & 0 & 1
\end{bmatrix}
\\ &=
\begin{bmatrix}
      D_1 &  & (\text{symmetric}) \\
      L_{21}D_1 & L^2_{21}D_1+D_2 &  \\
      L_{31}D_1 & L_{31}L_{21}D_1+L_{32}D_2 & L^2_{31}D_1+L^2_{32}D_2+D_3
\end{bmatrix}
\end{align*}
$$
where
$$
\begin{align*}
D_j &= A_{jj} - \sum^{j-1}_{k=1} L^2_{jk} D_k
\\
L_{ij} &= \frac{1}{D_j} \bigg( 
      A_{ij} - \sum^{j-1}_{k=1} L_{ik} L_{jk} D_{k}
\bigg) \quad i>j 
\end{align*}
$$

### Application

Given a system of linear equations in matrix form $A$ ($A$ is symmetric and positive definite) and we want to solve the linear system (find solution for $\bold{x}$)
$$
A\bold{x}=\bold{b}
$$

Given the decomposition $A=LL^*$, define a temp vector $\bold{y}$, $\bold{x}$ can be solved by forward and backward substitution:

1. solve $\bold{y}$ for $L\bold{y}=\bold{b}$
2. solve $\bold{x}$ for $L\bold{x}=\bold{y}$

## QR Decomposition/Factorization

Define a decomposition of a matrix $A$ into a product $A = QR$ of an orthogonal matrix $Q$ and an upper triangular matrix $R$.

It is useful to solve Linear least squares (LLS) approximation problems.

### Gram–Schmidt Process

Define a square matrix $A=[\bold{a}_1, \bold{a}_2, ...,\bold{a}_n]$, where $\bold{a}_k$ represent a column vector of $A$.

Denote a projection operator from vector $\bold{a}$ onto $\bold{u}$:
$$
proj_{\bold{u}}(\bold{a})
=
\frac{\langle \bold{u},\bold{a} \rangle}
{\langle \bold{u},\bold{u} \rangle}\bold{u}
$$
where $\langle\bold{u},\bold{a}\rangle=\bold{u}^\text{T}\bold{a}$ represents inner product operation.

$$
\begin{array}{cc}
    \bold{u}_1 = \bold{a}_1 & 
    \bold{e}_1=\frac{\bold{u}_1}{||\bold{u}_1||}
    \\
    \bold{u}_2 = \bold{a}_2 - proj_{\bold{u}_1}(\bold{a}_2) & 
    \bold{e}_2=\frac{\bold{u}_2}{||\bold{u}_2||}
    \\
    \bold{u}_3 = \bold{a}_3 - proj_{\bold{u}_1}(\bold{a}_3) - proj_{\bold{u}_2}(\bold{a}_3) & 
    \bold{e}_3=\frac{\bold{u}_3}{||\bold{u}_3||}
    \\
    \bold{u}_4 = \bold{a}_4 - proj_{\bold{u}_1}(\bold{a}_4) - proj_{\bold{u}_2}(\bold{a}_4) - proj_{\bold{u}_3}(\bold{a}_4) & 
    \bold{e}_4=\frac{\bold{u}_4}{||\bold{u}_4||}
    \\
    \space
    \\
    ... & ...
    \\
    \space
    \\
    \bold{u}_k = \bold{a}_k - \sum^{k-1}_{j}proj_{\bold{u}_j}(\bold{a}_k) &
    \bold{e}_k=\frac{\bold{u}_k}{||\bold{u}_k||}
\end{array}
$$

The figure below describes the above projection with this notation substitution $\bold{a}_k=\bold{v}_k$ to $proj_{\bold{u}}(\bold{v})$.

<div style="display: flex; justify-content: center;">
      <img src="imgs/Gram–Schmidt_process.svg.png" width="30%" height="30%" alt="Gram–Schmidt_process">
</div>
</br>

Express $\bold{a}_i$ over our newly computed orthonormal basis:
$$
\begin{align*}
      \bold{a}_1 &= \langle\bold{e}_1,\bold{a}_1\rangle \bold{e}_1
      \\
      \bold{a}_2 &= \langle\bold{e}_1,\bold{a}_2\rangle \bold{e}_1 + \langle\bold{e}_2,\bold{a}_2\rangle \bold{e}_2
      \\
      \bold{a}_3 &= \langle\bold{e}_1,\bold{a}_3\rangle \bold{e}_1 + \langle\bold{e}_2,\bold{a}_3\rangle \bold{e}_2 + \langle\bold{e}_3,\bold{a}_3\rangle \bold{e}_3
      \\
      & \vdots
      \\
      \bold{a}_n &= \sum^k_{j=1} \langle\bold{e}_j,\bold{a}_k\rangle \bold{e}_j
\end{align*}
$$

So that $A=QR$, where
$$
\begin{align*}
Q &= [\bold{e}_1, \bold{e}_2, ..., \bold{e}_n]
\\
R &= \begin{bmatrix}
      \langle\bold{e}_1,\bold{a}_1\rangle & 
      \langle\bold{e}_1,\bold{a}_2\rangle &
      \langle\bold{e}_1,\bold{a}_3\rangle &
      ... &
      \langle\bold{e}_1,\bold{a}_n\rangle \\
      0 & 
      \langle\bold{e}_2,\bold{a}_2\rangle &
      \langle\bold{e}_2,\bold{a}_3\rangle &
      ... &
      \langle\bold{e}_2,\bold{a}_n\rangle \\
      0 & 
      0 &
      \langle\bold{e}_3,\bold{a}_3\rangle &
      ... &
      \langle\bold{e}_3,\bold{a}_n\rangle \\
      \vdots & \vdots & \vdots & \ddots & \vdots \\
      0 & 
      0 &
      0 &
      ... &
      \langle\bold{e}_k,\bold{a}_n\rangle \\
\end{bmatrix}
\end{align*}
$$

The Gram-Schmidt algorithms can inherently fall victim to this: column $\bold{u}_j$ is successively reduced in length as components in the directions of $\{ \bold{e}_1, \bold{e}_2, ..., \bold{e}_{j-1} \}$ are subtracted,
leaving a small vector if $\bold{u}_j$ was almost in the span of the first $j$ columns of $A$. 
This is known as *catastrophic cancellation* (loss of precision when the number of projection subtraction grows for $\bold{u}_j$ computation).

### Householder Transformation

A Householder reflection $H$ (or Householder transformation) is a transformation that takes a vector and reflects (a.k.a, orthogonal transformation) it about some plane or hyperplane. 

Reflection takes into account "mirror" as the orthogonal basis change that does not decay as the number projection grows, thereby addressing the issue of catastrophic cancellation.

* Householder Transformation

Define a vector $\bold{z}$ (a.k.a. hyperplane) that is perpendicular to $\bold{u}$ (served as a normal vector to $\bold{z}$). 
Assume there is a vector $\bold{x}$, its projection on $\bold{u}$ is $\langle\bold{u},\bold{x}\rangle \bold{u}=\bold{u}^\text{H} \bold{x} \bold{u}$ (in real realm, there is $\bold{u}^\text{T} \bold{x} \bold{u}$). As a result, $\bold{x}$ can be expressed as $\bold{x}=\bold{z}+\bold{u}^\text{H} \bold{x} \bold{u}$.

The $\bold{x}$'s reflection based on the hyperplane $\bold{z}$ is $(I-2\bold{u}\bold{u}^\text{H})\bold{x}$, and this transformation $H=I-2\bold{u}\bold{u}^\text{H}$ is termed a reflector or Householder transformation.


<div style="display: flex; justify-content: center;">
      <img src="imgs/householder_transform.png" width="30%" height="30%" alt="householder_transform">
</div>
</br>

Easy to observe that

Any vector $\bold{z}$ that is perpendicular to $\bold{u}$ is left unchanged:
$$
\begin{align*}
(I-2\bold{u}\bold{u}^\text{H})\bold{z} &=
\bold{z} - 2\bold{u} \underbrace{\bold{u}^\text{H}\bold{z}}_{=0}
\\ &= \bold{z}
\end{align*}
$$

Any vector $\bold{x}$ can be written as $\bold{x}=\bold{z}+\bold{u}^\text{H} \bold{x} \bold{u}$, whose Householder transform/reflection can be computed as
$$
\begin{align*}
(I-2\bold{u}\bold{u}^\text{H})\bold{x} &=
(I-2\bold{u}\bold{u}^\text{H}) (\bold{z}+\bold{u}^\text{H} \bold{x} \bold{u})
\\ &=
\bold{z} + \bold{u}^\text{H} \bold{x} \bold{u} 
-2\bold{u} \underbrace{\bold{u}^\text{H}\bold{z}}_{=0} -2\bold{u}\bold{u}^\text{H} \bold{u}^\text{H} \bold{x} \bold{u}
\\ &=
\bold{z} + \bold{u}^\text{H} \bold{x} \bold{u} 
- \underbrace{
      2\bold{u}\bold{u}^\text{H} \bold{u}^\text{H} \bold{x} \bold{u}}_{= 2 \bold{u}^\text{H}\bold{x} \underbrace{
            \bold{u}^\text{H} \bold{u}}_{=1} \bold{u}}
\\ &=
\bold{z} + \bold{u}^\text{H} \bold{x} \bold{u} 
-2\bold{u}^\text{H} \bold{x} \bold{u} 
\\ &=
\bold{z} - \bold{u}^\text{H} \bold{x} \bold{u} 
\end{align*}
$$

Householder transformation $H$ is an orthogonal symmetric matrix
$$
\begin{align*}
\\ H &=
I-2\bold{u}\bold{u}^\text{H}
\\ &=
I-2\bold{u}^\text{H}\bold{u}
\\ &=
(I-2\bold{u}\bold{u}^\text{H})^\text{H}
\\ &=
H^\text{H}
\\ \space \\
HH^\text{H} &= 
(I-2\bold{u}\bold{u}^\text{H})
(I-2\bold{u}\bold{u}^\text{H})^\text{H}
\\ &=
(I-2\bold{u}\bold{u}^\text{H})
(I-2\bold{u}^\text{H}\bold{u})
\\ &=
I - 2 \underbrace{\bold{u}^\text{H}\bold{u}}_{=\bold{u}\bold{u}^\text{H}}
-2\bold{u}\bold{u}^\text{H}
+ 4\bold{u}\bold{u}^\text{H} \underbrace{\bold{u}^\text{H}\bold{u}}_{=1}
\\ &=
I
\end{align*}
$$

* Householder Decomposition Computation

Householder decomposition at the $i$-th step, multiplies an orthogonal matrix $Q_i$ to $A \in \mathbb{R}^{m \times n}$, so that the new matrix sees zeros located at $i+1$ to $m$ rows on the $i$-th column.

Householder transformation helps find this $Q_i$.

Define $A=[\bold{a}_1, \bold{a}_2, ..., \bold{a}_n]$. Start computation at $A$'s first column $\bold{a}_1$ with the first Householder reflection $H_1$

Since $H_1$ is orthogonal having $det(H_1)=\pm 1$, there is $||H_1\bold{a}_1||=||\bold{a}_1||$. Then define $\bold{e}_1=[1 \quad 0 \quad ... \quad 0]^\text{T} \in \mathbb{R}^n$ as a unit orthogonal vector such that $||\bold{e}_1||=1$. There is $H_1\bold{a}_1=\pm ||\bold{a}_1||\bold{e}_1$.

Recall Householder transformation:
$$
\left\{
      \begin{align*}
      H_1 &= I-2\bold{u}_1\bold{u}_1^\text{H}
      \\
      \bold{u}_1 &= \frac{\bold{x}_1}{||\bold{x}_1||}
      \end{align*}
\right.
$$

So that
$$
H_1 \bold{a}_1 =
( I-2\bold{u}_1\bold{u}_1^\text{H} ) \bold{a}_1 = 
\bold{a}_1 - 2 \frac{\bold{x}_1}{||\bold{x}_1||} (\bold{u}_1^\text{H} \bold{a}_1)
$$

Set $\bold{x}_1 = \beta\bold{a}_1 + \alpha\bold{e}_1$, there is
$$
\begin{align*}
H_1 \bold{a}_1 &=
\bold{a}_1 - 2 
\frac{(\beta\bold{a}_1 + \alpha\bold{e}_1)^\text{H} \bold{a}_1 }{(\beta\bold{a}_1 + \alpha\bold{e}_1)^2} (\beta\bold{a}_1 + \alpha\bold{e}_1)
\\ &=
\bold{a}_1 - 2 
\frac{(\beta\bold{a}_1 + \alpha\bold{e}_1)^\text{H} \bold{a}_1 }
{\beta^2\bold{a}^\text{H}_1\bold{a}_1 + \alpha^2\bold{e}^\text{H}_1\bold{e}_1+2\alpha\beta\bold{a}_1\bold{e}_1} 
(\beta\bold{a}_1 + \alpha\bold{e}_1)
\\ &=
\bigg(
  1- 2 
  \frac{\beta(\beta\bold{a}_1 + \alpha\bold{e}_1)^\text{H} \bold{a}_1 }
  {\beta^2\bold{a}^\text{H}_1\bold{a}_1 + \alpha^2\bold{e}^\text{H}_1\bold{e}_1+2\alpha\beta\bold{a}_1\bold{e}_1} 
\bigg)\bold{a}_1
-
\bigg(
  2 
  \frac{\alpha(\beta\bold{a}_1 + \alpha\bold{e}_1)^\text{H} \bold{a}_1 }
  {\beta^2\bold{a}^\text{H}_1\bold{a}_1 + \alpha^2\bold{e}^\text{H}_1\bold{e}_1+2\alpha\beta\bold{a}_1\bold{e}_1} 
\bigg) \bold{e}_1
\end{align*}
$$

For $\bold{e}_1$ and $\bold{a}_1$ are linearly independent (if not, just use $\bold{e}_1$ to represent $\bold{a}_1$, no need of $H_1$ computation), 
and $H_1\bold{a}_1=\pm ||\bold{a}_1||\bold{e}_1$, there are
$$
\begin{align*}
&\space
\left\{
\begin{align*}
1 - 2 
  \frac{\beta(\beta\bold{a}_1 + \alpha\bold{e}_1)^\text{H} \bold{a}_1 }
  {\beta^2\bold{a}^\text{H}_1\bold{a}_1 + \alpha^2\bold{e}^\text{H}_1\bold{e}_1+2\alpha\beta\bold{a}_1\bold{e}_1} 
&= 0
\\ 
- 2 
  \frac{\alpha(\beta\bold{a}_1 + \alpha\bold{e}_1)^\text{H} \bold{a}_1 }
  {\beta^2\bold{a}^\text{H}_1\bold{a}_1 + \alpha^2\bold{e}^\text{H}_1\bold{e}_1+2\alpha\beta\bold{a}_1\bold{e}_1} 
&= \pm ||\bold{a}_1||
\end{align*}
\right.
\\ \Rightarrow & \quad
\beta^2 \bold{a}^\text{H}_1 \bold{a} = \alpha^2
\end{align*}
$$

By taking $\beta=1$, there is $\alpha=\pm ||\bold{a}_1||$.

So that, $\bold{u}_1=\frac{\bold{a}_1 \pm ||\bold{a}_1||\bold{e}_1}{\big|\big| \bold{a}_1 \pm ||\bold{a}_1||\bold{e}_1 \big|\big|}$. Just to take the sign of $\bold{a}_1$ to replace $\pm$, then derive the final $\bold{u}_1$ such as
$$
\bold{u}_1=
\frac{\bold{a}_1 \space sgn(\bold{a}_1) ||\bold{a}_1||}
{\big|\big| \bold{a}_1 \space sgn(\bold{a}_1) ||\bold{a}_1||\bold{e}_1 \big|\big|}
\bold{e}_1
$$

Hence, the $H_1$ is
$$
\begin{align*}
H_1 &= I-2\bold{u}_1\bold{u}_1^\text{H}
\\ &=
I - 2 
\bigg( 
  \frac{\bold{a}_1 \space sgn(\bold{a}_1) ||\bold{a}_1||}
  {\big|\big| \bold{a}_1 \space sgn(\bold{a}_1) ||\bold{a}_1||\bold{e}_1 \big|\big|}\bold{e}_1
\bigg)
\bigg( 
  \frac{\bold{a}_1 \space sgn(\bold{a}_1) ||\bold{a}_1||}
  {\big|\big| \bold{a}_1 \space sgn(\bold{a}_1) ||\bold{a}_1||\bold{e}_1 \big|\big|} \bold{e}_1
\bigg)^\text{H}
\end{align*}
$$

The first Householder transformation can be written as
$$
H_1 A = 
\big[ H_1 \bold{a}_1 
\quad
H_1 A_1' \big]
=
\begin{bmatrix}
      * & * \\
      0 & \~{A_2}
\end{bmatrix}
$$
where $A'_1$ represents the $2$-th to $n$-th columns of $A$, and $*$ represents some computed scalar numbers.

Repeat the above process (set $\~{\bold{e}}_2=[1 \quad 0 \quad ... \quad 0]^\text{T} \in \mathbb{R}^{n-1}$) to find the Householder transformation $\~{H}_2$ corresponding to $\~{A_2} \in \mathbb{R}^{(n-1) \times (n-1)}$, 
and mark $H_2=\begin{bmatrix}      1 & 0 \\      0 & \~{H_2}\end{bmatrix}$, 
and define $\~{\bold{a}}_2 \in \mathbb{R}^{n-1}$ as the first column of $\~{A_2}$.
Then, the second Householder transformation result is
$$
H_2 H_1 A
=
\begin{bmatrix}
      * & * & * \\
      0 & * & * \\
      0 & 0 & \~{A_3}
\end{bmatrix}
$$

The sequence of operations goes on, hence, the Householder QR decomposition is
$$
\begin{align*}
&&
\underbrace{H_n H_{n-1} ... H_2 H_1}_{=Q^\text{H}} A &= R
\\ \Rightarrow && 
Q^\text{H}A &= R
\\ \Rightarrow &&
A &= QR
\end{align*}
$$

### Gram–Schmidt Process vs Householder

Given that Householder transformation $\~{H}_i$ is orthogonal having $det(\~{H}_i)=\pm 1$, and there is $\~{H}_i \~{\bold{a}}_i=\pm ||\~{\bold{a}}_i||\~{\bold{e}}_i$, Householder transformation $\~{H}_i$ maintains the length of each $\~{\bold{a}}_i$ at to each orthogonal basis vector $\~{\bold{e}}_i$. 

This is in stark contrast to the transformation by Gram–Schmidt process that $\bold{e}_k=\frac{\bold{u}_k}{||\bold{u}_k||}$ sees worsening loss of precision as dimensions grow for $\bold{u}_k$. As a result, $\langle\bold{e}_k,\bold{a}_n\rangle$ in $R$ might not retain full information.

### QR vs LU decomposition

For the full-rank case LU factorization is cheaper.  

QR is good for rank-deficient least-squares cases, since each QR step solves for one column, disregarding the number of rows. In least squares cases, often $rowNum \gg colNum$.