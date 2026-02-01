# Least Squares Problem

Given an objective function $f(\mathbf{x})$ to minimize:
$$
\underset{\mathbf{\phi}}{min} \space f(\mathbf{x})=\frac{1}{2}\sum^m_{j=1}{r}_j^2(\mathbf{x})
$$
where $r_j$ is a smooth function from $\mathbb{R}^n \rightarrow \mathbb{R}$ given $\mathbf{x} \in \mathbb{R}^n$. For least squares problem, there is $m \ge n$.

Residual $r_j$ is the error between function $\phi_j(\mathbf{x})$ and ground truth observation $y_j$:

$$
r_j(\mathbf{x}) = y_j - \phi_j(\mathbf{x})
$$

Assemble all $r_j$ to $\mathbf{r}$, define a mapping $\mathbb{R}^n \rightarrow \mathbb{R}^m$ as follows
$$
\mathbf{r}(\mathbf{x})=
(r_1(\mathbf{x}), r_2(\mathbf{x}), ..., r_m(\mathbf{x}))^\text{T}
$$

Rewrite $f$ to $f(\mathbf{x})=\frac{1}{2}||\mathbf{r}(\mathbf{x})||^2_2$, whose Jacobian is
$$
J(\mathbf{x})=
\big[\frac{\partial r_j}{\partial x\_i}\big]
_{
    \begin{array}{c}
    \footnotesize{j=1,2,3,...,m}
    \\
    \footnotesize{i=1,2,3,...,n}
    \end{array}
}=
\begin{bmatrix}
    \nabla r_1(\mathbf{x})^\text{T}
    \\
    \nabla r_2(\mathbf{x})^\text{T}
    \\
    \nabla r_3(\mathbf{x})^\text{T}
    \\
    ...
    \\
    \nabla r_m(\mathbf{x})^\text{T}
\end{bmatrix}
$$

Hence,
$$
\begin{align*}
\nabla f(\mathbf{x})&=
\sum^m_{j=1} r_j(\mathbf{x}) \nabla r_j(\mathbf{x})
\\ &=
J(\mathbf{x}^\text{T}) \mathbf{r}(\mathbf{x})
\end{align*}
$$

Its second degree derivative (Hessian) is
$$
\begin{align*}
\nabla^2f(\mathbf{x})&=
\sum^m_{j=1} \big( r_j(\mathbf{x}) \nabla r_j(\mathbf{x})\big)'
\\ &=
\sum^m_{j=1} r_j'(\mathbf{x}) \nabla r_j(\mathbf{x}) + r_j(\mathbf{x}) \nabla r_j'(\mathbf{x})
\\ &=
\sum^m_{j=1} \nabla r_j(\mathbf{x}) \nabla r_j(\mathbf{x})^\text{T} +
\sum^m_{j=1} r_j(\mathbf{x}) \nabla^2 r_j(\mathbf{x})
\\ &=
J(\mathbf{x})^\text{T} J(\mathbf{x}) + \sum^m_{j=1} r_j(\mathbf{x}) \nabla^2 r_j(\mathbf{x})
\end{align*}
$$

## Linear Least Squares Problem


### Over-determined vs under-determined

We have a $m \times n$ linear system matrix $A$ and $m \times 1$ vector $\mathbf{b}$ such as
$$
A\mathbf{x}=\mathbf{b}
$$

If $m = n$, the solution is $\mathbf{x}=A^{-1}\mathbf{b}$ 

If $m > n$, there are more equations than unknown $\mathbf{x}$, solution to $\mathbf{x}$ is over-determined

If $m < n$, there are less equations than unknown $\mathbf{x}$, solution to $\mathbf{x}$ is under-determined

### Residuals as linears

Given residual expression $r_j(\mathbf{x}) = y_j - \phi_j(\mathbf{x})$, if $r_j$ is linear ($\phi_j(\mathbf{x})$ is represented in linear forms), the minimization becomes a *linear least squares problem*. Residual can be expressed as
$$
\mathbf{r}(\mathbf{x})=
A\mathbf{x} - \mathbf{y}
$$

For convenience, replace $\mathbf{y}$ with $\mathbf{b}$, so that $\mathbf{r}(\mathbf{x}) = A\mathbf{x} - \mathbf{b}$

### Solution

Since $A\mathbf{x}=\mathbf{b}$ is over-determined, it cannot be solved directly. Define $\hat{\mathbf{x}}$ as the solution when $\mathbf{e}=\mathbf{b}-A\mathbf{x}$ is small enough.

Given 
$$
\begin{align*}
\mathbf{r}^2(\mathbf{x}) &= 
(A\mathbf{x}-\mathbf{b})^\text{T}(A\mathbf{x}-\mathbf{b})
\\ &=
\big((A\mathbf{x})^\text{T}-\mathbf{b}^\text{T}\big)(A\mathbf{x}-\mathbf{b})
\\ &=
(A\mathbf{x})^\text{T}(A\mathbf{x})-(A\mathbf{x})^\text{T}\mathbf{b}-(A\mathbf{x})\mathbf{b}^\text{T}+\mathbf{b}^\text{T}\mathbf{b}
\end{align*}
$$

Both $A\mathbf{x}$ and $\mathbf{b}$ are vectors, by the rule $a^\text{T}b=b^\text{T}a$, where $a$ and $b$ are vectors, so that
$$
\begin{align*}
\mathbf{r}^2(\mathbf{x}) &=
(A\mathbf{x})^\text{T}(A\mathbf{x})-2(A\mathbf{x})^\text{T}\mathbf{b}+\mathbf{b}^\text{T}\mathbf{b}
\end{align*}
$$

Minimized $\mathbf{e}$ should see $\frac{\partial \mathbf{r}^2(\mathbf{x})}{\partial \mathbf{x}}=0$, Set the optimal as $\mathbf{x}^*$, so that
$$
\begin{align*}
\frac{\partial \mathbf{r}^2(\mathbf{x})}{\partial \mathbf{x}}&= 0 
\\
2A^\text{T}A{\mathbf{x}^*} - 2A^\text{T}\mathbf{b} &= 0
\\
A^\text{T}A{\mathbf{x}^*} &= A^\text{T}\mathbf{b}
\\
\mathbf{x}^*=\frac{A^\text{T}\mathbf{b}}{A^\text{T}A}
\end{align*}
$$

When $A$ is 
* each column is linearly independent
* $A^\text{T}A$ is invertible

Then, for $A^\text{T}A{\mathbf{x}^*} = A^\text{T}\mathbf{b}$, can use typical matrix decomposition methods such as Cholesky decomposition to solve this linear equations for $\mathbf{x}^*$
 
