# Least Squares Problem

Given an objective function $f(\bold{x})$ to minimize:
$$
\underset{\bold{\phi}}{min} \space f(\bold{x})=\frac{1}{2}\sum^m_{j=1}{r}_j^2(\bold{x})
$$
where $r_j$ is a smooth function from $\mathbb{R}^n \rightarrow \mathbb{R}$ given $\bold{x} \in \mathbb{R}^n$. For least squares problem, there is $m \ge n$.

Residual $r_j$ is the error between function $\phi_j(\bold{x})$ and ground truth observation $y_j$:
$$
r_j(\bold{x}) = y_j - \phi_j(\bold{x})
$$

Assemble all $r_j$ to $\bold{r}$, define a mapping $\mathbb{R}^n \rightarrow \mathbb{R}^m$ as follows
$$
\bold{r}(\bold{x})
=
(r_1(\bold{x}), r_2(\bold{x}), ..., r_m(\bold{x}))^\text{T}
$$

Rewrite $f$ to $f(\bold{x})=\frac{1}{2}||\bold{r}(\bold{x})||^2_2$, whose Jacobian is
$$
J(\bold{x})
=
\big[\frac{\partial r_j}{\partial x_i}\big]
_{
    \begin{array}{c}
    \footnotesize{j=1,2,3,...,m}
    \\
    \footnotesize{i=1,2,3,...,n}
    \end{array}
}
=
\begin{bmatrix}
    \triangledown r_1(\bold{x})^\text{T}
    \\
    \triangledown r_2(\bold{x})^\text{T}
    \\
    \triangledown r_3(\bold{x})^\text{T}
    \\
    ...
    \\
    \triangledown r_m(\bold{x})^\text{T}
\end{bmatrix}
$$

Hence,
$$
\begin{align*}
\triangledown f(\bold{x})
&=
\sum^m_{j=1} r_j(\bold{x}) \triangledown r_j(\bold{x})
\\ &=
J(\bold{x}^\text{T}) \bold{r}(\bold{x})
\end{align*}
$$

Its second degree derivative (Hessian) is
$$
\begin{align*}

\triangledown^2f(\bold{x})
&=
\sum^m_{j=1} \big( r_j(\bold{x}) \triangledown r_j(\bold{x})\big)'
\\ &=
\sum^m_{j=1} r_j'(\bold{x}) \triangledown r_j(\bold{x}) + r_j(\bold{x}) \triangledown r_j'(\bold{x})
\\ &=
\sum^m_{j=1} \triangledown r_j(\bold{x}) \triangledown r_j(\bold{x})^\text{T}
+
\sum^m_{j=1} r_j(\bold{x}) \triangledown^2 r_j(\bold{x})
\\ &=
J(\bold{x})^\text{T} J(\bold{x}) + \sum^m_{j=1} r_j(\bold{x}) \triangledown^2 r_j(\bold{x})
\end{align*}
$$

## Linear Least Squares Problem


### Over-determined vs under-determined

We have a $m \times n$ linear system matrix $A$ and $m \times 1$ vector $\bold{b}$ such as
$$
A\bold{x}=\bold{b}
$$

If $m = n$, the solution is $\bold{x}=A^{-1}\bold{b}$ 

If $m > n$, there are more equations than unknown $\bold{x}$, solution to $\bold{x}$ is over-determined

If $m < n$, there are less equations than unknown $\bold{x}$, solution to $\bold{x}$ is under-determined

### Residuals as linears

Given residual expression $r_j(\bold{x}) = y_j - \phi_j(\bold{x})$, if $r_j$ is linear ($\phi_j(\bold{x})$ is represented in linear forms), the minimization becomes a *linear least squares problem*. Residual can be expressed as
$$
\bold{r}(\bold{x})
=
A\bold{x} - \bold{y}
$$

For convenience, replace $\bold{y}$ with $\bold{b}$, so that $\bold{r}(\bold{x}) = A\bold{x} - \bold{b}$

### Solution

Since $A\bold{x}=\bold{b}$ is over-determined, it cannot be solved directly. Define $\hat{\bold{x}}$ as the solution when $\bold{e}=\bold{b}-A\bold{x}$ is small enough.

Given 
$$
\begin{align*}
\bold{r}^2(\bold{x}) 
&= 
(A\bold{x}-\bold{b})^\text{T}(A\bold{x}-\bold{b})
\\ &=
\big((A\bold{x})^\text{T}-\bold{b}^\text{T}\big)(A\bold{x}-\bold{b})
\\ &=
(A\bold{x})^\text{T}(A\bold{x})-(A\bold{x})^\text{T}\bold{b}-(A\bold{x})\bold{b}^\text{T}+\bold{b}^\text{T}\bold{b}
\end{align*}
$$

Both $A\bold{x}$ and $\bold{b}$ are vectors, by the rule $a^\text{T}b=b^\text{T}a$, where $a$ and $b$ are vectors, so that
$$
\begin{align*}
\bold{r}^2(\bold{x}) 
&=
(A\bold{x})^\text{T}(A\bold{x})-2(A\bold{x})^\text{T}\bold{b}+\bold{b}^\text{T}\bold{b}
\end{align*}
$$

Minimized $\bold{e}$ should see $\frac{\partial \bold{r}^2(\bold{x})}{\partial \bold{x}}=0$, so that
$$
\begin{align*}
\frac{\partial \bold{r}^2(\bold{x})}{\partial \bold{x}}
&= 0 
\\
2A^\text{T}A{\bold{x}} - 2A^\text{T}\bold{b} &= 0
\\
A^\text{T}A{\bold{x}} &= A^\text{T}\bold{b}
\\
\bold{x}=\frac{A^\text{T}\bold{b}}{A^\text{T}A}
\end{align*}
$$

When $A$ is 
* each column is linearly independent
* $A^\text{T}A$ is invertible


 
