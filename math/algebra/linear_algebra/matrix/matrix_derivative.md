# Matrix Derivative

Given a matrix $A\in \mathbb{R}^{m \times n}$, whose columns range from $\bold{a}_1$ to $\bold{a}_n$ and rows range from $\~{\bold{a}}^\text{T}_1$ to $\~{\bold{a}}^\text{T}_m$, and its entries are denoted as $a_{ij}$.

Matrix multiplication $AA^\text{T}$ has the following relationship.
$$
\begin{align*}
AA^\text{T} &=
\begin{bmatrix}
\sum^n_{p=1} a_{p1}a_{p1} & \sum^n_{p=1} a_{p1}a_{p2} & ... & \sum^n_{p=1} a_{p1}a_{pn} \\
\sum^n_{p=1} a_{p2}a_{p1} & \sum^n_{p=1} a_{p2}a_{p2} & ... & \sum^n_{p=1} a_{p2}a_{pn} \\
\vdots & \vdots & \ddots & \vdots \\
\sum^n_{p=1} a_{pm}a_{p1} & \sum^n_{p=1} a_{pm}a_{p2} & ... & \sum^n_{p=1} a_{pm}a_{pn}
\end{bmatrix}
\end{align*}
$$
whose elements can be expressed as 
$$
(AA^\text{T})_{ij} =
\sum^n_{p=1} a_{pi}a_{pj}
$$

## Gradient of Linear and Quadratic Functions

### Linear Function Derivative

Consider $Ax$, where $A\in \mathbb{R}^{m \times n}$ and $x \in \mathbb{R}^n$, that the gradient on $x$ is defined
$$
\begin{align*}

\nabla Ax 
&=
\begin{bmatrix}
\nabla \~{\bold{a}}^\text{T}_1 x \\
\nabla \~{\bold{a}}^\text{T}_2 x \\
\vdots \\
\nabla \~{\bold{a}}^\text{T}_m x \\
\end{bmatrix}

\\ &=
[\~{\bold{a}}_1, \~{\bold{a}}_2, ..., \~{\bold{a}}_m]

\\ &=
A^\text{T}
\end{align*}
$$

### Quadratic Function Derivative

Consider $x^\text{T}Ax$ for $A \in \mathbb{R}^{n \times n}$ and $x \in \mathbb{R}^n$, such that
$$
\begin{align*}
x^\text{T} A x 
&=
x^\text{T} 
[\~{\bold{a}}^\text{T}_1 x, \~{\bold{a}}^\text{T}_2 x, ..., \~{\bold{a}}^\text{T}_n x]^\text{T}
\\ &=
x_1 \~{\bold{a}}^\text{T}_1 x +
x_2 \~{\bold{a}}^\text{T}_2 x +
... +
x_n \~{\bold{a}}^\text{T}_n x
\end{align*}
$$

Take the derivative with respect to $x_l$ (a scalar value, not a vector). For each $\~{\bold{a}}_i$, the $l$-th element is $a_{il}$, hence, derived the term $x_l \~{\bold{a}}_l^\text{T}x$, and
$$
\begin{align*}

\frac{\partial }{\partial x_l} x^\text{T} A x
&= 
\frac{\partial}{\partial x_l}x_1 \~{\bold{a}}^\text{T}_1 x +
\frac{\partial}{\partial x_l}x_2 \~{\bold{a}}^\text{T}_2 x +
... +
\frac{\partial}{\partial x_l}x_n \~{\bold{a}}^\text{T}_n x

\\ &=
x_1 a_{1l} +
x_2 a_{2l} +
... +
\frac{\partial}{\partial x_l}x_l \~{\bold{a}}^\text{T}_l x +
... +
x_n a_{nl}

\\ &=
\sum^n_{i=1} x_i a_{il} + \~{\bold{a}}_l^\text{T}x

\\  &=
\bold{a}_l^\text{T} x + \~{\bold{a}}_l^\text{T}x
\end{align*}
$$

Apply $x_l$ to all $x \in \mathbb{R}^n$, there is
$$
\nabla_x x^\text{T} A x
=
Ax + A^\text{T} x
$$

## Derivative and Trace

Given the definition of derivative of a vectored function $\bold{f}$ with vectored input $\bold{x}$:
$$
\bold{f}(\bold{x}+d\bold{x})
=
\bold{f}(\bold{x}) +
\bold{f}'(\bold{x})d\bold{x} +
\bold{f}''(\bold{x})d\bold{x}^2 +
...
$$
where $\bold{f}'$ is the derivative/Jacobian.

Define $A \in \mathbb{R}^{m \times m}$ and $X \in \mathbb{R}^{m \times n}$, there exists this relationship
$$
\begin{align*}

\frac{tr(AdX)}{dX} &=
\frac{
    tr \begin{bmatrix}
        \~{\bold{a}}^\text{T}_1 d \bold{x}_1 & \~{\bold{a}}^\text{T}_1 d \bold{x}_2 & & \\
        \~{\bold{a}}^\text{T}_2 d \bold{x}_1 & \~{\bold{a}}^\text{T}_2 d \bold{x}_2 & & \\
        & & \ddots & \\
        & & & \~{\bold{a}}^\text{T}_n d \bold{x}_n \\
    \end{bmatrix}
}{dX}

\\ &=
\frac{
    \sum^n_{i=1} \~{\bold{a}}^\text{T}_i d \bold{x}_i
}{dX}
\end{align*}
$$
where $tr$ denotes the trace operation.

For the $ij$-th element, there is
$$
\begin{align*}

\bigg[
    \frac{tr(AdX)}{dX} 
\bigg]_{ij}
&=
\bigg[
\frac{
    \sum^n_{i=1} \~{\bold{a}}^\text{T}_i d \bold{x}_i
}{dX}
\bigg]_{ij}

\\ &=
\frac{
    \sum^n_{i=1} \~{\bold{a}}^\text{T}_i d \bold{x}_i
}{\partial x_{ij}}

\\ &=
a_{ij}
\end{align*}
$$

So that for all $a_{ij}$ in $A$,
$$
\frac{tr(AdX)}{dX}
=
A
$$

### Derivative of product in trace

There exists $\nabla_A tr(AB) = B^\text{T}$. Here is the proof

Define $A \in \mathbb{R}^{n \times m}$ and $B \in \mathbb{R}^{k \times n}$
$$
\begin{align*}
tr(AB) &= 
tr \bigg(
    \begin{bmatrix}
    \bold{a_1} \\
    \bold{a_2} \\
    \vdots \\
    \bold{a_n}
\end{bmatrix}
\begin{bmatrix}
    \bold{b_1} \\
    \bold{b_2} \\
    \vdots \\
    \bold{b_n}
\end{bmatrix}^\text{T}
\bigg)

\\ &=
tr \begin{bmatrix}
    \bold{a_1}\bold{b_1}^\text{T} & \bold{a_1}\bold{b_2}^\text{T} & &  \\
    \bold{a_2}\bold{b_1}^\text{T} & \bold{a_2}\bold{b_2}^\text{T} & & \\
    & & \ddots & \\
    & & & \bold{a_n}\bold{b_n}^\text{T}
\end{bmatrix}

\\ &=
\sum^n_i \bold{a_i}\bold{b_i}^\text{T}

\end{align*}
$$

So that (notice here the subscripts of $a_{ij}$ and $b_{ji}$ are in a reverse order. 
$$
\frac{\partial tr(AB)}{\partial a_{ij}}
=
b_{ji}
$$

To expand to all elements, there is $\nabla_A tr(AB) = B^\text{T}$. 