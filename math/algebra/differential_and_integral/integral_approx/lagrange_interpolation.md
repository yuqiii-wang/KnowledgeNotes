# Lagrange Interpolation

A short summary: Lagrange interpolation considers monomial as the bases to form a unique polynomial of degree $n$ that passes through a number of $n+1$ points $\mathbf{x}=[x_0, x_1, . . . , x_n]$.

## Definition

An interpolation polynomial is defined as $p_n(x) \in \mathbb{R}^n$ that satisfies $p_n(x\_i)=y_i$ for $i=0,1,2,...,n$.
The points $x_0, x_1, ..., x_n$ are called *interpolation points*.

If the interpolation points $x_0, x_1, ..., x_n$ are distinct, then the process of finding a polynomial that passes through the points $(x\_i, y_i)$ for $i=0,1,2,...,n$ is equivalent to solving a system of linear equations $A\mathbf{x}=\mathbf{b}$ that has a unique solution.

Lagrange interpolation approach defines $b_i=y_i$ and $a_{ij}=p_j(x\_i)$ for $i=0,1,2,...,n$, and uses *monomial basis* $\{1,x,x^2,...,x^n\}$ as the basis for the polynomial space. The corresponding matrix $A$ is called *Vandermonde matrix*.

$$
A = Vandermonde([x_0, x_1, . . . , x_n]) =
\begin{bmatrix}
    1 & x_0 & x_0^2 & ... & x_0^n \\
    1 & x_1 & x_1^2 & ... & x_2^n \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    1 & x_n & x_n^2 & ... & x_n^n \\
\end{bmatrix}
$$

$p_n(x)$ can be written as
$$
p_n(x)=
\sum^n\_{j=0} y_i L\_{n,j}(x)
$$
where polynomials $\{L\_{n,j}\}$ for $j=0,1,2,...,n$ are called *Lagrange Polynomials* for the interpolation points $x_0, x_1, ..., x_n$
$$
L\_{n,j}(x) =
\prod_{k=0, k \ne j}^n \frac{x-x_k}{x_j-x_k}
$$

## Example

Find the unique polynomial $p_3(x)$ of degree $3$ or less that satisfies the below points.

|$i$|$x\_i$|$y_i$|
|-|-|-|
|$0$|$-1$|$3$|
|$1$|$0$|$-4$|
|$2$|$1$|$5$|
|$3$|$2$|$6$|

This is identical to find $p_3(x)$ so that $p_3(-1)=3$, $p_3(0)=-4$, $p_3(1)=5$ and $p_3(2)=6$.
In Vandermonde matrix expression, there is
$$
\begin{align*}
A\mathbf{x} &= \mathbf{b}
\\
\begin{bmatrix}
1 & x_0 & x_0^2 & x_0^3 \\
1 & x_1 & x_1^2 & x_1^3 \\
1 & x_2 & x_2^2 & x_2^3 \\
1 & x_3 & x_3^2 & x_3^3 
\end{bmatrix}
\begin{bmatrix}
w_0 \\
w_1 \\
w_2 \\
w_3 \\
\end{bmatrix}&=
\begin{bmatrix}
y_0 \\
y_1 \\
y_2 \\
y_3 \\
\end{bmatrix}
\end{align*}
$$
where $A$ is the Vandermonde matrix and $\mathbf{x}=\{w_0, w_1, w_2, w_3\}$ is the assigned weights.

Compute and find the solution
$$
\begin{align*}
L\_{3,0}(x) &=
\prod_{k=0, k \ne j}^3 \frac{x-x_k}{x_j-x_k}
\\ &= 
\frac{(x-x_1)(x-x_2)(x-x_3)}{(x_0-x_1)(x_0-x_2)(x_0-x_3)}
\\ &= 
\frac{(x-0)(x-1)(x-2)}{(-1-0)(-1-1)(-1-2)}
\\ &=
-\frac{1}{6} (x^3-3x^2+2x)
\\
L\_{3,1}(x) &=
\prod_{k=0, k \ne j}^3 \frac{x-x_k}{x_j-x_k}
\\ &= 
\frac{(x-x_1)(x-x_2)(x-x_3)}{(x_1-x_1)(x_1-x_2)(x_1-x_3)}
\\ &= 
\frac{(x+1)(x-1)(x-2)}{(0+1)(0-1)(0-2)}
\\ &=
\frac{1}{2} (x^3-2x^2-x+2)
\\
L\_{3,2}(x) &=
\prod_{k=0, k \ne j}^3 \frac{x-x_k}{x_j-x_k}
\\ &= 
\frac{(x-x_1)(x-x_2)(x-x_3)}{(x_1-x_1)(x_1-x_2)(x_1-x_3)}
\\ &= 
\frac{(x+1)(x+0)(x-2)}{(1+1)(1+0)(1-2)}
\\ &=
-\frac{1}{2} (x^3-x^2-2x)
\\
L\_{3,3}(x) &=
\prod_{k=0, k \ne j}^3 \frac{x-x_k}{x_j-x_k}
\\ &= 
\frac{(x-x_1)(x-x_2)(x-x_3)}{(x_1-x_1)(x_1-x_2)(x_1-x_3)}
\\ &= 
\frac{(x+1)(x+0)(x-1)}{(2+1)(2+0)(2-1)}
\\ &=
-\frac{1}{6} (x^3-x)
\end{align*}
$$

By taking the sum of the above expressions
$$
\begin{align*}
p_3(x)&=
\sum^3_{j=0} y_i L\_{n,j}(x)
\\ &=
y_0 L\_{3,0}(x) + y_1 L\_{3,1}(x) + y_2 L\_{3,2}(x) + y_3 L\_{3,3}(x)
\\ &=
-6x^3+8x^2+7x-4
\end{align*}
$$