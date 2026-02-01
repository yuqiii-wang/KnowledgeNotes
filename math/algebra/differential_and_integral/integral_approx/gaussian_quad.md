# Gaussian quadrature

A quadrature rule is an approximation of the definite integral of a function, usually stated as a weighted sum of function values $\sum^n\_{i=1} w_i f(x\_i)$ at specified points $x\_i$ within the domain of integration $[a,b]$. 

Gaussian quadrature rule can yield an exact result for polynomials of degree $2n+1$ or less by a suitable choice of the nodes $x\_i$ and weights $w_i$ for $i=0,1,...,n$.

* Intuition

Assume a function $f$ is a polynomial or continuous differentiable on an interval $[a,b]$ at most $2n-1$ degrees. $f$ can be decomposed to two orthogonal polynomials's product $p(x)q(x)$ and the rest is called remainder $r(x)$. 

$r(x)$ can be computed by Lagrange polynomial interpolation, while $p(x)q(x)$ is cancelled out to $0$.

* Benefits:

We use only $n$ nodes to estimate $2n-1$ power of polynomial exactly.  If we want to calculate a $19$-power polynomial we only need $10$ nodes.

## Definition

Integral of a function $h$ can be approximated via the form ($w(x)$ and $w_i(x)$ refer to two different things)
$$
\int^b_a w(x) f(x) dx
\approx
\sum_{i=0}^n w_i(x) f_i(x) 
$$

Some choices of $w(x)$ and $f_i(x)$ are

|Interval|$w(x)$|Orthogonal Polynomials|
|-|-|-|
|$[-1,1]$|$1$|Legendre Polynomials|
|$(-1,1)$|$(1-x)^\alpha (1+x)^\beta$|Jacobi Polynomials|
|$[-\infty,+\infty]$|$e^{x^2}$|Hermite Polynomials|


## Derivations

Assume $f$ is a polynomial of degree at most $2n + 1$, define $q(x)$ and $r(x)$ have degrees at most $n$, $f$ can be decomposed as the below expression.
$$
f(x) = q(x)p(x)+r(x)
$$
where $r(x)$ is the remainder after $f(x)$ being approximated by $p(x) q(x)$. For example, By Legendre polynomial interpolation, there are $p(x)=L_n(x)$ and $q(x)=L_m(x)$.

$r(x)$ can be interpolated via Lagrange polynomials $l_i(x)$ such as

$$
\begin{align*}
r(x) &=
\sum_{i=1}^n l_i(x) r(x\_i)
\\\\ &=
\sum_{i-1}^n
  \prod_{i \ne j} \frac{x-x_j}{x\_i-x_j}
  r(x\_i)
\end{align*}
$$  

Take $x\_i^*$ as the zeros of $p(x)$, then $p(x\_i^*)q(x\_i^*)=0$ for $i=0,1,...,n$, hence

$$
\begin{align*}
f(x\_i^*) &= q(x\_i^*)p(x\_i^*)+r(x\_i^*)
\\\\ &=r(x\_i^*)
\end{align*}
$$

For the integral, by orthogonality $p(x\_i^*)q(x\_i^*)=0$ (such as by Legendre polynomials $\int L_n(x)L_m(x)=0$ for $m\ne m$) and $r(x)$ as the remainder polynomial of at most $n$ degree interpolated by Lagrange polynomial interpolation, there is 
$$
\begin{align*}
\int^b_a f(x)w(x)dx&=
\int^b_a \big(
    q(x)p(x)+r(x) w(x)
    \big)
    dx
\\\\ &=
0 + \int^b_a r(x)w(x) dx
\\\\ &=
\int^b_a r(x)w(x) dx
\\\\ &=
\int^b_a w(x) \sum_{i-1}^n l_i(x) r(x\_i^*)
 dx
\\\\ &=
\int^b_a r(x\_i^*)
\sum_{i-1}^n l_i(x) w(x)
\\\\ &=
\sum^n\_{i=0} w_i r(x\_i^*)
\\\\ &=
\sum^n\_{i=0} w_i f(x\_i^*)
\end{align*}
$$

## Example by Legendre polynomials

For $n=2$ on the interval $[-1,1]$, approximate the integral of a function $f(x)$ such as $\int^1_{-1} f(x)dx$ to be a precision level of degree $3$. The approximation method is by Legendre polynomials.

Given Legendre polynomial approximation, there is
$$
\int^1_{-1} 1 \cdot f(x)dx
\approx
w_0 f(x_0^*) + w_1 f(x_1^*)
$$

Given the employment of Legendre polynomials, define $x_0^*$ and $x_1^*$ as the roots of $p_2(x) = \frac{1}{2}( x^2 - \frac{1}{3} ) = 0$, there are
$$
x_0^*=\frac{1}{\sqrt{3}} 
\quad 
x_1^*=-\frac{1}{\sqrt{3}} 
$$

By Lagrange polynomial interpolation, there are

$$
\begin{align*}
\int^1_{-1} 1 \space dx&= w_0 + w_1
\\\\ &= 2 \\\\
\int^1_{-1} x \space dx&= w_0 x_0^* + w_1 x_1^*
\\\\ &= 0
\end{align*}
$$ 

Solve $w_i$ such as

$$
\begin{bmatrix}
1 & 1 \\\\
x_0^* & x_1^*
\end{bmatrix}
\begin{bmatrix}
w_0 \\\\
w_1
\end{bmatrix}=
\begin{bmatrix}
2 \\\\
0
\end{bmatrix}
$$

$w_0=1$ and $w_1=1$ are the solutions. So that

$$
\begin{align*}
\int^1_{-1} f(x)dx
& \approx
w_0 f(x_0^*) + w_1 f(x_1^*)
\\\\ &=
1 \cdot f(\frac{1}{\sqrt{3}}) + 1 \cdot f(-\frac{1}{\sqrt{3}}) 
\end{align*}
$$

This result is of precision polynomial degree $3$.

Below experiments tested three functions 
* $f(x)=e^x$
* $f(x)=3x^2+2x+1$
* $f(x)=5x^4+4x^3+3x^2+2x+1$
 
that $f(x)=3x^2+2x+1$ yields an exact result.

### If $f(x)=e^x$

By approximation: 
$$
\begin{align*}
\int^1_{-1} f(x)dx
&\approx
e^{(\frac{1}{\sqrt{3}})}+e^{(-\frac{1}{\sqrt{3}})}
\\\\ &=
2.3426960879097307
\end{align*}
$$

It true value form and result

$$
\begin{align*}
\int^1_{-1} f(x)dx
& =
e+e^{-1}
\\\\ &=
2.350402387287603
\end{align*}
$$

### If $f(x)=3x^2+2x+1$

By approximation: 

$$
\begin{align*}
\int^1_{-1} f(x)dx
&\approx
f(\frac{1}{\sqrt{3}}) +f(-\frac{1}{\sqrt{3}}) 
\\\\ &=
3.154700538379252+0.8452994616207485
\\\\ &=
4.0.
\end{align*}
$$

It true value form and result

$$
\begin{align*}
\int^1_{-1} f(x)dx
& =
x^3+x^2+x
\\\\ &=
4
\end{align*}
$$

### If $f(x)=5x^4+4x^3+3x^2+2x+1$

By approximation: 
$$
\begin{align*}
\int^1_{-1} f(x)dx
&\approx
f(\frac{1}{\sqrt{3}}) +f(-\frac{1}{\sqrt{3}}) 
\\\\ &=
4.480056452854309+0.6310546582568031
\\\\ &=
5.111111111111113
\end{align*}
$$

It true value form and result

$$
\begin{align*}
\int^1_{-1} f(x)dx
& =
x^5+x^4+x^3+x^2+x
\\\\ &=
6
\end{align*}
$$