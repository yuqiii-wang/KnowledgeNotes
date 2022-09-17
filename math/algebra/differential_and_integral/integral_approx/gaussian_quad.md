# Gaussian quadrature

A quadrature rule is an approximation of the definite integral of a function, usually stated as a weighted sum of function values $\sum^n_{i=1} w_i f(x_i)$ at specified points $x_i$ within the domain of integration $[a,b]$. 

Gaussian quadrature rule can yield an exact result for polynomials of degree $2n+1$ or less by a suitable choice of the nodes $x_i$ and weights $w_i$ for $i=0,1,...,n$.

* Intuition and use case



* Benefits:

We use only $n$ nodes to estimate $2n-1$ power of polynomial exactly.  If we want to calculate a $19$-power polynomial we only need $10$ nodes.

## Definition

Let $p_{n+1}$ be a nonzero polynomial of degree $n + 1$ and $w$ a positive weight function, $x^k$ is the degree promotion term, such that:
$$
\int^b_a
x^k p_{n+1}(x) w(x) dx = 0
\quad k=0,1,2,...,n
$$

The above expression is equivalent to the below given a $q(x)$ of degree at most $n$
$$
\int^b_a
p_{n+1}(x) q(x) w(x) dx = 0
$$

Define $f$ as a polynomial of degree at most $2n + 1$, there is
$$
f(x) = p(x) q(x) + r(x)
$$
where $r(x)$ is the remainder after $f(x)$ being approximated by $p(x) q(x)$.

Denotes $x_i^*$ for $i=0,1,..,n$ as the zeros of $q$, there is
$$
\int^b_a f(x) w(x) dx
\approx
\sum^n_{i=0} A_i f(x_i^*)
$$
where 
$$
A_i = \int^b_a l_i(x) w(x) dx 
\quad i=0,1,...,n
$$
where $l_i$ are Lagrange interpolating polynomials. 

The above approximation is exact for all polynomials of degree at most $2n+1$.

## Derivations

Assume $f$ is a polynomial of degree at most $2n + 1$, and show
$$
\sum^n_{i=0} A_i f(x_i)
=
\int^b_a f(x) w(x) dx 
$$
where $p(x)$ has degree $n+1$, and $q(x)$ and $r(x)$ have degrees at most $n$
$$
f(x) = q(x)p(x)+r(x)
$$

Take $x_i^*$ as the zeros of $p(x)$, then $p(x_i^*)q(x_i^*)=0$ for $i=0,1,...,n$, hence
$$
f(x_i^*)=r(x_i^*)
$$

For the integral, by orthogonality $p(x_i^*)q(x_i^*)=0$ and $r(x)$ as the remainder polynomial of at most $n$ degree interpolated by Lagrange interpolation, there is 
$$
\begin{align*}
\int^b_a f(x)w(x)dx
&=
\int^b_a \big(
    q(x)p(x)+r(x) w(x)
    \big)
    dx
\\ &=
0 + \int^b_a r(x)w(x) dx
\\ &=
\int^b_a r(x)w(x) dx
\\ &=
\sum^n_{i=0} A_i r(x_i^*)
\end{align*}
$$