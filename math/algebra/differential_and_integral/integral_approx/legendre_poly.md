
# Legendre polynomials

The polynomials are defined as an orthogonal system with respect to the weight function $w(x)=1$ over the interval $[-1,1]$, and $L_n$ is a polynomial of degree $n$

$$
\int^1_{-1} L_m(x) L_n(x) dx = 0
\quad n \ne m
$$

## Orthogonality

Orthogonality of two functions is defined as
$$
<f,g>=
\int f(x) g(x) dx = 0
$$

Legendre polynomials are orthogonal. 
For example, given a space $\bold{L}=\{1,x,x^2\}$, 

1. For $L_0$ and $L_1$

Given $L_i$ 's definitions $L_0(x) = 1, L_1(x) = x$

and the integral
$$
\begin{align*}
\int^1_{-1} L_0(x) \times L_1(x) dx &=
\int^1_{-1} x dx 
\\ &= 0
\end{align*}
$$

hence perpendicular to each other $
L_0(x) \perp L_1(x)
$
 
2. For $L_0$ and $L_2$

Given $L_i$ 's definitions $L_0(x) = 1, L_2(x) = x^2$

and the integral
$$
\begin{align*}
\int^1_{-1} L_0(x) \times L_2(x) dx &=
\int^1_{-1} x^2 dx 
\\ &= 0
\end{align*}
$$
hence perpendicular to each other $L_0(x) \perp L_2(x)$
 
3. For $L_1$ and $L_2$

Given $L_i$ 's definitions $L_1(x) = x, L_2(x) = x^2 $

and the integral
$$
\begin{align*}
\int^1_{-1} L_1(x) \times L_2(x) dx &=
\int^1_{-1} x^3 dx 
\\ &= 0
\end{align*}
$$
hence perpendicular to each other $L_1(x) \perp L_2(x)$

## Change of interval

Change of interval rule allows mapping between $[a,b]$ and $[-1,1]$

$$
\int^b_a f(x) dx
=
\int^{1}_{-1} 
f(\frac{b-a}{2} \xi \frac{a+b}{2}) \frac{dx}{d \xi} d \xi
$$
where $\frac{dx}{d \xi}=\frac{b-a}{2}$