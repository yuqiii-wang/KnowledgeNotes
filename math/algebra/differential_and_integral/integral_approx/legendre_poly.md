
# Legendre polynomials

The polynomials are defined as an orthogonal complete system with the below definitions:

$$
L_n (x) = 
\frac{1}{2^n n!} \cdot 
\frac{d^n}{dx^n} \big(
              (x^2-1)^n
              \big)
$$
or recursively starting with $L_0(x)=1, L_1(x)=x$, there is
$$
(n+1)L\_{n+1}(x) = 
(2n+1)x L_n (x) - n L\_{n-1}(x)
$$

For example, for $n=1$, there is
$$
\begin{align*}
(1+1) L_2(x) &= (2+1)x L_1(x) - L_0(x)
\\
2 L_2(x) &= (3)x^2 - 1
\\
L_2(x) &= \frac{3}{2}x^2 - \frac{1}{2}
\end{align*}
$$ 

Solve the above equations by $L_n(x)=0$ with different degree $n$ and obtain roots $x\_i$ 

For weights $w_i$, there is
$$
w_i=
\frac{2}{(1-x\_i^2)\big(L_n'(x\_i)\big)^2}
$$ 

Here is a table of result summary.

|$n$|$x\_i$|$w_i$|
|-|-|-|
|$1$|$0$|$2$|
|$2$|$\frac{1}{\sqrt{3}}, -\frac{1}{\sqrt{3}}$|$1,1$|
|$3$|$0,\sqrt{\frac{3}{5}}, -\sqrt{\frac{3}{5}}$|$\frac{8}{9},\frac{5}{9},\frac{5}{9}$|
|$4$|$\sqrt{\frac{3}{7}-\frac{2}{7}\sqrt{\frac{6}{5}}},-\sqrt{\frac{3}{7}-\frac{2}{7}\sqrt{\frac{6}{5}}}, \sqrt{\frac{3}{7}+\frac{2}{7}\sqrt{\frac{6}{5}}},-\sqrt{\frac{3}{7}+\frac{2}{7}\sqrt{\frac{6}{5}}}$|$\frac{18+\sqrt{30}}{36},\frac{18+\sqrt{30}}{36},\frac{18-\sqrt{30}}{36},\frac{18-\sqrt{30}}{36}$|
||$\vdots$||

## Orthogonality

Orthogonality of two functions is defined as
$$
<f,g>=
\int f(x) g(x) dx = 0
$$

Legendre polynomials are orthogonal. 
For example, given a space $\mathbf{L}=\{1,x,\frac{3}{2}x^2 - \frac{1}{2}\}$, 

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

Given $L_i$ 's definitions $L_0(x) = 1, L_2(x) = \frac{3}{2}x^2 - \frac{1}{2}$

and the integral
$$
\begin{align*}
\int^1_{-1} L_0(x) \times L_2(x) dx &=
\int^1_{-1} 1 \cdot (\frac{3}{2}x^2 - \frac{1}{2}) dx 
\\ &= 0
\end{align*}
$$
hence perpendicular to each other $L_0(x) \perp L_2(x)$
 
3. For $L_1$ and $L_2$

Given $L_i$ 's definitions $L_1(x) = x, L_2(x) = \frac{3}{2}x^2 - \frac{1}{2}$

and the integral
$$
\begin{align*}
\int^1_{-1} L_1(x) \times L_2(x) dx &=
\int^1_{-1} x (\frac{3}{2}x^2 - \frac{1}{2}) dx 
\\ &= 0
\end{align*}
$$
hence perpendicular to each other $L_1(x) \perp L_2(x)$

## Change of interval

Change of interval rule allows mapping between $[a,b]$ and $[-1,1]$

$$
\int^b_a f(x) dx=
\int^{1}\_{-1} 
f(\frac{b-a}{2} \xi \frac{a+b}{2}) \frac{dx}{d \xi} d \xi
$$
where $\frac{dx}{d \xi}=\frac{b-a}{2}$

## First few Legendre polynomials

$$
\begin{align*}
l_0(x) &= 1
\\
l_1(x) &= x
\\
l_2(x) &= \frac{1}{2} (3x^2-1)
\\
l_3(x) &= \frac{1}{2} (5x^3-3x)
\\
l_4(x) &= \frac{1}{8} (35x^4-30x^2+3)
\end{align*}
$$