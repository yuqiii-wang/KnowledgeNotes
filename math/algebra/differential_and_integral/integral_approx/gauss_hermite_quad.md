# Gauss–Hermite Quadrature

A quadrature rule is an approximation of the definite integral of a function, usually stated as a weighted sum of function values at specified points within the domain of integration. 

In Gauss-Hermite quadrature, instead of having set $w(x)=1$ such as the Legendre method's, Hermite sets $w(x)=e^{-x^2}$, and the integration interval is $(-\infty, +\infty)$

$$
\begin{align*}
\int_a^b w(x) f(x) dx
&=
\int_{-\infty}^{+\infty} e^{-x^2} f(x) dx
\\ &\approx
\sum_{i=1}^n w_i f(x_i)
\end{align*}
$$

An $n$-point ($x_i$ for $i=0,1,...,n$) Gaussian quadrature rule measures polynomials of degree $2n-1$ or less.

* Benefits and intuition

A typical Gaussian distribution probability density function is infinite-order differentiable, hence eligible to be approximated by polynomials.

Just give $n$ points, Gauss-Hermite quadrature can interpolate the Gaussian distribution with $2n-1$ degree polynomial precision.

## Hermite polynomials

*physicist's Hermite polynomials*:

$$
h_n(x) =
(-1)^n e^{x^2}
\frac{d^n}{dx^n} e^{-x^2}
$$

### Orthogonality

Orthogonality of two functions is defined as
$$
<f,g>=
\int f(x) g(x) dx = 0
$$

$h_n(x)$ for $n=0,1,2,...$ are orthogonal with respect to the weight function $w(x)=e^{-x^2}$, so that

$$
\int_{-\infty}^{+\infty} 
h_m(x) h_n(x) w(x) dx 
= 0
\quad
\forall \space m \ne n
$$

### First few Hermite polynomials

$$
\begin{align*}

h_0(x) &= 1
\\
h_1(x) &= 2x
\\
h_2(x) &= 4x^2-2
\\
h_3(x) &= 8x^3-12x
\\
h_4(x) &= 16x^4-48x^2+12

\end{align*}
$$

##  Gauss–Hermite quadrature definition

Gauss–Hermite quadrature is a form of Gaussian quadrature for approximating the value of integrals of the following kind (in contrast to Legendre polynomial that $e^{-x^2}$ is replaced with $1$):
$$
\int_{-\infty}^{+\infty} 
e^{-x^2} f(x) dx
\approx
\sum_{i=1}^n 
w_i f(x_i)
$$
where, given $x_i^*$ are the roots of physicist's Hermite polynomial $h_{n}(x)$, $w_i$ can be expressed as
$$
w_i =
\frac{
    2^{n-1} n! \sqrt{\pi}
}{
    n^2 \big(
        h_{n-1}(x^*_i)
        \big)^2
}
$$

## Example: two point approximation

Define $x_i^*$ as the roots of Hermite polynomials. For $n=2$, there is $h_2(x) = 4x^2-2$, whose zeros are $x_0^*=\frac{1}{\sqrt{2}}$ and $x_1^*=-\frac{1}{\sqrt{2}}$

$$
\begin{align*}
\int_a^b w(x) f(x) dx
&=
\int_{-\infty}^{+\infty} e^{-x^2} f(x) dx
\\ &\approx
\sum_{i=1}^n w_i f(x_i^*)
\\ &=
w_0 f(x_0^*) + w_1 f(x_1^*)
\\ &=
w_0 f(\frac{1}{\sqrt{2}}) + w_1 f(-\frac{1}{\sqrt{2}})
\end{align*}
$$

Set $n=2$, and $x_0^*=\frac{1}{\sqrt{2}}$ and $x_1^*=-\frac{1}{\sqrt{2}}$ to compute $w_i$
$$
w_i =
\frac{
    2^{n-1} n! \sqrt{\pi}
}{
    n^2 \big(
        h_{n-1}(x^*_i)
        \big)^2
}
$$
There are 
$$
\begin{align*}
w_0 &=
\frac{
    2^{2-1} 2! \sqrt{\pi}
}{
    2^2 \big(
        h_{2-1}(\frac{1}{\sqrt{2}})
        \big)^2
}
& w_1 &=
\frac{
    2^{2-1} 2! \sqrt{\pi}
}{
    2^2 \big(
        h_{2-1}(-\frac{1}{\sqrt{2}})
        \big)^2
}
\\ &=
\frac{4\sqrt{\pi}}{4 (\sqrt{2})^2}
& &=
\frac{4\sqrt{\pi}}{4 (-\sqrt{2})^2}
\\ &=
\frac{\sqrt{\pi}}{2}
& &=
\frac{\sqrt{\pi}}{2}

\end{align*}
$$

Finally,
$$
\begin{align*}
\int_a^b w(x) f(x) dx
&=
\int_{-\infty}^{+\infty} e^{-x^2} f(x) dx
\\ &\approx
w_0 f(\frac{1}{\sqrt{2}}) + w_1 f(-\frac{1}{\sqrt{2}})
\\ &=
\frac{\sqrt{\pi}}{2} f(\frac{1}{\sqrt{2}})
+ 
\frac{\sqrt{\pi}}{2} f(-\frac{1}{\sqrt{2}})
\end{align*}
$$


## Example: Gaussian distribution transformation

Consider a function $f(y)$, where the variable y is Normally distributed: $y \sim N(\mu, \sigma^2)$. $h$ can be linear or non-linear transformation of the normal distribution input $y$.

Its expectation is
$$
E\big(f(y)\big)
=
\int_{-\infty}^{+\infty}
\frac{1}{\sigma \sqrt{2\pi}}
e^{-\frac{(y-\mu)^2}{2\sigma^2}}
f(y) dy
$$ 

To make the integral compliant with the form of Hermite polynomial by
$$
\begin{align*}
x&=
\frac{y-\mu}{\sqrt{2}\sigma}
\\
y&=
\sqrt{2} \sigma x + \mu
\end{align*}
$$
and by $y=\phi(x)$, then
$$
\int_a^b g(y) dy = 
\int_{\phi^{-1}(a)}^{\phi^{-1}(b)}
g(\phi(x))\phi'(x) dx
$$

Coupled with the integration by substitution,
$$
\begin{align*}

E\big(f(y)\big)
&=
\int_{-\infty}^{+\infty}
\frac{1}{\sigma \sqrt{2\pi}}
e^{-\frac{(y-\mu)^2}{2\sigma^2}}
f(y) dy
\\ &=
\int_{-\infty}^{+\infty}
\frac{1}{\sqrt{\pi}}
e^{-x^2}
f(\sqrt{2} \sigma x + \mu)
dx
\\ &\approx
\frac{1}{\sqrt{\pi}}
\sum_{i=1}^n w_i f(\sqrt{2} \sigma x_i + \mu)

\end{align*}
$$

