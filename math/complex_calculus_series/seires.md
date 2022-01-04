# Series

## Series of Power over Factorial Converges

For $x=0$ there is 
$$
\forall n \geq 1 : \frac{0^n}{n!}=0
$$

For $x \neq 0$, there is
$$
lim_{n \rightarrow \infty} \bigg| \frac{\frac{x^{n+1}}{(x+1)!}}{\frac{x^{n}}{x!}}\bigg| = \frac{|x|}{n+1} = 0
$$

Hence by the Ratio Test: $\sum_{n=0}^{\infty} \frac {x^n}{n!}$ converges. 

## Taylor-Maclaurin series expansion

Let $f$ be a real or complex-valued function which is smooth on the open interval, so Taylor-Maclaurin series expansion of $f$ is: 
$$
\sum_{n=0}^{\infty} \frac {f^{(n)}(a)}{n!} (x-a)^n
$$

When $a=0$, the above expression is called Maclaurin series expansion:
$$
\sum_{n=0}^{\infty} \frac {f^{(n)}(0)x^n}{n!}
$$

Taylor series draw inspiration from generalization of the Mean Value Theorem: given $f: [a,b]$ in $\R$ is differentiable, there exists $c \in (a,b)$ such that
$$
\frac{f(b)-f(a)}{b-a} = f'(c) 
$$

hence
$$\begin{align*}
f(b)
& = f(a) + f'(c)(b-a) \\
& = f(a) + \frac{f'(a)}{1}{(b-a)^1} + \frac{f''(c)}{2}(b-a)^2 \\
& = f(a) + \frac{f'(a)}{1}{(b-a)^1} + \frac{f''(c)}{2}(b-a)^2 + ... + \frac {f^{(n)}(c)}{n!} (b-a)^n
\end{align*}$$

### Approximation
If the Taylor series of a function is convergent over $[a,b]$, its sum is the limit of the infinite sequence of the Taylor polynomials.

## Power Series Expansion for Sine/Cosine Function

For $m \in \N$ and $k \in \Z$:
$$\begin{align*}
m = 4k : \frac{d^m}{dx^m}cos(x) = cos(x) \\
m = 4k + 1 : \frac{d^m}{dx^m}cos(x) = -sin(x) \\
m = 4k + 2 : \frac{d^m}{dx^m}cos(x) = -cos(x) \\
m = 4k + 3 : \frac{d^m}{dx^m}cos(x) = sin(x)
\end{align*}$$

Take into consideration Taylor-Maclaurin series expansion:
$$\begin{align*}
sin(x) 
& = \sum_{k=0}^{\infty} \bigg( \frac{x^{4k}}{(4k)!}cos(0) - \frac{x^{4k+1}}{(4k+!)!}sin(0) - \frac{x^{4k+2}}{(4k+2)!}cos(0) + \frac{x^{4k+3}}{(4k+3)!}sin(0)\bigg) \\
& = \sum_{n=0}^{\infty} (-1)^{n} \frac{x^{2n+1}}{(2n+1)!}
\end{align*}$$
where $n=2k$.

Similarly, cosine is derived as below
$$
cos(x) = \sum_{n=0}^{\infty} (-1)^{n} \frac{x^{2n}}{(2n)!}
$$

## Exponential function $e^x$ by Maclaurin series expansion (Euler's formula)

* Euler's formula:
$$\begin{align*}
e^x = 
& \sum_{n=0}^{\infty} \frac{x^n}{n!} \\
&  = cos(x) + i \space sin(x)
\end{align*}$$
where $i$ represents imaginary part of a complex number.