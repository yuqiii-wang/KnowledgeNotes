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

### Intuition about $e^{Ax}$ 

Given $e^{Ax}$ where $A$ is a matrix, for example
$$
e^{A_{3 \times 3}
}
=
e^{
\begin{bmatrix}
      a_{1,1} & a_{1,2} & a_{1,3} \\
      a_{1,2} & a_{2,2} & a_{2,3} \\
      a_{1,3} & a_{3,2} & a_{3,3}
\end{bmatrix}
}
$$

Take Euler's formula into consideration
$$
e^x = 
\sum_{n=0}^{\infty} \frac{x^n}{n!}
$$

Derived with $n \rightarrow \infty$:
$$
e^{A_{3 \times 3}x} =
x^0 {\begin{bmatrix}
      a_{1,1} & a_{1,2} & a_{1,3} \\
      a_{1,2} & a_{2,2} & a_{2,3} \\
      a_{1,3} & a_{3,2} & a_{3,3}
\end{bmatrix}}^0
+
x^1 {\begin{bmatrix}
      a_{1,1} & a_{1,2} & a_{1,3} \\
      a_{1,2} & a_{2,2} & a_{2,3} \\
      a_{1,3} & a_{3,2} & a_{3,3}
\end{bmatrix}}^1
+ \\
\frac{x^2}{2}
{\begin{bmatrix}
      a_{1,1} & a_{1,2} & a_{1,3} \\
      a_{1,2} & a_{2,2} & a_{2,3} \\
      a_{1,3} & a_{3,2} & a_{3,3}
\end{bmatrix}}^2
+ ... +
\frac{x^n}{n!}
{\begin{bmatrix}
      a_{1,1} & a_{1,2} & a_{1,3} \\
      a_{1,2} & a_{2,2} & a_{2,3} \\
      a_{1,3} & a_{3,2} & a_{3,3}
\end{bmatrix}}^n
$$

### Derivatives

For $x(t) \in \mathbb{R}$ being a linear function, $a \in \mathbb{R}$, the derivative of $x(t)$ can be expressed as
$$
x'(t) = ax(t)
$$

hence the integral:
$$
x(t) = e^{at}x_0
$$

Remember
$$
e^{at} = 
\sum_{n=0}^{\infty} \frac{(at)^n}{n!} \\
=1 + at + \frac{(at)^2}{2!} + \frac{(at)^3}{3!} + ...
$$

This holds true for matrix as well
$$
x'(t) = Ax(t)
$$

$$
x(t) = e^{At}x_0
$$

## Riemann Series Theorem (also known as Riemann Rearrangement Theorem)

If an infinite series of real numbers is conditionally convergent, then its terms can be arranged in a permutation so that the new series converges to an arbitrary real number, or diverges. 

* Conditional Convergence: sum of series is a finite number disregarding its element rearrangement:

Example: alternating harmonic series
$$
\begin{align*}
& 1 - \frac{1}{2} + \frac{1}{3} - \frac{1}{4} + \frac{1}{5} - ... = \ln 2
\\
& 1 + (- \frac{1}{2} + \frac{1}{3}) + (- \frac{1}{4} + \frac{1}{5}) - ... = \frac{3}{2} \ln 2
\end{align*}
$$

Example: alternating sign unit sum
$$
\begin{align*}
& 1 - 1 + 1 - 1 + 1 - ... = 0
\\
& 1 + (- 1 + 1) + (- 1 + 1) - ... = 1
\end{align*}
$$

* The new series converges to an arbitrary real number, or diverges

For example, for $1 - 1 + 1 - 1 + 1 - ...$, the result can be an arbitrary real number.
$$
\begin{align*}
& 1 - 1 + 1 - 1 + 1 - ... = 0
\\
& 1 + (- 1 + 1) + (- 1 + 1) - ... = 1
\\
& 1 + 1 + 1 + 1 + (-1 - 1 - 1 - 1) + (1 + 1 + 1 + 1) - ... = 4
\\
& -1 - 1 - 1 - 1 + (1 + 1 + 1 + 1) + (-1 - 1 - 1 - 1) - ... = -4
\end{align*}
$$

* Funny Application: given two sets: $S_1=[-1, 0]$ and $S_2=[0, 1]$, the sum of the two sets might not be zero, but undefined: $\sum_{x_{1i} \in S_1}x_{1i} + \sum_{x_{2i} \in S_2}x_{2i}=\text{arbitrary}$

Intuition: 

Set $x_1 \in [-1, 0)$, there is $0.9 - \frac{x_1}{10} \in (0.9, 1]$.

So that $x_1 + 0.9 - \frac{x_1}{10} \in [0, 0.9)$ is greater than/equal to zero.

Easy to say the sum of elements in $[-1, 0)$ and $(0.9, 1]$ is greater than/equal to zero.

Consider all elements in $(0, 0.9]$ are greater than zero, so that for $S_2=(0, 0.9] \cup (0.9, 1]$, easy to say $\sum_{x_{1i} \in S_1}x_{1i} + \sum_{x_{2i} \in S_2}x_{2i} > 0$.

Actually by Riemann Series Theorem, the $\sum_{x_{1i} \in S_1}x_{1i} + \sum_{x_{2i} \in S_2}x_{2i}$ is undefined.