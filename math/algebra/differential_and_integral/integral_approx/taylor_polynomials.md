# Approximating Integrals using Taylor Polynomials

Define $n$-th order Taylor polynomial $T_n$ of $f(x) \in \mathbb{R}^n$ at a point $c$ is
$$
T_n(f)(x)=
\sum^n\_{k=1} \frac{f^{(k)}(c)}{k!} (x-c)^k
$$

When $x$ is very near to $c$, there is $T_n(f)(x) \approx f(x)$.

Define $n$-th order remainder $R_n$ of $f(x)$ after removed of $T_n(f)(x)$ as one degree higher than $T_n(f)(x)$ (this demands $f(x)$ should be at least $n+1$ order differentiable)

$$
\begin{align*}
R_n(f)(x) &=
f(x) - T_n(f)(x)
\\\\ & \approx
\int^x_c \frac{f^{(n+1)}(y)}{n!} (x-y)^n dy
\end{align*}
$$

When $R_n(f)(x) \approx 0$, $T_n(f)(x)$ is a good approximation to $f(x)$.

## Example

Approximate $\int^{\frac{1}{3}}_0 e^{-x^2}$ to within $10^{-6}$ precision.

Solution: find $T_n(f)(x)$ that satisfies $|\int^{\frac{1}{3}}_0 R_n(x) dx| < 10^{-6}$
$$
\int^{\frac{1}{3}}_0 e^{-x^2}=
\int^{\frac{1}{3}}_0 T_n(x) dx 
+
\int^{\frac{1}{3}}_0 R_n(x) dx 
$$