# Differential and Integral

## Differential vs Derivative

Derivative refers to "rate of change" while differential refers to "trivial amount".

### Partial Derivative

When input/output is $\mathbb{R}^1 \rightarrow \mathbb{R}^1$ such that $w=f\big(g(x)\big)$, there is a chian rule:
$$
\frac{dw}{dx}=\frac{df}{dg}\frac{dg}{dx}
$$

When $w=f\big(a(x),b(x),c(x)\big)$, the chain rule becomes

$$
\frac{dw}{dx}=
\frac{\partial w}{\partial a}\frac{d a}{d x} +
\frac{\partial w}{\partial b}\frac{d b}{d x} +
\frac{\partial w}{\partial c}\frac{d c}{d x}
$$

In this expression, $\frac{\partial w}{\partial x_i}dx_i$ are called *partial differentials*.

## Gradient vs Derivative

*Directional derivative* refers to "total" derivative.

For example, suppose $z=f(x,y)$ is a function of two variables with a domain of $D$. Set $\overrightarrow{\bold{u}}=\cos\theta \overrightarrow{\bold{i}}+\sin\theta \overrightarrow{\bold{j}}$, the directional derivative of $f$ in the direction of $\overrightarrow{\bold{u}}$ is as below.

$$
\begin{align*}
D_{\overrightarrow{\bold{u}}} f(x,y) &=
\lim_{h \rightarrow 0} \frac{f(x+h\cos\theta, y+h\sin\theta) - f(x,y)}{h}
\\ &=
f_x(x,y)\cos\theta + f_y(x,y)\sin\theta
\end{align*}
$$

where $f_x$ and $f_y$ are the partial derivatives, respectively.

Directional derivative can be decomposed to
$D_{\overrightarrow{\bold{u}}} f(x,y)=\overrightarrow{\bold{\nabla}} f(x,y) \cdot \overrightarrow{\bold{u}}$, where $\overrightarrow{\bold{u}}$ is a normal vector such that $|\overrightarrow{\bold{u}}|=1$, and $\overrightarrow{\bold{\nabla}} f(x,y)$ is called *gradient*.

## Ordinary Differential Equations (ODE) vs Partial Differential Equations (PDE)

* *Ordinary differential equations* (ODE) are equations where the derivatives are taken with respect to only one variable.
* *Partial differential equations* (PDE) are equations that depend on partial derivatives of several variables.

## Multiple Integral

A multiple integral is a definite integral of a function $f$ of more than one variables $\bold{x} \in \mathbb{R}^n, n>1$ over a domain $D$.

$$
\int_D f(\bold{x}) d^n \bold{x} =
\int ... \int_D f(x_1, x_2, ..., x_n) d x_1 d x_2 ... d x_n
$$

In particular, $\bold{x} \in \mathbb{R}^2$ is called *double integrals*, and $\bold{x} \in \mathbb{R}^3$ is called *triple integrals*.

### Computation Example

For example, let $f(x,y)=2$ and compute $D=\{ (x,y) \in \mathbb{R}^2: 2 \le x \le 4; 3 \le x \le 6 \}$.
The solution is

$$
\int^6_3 \int^4_2 2\space dx\space dy =
2 \int^6_3 \int^4_2 1\space dx\space dy =
2 \times (3 \times 2) =
12
$$