# Terminologies in Differential and Integral

## Operator Explained $\frac{d^n}{d x^n}f$

## Derivative vs Differential vs Gradient

Given a function $f(\mathbf{x})$ where $\mathbf{x}\in\mathbb{R}^D$ is an input vector.

* Derivative: the change rate of an dimension $x\_i$ to the function $f(x\_i)$.

$$
f'(x\_i)=\lim_{h \rightarrow 0} \frac{f(x\_i+h)-f(x\_i)}{h}
$$

* Differential: find trivial function change $df$ (often) by first order derivative approximation

$$
df=f'(\mathbf{x})d\mathbf{x}=
\frac{\partial f}{\partial x_1} d x_1+\frac{\partial f}{\partial x_2} d x_2+\frac{\partial f}{\partial x_3} d x_3+...
$$

* Gradient: a vector that collects all the partial derivatives, indicative of each dimensional change rate, contained directional info

$$
\nabla f=\frac{\partial f}{\partial \mathbf{x}}=
\Big(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \frac{\partial f}{\partial x_3}, ...\Big)
$$

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

In this expression, $\frac{\partial w}{\partial x\_i}dx_i$ are called *partial differentials*.

### Directional Derivative

*Directional derivative* refers to "total" derivative.

For example, suppose $z=f(x,y)$ is a function of two variables with a domain of $D$. Set $\overrightarrow{\mathbf{u}}=\cos\theta \overrightarrow{\mathbf{i}}+\sin\theta \overrightarrow{\mathbf{j}}$, the directional derivative of $f$ in the direction of $\overrightarrow{\mathbf{u}}$ is as below.

$$
\begin{align*}
D_{\overrightarrow{\mathbf{u}}} f(x,y) &=
\lim_{h \rightarrow 0} \frac{f(x+h\cos\theta, y+h\sin\theta) - f(x,y)}{h}
\\ &=
f_x(x,y)\cos\theta + f_y(x,y)\sin\theta
\end{align*}
$$

where $f_x$ and $f_y$ are the partial derivatives, respectively.

Directional derivative can be decomposed to
$D_{\overrightarrow{\mathbf{u}}} f(x,y)=\overrightarrow{\mathbf{\nabla}} f(x,y) \cdot \overrightarrow{\mathbf{u}}$, where $\overrightarrow{\mathbf{u}}$ is a normal vector such that $|\overrightarrow{\mathbf{u}}|=1$, and $\overrightarrow{\mathbf{\nabla}} f(x,y)$ is called *gradient*.

## Ordinary Differential Equations (ODE) vs Partial Differential Equations (PDE)

* *Ordinary differential equations* (ODE) are equations where the derivatives are taken with respect to only one variable.
* *Partial differential equations* (PDE) are equations that depend on partial derivatives of several variables.

## Multiple Integral

A multiple integral is a definite integral of a function $f$ of more than one variables $\mathbf{x} \in \mathbb{R}^n, n>1$ over a domain $D$.

$$
\int_D f(\mathbf{x}) d^n \mathbf{x} =
\int ... \int_D f(x_1, x_2, ..., x_n) d x_1 d x_2 ... d x_n
$$

In particular, $\mathbf{x} \in \mathbb{R}^2$ is called *double integrals*, and $\mathbf{x} \in \mathbb{R}^3$ is called *triple integrals*.

### Computation Example

For example, let $f(x,y)=2$ and compute $D=\{ (x,y) \in \mathbb{R}^2: 2 \le x \le 4; 3 \le x \le 6 \}$.
The solution is

$$
\int^6_3 \int^4_2 2\space dx\space dy =
2 \int^6_3 \int^4_2 1\space dx\space dy =
2 \times (3 \times 2) =
12
$$