# Ordinary Differential Equations (ODE) vs Partial Differential Equations (PDE)

## Differential vs Partial Derivative

Derivative refers to "rate of change" while differential refers to "trivial amount".

If input/output is $1$-d to $1$-d $w=f\big(g(x)\big)$, there is a chian rule:
$$
\frac{dw}{dx}=\frac{df}{dg}\frac{dg}{dx}
$$

If $w=f\big(a(x),b(x),c(x)\big)$, the chain rule becomes
$$
\frac{dw}{dx}=
\frac{\partial w}{\partial a}\frac{d a}{d x} +
\frac{\partial w}{\partial b}\frac{d b}{d x} +
\frac{\partial w}{\partial c}\frac{d c}{d x}
$$

In this expression, $\frac{\partial w}{\partial x_i}dx_i$ are called *partial differentials*.