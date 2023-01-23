#  Matrix/vector Multiplication and Derivative Rules

## Multiplications

|Multiplications|Comments|
|-|-|
|$(AB)^T = B^T A^T$|transpose|
|$(a^T B c)^T = c^T B^T a$|transpose|
|$a^T b = b^T a$|scalar product|
|$(A+B)C = AC + BC$|distributive|
|$(a+b)^T C = a^T C + b^T C$|distributive|
|$AB \ne BA$|non-commutative|

## Derivative

* Compute $\frac{\partial }{\partial \bold{w}}\big(\bold{x}^\top\bold{w}\big)^2$

$$
\begin{align*}
    \frac{\partial }{\partial \bold{w}}\big(\bold{x}^\top\bold{w}\big)^2
    &=
    2 (\bold{x}^\top\bold{w})
    \frac{\partial }{\partial \bold{w}}(\bold{x}^\top\bold{w})
    \\ &=
    2 (\bold{x}^\top\bold{w}) \bold{x}^\top
    \\ &=
    2 \bold{w}^\top ( \bold{x} \bold{x}^\top )
\end{align*}
$$