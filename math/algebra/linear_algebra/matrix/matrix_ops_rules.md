# Matrix/vector Multiplication and Derivative Rules

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

* Compute $\frac{\partial }{\partial \mathbf{w}}\big(\mathbf{x}^\top\mathbf{w}\big)^2$

$$
\begin{align*}
    \frac{\partial }{\partial \mathbf{w}}\big(\mathbf{x}^\top\mathbf{w}\big)^2
    &=
    2 (\mathbf{x}^\top\mathbf{w})
    \frac{\partial }{\partial \mathbf{w}}(\mathbf{x}^\top\mathbf{w})
    \\\\ &=
    2 (\mathbf{x}^\top\mathbf{w}) \mathbf{x}^\top
    \\\\ &=
    2 \mathbf{w}^\top ( \mathbf{x} \mathbf{x}^\top )
\end{align*}
$$