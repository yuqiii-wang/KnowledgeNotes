# Dual Space

## Dual Numbers

Dual numbers introduce an *infinitesimal* unit $\epsilon$ that has the property $\epsilon^2=0$ (analogously to imaginary number having the property $i^2=-1$). A dual number $a+v\epsilon$ has two components, the real component $a$ and the infinitesimal component $v$.

This simple change leads to a convenient method for computing exact derivatives without needing to manipulate complicated symbolic expressions.

For example, $f(x+\epsilon)$ can be expressed as
$$
\begin{align*}
    f(x+\epsilon) &=
    f(x) + kf(x)\epsilon + k^2f(x)\frac{\epsilon^2}{2} + k^3f(x)\frac{\epsilon^3}{6}
    \\ &=
    f(x) + kf(x)\epsilon 
\end{align*}
$$

## Dual Vectors

Any vector space $V$ has a corresponding dual vector space (or just dual space for short) consisting of all linear forms on $V$.

Define a  Jet: a Jet is a $n$-dimensional dual number: $\bold{\epsilon}=[\epsilon_1, \epsilon_2, ..., \epsilon_n]$ with the property $\forall i,j: \epsilon_i \epsilon_j=0$. Then a Jet consists of a real part $a$ and a $n$-dimensional infinitesimal part $\bold{v}$, i.e.,
$$
\begin{align*}
x &= a+\sum_{i=1}^{n} v_i \epsilon_i
\\ &= a + \bold{v}
\end{align*}
$$

Then, using the same Taylor series expansion used above, there is
$$
f(a+\bold{v})=f(a)+kf(a)\bold{v}
$$

Expand to a multivariate function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ evaluated on $x_j=a_j+\bold{v}_j$

$$
\begin{align*}
f(\bold{x}) &=
f(x_1, x_2, ..., x_n)
\\&=
f(a_1, a_2, ..., a_n) + \sum_{j=1}^{n} k_j f(a_1, a_2, ..., a_n)\bold{v}_j
\end{align*}
$$

Set $\bold{v}_j=\bold{e}_j$ to be a unit vector, there is
$$
f(x_1, x_2, ..., x_n) =
f(a_1, a_2, ..., a_n) + \sum_{j=1}^{n} k_j f(a_1, a_2, ..., a_n)\bold{\epsilon}_j
$$
