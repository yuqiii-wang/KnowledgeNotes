# Matrix Derivative

## Inspiration: Derivative of Scalar $f$ over Matrix $X$

Simply, the derivative of a scalar function $f$ over a matrix $X$ can be defined as $\frac{\partial f}{\partial X}=\big[ \frac{\partial f}{\partial x_{ij}} \big]$.
However, such a definition is not handy since it requires computation of each $\partial x_{ij}$.

The differential of a scalar function $df$ can be expressed as the sum of respective differential $d x_i$.

$$
df = \sum^n_{i=1} \frac{\partial f}{\partial x_i} d x_i= \big(\frac{\partial f}{\partial \mathbf{x}}\big)^\text{T} d\mathbf{x}
$$

where, by vector representation, there is $\frac{\partial f}{\partial \mathbf{x}} \in \mathbb{R}^{n \times 1}$ and $d\mathbf{x} \in \mathbb{R}^{n \times 1}$. so that $df$ is the inner product of $\frac{\partial f}{\partial \mathbf{x}} \cdot d\mathbf{x}$.

Given this inspiration, expand the vector to a matrix space, so that

$$
df = \sum^m_{i=1} \sum^n_{j=1} \frac{\partial f}{\partial X_{ij}} d X_{ij}= tr\bigg(\big(\frac{\partial f}{\partial X}\big)^\text{T} dX \bigg)
$$

As a result $df$ is the trace of inner product between $\frac{\partial f}{\partial X} d X$.

## Compute $\frac{\partial f}{\partial Y}\frac{\partial Y}{\partial X}$

Intuitively speaking, to compute $\frac{\partial f}{\partial X}=\frac{\partial f}{\partial Y}\frac{\partial Y}{\partial X}$, where $Y$ is a function of matrix $X$, and $f$ only has derivative definition over $Y$, this requires the computation of this undefined term $\frac{\partial Y}{\partial X}$. Commonly, denote $Y=AXB$, where $A$ and $B$ are constant transformations (give $dX\ne \mathbf{0}$, $dA=\mathbf{0}$ and $dB=\mathbf{0}$) around the variable $X$. This is useful to define $\frac{\partial Y}{\partial X}$ by trace operation.

Recall some trace operation properties:
* Transpose: $tr(A^\text{T})=tr(A)$
* Linearity: $tr(A\pm B)=tr(A)\pm tr(B)$
* Associative: $tr(AB)=tr(BA)$

So that, 
$$
\begin{align*}
df &= tr \big(
    (\frac{\partial f}{\partial Y})^\text{T} dY
\big)
\\\\ &=
tr \big(
    (\frac{\partial f}{\partial Y})^\text{T} d(AXB)
\big)
\\\\ &=
tr \bigg(
    (\frac{\partial f}{\partial Y})^\text{T} \space \big( \underbrace{(d A) X B}_{=\mathbf{0}\text{, for }dA=\mathbf{0}} + A (d X) B +  \underbrace{A X (dB)}_{=\mathbf{0}\text{, for }dB=\mathbf{0}} \big)
\bigg)
\\\\ &=
tr \big(
    (\frac{\partial f}{\partial Y})^\text{T} A \space dX B
\big)
\\\\ &=
tr \big(
    B (\frac{\partial f}{\partial Y})^\text{T} A \space dX
\big)
\\\\ &=
tr \big(
    (A^\text{T} (\frac{\partial f}{\partial Y})^\text{T} B)^\text{T} \space dX
\big)
\\\\ &=
tr \big(
    (\frac{\partial f}{\partial X})^\text{T} \space dX
\big)
\end{align*}
$$

This gives $\frac{\partial f}{\partial X}=A^\text{T} (\frac{\partial f}{\partial Y})^\text{T} B$. Remember, $\frac{\partial f}{\partial Y}$ is defined such as $\frac{\partial f}{\partial Y}=tr\big((\frac{\partial f}{\partial Y})^\text{T}\big) dY$.

### Example: Find $\frac{\partial f}{\partial X}$ where $f=\mathbf{a}^\text{T}e^{X\mathbf{b}}$

First, define element-wise multiplication operator $\odot$.
There is $d\sigma(X)=\sigma'(X) \odot dX$.
For example, given a matrix $X$ and compute $d \space sin(X)$, there is

$$
X = \begin{bmatrix}
    x_{11} & x_{12} \\\\
    x_{21} & x_{22}
\end{bmatrix}
, \quad
d \space sin(X) = 
\begin{bmatrix}
    cos(x_{11}) d x_{11} & cos(x_{12}) d x_{12} \\\\
    cos(x_{21}) d x_{21} & cos(x_{22}) d x_{22}
\end{bmatrix}=
cos(X) \odot dX
$$

Back to $f$, first compute $df$ by $\odot$ ($d\space e^{X\mathbf{b}}=e^{X\mathbf{b}} \odot dX\mathbf{b}$). Then apply trace operations.

Regarding the size, there are $\mathbf{a} \in \mathbb{R}^{m \times 1}, X \in \mathbb{R}^{m \times n}, \mathbf{b} \in \mathbb{R}^{n \times 1}$. $e^{X}$ has the same size of $X$'s since the exponential $e^{X}$ is an element-wise operation on each element $x_{ij}$ of $X$. Besides, $X\mathbf{b} \in \mathbb{R}^{m \times 1}$.

$$
\begin{align*}
    df &= \mathbf{a}^\text{T} (e^{X\mathbf{b}} \odot dX\mathbf{b})
    \\\\ &=
    tr \big(
        \mathbf{a}^\text{T} (e^{X\mathbf{b}} \odot dX\mathbf{b})
    \big)
    \\\\ &=
    tr \big(
        (\mathbf{a} \odot e^{X\mathbf{b}})^\text{T} dX\mathbf{b}
    \big)
    \\\\ &=
    tr \big(
        \mathbf{b} (\mathbf{a} \odot e^{X\mathbf{b}})^\text{T} dX
    \big)
    \\\\ &=
    tr \big(
        ((\mathbf{a} \odot e^{X\mathbf{b}}) \mathbf{b}^\text{T})^\text{T} dX
    \big)
\end{align*}
$$

Given the above result $df  =  tr \big(   ((\mathbf{a} \odot e^{X\mathbf{b}}) \mathbf{b}^\text{T})^\text{T} dX    \big)$ comparing against $df = tr\bigg(\big(\frac{\partial f}{\partial X}\big)^\text{T} dX \bigg)$, derive the final solution that $\frac{\partial f}{\partial X}=(\mathbf{a} \odot e^{X\mathbf{b}}) \mathbf{b}^\text{T}$

### Appendix: Proof of $\nabla_A tr(AB) = B^\text{T}$

Define $A \in \mathbb{R}^{n \times m}$ and $B \in \mathbb{R}^{k \times n}$

$$
\begin{align*}
tr(AB) &=
tr \bigg(
    \begin{bmatrix}
    \mathbf{a_1} \\\\
    \mathbf{a_2} \\\\
    \vdots \\\\
    \mathbf{a_n}
\end{bmatrix}
\begin{bmatrix}
    \mathbf{b_1} \\\\
    \mathbf{b_2} \\\\
    \vdots \\\\
    \mathbf{b_n}
\end{bmatrix}^\text{T}
\bigg)\\\\ &=
tr \begin{bmatrix}
    \mathbf{a_1}\mathbf{b_1}^\text{T} & \mathbf{a_1}\mathbf{b_2}^\text{T} & &  \\\\
    \mathbf{a_2}\mathbf{b_1}^\text{T} & \mathbf{a_2}\mathbf{b_2}^\text{T} & & \\\\
    & & \ddots & \\\\
    & & & \mathbf{a_n}\mathbf{b_n}^\text{T}
\end{bmatrix}\\\\ &=
\sum^n_i \mathbf{a_i}\mathbf{b_i}^\text{T}
\end{align*}
$$

So that (notice here the subscripts of $a_{ij}$ and $b_{ji}$ are in a reverse order)

$$
\frac{\partial tr(AB)}{\partial a_{ij}}=
b_{ji}
$$

To expand to all elements, there is $\nabla_A tr(AB) = B^\text{T}$. 
