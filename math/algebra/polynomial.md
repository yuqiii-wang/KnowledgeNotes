# Polynomial

## Complete homogeneous symmetric polynomial

A polynomial is complete homogeneous symmetric when satisfying the below expression 

$$
h_k(x_1,x_2,...,x_n)=
\sum_{1 \le i_1 , i_2 , ..., \le i_n}
\frac{m_1!m_2!...m_n!}{k!} 
x_{i_1} x_{i_2} ...  x_{i_n}
$$

which is basically the full combinations of variables and their powers.

### Example

For $n=3$:
$$
\begin{align*}
h_1(x_1, x_2, x_3) &= x_1 + x_2 + x_3
\\
h_2(x_1, x_2, x_3) &= x_1^2 + x_2^2 + x_3^2 + x_1 x_2 + x_2 x_3 + x_1 x_3
\\
h_3(x_1, x_2, x_3) &= x_1^3 + x_2^3 + x_3^3 +x_1^2 x_2 + x_2^2 x_3 + x_1^2 x_3 +x_2^2 x_1 + x_3^2 x_1 + x_3^2 x_2+x_1 x_2 x_3
\end{align*}
$$

## Vieta's Formulas

For a polynomial of degree $n$:

$$
\text{poly}(x)=
a_n x^n + a_{n-1} x^{n-1} + ... + a_1 x + a_0
$$

Vieta's formulas establish relationships between the coefficients of a polynomial and the sums/products of its roots (let $r_1, r_2, ..., r_n$ be the roots):

$$
\begin{align*}
r_1+r_2+...+r_n&=(-1)^{1}\frac{a_{n-1}}{a_n} \\
r_1r_2+r_1r_3+...+r_{n-2}r_{n-1}+r_{n-1}r_{n}&=(-1)^{2}\frac{a_{n-2}}{a_n} \\
r_1r_2r_3+r_1r_2r_4+...+r_{n-2}r_{n-1}r_{n}&=(-1)^{3}\frac{a_{n-3}}{a_n} \\
... \\
r_1r_2r_3...r_{n-1}r_{n}&=(-1)^{n}\frac{a_{0}}{a_n}
\end{align*}
$$

### Derivation via Factored Form

Given roots $r_1, r_2, ..., r_n$, any polynomial can be expanded as

$$
\begin{align*}
\text{poly}(x)&=a_n(x-r_1)(x-r_2)...(x-r_n) \\
&= a_n \bigg(
 x_n-\Big(\sum_{i=1}^{n}r_i\Big)x^{n-1}+\Big(\sum_{1\le i\le j\le n}r_i r_j\Big)x^{n-2}-\Big(\sum_{1\le i\le j\le k\le n}r_i r_j r_k\Big)x^{n-3}
    +...+(-1)^{n}\Big(r_1r_2r_3...r_{n-1}r_{n}\Big)
\bigg)
\end{align*}
$$

Compare and settle down by matching individual coefficients

$$
\begin{align*}
a_n\Big(\sum_{i=1}^{n}r_i\Big)=a_{n-1} &\quad\Rightarrow\quad \frac{a_{n-1}}{a_n}=(-1)^{1}\sum_{i=1}^{n}r_i \\
a_n\Big(\sum_{1\le i\le j\le n}r_i r_j\Big)=a_{n-2} &\quad\Rightarrow\quad \frac{a_{n-2}}{a_n}=(-1)^{2}\sum_{1\le i\le j\le n}r_i r_j \\
a_n\Big(\sum_{1\le i\le j\le k\le n}r_i r_j r_k\Big)=a_{n-2} &\quad\Rightarrow\quad \frac{a_{n-3}}{a_n}=(-1)^{3}\sum_{1\le i\le j\le k\le n}r_i r_j r_k \\
... \\
a_n \big(r_1r_2r_3...r_{n-1}r_{n}\big)=a_0 &\quad\Rightarrow\quad \frac{a_{0}}{a_n}=(-1)^{n}r_1r_2r_3...r_{n-1}r_{n}
\end{align*}
$$
