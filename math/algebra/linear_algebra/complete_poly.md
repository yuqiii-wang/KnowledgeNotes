# Complete homogeneous symmetric polynomial

A polynomial is complete homogeneous symmetric when satisfying the below expression 

$$
h_k(x_1,x_2,...,x_n)
=
\sum_{1 \le i_1 , i_2 , ..., \le i_n}
\frac{m_1!m_2!...m_n!}{k!} 
x_{i_1} x_{i_2} ...  x_{i_n}
$$

## Example

For $n=3$:
$$
\begin{align*}
h_1(x_1, x_2, x_3) &= x_1 + x_2 + x_3
\\
h_2(x_1, x_2, x_3) &= x_1^2 + x_2^2 + x_3^2 + x_1 x_2 + x_2 x_3 + x_1 x_3
\\
h_3(x_1, x_2, x_3) &= x_1^3 + x_2^3 + x_3^3 
+ x_1^2 x_2 + x_2^2 x_3 + x_1^2 x_3 
+ x_2^2 x_1 + x_3^2 x_1 + x_3^2 x_2
+ x_1 x_2 x_3
\end{align*}
$$