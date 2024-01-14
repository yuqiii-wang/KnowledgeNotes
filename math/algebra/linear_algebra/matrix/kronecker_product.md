# Kronecker Product

Given $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{p \times q}$, the *Kronecker product* is defined as $C=A \otimes B$ whose result is $C \in \mathbb{R}^{pm \times qn}$, suhc as
$$
A \otimes B = 
\begin{bmatrix}
    a_{11} B & a_{12} B &  & a_{1n} B \\
    a_{21} B & a_{22} B &  & a_{2n} B \\
    & & \ddots & \\
    a_{m1} B & a_{m2} B &  & a_{mn} B \\
\end{bmatrix}
$$

For example,
$$
\begin{align*}
\begin{bmatrix}
    1 & 2 \\
    3 & 4
\end{bmatrix}
\otimes
\begin{bmatrix}
    0 & 5 \\
    6 & 7
\end{bmatrix}&=
\begin{bmatrix}
    1 \begin{bmatrix}
        0 & 5 \\
        6 & 7
    \end{bmatrix}
    &
    2 \begin{bmatrix}
        0 & 5 \\
        6 & 7
    \end{bmatrix} 
    \\
    3 \begin{bmatrix}
        0 & 5 \\
        6 & 7
    \end{bmatrix}
    & 
    4 \begin{bmatrix}
        0 & 5 \\
        6 & 7
    \end{bmatrix}
\end{bmatrix}
\\ &=
\begin{bmatrix}
    1 \times 0 & 1 \times 5 & 2 \times 0 & 2 \times 5 \\
    1 \times 6 & 1 \times 7 & 2 \times 6 & 2 \times 7 \\
    3 \times 0 & 3 \times 5 & 4 \times 0 & 4 \times 5 \\
    3 \times 6 & 3 \times 7 & 4 \times 6 & 4 \times 7 \\
\end{bmatrix}
\\ &=
\begin{bmatrix}
    0 & 5 & 0 & 10 \\
    6 & 7 & 12 & 14 \\
    0 & 15 & 0 & 20 \\
    18 & 21 & 24 & 28 \\
\end{bmatrix}
\end{align*}
$$