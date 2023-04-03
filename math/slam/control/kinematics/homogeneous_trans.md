# Homogeneous Transformation

For rigid motion, there are rotations and translation.

Given a three-dimensional space, there is unit orthogonal basis vector for a global coordinate $\bold{e}$
$$
\begin{bmatrix}
e_1 , e_2, e_3
\end{bmatrix}
$$

Given robot as origin, there exists another robot-based coordinate $\bold{e^{'}}$
$$
\begin{bmatrix}
e^{'}_1, e^{'}_2, e^{'}_3
\end{bmatrix}
$$

To represent an object in the two aforementioned coordinates, there are $\bold{a}$ and $\bold{a^{'}}$
$$
\begin{bmatrix}
a_1 \\
a_2 \\
a_3
\end{bmatrix}
and
\begin{bmatrix}
a^{'}_1 \\
a^{'}_2 \\
a^{'}_3
\end{bmatrix}
$$

In practice, when a robot starts, its origin overlaps with the origin of the global coordinate
$$
\begin{bmatrix}
e_1 , e_2, e_3
\end{bmatrix}
\begin{bmatrix}
a_1 \\
a_2 \\
a_3
\end{bmatrix}
=
\begin{bmatrix}
e^{'}_1, e^{'}_2, e^{'}_3
\end{bmatrix}
\begin{bmatrix}
a^{'}_1 \\
a^{'}_2 \\
a^{'}_3
\end{bmatrix}
$$

When there is a robot movement
$$
\begin{bmatrix}
a_1 \\
a_2 \\
a_3
\end{bmatrix}
=
\begin{bmatrix}
e^T_1 e^{'}_1 & e^T_1 e^{'}_2 & e^T_1 e^{'}_3 \\
e^T_2 e^{'}_1 & e^T_2 e^{'}_2 & e^T_2 e^{'}_2 \\
e^T_3 e^{'}_1 & e^T_3 e^{'}_2 & e^T_3 e^{'}_3
\end{bmatrix}
\begin{bmatrix}
a^{'}_1 \\
a^{'}_2 \\
a^{'}_3
\end{bmatrix}
$$

To put it in a simple way
$$
\bold{a} = R \bold{a^{'}}
$$

Remember, $R$'s determinant is $1$: $det(R)=1$, and $R^{-1} = R^T$, hence $R^T R = I$

It is coined *Special Orthogonal Group*. Given $3$ as the dimensionality, it is denoted as $SO(3)$.

Consider translation:

$$
\begin{bmatrix}
a_1 \\
a_2 \\
a_3
\end{bmatrix}
=
\begin{bmatrix}
e^T_1 e^{'}_1 & e^T_1 e^{'}_2 & e^T_1 e^{'}_3 \\
e^T_2 e^{'}_1 & e^T_2 e^{'}_2 & e^T_2 e^{'}_2 \\
e^T_3 e^{'}_1 & e^T_3 e^{'}_2 & e^T_3 e^{'}_3
\end{bmatrix}
\begin{bmatrix}
a^{'}_1 \\
a^{'}_2 \\
a^{'}_3
\end{bmatrix}
+
\begin{bmatrix}
t_1 \\
t_2 \\
t_3
\end{bmatrix}
$$

To put it in a simple way
$$
\bold{a} = T \bold{a^{'}}
$$

Here introduces *Homogeneous Transformation* that takes translation and rotation merged as one: 

$$
\begin{bmatrix}
a_1 \\
a_2 \\
a_3 \\
1
\end{bmatrix}
=
\begin{bmatrix}
e^T_1 e^{'}_1 & e^T_1 e^{'}_2 & e^T_1 e^{'}_3 & t_1 \\
e^T_2 e^{'}_1 & e^T_2 e^{'}_2 & e^T_2 e^{'}_2 & t_2 \\
e^T_3 e^{'}_1 & e^T_3 e^{'}_2 & e^T_3 e^{'}_3 & t_3 \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
a^{'}_1 \\
a^{'}_2 \\
a^{'}_3 \\
1
\end{bmatrix}
$$

This set of transform matrix is also known
as the *special Euclidean group*:

$$
SE(3) = 
\bigg\{
    T = 
    \begin{bmatrix}
        R & T \\
        0 & 1
    \end{bmatrix}
    \in \mathbb{R}^{4 \times 4}
    |
    R \in SO(3), T \in \mathbb{R}^3
\bigg\}
$$
