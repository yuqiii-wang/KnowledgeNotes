# Derivative of A Rotation Matrix

Consider a rotation matrix $R_x(\theta)$ on the $x$-axis.
$$
R_x(\theta) = \begin{bmatrix}
    1 & 0 & 0 \\
    0 & \cos\theta & -\sin\theta \\
    0 & \sin\theta & \cos\theta
\end{bmatrix}
$$

By rotation property, $R_x^\top(\theta)=R_x^{-1}(\theta)$, there is
$$
R_x(\theta) R_x^\top(\theta) = I
$$

Take derivative, and replace the multiplication term with $S_x$ there is
$$
\begin{align*}
&&
    \Big(\frac{d}{d\theta} R_x(\theta) \Big) R_x^\top(\theta)
    + R_x(\theta) \Big(\frac{d}{d\theta} R^\top_x(\theta) \Big) &= \bold{0}
\\ \Rightarrow &&
    \underbrace{\Big(\frac{d}{d\theta} R_x(\theta) \Big) R_x^\top(\theta)}_{S_x}
    + 
    \underbrace{\Bigg(\Big(\frac{d}{d\theta} R_x(\theta) \Big) R^\top_x(\theta)\Bigg)^\top}_{S_x^\top} 
    &= \bold{0}
\\ \Rightarrow &&
    S_x + S_x^\top &= \bold{0}
\end{align*}
$$

For $S$ is anti-symmetric, generalize $S_x \rightarrow S([1\quad 0\quad 0])$ such as below
$$
\begin{align*}
&& S(\bold{v}) &=
\begin{bmatrix}
    0 & -v_z & v_y \\
    v_z & 0 & -v_x \\
    -v_y & v_x & 0
\end{bmatrix} 
\\
\text{For } S_x \rightarrow S([1\quad 0\quad 0]) :
\\ && 
S([1\quad 0\quad 0]) &= \begin{bmatrix}
    1 & 0 & 0 \\
    0 & -\sin\theta & -\cos\theta \\
    0 & \cos\theta & -\sin\theta
\end{bmatrix} 
\begin{bmatrix}
    1 & 0 & 0 \\
    0 & \cos\theta & \sin\theta \\
    0 & -\sin\theta & \cos\theta
\end{bmatrix}
\\ && &=
\begin{bmatrix}
    0 & 0 & 0 \\
    0 & 0 & -1 \\
    0 & 1 & 0
\end{bmatrix}
\\
\text{Re-express the derivative by } S_x :
\\
&& \frac{d}{d\theta} R_x(\theta) &= S([1\quad 0\quad 0]) R_x(\theta)
\\
\text{Similarly for } S_y \text{ and } S_z:
\\
&& \frac{d}{d\theta} R_y(\theta) &= S([0\quad 1\quad 0]) R_y(\theta)
\\
&& \frac{d}{d\theta} R_z(\theta) &= S([0\quad 0\quad 1]) R_z(\theta)
\end{align*}
$$