# Rotation

## Rodrigues' rotation formula

This formula provides a shortcut to compute exponential map from $so(3)$ (*Special Orthogonal Group*), the Lie algebra of $SO(3)$, to $SO(3)$ without actually computing the full matrix exponential.

In other words, it helps transform a vector to its matrix representation.

Representing a spacial point as $\bold{v}$, and $\bold{k} \times \bold{v}$ as column matrices ($\bold{k}$ is a unit vector), the cross product can be expressed as a matrix product

$$
\begin{bmatrix}
      (\bold{k} \times \bold{v})_x \\
      (\bold{k} \times \bold{v})_y \\
      (\bold{k} \times \bold{v})_z
\end{bmatrix}
=
\begin{bmatrix}
      k_y v_z - k_z v_y \\
      k_z v_x - k_x v_z \\
      k_x v_y - k_y v_x
\end{bmatrix}
=
\begin{bmatrix}
      0 & -k_z & k_y \\
      k_z & 0 & -k_x \\
      -k_y & k_x & 0
\end{bmatrix}
\begin{bmatrix}
      v_x \\
      v_y \\
      v_z
\end{bmatrix}
$$
where, 
$$
\bold{K}=
\begin{bmatrix}
      0 & -k_z & k_y \\
      k_z & 0 & -k_x \\
      -k_y & k_x & 0
\end{bmatrix}
$$

Now, the rotation matrix can be written in terms of $\bold{K}$ as
$$
\bold{Q}=e^{\bold{K}\theta}
=
\bold{I}+\bold{K}sin(\theta)+\bold{K}^2\big(1-cos(\theta)\big)
$$
where $\bold{K}$ is rotation direction unit matrix while $\theta$ is the angle magnitude.

* Vector Form

Define $\bold{v}$ is a vector $\bold{v} \in \mathbb{R}^3$, $\bold{k}$ is a unit vector describing an axis of rotation about which $\bold{v}$ rotates by an angle $\theta$

$$
\bold{v}_{rot}
=
\bold{v} cos\theta + (\bold{k} \times \bold{v})sin\theta + \bold{k}(\bold{k} \cdot \bold{v})(1-cos\theta)
$$

* Matrix Form

Define $\wedge$ as the skew-symmetric matrix representation of a vector (same as $\bold{K}=\bold{k}^{\wedge}$)

$$
\bold{Q} =
cos \theta \bold{I} + (1-cos \theta) \bold{k}\bold{k}^\text{T} + sin\theta \bold{k}^{\wedge}
$$

* Angle Computation

Define $tr$ as the trace operation; take the trace of the equation (Rodrigues' rotation in matrix form), so that the angle can be computed as
$$
\begin{align*}
tr(\bold{Q}) &= tr \big(
      cos \theta \bold{I} + (1-cos \theta) \bold{k}\bold{k}^\text{T} + sin\theta \bold{k}^{\wedge}
\big)

\\ &=
cos\theta \space tr(\bold{I})
+ (1-cos \theta) tr(\bold{k}\bold{k}^\text{T})
+ sin \theta \space tr(\bold{k}^{\wedge})

\\ &=
3 cos\theta + (1-cos \theta)

\\ &=
1+2cos\theta
\end{align*}
$$

Therefore,
$$
\theta = arccos \bigg(
      \frac{tr(\bold{Q})-1}{2}
\bigg)
$$

### Taylor Expansion Explanation

$$
e^{\bold{K}\theta}=
\bold{I}+\bold{K}\theta
+\frac{(\bold{K}\theta)^2}{2!}
+\frac{(\bold{K}\theta)^3}{3!}
+\frac{(\bold{K}\theta)^4}{4!}
+ ...
$$

Given the properties of $\bold{K}$ being an antisymmentric matrix, there is $\bold{K}^3=-\bold{K}$, so that
$$
\begin{align*}
e^{\bold{K}\theta}
 &=
\bold{I}
+\big(
    \bold{K}\theta
    -\frac{\bold{K}\theta^3}{3!}
    +\frac{\bold{K}\theta^5}{5!}
    -\frac{\bold{K}\theta^7}{7!}
    +\frac{\bold{K}\theta^9}{9!}
    +...
\big)
\\ &
\space \space \space \space 
+\big(
    \frac{\bold{K}^2\theta^2}{2!}
    -\frac{\bold{K}^4\theta^4}{4!}
    +\frac{\bold{K}^6\theta^6}{6!}
    -\frac{\bold{K}^8\theta^8}{8!}
    +...
\big)
\\ &=
\bold{I} +
\bold{K}\big(
    \theta
    -\frac{\theta^3}{3!}
    +\frac{\theta^5}{5!}
    -\frac{\theta^7}{7!}
    +\frac{\theta^9}{9!}
    +...
\big) 
\\ &
\space \space \space \space 
+ \bold{K}^2\big(
    -\frac{\theta^2}{2!}
    +\frac{\theta^4}{4!}
    -\frac{\theta^6}{6!}
    +\frac{\theta^8}{8!}
    +...
\big)
\\ &=
\bold{I}+\bold{K}sin(\theta)+\bold{K}^2\big(1-cos(\theta)\big)
\end{align*}
$$

## Derivative of A Rotation Matrix

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