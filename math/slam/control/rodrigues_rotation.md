# Rodrigues' rotation formula

This formula provides a shortcut to compute exponential map from $so(3)$ (*Special Orthogonal Group*), the Lie algebra of $SO(3)$, to $SO(3)$ without actually computing the full matrix exponential.

Representing $\bold{v}$ and $\bold{k} \times \bold{v}$ as column matrices ($\bold{k}$ is a unit vector), the cross product can be expressed as a matrix product

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

## Taylor Expansion Explanation

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
\\ &=
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