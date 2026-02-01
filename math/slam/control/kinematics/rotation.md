# Rotation

## Vector Representation of Rotations

Different from Euler's formula that describes rotation according to the three axis $x$-th, $y$-th and $z$-th as $\theta_{roll}$, $\theta_{pitch}$ and $\theta_{yaw}$, 
the vector representation first defines a unit rotation vector $\mathbf{u}$ and an angle $\theta$. 
A spatial point $\mathbf{p}$ rotates about $\mathbf{u}$ by $\theta$, and the result $\mathbf{p}^*$ can be found as below.

For rotation about non-unit vector $\mathbf{r}$, should first normalize it: $\mathbf{u}=\frac{\mathbf{r}}{||\mathbf{r}||}$.

Then, decompose $\mathbf{p}$ to two orthogonal vectors $\mathbf{a}$ (along the direction of $\mathbf{u}$) and $\mathbf{b}$ (perpendicular to $\mathbf{u}$):
$$
\begin{align*}
\mathbf{a} &= \mathbf{u}\mathbf{u}^{\top}\mathbf{p} \\\\
\mathbf{b} &= \mathbf{p} - \mathbf{a} = (1-\mathbf{u}\mathbf{u}^{\top})\mathbf{p}    
\end{align*}
$$

For rotation has no effect on $\mathbf{a}$, but rotates $\mathbf{b}$ by $\theta$ to $\mathbf{b}'$, here define the $\mathbf{c}$ perpendicular to $\mathbf{b}$ on the same rotation plane.

$$
\mathbf{c} = \mathbf{u} \times \mathbf{p}
$$

So that the after rotation spatial point $\mathbf{p}'$ can be computed by *Rodrigues' formula*
$$
\begin{align*}
\mathbf{p}' &= \mathbf{a} + \mathbf{b}'
\\\\ &=
\mathbf{a} + \mathbf{b}\cos\theta + \mathbf{c}\sin\theta
\\\\ &=
\mathbf{u}\mathbf{u}^{\top}\mathbf{p} + (1-\mathbf{u}\mathbf{u}^{\top})\mathbf{p}\cos\theta + \mathbf{u} \times \mathbf{p}\sin\theta
\\\\ &=
\big( I\cos\theta + (1-\cos\theta)\mathbf{u}\mathbf{u}^{\top} + \mathbf{u}^{\wedge}\sin\theta \big) \mathbf{p}
\end{align*}
$$

where $\space^{\wedge}$ is denoted as the skew-symmetric representation of a vector.

<div style="display: flex; justify-content: center;">
      <img src="imgs/rodrigues_formula_derivation.png" width="30%" height="30%" alt="rodrigues_formula_derivation" />
</div>
</br>

The rotation $I\cos\theta + (1-\cos\theta)\mathbf{u}\mathbf{u}^{\top} + \mathbf{u}^{\wedge}\sin\theta$ has two terms:
$I\cos\theta + (1-\cos\theta)\mathbf{u}\mathbf{u}^{\top}$ is symmetric, and $\mathbf{u}^{\wedge}\sin\theta$ is anti-symmetric.

So that in $R-R^{\top}$, the symmetric term  is canceled out for $\Big(I\cos\theta + (1-\cos\theta)\mathbf{u}\mathbf{u}^{\top}\Big) - \Big(I\cos\theta + (1-\cos\theta)\mathbf{u}\mathbf{u}^{\top}\Big)^{\top} = 0$, leaving only $\mathbf{u}^{\wedge}\sin\theta$ untouched.

$$
\begin{align*}
R - R^{-1} &= R - R^{\top}
\\\\ &= 
2\mathbf{u}^{\wedge}\sin\theta
\\\\ &= 2 \begin{bmatrix}
    0 & -u3 & u_2 \\\\
    u_3 & 0 & -u_1 \\\\
    -u_2 & u_1 & 0
\end{bmatrix}
\sin\theta
\\\\ &= 2 \begin{bmatrix}
    0 & -p_3 & p_2 \\\\
    p_3 & 0 & -p_1 \\\\
    -p_2 & p_1 & 0
\end{bmatrix}
\end{align*}
$$

The vector $[p_1\quad p_2\quad p_3]$ has the norm of $\sin\theta$ aligned to the $\mathbf{u}$'s direction.

Given a typical rotation matrix $R=\begin{bmatrix} r_{11} & r_{12} & r_{13} \\\\ r_{21} & r_{22} & r_{23} \\\\ r_{31} & r_{32} & r_{33} \end{bmatrix}$, there is $\text{tr}(R) = r_{11} + r_{22} + r_{33} = 2 \cos\theta + 1$, where $\theta$ represents the angle of the rotation in axis/angle form (for derivation see below *Angle Computation* for Rodrigues' formula).


## Rodrigues' Rotation Formula

This formula provides a shortcut to compute exponential map from $so(3)$ (*Special Orthogonal Group*), the Lie algebra of $SO(3)$, to $SO(3)$ without actually computing the full matrix exponential.

In other words, it helps transform a vector to its matrix representation.

Representing a spacial point as $\mathbf{v}$, and $\mathbf{k} \times \mathbf{v}$ as column matrices ($\mathbf{k}$ is a unit vector), the cross product can be expressed as a matrix product

$$
\begin{bmatrix}
      (\mathbf{k} \times \mathbf{v})_x \\\\
      (\mathbf{k} \times \mathbf{v})_y \\\\
      (\mathbf{k} \times \mathbf{v})_z
\end{bmatrix}=
\begin{bmatrix}
      k_y v_z - k_z v_y \\\\
      k_z v_x - k_x v_z \\\\
      k_x v_y - k_y v_x
\end{bmatrix}=
\begin{bmatrix}
      0 & -k_z & k_y \\\\
      k_z & 0 & -k_x \\\\
      -k_y & k_x & 0
\end{bmatrix}
\begin{bmatrix}
      v_x \\\\
      v_y \\\\
      v_z
\end{bmatrix}
$$

where, 
$$
\mathbf{K}=
\begin{bmatrix}
      0 & -k_z & k_y \\\\
      k_z & 0 & -k_x \\\\
      -k_y & k_x & 0
\end{bmatrix}
$$

Now, the rotation matrix can be written in terms of $\mathbf{K}$ as
$$
\mathbf{Q}=e^{\mathbf{K}\theta}=
\mathbf{I}+\mathbf{K}sin(\theta)+\mathbf{K}^2\big(1-cos(\theta)\big)
$$
where $\mathbf{K}$ is rotation direction unit matrix while $\theta$ is the angle magnitude.

* Vector Form

Define $\mathbf{v}$ is a vector $\mathbf{v} \in \mathbb{R}^3$, $\mathbf{k}$ is a unit vector describing an axis of rotation about which $\mathbf{v}$ rotates by an angle $\theta$

$$
\mathbf{v}\_{rot}=
\mathbf{v} cos\theta + (\mathbf{k} \times \mathbf{v})sin\theta + \mathbf{k}(\mathbf{k} \cdot \mathbf{v})(1-cos\theta)
$$

The rotation of $\mathbf{v}$ about $\mathbf{k}$ by an angle $\theta$ follows the right hand rule

* Matrix Form

Define $\wedge$ as the skew-symmetric matrix representation of a vector (same as $\mathbf{K}=\mathbf{k}^{\wedge}$)

$$
\mathbf{Q} =
cos \theta \mathbf{I} + (1-cos \theta) \mathbf{k}\mathbf{k}^\text{T} + sin\theta \mathbf{k}^{\wedge}
$$

* Angle Computation

Define $tr$ as the trace operation; take the trace of the equation (Rodrigues' rotation in matrix form), so that the angle can be computed as

$$
\begin{align*}
tr(\mathbf{Q}) &= tr \big(
      cos \theta \mathbf{I} + (1-cos \theta) \mathbf{k}\mathbf{k}^\text{T} + sin\theta \mathbf{k}^{\wedge}
\big)\\\\ &=
cos\theta \space tr(\mathbf{I})+(1-cos \theta) tr(\mathbf{k}\mathbf{k}^\text{T})+sin \theta \space tr(\mathbf{k}^{\wedge})\\\\ &=
3 cos\theta + (1-cos \theta)\\\\ &=
1+2cos\theta
\end{align*}
$$

Therefore,
$$
\theta = arccos \bigg(
      \frac{tr(\mathbf{Q})-1}{2}
\bigg)
$$

### Taylor Expansion Explanation

$$
e^{\mathbf{K}\theta}=
\mathbf{I}+\mathbf{K}\theta
+\frac{(\mathbf{K}\theta)^2}{2!}
+\frac{(\mathbf{K}\theta)^3}{3!}
+\frac{(\mathbf{K}\theta)^4}{4!}+...
$$

Given the properties of $\mathbf{K}$ being an antisymmentric matrix, there is $\mathbf{K}^3=-\mathbf{K}$, so that

$$
\begin{align*}
e^{\mathbf{K}\theta}
 &=
\mathbf{I}
+\big(
    \mathbf{K}\theta
    -\frac{\mathbf{K}\theta^3}{3!}
    +\frac{\mathbf{K}\theta^5}{5!}
    -\frac{\mathbf{K}\theta^7}{7!}
    +\frac{\mathbf{K}\theta^9}{9!}
    +...
\big)
\\\\ &
\space \space \space \space 
+\big(
    \frac{\mathbf{K}^2\theta^2}{2!}
    -\frac{\mathbf{K}^4\theta^4}{4!}
    +\frac{\mathbf{K}^6\theta^6}{6!}
    -\frac{\mathbf{K}^8\theta^8}{8!}
    +...
\big)
\\\\ &=
\mathbf{I} +
\mathbf{K}\big(
    \theta
    -\frac{\theta^3}{3!}
    +\frac{\theta^5}{5!}
    -\frac{\theta^7}{7!}
    +\frac{\theta^9}{9!}
    +...
\big) 
\\\\ &
\space \space \space \space +\mathbf{K}^2\big(
    -\frac{\theta^2}{2!}
    +\frac{\theta^4}{4!}
    -\frac{\theta^6}{6!}
    +\frac{\theta^8}{8!}
    +...
\big)
\\\\ &=
\mathbf{I}+\mathbf{K}sin(\theta)+\mathbf{K}^2\big(1-cos(\theta)\big)
\end{align*}
$$

## Derivative of Rotation Over Time: Angular Velocity $\overrightarrow{\mathbf{\omega}}$ and Linear Velocity $\overrightarrow{\mathbf{v}}$

Vector rotation rotating $\overrightarrow{\mathbf{r}}$ about the $Z$-th axis over some degree $\theta$ can be described as below.

<div style="display: flex; justify-content: center;">
      <img src="imgs/rotation_trigonometry.png" width="30%" height="30%" alt="rotation_trigonometry" />
</div>
</br>

Compute the trigonometry of rotation for $\Delta \theta$ to approximate the angular velocity as $\Delta t \rightarrow 0$
$$
\begin{align*}
    {\mathbf{\omega}} =
    \lim_{\Delta t \rightarrow 0}\Delta \theta
    &= 
    \Big|\Big|\overrightarrow{\mathbf{r}} \Big|\Big| \tan(\frac{d \theta}{dt} \Delta t)
    \\\\ &=
    \Big|\Big| \overrightarrow{\mathbf{r}} \Big|\Big| \frac{\sin(\frac{d \theta}{dt} \Delta t)}{\cos(\frac{d \theta}{dt} \Delta t)}
    \\\\ &\approx
    \Big|\Big| \overrightarrow{\mathbf{r}} \Big|\Big| \frac{\frac{d \theta}{dt}}{1}
    && \qquad \text{for } \lim_{t \rightarrow 0} \sin(\theta t) = \theta \text{ and } \lim_{t \rightarrow 0} \cos(\theta t) = 1
    \\\\ &=
    \Big|\Big| \overrightarrow{\mathbf{r}} \Big|\Big| \frac{d \theta}{dt}
\end{align*}
$$


The linear velocity is defined $\overrightarrow{\mathbf{v}}={\mathbf{\omega}} \overrightarrow{\mathbf{d}}$, 
where $\overrightarrow{\mathbf{d}}$ is the rotation direction computed by the normalized cross product $\overrightarrow{\mathbf{d}}= \frac{1}{||\overrightarrow{\mathbf{r}}||} \big( Z \times \overrightarrow{\mathbf{r}} \big)$.

$$
\begin{align*}
    \overrightarrow{\mathbf{v}} &= {\mathbf{\omega}} \overrightarrow{\mathbf{d}}
    \\\\ &=
    \Big|\Big| \overrightarrow{\mathbf{r}} \Big|\Big| \frac{d \theta}{dt} 
    \cdot \frac{1}{||\overrightarrow{\mathbf{r}}||} \big( Z \times \overrightarrow{\mathbf{r}} \big)
    \\\\ &= 
    \underbrace{\frac{d \theta}{dt} Z}\_{=\overrightarrow{\mathbf{\omega}}=\frac{d \theta}{dt} \begin{bmatrix} 0 & 0 & 1 \end{bmatrix}} 
    \times \overrightarrow{\mathbf{r}}  
\end{align*}
$$

This indicates that the angular velocity has a direction pointing upward as $Z$ such that $\overrightarrow{\mathbf{\omega}}=\frac{d \theta}{dt} \begin{bmatrix} 0 & 0 & 1 \end{bmatrix}$.
Linear velocity $\overrightarrow{\mathbf{v}}$ is perpendicular to $\overrightarrow{\mathbf{r}}$.

### Matrix Representation

Rotating $\overrightarrow{\mathbf{r}}$ from its origin position $\overrightarrow{\mathbf{r}}_0$ at the angular speed $\overrightarrow{\mathbf{\omega}}$ is basically to compute $\overrightarrow{\mathbf{r}}(t)=R(t) \overrightarrow{\mathbf{r}}_0$,
where $R(t) = \exp(\overrightarrow{\mathbf{\omega}}^{\wedge}t)$.

Recall Rodrigues' rotation formula, here decomposes $\overrightarrow{\mathbf{\omega}}t = \frac{\overrightarrow{\mathbf{\omega}}}{||\overrightarrow{\mathbf{\omega}}||} \cdot ||\overrightarrow{\mathbf{\omega}}||t$, where $Z=\frac{\overrightarrow{\mathbf{\omega}}}{||\overrightarrow{\mathbf{\omega}}||}$ is the $Z$-th rotation axis, and $\theta=||\overrightarrow{\mathbf{\omega}}||t$ is the scalar rotation angle.

$$
\begin{align*}
&&
\mathbf{Q} =
e^{\mathbf{K}\theta}&=
\mathbf{I}+\mathbf{K}sin(\theta)+\mathbf{K}^2\big(1-cos(\theta)\big)
\\\\ \text{Substitutions } \Rightarrow &&
R(t) =
\exp(\overrightarrow{\mathbf{\omega}}^{\wedge}t)&=
\mathbf{I} + Z^{\wedge} \sin(||\overrightarrow{\mathbf{\omega}}||t) + \big(1-\cos(||\overrightarrow{\mathbf{\omega}}||t)\big) Z^{\wedge} Z^{\wedge} 
\end{align*}
$$

where $Z^{\wedge}=\begin{bmatrix} 0 & 0 & 0 \\\\ 0 & 0 & 1 \\\\ 0 & -1 & 0 \end{bmatrix}$ is the skew matrix representation of $Z = \begin{bmatrix} 0 & 0 & 1 \end{bmatrix}$.

To get the first order derivative of $\dot{R}(t)$, when $t \rightarrow 0$, there are $\sin(||\overrightarrow{\mathbf{\omega}}||t) \rightarrow ||\overrightarrow{\mathbf{\omega}}||$ and $\cos(||\overrightarrow{\mathbf{\omega}}||t) \rightarrow 1$, so that
$$
\dot{R}(t) = \frac{d \space R(t)}{dt} = \mathbf{I} + Z^{\wedge} ||\overrightarrow{\mathbf{\omega}}||
$$

## Derivative of Rotation Over $\theta$

Consider a rotation matrix $R_x(\theta)$ on the $x$-axis.

$$
R_x(\theta) = 
\exp \Bigg( \theta \begin{bmatrix}
    0 & 0 & 0 \\\\
    0 & 0 & -1 \\\\
    0 & 1 & 0
\end{bmatrix} \Bigg)=
\begin{bmatrix}
    1 & 0 & 0 \\\\
    0 & \cos\theta & -\sin\theta \\\\
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
    + R_x(\theta) \Big(\frac{d}{d\theta} R^\top_x(\theta) \Big) &= \mathbf{0}
\\\\ \Rightarrow &&
    \underbrace{\Big(\frac{d}{d\theta} R_x(\theta) \Big) R_x^\top(\theta)}\_{S_x}
    + 
    \underbrace{\Bigg(\Big(\frac{d}{d\theta} R_x(\theta) \Big) R^\top_x(\theta)\Bigg)^\top}\_{S_x^\top} 
    &= \mathbf{0}
\\\\ \Rightarrow &&
    S_x + S_x^\top &= \mathbf{0}
\end{align*}
$$

For $S$ is anti-symmetric, generalize $S_x \rightarrow S([1\quad 0\quad 0])$ such as below

$$
\begin{align*}
&& S(\mathbf{v}) &=
\begin{bmatrix}
    0 & -v_z & v_y \\\\
    v_z & 0 & -v_x \\\\
    -v_y & v_x & 0
\end{bmatrix}  \\\\
\text{For } S_x \rightarrow S([1\quad 0\quad 0]) :
\\\\ && 
S([1\quad 0\quad 0]) &= \begin{bmatrix}
    1 & 0 & 0 \\\\
    0 & -\sin\theta & -\cos\theta \\\\
    0 & \cos\theta & -\sin\theta
\end{bmatrix} 
\begin{bmatrix}
    1 & 0 & 0 \\\\
    0 & \cos\theta & \sin\theta \\\\
    0 & -\sin\theta & \cos\theta
\end{bmatrix}
\\\\ && &=
\begin{bmatrix}
    0 & 0 & 0 \\\\
    0 & 0 & -1 \\\\
    0 & 1 & 0
\end{bmatrix} \\\\
\text{Re-express the derivative by } S_x : \\\\
&& \frac{d}{d\theta} R_x(\theta) &= S([1\quad 0\quad 0]) R_x(\theta) \\\\
\text{Similarly for } S_y \text{ and } S_z: \\\\
&& \frac{d}{d\theta} R_y(\theta) &= S([0\quad 1\quad 0]) R_y(\theta) \\\\
&& \frac{d}{d\theta} R_z(\theta) &= S([0\quad 0\quad 1]) R_z(\theta)
\end{align*}
$$

In conclusion, the derivative of a rotation matrix with respects to the $x$-th, $y$-th and $z$-th axis is some signs' changes to their respective cells, 
such as 
$$
\begin{align*}
    \frac{d}{d\theta} R_x(\theta) &= S([1\quad 0\quad 0]) R_x(\theta) 
    \\\\ &= 
    \begin{bmatrix} 0 & 0 & 0 \\\\ 
    0 & 0 & -1 \\\\ 
    0 & 1 & 0 
    \end{bmatrix} 
    \begin{bmatrix} 1 & 0 & 0 \\\\ 
    0 & \cos\theta & -\sin\theta \\\\ 
    0 & \sin\theta & \cos\theta 
    \end{bmatrix}
\end{align*}
$$