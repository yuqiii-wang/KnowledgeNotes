# Quaternion

## Euler Angle and Singularity

Euler angle provides an intuitive perspective into rotation that 
it uses primal axes to decompose a rotation into three rotations around different axes.

* Rotate around the $Z$ axis of the object to get the yaw angle $\theta_{yaw}=y$;
* Rotate around the $Y$ axis of the object to get the pitch angle $\theta_{pitch}=p$;
* Rotate around the $X$ axis of the object to get the roll angle $\theta_{roll}=r$;

However, it suffers from *Gimbal lock* (singularity problem), that results in loss of one degree of freedom. 
Shown as below, when the pitch (green) and yaw (magenta) gimbals become aligned, changes to roll (blue) and yaw apply the same rotation to the airplane.

![Gimbal_Lock_Plane](imgs/Gimbal_Lock_Plane.gif "Gimbal_Lock_Plane")

Euler angle's $\theta$ operates on a 2-d plane at a time. If all three axes's $\theta$ s are aligned to one 2-d plane, gimbal lock happens.

Quaternion uses four elements to represent rotation that avoids Gimbal lock issues. 

## Quaternion Definition

A rotation of angle $\theta$ in the three dimensional space given three bases $u_x\overrightarrow{i}, u_y\overrightarrow{j}, u_z\overrightarrow{k}$ are defined by the unit vector
$$
\overrightarrow{u}=(u_x, u_y, u_z)=u_x \overrightarrow{i} + u_y \overrightarrow{j} + u_z \overrightarrow{k}
$$
can be represented by a quaternion using an extension of Euler's formula:
$$
\begin{align*}
\bold{q}&=
e^{\frac{\theta}{2}(u_x \overrightarrow{i} + u_y \overrightarrow{j} + u_z \overrightarrow{k})}
\\&=
cos\frac{\theta}{2} + (u_x \overrightarrow{i} + u_y \overrightarrow{j} + u_z \overrightarrow{k})sin\frac{\theta}{2}
\end{align*}
$$

## Quaternion Operations

Quaternion given the above can be expressed in the below general form

$$
\bold{q} = 
[s, \bold{v}]^\text{T},
\quad s=q_0 \in \mathbb{R},
\quad \bold{v}=[x \overrightarrow{i}, y \overrightarrow{j}, z \overrightarrow{k}]^\text{T} \in \mathbb{R}^3
$$
where $\overrightarrow{i},\overrightarrow{j},\overrightarrow{k}$ represent imaginary parts in respect to the three dimensions.

* Addition/Subtraction

$$
\bold{q}_a + \bold{q}_b =
[s_a \pm s_b, \bold{v}_a \pm \bold{v}_b ]^\text{T}
$$

* Multiplication: 

Given the relationship of the respective imaginary parts (derived from cross product that sees $a \times b$ being perpendicular to the $<a , b>$ plane),
$$
\left\{\begin{array}{cc}
    \overrightarrow{i}^2 = \overrightarrow{j}^2 = \overrightarrow{k}^2 = -1 \\
    \overrightarrow{i}\overrightarrow{j} = \overrightarrow{k}, 
    \quad \overrightarrow{j}\overrightarrow{i} = -\overrightarrow{k} \\
    \overrightarrow{j}\overrightarrow{k} = \overrightarrow{i}, 
    \quad \overrightarrow{k}\overrightarrow{j} = -\overrightarrow{i} \\
    \overrightarrow{k}\overrightarrow{i} = \overrightarrow{j}, 
    \quad \overrightarrow{i}\overrightarrow{k} = -\overrightarrow{i} \\
\end{array}\right.
$$

define multiplication $\mathbb{R}^3 \rightarrow \mathbb{R}^3$:
$$
\begin{align*}
\bold{q}_a  \bold{q}_b &=
s_a s_b - x_a x_b - y_a y_b - z_a z_b
\\ & \quad +
(s_a x_b + x_a s_b + y_a z_b - z_a y_b)\overrightarrow{i}
\\ & \quad +
(s_b y_b - x_a z_b + y_a z_b - z_a x_b)\overrightarrow{j}
\\ & \quad +
(s_c z_b + x_a y_b + y_a z_b - z_a s_b)\overrightarrow{k}
\end{align*}
$$

## Quaternion Example

Consider a rotation $\overrightarrow{{u}}$ around $\overrightarrow{v}=\overrightarrow{i} + \overrightarrow{j} + \overrightarrow{k}$ with a rotation angle of $\theta=\frac{2\pi}{3}$. The length of $\overrightarrow{v}$ is $\sqrt{3}$.

<div style="display: flex; justify-content: center;">
      <img src="imgs/quat_rotation.png" width="20%" height="20%" alt="quat_rotation" />
</div>
</br>

Hence
$$
\begin{align*}
\overrightarrow{u}&=
cos \frac{\theta}{2} + sin \frac{\theta}{2} \cdot \frac{\overrightarrow{v}}{||\overrightarrow{v}||}
\\ &=
cos\frac{\pi}{3} + sin\frac{\pi}{3} \cdot \frac{\overrightarrow{v}}{||\sqrt{3}||}
\\ &=
cos\frac{\pi}{3} + sin\frac{\pi}{3} \cdot \frac{\overrightarrow{i} + \overrightarrow{j} + \overrightarrow{k}}{||\sqrt{3}||}
\\ &=
\frac{1+\overrightarrow{i} + \overrightarrow{j} + \overrightarrow{k}}{2}
\end{align*}
$$

This result's Euler angle is $(\frac{\pi}{2}, 0, \frac{\pi}{2})$.

## Quaternion Derivative

Rotation change about the instantaneous axis $\bold{u}=\frac{\bold{\omega}}{||\bold{\omega}||}$ through $\theta=||\bold{\omega}||\Delta t$ can be described by
$$
\begin{align*}
\Delta \bold{q} &= \cos\frac{\theta}{2} + \bold{u} \sin\frac{\theta}{2}
\\ &=
\cos\frac{||\bold{\omega}||\Delta t}{2} + \frac{\bold{\omega}}{||\bold{\omega}||} \sin\frac{||\bold{\omega}||\Delta t}{2}
\end{align*}
$$

By the nature of quaternion rotation, there is $\bold{q}(t+\Delta t)=\Delta\bold{q}\bold{q}(t)$.
Then, define the difference:
$$
\begin{align*}
\bold{q}(t+\Delta t)-\bold{q}(t) &= 
\Big( \cos\frac{||\bold{\omega}||\Delta t}{2} + \frac{\bold{\omega}}{||\bold{\omega}||} \sin\frac{||\bold{\omega}||\Delta t}{2}
 \Big) \bold{q} - \bold{q}
\\ &=
\Big( \cos\frac{||\bold{\omega}||\Delta t}{2} + \frac{\bold{\omega}}{||\bold{\omega}||} \sin\frac{||\bold{\omega}||\Delta t}{2}
 - 1 \Big) \bold{q}
\\ &=
\Big( -2\sin^2\frac{||\bold{\omega}||\Delta t}{4} + \frac{\bold{\omega}}{||\bold{\omega}||} \sin\frac{||\bold{\omega}||\Delta t}{2}
\Big) \bold{q}
\end{align*}
$$
where, for quaternion $\bold{q} \in \mathbb{H}^4$ multiplying with angular velocity $\bold{\omega} \in \mathbb{R}^3$, here redefines $\bold{\omega}=[0\quad \omega_x\quad \omega_y\quad \omega_z] \in \mathbb{H}^4$.

Finally, develop the time-derivative of the quaternions.
$$
\begin{align*}
\frac{d\bold{q}}{d\Delta t} &= \lim_{\Delta t \rightarrow 0} \frac{\bold{q}(t+\Delta t)-\bold{q}(t)}{\Delta t}
\\ &=
\lim_{\Delta t \rightarrow 0}\frac{1}{\Delta t}
\Big( -2\sin^2\frac{||\bold{\omega}||\Delta t}{4} + \frac{\bold{\omega}}{||\bold{\omega}||} \sin\frac{||\bold{\omega}||\Delta t}{2}
\Big) \bold{q}
\\ &=
\Big(0 + \frac{\bold{\omega}}{||\bold{\omega}||} \frac{||\bold{\omega}||}{2} \Big) \bold{q}
&\qquad \text{Apply L'Hopital's Rule, there are } \lim_{x \rightarrow 0}\frac{\sin^2(x)}{x}=0 \text{ and } \lim_{x \rightarrow 0}\frac{\sin(x)}{x}=1
\\ &=
\frac{1}{2} \bold{\omega} \bold{q}
\end{align*}
$$

Define the skew-symmetric of $\bold{\omega}$, such that
$$
\Omega(\bold{\omega}) = 
\begin{bmatrix}
    0 & -\bold{\omega}^{\top} \\
    \bold{\omega} & \bold{\omega}^{\wedge}
\end{bmatrix}=\begin{bmatrix}
    0 & -\omega_x & -\omega_y & -\omega_z \\
    \omega_x & 0 & \omega_z & -\omega_y \\
    \omega_y & -\omega_z & 0 & -\omega_x \\
    \omega_z & \omega_y & -\omega_x & 0\\
\end{bmatrix}
$$

The quaternion derivative can be expressed as $\frac{d\bold{q}}{d\Delta t}=\frac{1}{2} \Omega(\bold{\omega}) \bold{q}$

## Quaternion Integral

The $\bold{q}(t+\Delta t)=\bold{q}_{t+1}$ can be derived from Taylor series 
$$
\begin{align*}
\bold{q}_{t+1} &= \bold{q}_t + \frac{d\bold{q}_{t}}{d\Delta t} \Delta t
+ \frac{1}{2!}\frac{d^2\bold{q}_{t}}{d\Delta t^2} \Delta t^2
+ \frac{1}{3!}\frac{d^3\bold{q}_{t}}{d\Delta t^3} \Delta t^3 + ...
\\ &=
\Big( I_{4 \times 4} + \frac{1}{2} \Omega(\bold{\omega})\Delta t
+ \frac{1}{2!} \big(\frac{1}{2} \Omega(\bold{\omega})\Delta t\big)^2 
+ \frac{1}{3!} \big(\frac{1}{2} \Omega(\bold{\omega}) \Delta t \big)^3 + ... \Big) \bold{q}_t
&& \qquad \text{Apply } \frac{d\bold{q}}{d\Delta t}=\frac{1}{2} \Omega(\bold{\omega}) \bold{q} \text{ to all derivatives recursively to all degrees}
\\ &\qquad +
\frac{1}{4} \frac{d\Omega(\omega)}{d\Delta t}\Delta t^2 \bold{q}_t
+ \Big( \frac{1}{12}\frac{d\Omega(\omega)}{d\Delta t}\Omega(\omega) 
+ \frac{1}{24}\Omega(\omega)\frac{d\Omega(\omega)}{d\Delta t} 
+ \frac{1}{12} \frac{d^2\Omega(\omega)}{d\Delta t^2} \Big) \Delta t^3 \bold{q}_t + ...
\\ &=
\Big( I_{4 \times 4} + \frac{1}{2} \Omega(\bold{\omega})\Delta t
+ \frac{1}{2!} \big(\frac{1}{2} \Omega(\bold{\omega})\Delta t\big)^2 
+ \frac{1}{3!} \big(\frac{1}{2} \Omega(\bold{\omega}) \Delta t \big)^3 + ... \Big) \bold{q}_t
&& \qquad \text{Angular velocity is assumed constant } \frac{d\Omega(\omega)}{d\Delta t}=0
\\ &=
\Big( I_{4 \times 4} + \frac{1}{2} \Omega(\bold{\omega})\Delta t \Big) \bold{q}_t
&& \qquad \text{Removed higher order terms}
\end{align*}
$$

## Quaternion Derivation

Define $\bold{q}^+$ and $\bold{q}^{\oplus}$ as the matrix representation of quaternion.

$$
\bold{q}^+ =
\begin{bmatrix}
    s & - \bold{v}^\text{T} \\
    \bold{v} & sI+\bold{v}^\wedge
\end{bmatrix}
, \quad
\bold{q}^{\oplus} =
\begin{bmatrix}
    s & - \bold{v}^\text{T} \\
    \bold{v} & sI-\bold{v}^\wedge
\end{bmatrix}
$$
where $\wedge$ denotes the skew-symmetric matrix representation of the vector and $I$ is the identity matrix.

Derivation shows as below.
$$
\bold{q}_a^+ \bold{q}_b=
\begin{bmatrix}
    s_a & - \bold{v}_a^\text{T} \\
    \bold{v}_a & s_a I+\bold{v}_a^\wedge
\end{bmatrix}
\begin{bmatrix}
    s_b \\
    \bold{v}_b
\end{bmatrix}=
\begin{bmatrix}
    -\bold{v}_a^\text{T} \bold{v}_b + s_a s_b \\
    s_b \bold{a} + s_b \bold{v}_b + \bold{v}^{\wedge}_a \bold{v}_b
\end{bmatrix}=
\bold{q}_a \bold{q}_b
$$

Similarly, there is
$$
\bold{q}_a \bold{q}_b=
\bold{q}_a^+ \bold{q}_b=
\bold{q}_a \bold{q}_b^{\oplus}
$$

Define a spacial point represented in quaternion $\bold{p}=[0,\bold{v}_p] \in \mathbb{R}^3$ whose rotation is $\bold{p}'=\bold{q}\bold{p}\bold{q}^{-1}$, where $\bold{q}^{-1}$ is the matrix normalization term, there is
$$
\begin{align*}
    \bold{p}'&=\bold{q}\bold{p}\bold{q}^{-1}
    \\ &=
    \bold{q}^+\bold{p}^+\bold{q}^{-1}
    \\ &=
    \bold{q}^+\bold{q}^{-1^\oplus}\bold{p}
\end{align*}
$$

Here computes $\bold{q}^+\bold{q}^{-1^\oplus}$:
$$
\begin{align*}
    
\bold{q}^+\bold{q}^{-1^\oplus}&=
\begin{bmatrix}
    s_a & - \bold{v}_a^\text{T} \\
    \bold{v}_a & s_a I+\bold{v}_a^\wedge
\end{bmatrix}
\begin{bmatrix}
    s & - \bold{v}^\text{T} \\
    \bold{v} & sI-\bold{v}^\wedge
\end{bmatrix}\\ &=
\begin{bmatrix}
    1 & 0 \\
    \bold{0} & \bold{v}\bold{v}^\text{T}+s^2I+ 2s\bold{v}^{\wedge}+(\bold{v}^{\wedge})^2
\end{bmatrix}
\end{align*}
$$

Since $\bold{q}$ is defined as purely imaginary, so that the *quaternion-to-rotation matrix* can be defined as
$$
R =
\bold{v}\bold{v}^\text{T}+s^2I+ 2s\bold{v}^{\wedge}+(\bold{v}^{\wedge})^2
$$

In order to compute $\theta$, trace operation is performed as below
$$
\begin{align*}
    tr(R) &=
    tr \big(
        \bold{v}\bold{v}^\text{T}+s^2I+ 2s\bold{v}^{\wedge}+(\bold{v}^{\wedge})^2
    \big)

    \\ &=
    v_x^2 + v_y^2 + v_z^2 + 3s^2 + 0
    -2(v_x^2 + v_y^2 + v_z^2)

    \\ &=
    (1-s^2) + 3s^2 -2(1-s^2)

    \\ &=
    4s^2-1
\end{align*}
$$

According to *Rodrigues' rotation formula*, $\theta$ can be computed as 
$$
\begin{align*}
    
\theta &= \arccos \bigg(
      \frac{tr(R)-1}{2}
\bigg)
\\ &=
\arccos(2s^2-1)
\\ &=
2 \space \arccos \space s

\end{align*}
$$

