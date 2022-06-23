# Kalman Filter

Motivation: assumed a system having Gaussian noises on state update calculation and state observation, given a number of iterations, the conclusion on the actual state measurement converges.

## Example

Distance $x$ and velocity $\dot{x}$ of a vehicle is given below
$$
x = 
\begin{bmatrix}
x \\
\dot{x}
\end{bmatrix}
$$

Vehicle drives with a constant acceleration $a_k$ between two timesteps $k-1$ and $k$, following normal distribution with mean $0$ and standard deviation $\sigma_a$. Given Newton's laws of motion:
$$
x_k = F x_{k-1} + G a_k
$$
where
$$
F = 
\begin{bmatrix}
1 & \Delta t \\
0 & 1
\end{bmatrix}
, \space
G = 
\begin{bmatrix}
\frac{1}{2} \Delta t^2 \\
\Delta t
\end{bmatrix}
$$

Given $a_k$ following normal distribution, there is 
$$
x_k = F x_{k-1} + w_k
$$
where $w_k \sim N(0, Q)$, in which 
$$
Q = \sigma_{a_k}G G^T \sigma_{a_k} =
\begin{bmatrix}
\frac{1}{4}\Delta t^4 & \frac{1}{2}\Delta t^3 \\
\frac{1}{2}\Delta t^3 & \Delta t^2
\end{bmatrix}
\sigma_{a_k}^2
$$

Since $GG^T$ is not full ranked ($R_1 = [\frac{1}{4}\Delta t^4, \frac{1}{2}\Delta t^3] = \frac{1}{2}\Delta t^3 R_2$) hence 
$w_k \sim G \cdot N(0, Q) \sigma_{a_k}^2 \sim G \cdot N (0, \sigma_{a_k}^2)$

Here defines Observation 

$z_k = H x_k + v_k$

where $v_k$ follows $N(0, \sigma_z)$ and $H$ is observation transformation matrix 
$$
H = 
\begin{bmatrix}
1 \\
0
\end{bmatrix}
$$

Here $R = E[v_k v_k^T] = \sigma_{z}^2$.