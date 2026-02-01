# LQR (Linear Quadratic Regulator)

Consider this linear system:
$$
\begin{align*}
\dot{\mathbf{x}}(t) &= A\mathbf{x}(t) + B \mathbf{u}(t) \\\\
\mathbf{y}(t) &= C{\mathbf{x}}(t)
\end{align*}
$$

for $\mathbf{x}(0)=\mathbf{x}_0 \ne \mathbf{0}$.

<div style="display: flex; justify-content: center;">
      <img src="imgs/lqr.png" width="40%" height="40%" alt="lqr" />
</div>
</br>

By this applied correction $\mathbf{u}(t) = K \mathbf{x}(t)$, LQR attempts to minimize this cost function:

$$
\min \frac{1}{2} \mathbf{x}^\top(t_f) S \mathbf{x}(t_f)+\frac{1}{2} \int^{t_f}\_{t_0} \Big( 
\mathbf{x}(t)^\top Q \mathbf{x}(t) + \mathbf{u}^\top (t)R \mathbf{u}(t)
\Big) dt
$$

subject to
$$
\dot{\mathbf{x}} = A\mathbf{x} + B \mathbf{u}
,\qquad
\mathbf{x}(0)=\mathbf{x}_0 \ne \mathbf{0}
$$

where $\frac{1}{2} \mathbf{x}\_{t_f}^\top S \mathbf{x}\_{t_f}$ represents the error of when the system reaches a stable state at the time $t_f$; 
$\frac{1}{2} \int^{t_f}\_{t_0} \Big( 
\mathbf{x}^\top Q \mathbf{x} + \mathbf{x}^\top R \mathbf{x} \Big) dt$ is the cost to be reduced to zero.
In other words, over the integrated period $[t_0, t_f]$, the total $R$-transformed action $\mathbf{u}$ should cancel out the total $Q$-transformed state error $\mathbf{x}$

$R$ is positive definite; $Q$ and $S$ are semi-positive definite.

Solution to this system can be found by *Riccati* equations:
$$
\dot{P}(t) = 
-P(t)A -A^\top +P(t)BR^{-1}B^\top P(t) -Q
$$

then set
$$
\mathbf{u}(t) = R^{-1}B^\top P(t) \mathbf{x}(t)=K\mathbf{x}(t)
$$

from here update the system's close loop coefficient $A_{cl}=A-BK$.

## Example in Vehicle Horizontal Adjustment

Assumed a vehicle moving forward at a speed $v$, has a horizontal reference error (to the road central positioning) $e_h$, and a small heading angle error $e_{\theta}$.

Approximate the trivial heading angle error $\sin(e_{\theta})\approx e_{\theta}$.

Horizontal position translation is $S$ and its velocity is $\dot{S}=S \cdot \sin(e_{\theta})$, 
and vehicle angular velocity is $\dot{\theta}=\omega$.

Now, want to adjust $\omega$ to get the vehicle back to the road reference line $e_h=0$.

The system is defined as

$$
\begin{align*}
\begin{bmatrix}
    \dot{S} \\\\
    \dot{\theta}
\end{bmatrix}&=
\underbrace{\begin{bmatrix}
    0 & v \\\\
    0 & 0
\end{bmatrix}}\_{A}
\begin{bmatrix}
    S \\\\
    \theta
\end{bmatrix}
+
\underbrace{\begin{bmatrix}
    0 \\\\
    1
\end{bmatrix}}\_{B}
\omega
\end{align*}
$$

The output $\mathbf{y}=S$ is simply the horizontal positioning translation such that $\mathbf{y}=C\mathbf{x}$, where $C=\begin{bmatrix}  1 & 0 \\\\ 0 & 0 \end{bmatrix}$.

The cost coefficient matrices can be defined as
$$
Q=\begin{bmatrix}
    1 & 0 \\\\
    0 & 1
\end{bmatrix}
,\qquad
R=\begin{bmatrix}
    1 & 0 \\\\
    0 & 0
\end{bmatrix}
$$

By Riccati Equation, $K$ can be computed to update the new close loop system coefficient matrix $A_{cl}=A-BK$.

Finally, by this new $A_{cl}$, the output $\mathbf{y}(t)$ should gradually converge to the horizon reference line as $t \rightarrow t_f$.

$$
\begin{align*}
\dot{\mathbf{x}}(t) &= A_{cl}\mathbf{x}(t) + B \mathbf{u}(t) \\\\
\mathbf{y}(t) &= C{\mathbf{x}}(t)
\end{align*}
$$

In other words, when $t=t_f$, there is $\mathbf{y}(t_f)=S_{t_f}=0$