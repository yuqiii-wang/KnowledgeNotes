# Triangulation

PnP (Perspective-n-Point) describes how a 3d world point $\bold{X}=[X\quad Y\quad Z\quad 1]^\top$ is projected to an image pixel $\bold{x}=[u\quad v\quad 1]^\top$ scaled by $s$.
Denote the projection as $\bold{P}$.
$$
s \underbrace{\begin{bmatrix}
    u \\
    v \\
    1
\end{bmatrix}}_{\bold{x}}
=
\underbrace{
\begin{bmatrix}
    f_x & 0 & c_x \\
    0 & f_y & c_y \\
    0 & 0 & 1
\end{bmatrix}
\underbrace{
    \begin{bmatrix}
        t_1 & t_2 & t_3 & t_4 \\
        t_5 & t_6 & t_7 & t_8 \\
        t_9 & t_{10} & t_{11} & t_{12} \\
    \end{bmatrix}
}_{[\bold{R}|\bold{t}]}
}_{\bold{P}=\begin{bmatrix}
        p_1 & p_2 & p_3 & p_4 \\
        p_5 & p_6 & p_7 & p_8 \\
        p_9 & p_{10} & p_{11} & p_{12} \\
    \end{bmatrix}=\begin{bmatrix}
        \bold{p}_1^\top \\
        \bold{p}_2^\top \\
        \bold{p}_3^\top \\
    \end{bmatrix}
    }
\underbrace{\begin{bmatrix}
    X \\
    Y \\
    Z \\
    1
\end{bmatrix}}_{\bold{X}}
$$
where $\bold{p}_1^\top=[p_1\quad p_2\quad p_3\quad p_4],\qquad \bold{p}_2^\top=[p_5\quad p_6\quad p_7\quad p_8], \qquad \bold{p}_3^\top=[p_9\quad p_{10}\quad p_{11}\quad p_{12}]$.

So that 
$$
\begin{align*}
&&
s \begin{bmatrix}
    u \\
    v \\
    1
\end{bmatrix} &=
\begin{bmatrix}
    \bold{p}_1^\top \\
    \bold{p}_2^\top \\
    \bold{p}_3^\top \\
\end{bmatrix}
\bold{X}
\\ \Rightarrow && &=
\begin{bmatrix}
    \bold{p}_1^\top \bold{X} \\
    \bold{p}_2^\top \bold{X} \\
    \bold{p}_3^\top \bold{X} \\
\end{bmatrix}
\end{align*}
$$

Triangulation attempts to find the 3d world point $\bold{X}$ given known $\bold{x}$ and $\bold{P}$.
In other words, solve for $\bold{X}$ by
$$
\underset{\text{known}}{\bold{x}}
=
\underset{\text{known}}{\bold{P}}
\bold{X}
$$

Given two images' pixels denoted as $\bold{x}'=\bold{X}_{\text{L}}$ and $\bold{x}=\bold{X}_{\text{R}}$, there are
$$
\bold{x}'=\bold{P}'\bold{X}
\qquad
\bold{x}=\bold{P}\bold{X}
$$

<div style="display: flex; justify-content: center;">
      <img src="imgs/epipolar_geo.png" width="40%" height="40%" alt="epipolar_geo" />
</div>
</br>

Recall the epipolar geometry that $\bold{x} \times \bold{P}\bold{X}=\bold{0}$ for $\bold{x}$ and $\bold{P}\bold{X}$ are co-planar and their cross product should be zero. Removed the scale $s$, there is
$$
\begin{align*}
&& 
\bold{x} \times \bold{P}\bold{X} &= \bold{0}
\\ \Rightarrow && 
\begin{bmatrix}
    u \\
    v \\
    1
\end{bmatrix} \times
\begin{bmatrix}
    \bold{p}_1^\top \bold{X} \\
    \bold{p}_2^\top \bold{X} \\
    \bold{p}_3^\top \bold{X} \\
\end{bmatrix}
&=
\begin{bmatrix}
    v\bold{p}_3^\top \bold{X} - \bold{p}_1^\top \bold{X} \\
    \bold{p}_1^\top \bold{X} - u\bold{p}_3^\top \bold{X}  \\
    u\bold{p}_2^\top \bold{X} - v\bold{p}_1^\top \bold{X} 
\end{bmatrix}
=
\begin{bmatrix}
    0 \\
    0 \\
    0
\end{bmatrix}
\end{align*}
$$

The third row $u\bold{p}_2^\top \bold{X} - v\bold{p}_1^\top \bold{X}$ is actually the linear combination of the first two rows. Remove the third row, derive the linearly independent rows:

$$
\begin{bmatrix}
    v\bold{p}_3^\top \bold{X} - \bold{p}_1^\top \bold{X} \\
    \bold{p}_1^\top \bold{X} - u\bold{p}_3^\top \bold{X}
\end{bmatrix}
=
\begin{bmatrix}
    0 \\
    0
\end{bmatrix}
$$

Consider the other side image pixels $(u', v')$ and its corresponding projection matrix $\bold{P'}$, there is $A\bold{X}=\bold{0}$
$$
\begin{bmatrix}
    v\bold{p}_3^\top \bold{X} - \bold{p}_1^\top \bold{X} \\
    \bold{p}_1^\top \bold{X} - u\bold{p}_3^\top \bold{X} \\
    v'\bold{p'}_3^\top \bold{X} - \bold{p'}_1^\top \bold{X} \\
    \bold{p'}_1^\top \bold{X} - u'\bold{p'}_3^\top \bold{X}
\end{bmatrix}
\overset{\text{take out }\bold{X}}{=}
\underbrace{\begin{bmatrix}
    v\bold{p}_3^\top - \bold{p}_1^\top \\
    \bold{p}_1^\top - u\bold{p}_3^\top \\
    v'\bold{p'}_3^\top - \bold{p'}_1^\top \\
    \bold{p'}_1^\top - u'\bold{p'}_3^\top
\end{bmatrix}}_{A}
\bold{X}
=
\begin{bmatrix}
    0 \\
    0 \\
    0 \\
    0
\end{bmatrix}
$$

In practice, there could be many more image pixels: one 3d world point $\bold{X}$ can be co-observed by many camera frames, where each frame has a pixel $(u, v)$ describing it. So that
$$

\underbrace{\begin{bmatrix}
    v\bold{p}_3^\top - \bold{p}_1^\top \\
    \bold{p}_1^\top - u\bold{p}_3^\top \\
    v'\bold{p'}_3^\top - \bold{p'}_1^\top \\
    \bold{p'}_1^\top - u'\bold{p'}_3^\top \\
    v''\bold{p''}_3^\top - \bold{p''}_1^\top \\
    \bold{p''}_1^\top - u''\bold{p''}_3^\top \\
    \vdots
\end{bmatrix}}_{A}
\bold{X}
=
\begin{bmatrix}
    0 \\
    0 \\
    0 \\
    0 \\
    0 \\
    0 \\
    \vdots
\end{bmatrix}
$$

Apparently, $\bold{X}$ is over-determined. 
To solve $A\bold{X}=\bold{0}$, formulate it to a least squares problem.
$$
\begin{align*}
&&
    \min \big|\big| A\bold{X} \big|\big|^2
    &= \sum_i (\bold{a}_i \bold{X})^2
\\ \text{subject to } &&
\big|\big| \bold{X} \big|\big|^2 &= 1
\end{align*}
$$

This is equivalent to $\min_{\bold{X}} \frac{\big|\big| A\bold{X} \big|\big|^2}{\big|\big| \bold{X} \big|\big|^2}$.
The constraint $\big|\big| \bold{X} \big|\big|^2 = 1$ is introduced by setting the scaling to $s=1$.

### Rayleigh Quotient

The above optimization problem can be solved by Rayleigh quotient.

$$
\min_{\bold{X}} \frac{\big|\big| A\bold{X} \big|\big|^2}{\big|\big| \bold{X} \big|\big|^2} 
=
\frac{\bold{X}^\top A^\top A \bold{X}}{\bold{X}^\top \bold{X}}
$$

Rayleigh quotient states that, the quotient reaches its minimum when $\bold{X}=\bold{v}_{min}$, where $\bold{v}_{min}$ is the eigenvector corresponding to the smallest eigenvalue $\sigma_{min}$ of $A^\top A$.
Similarly, it reaches maximum when $\bold{X}=\bold{v}_{max}$ corresponding to $\sigma_{max}$.

For the minimization problem, the optimum is $\bold{X}^*=\bold{v}_{min}$ computed from $A^\top A$.

$\bold{v}_{min} = [v_1\quad v_2\quad v_3\quad v_4]$ has four elements, and the fourth element relates to the scaling factor $s$. 
To take the scaling into account, the actual position of a 3d world point $(X, Y, Z)$ should be scaled by $s$, such that for depth, there is $Z=\frac{v_3}{v_4}$.

