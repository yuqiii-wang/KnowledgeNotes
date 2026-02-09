# Triangulation

PnP (Perspective-n-Point) describes how a 3d world point $\mathbf{X}=[X\quad Y\quad Z\quad 1]^\top$ is projected to an image pixel $\mathbf{x}=[u\quad v\quad 1]^\top$ scaled by $s$.
Denote the projection as $\mathbf{P}$.

$$
s \underbrace{\begin{bmatrix}
    u \\\\
    v \\\\
    1
\end{bmatrix}}_{\mathbf{x}} =
\underbrace{
\begin{bmatrix}
    f_x & 0 & c_x \\\\
    0 & f_y & c_y \\\\
    0 & 0 & 1
\end{bmatrix}
\underbrace{
    \begin{bmatrix}
        t_1 & t_2 & t_3 & t_4 \\\\
        t_5 & t_6 & t_7 & t_8 \\\\
        t_9 & t_{10} & t_{11} & t_{12} \\\\
    \end{bmatrix}
}_{[\mathbf{R}|\mathbf{t}]}
}_{\mathbf{P}=\begin{bmatrix}
        p_1 & p_2 & p_3 & p_4 \\\\
        p_5 & p_6 & p_7 & p_8 \\\\
        p_9 & p_{10} & p_{11} & p_{12} \\\\
    \end{bmatrix}=\begin{bmatrix}
        \mathbf{p}_1^\top \\\\
        \mathbf{p}_2^\top \\\\
        \mathbf{p}_3^\top \\\\
    \end{bmatrix}
    }
\underbrace{\begin{bmatrix}
    X \\\\
    Y \\\\
    Z \\\\
    1
\end{bmatrix}}_{\mathbf{X}}
$$

where $\mathbf{p}_1^\top=[p_1\quad p_2\quad p_3\quad p_4],\qquad \mathbf{p}_2^\top=[p_5\quad p_6\quad p_7\quad p_8], \qquad \mathbf{p}_3^\top=[p_9\quad p_{10}\quad p_{11}\quad p_{12}]$.

So that 

$$
\begin{align*}
&& s \begin{bmatrix}
    u \\\\
    v \\\\
    1
\end{bmatrix} &=
\begin{bmatrix}
    \mathbf{p}_1^\top \\\\
    \mathbf{p}_2^\top \\\\
    \mathbf{p}_3^\top \\\\
\end{bmatrix}
\mathbf{X}
\\\\ \Rightarrow && &=
\begin{bmatrix}
    \mathbf{p}_1^\top \mathbf{X} \\\\
    \mathbf{p}_2^\top \mathbf{X} \\\\
    \mathbf{p}_3^\top \mathbf{X} \\\\
\end{bmatrix}
\end{align*}
$$

Triangulation attempts to find the 3d world point $\mathbf{X}$ given known $\mathbf{x}$ and $\mathbf{P}$.
In other words, solve for $\mathbf{X}$ by

$$
\underset{\text{known}}{\mathbf{P}}
\mathbf{X}
$$

Given two images' pixels denoted as $\mathbf{x}'=\mathbf{X}_{\text{L}}$ and $\mathbf{x}=\mathbf{X}_{\text{R}}$, there are

$$
\mathbf{x}'=\mathbf{P}'\mathbf{X}
\qquad
\mathbf{x}=\mathbf{P}\mathbf{X}
$$

<div style="display: flex; justify-content: center;">
      <img src="imgs/epipolar_geo.png" width="40%" height="40%" alt="epipolar_geo" />
</div>
</br>

Recall the epipolar geometry that $\mathbf{x} \times \mathbf{P}\mathbf{X}=\mathbf{0}$ for $\mathbf{x}$ and $\mathbf{P}\mathbf{X}$ are co-planar and their cross product should be zero. Removed the scale $s$, there is

$$
\begin{align*}
&& 
\mathbf{x} \times \mathbf{P}\mathbf{X} &= \mathbf{0}
\\\\ \Rightarrow && 
\begin{bmatrix}
    u \\\\
    v \\\\
    1
\end{bmatrix} \times
\begin{bmatrix}
    \mathbf{p}_1^\top \mathbf{X} \\\\
    \mathbf{p}_2^\top \mathbf{X} \\\\
    \mathbf{p}_3^\top \mathbf{X} \\\\
\end{bmatrix}&=
\begin{bmatrix}
    v\mathbf{p}_3^\top \mathbf{X} - \mathbf{p}_1^\top \mathbf{X} \\\\
    \mathbf{p}_1^\top \mathbf{X} - u\mathbf{p}_3^\top \mathbf{X}  \\\\
    u\mathbf{p}_2^\top \mathbf{X} - v\mathbf{p}_1^\top \mathbf{X} 
\end{bmatrix} =
\begin{bmatrix}
    0 \\\\
    0 \\\\
    0
\end{bmatrix}
\end{align*}
$$

The third row $u\mathbf{p}_2^\top \mathbf{X} - v\mathbf{p}_1^\top \mathbf{X}$ is actually the linear combination of the first two rows. Remove the third row, derive the linearly independent rows:

$$
\begin{bmatrix}
    v\mathbf{p}_3^\top \mathbf{X} - \mathbf{p}_1^\top \mathbf{X} \\\\
    \mathbf{p}_1^\top \mathbf{X} - u\mathbf{p}_3^\top \mathbf{X}
\end{bmatrix} =
\begin{bmatrix}
    0 \\\\
    0
\end{bmatrix}
$$

Consider the other side image pixels $(u', v')$ and its corresponding projection matrix $\mathbf{P'}$, there is $A\mathbf{X}=\mathbf{0}$

$$
\begin{bmatrix}
    v\mathbf{p}_3^\top \mathbf{X} - \mathbf{p}_1^\top \mathbf{X} \\\\
    \mathbf{p}_1^\top \mathbf{X} - u\mathbf{p}_3^\top \mathbf{X} \\\\
    v'\mathbf{p'}_3^\top \mathbf{X} - \mathbf{p'}_1^\top \mathbf{X} \\\\
    \mathbf{p'}_1^\top \mathbf{X} - u'\mathbf{p'}_3^\top \mathbf{X}
\end{bmatrix}
\overset{\text{take out }\mathbf{X}}{=}
\underbrace{\begin{bmatrix}
    v\mathbf{p}_3^\top - \mathbf{p}_1^\top \\\\
    \mathbf{p}_1^\top - u\mathbf{p}_3^\top \\\\
    v'\mathbf{p'}_3^\top - \mathbf{p'}_1^\top \\\\
    \mathbf{p'}_1^\top - u'\mathbf{p'}_3^\top
\end{bmatrix}}_{A}
\mathbf{X} =
\begin{bmatrix}
    0 \\\\
    0 \\\\
    0 \\\\
    0
\end{bmatrix}
$$

In practice, there could be many more image pixels: one 3d world point $\mathbf{X}$ can be co-observed by many camera frames, where each frame has a pixel $(u, v)$ describing it. So that

$$
\underbrace{\begin{bmatrix}
    v\mathbf{p}_3^\top - \mathbf{p}_1^\top \\\\
    \mathbf{p}_1^\top - u\mathbf{p}_3^\top \\\\
    v'\mathbf{p'}_3^\top - \mathbf{p'}_1^\top \\\\
    \mathbf{p'}_1^\top - u'\mathbf{p'}_3^\top \\\\
    v''\mathbf{p''}_3^\top - \mathbf{p''}_1^\top \\\\
    \mathbf{p''}_1^\top - u''\mathbf{p''}_3^\top \\\\
    \vdots
\end{bmatrix}}_{A}
\mathbf{X} =
\begin{bmatrix}
    0 \\\\
    0 \\\\
    0 \\\\
    0 \\\\
    0 \\\\
    0 \\\\
    \vdots
\end{bmatrix}
$$

Apparently, $\mathbf{X}$ is over-determined. 
To solve $A\mathbf{X}=\mathbf{0}$, formulate it to a least squares problem.

$$
\begin{align*}
&&
    \min \big|\big| A\mathbf{X} \big|\big|^2
    &= \sum_i (\mathbf{a}_i \mathbf{X})^2
\\\\ \text{subject to } &&
\big|\big| \mathbf{X} \big|\big|^2 &= 1
\end{align*}
$$

This is equivalent to $\min_{\mathbf{X}} \frac{\big|\big| A\mathbf{X} \big|\big|^2}{\big|\big| \mathbf{X} \big|\big|^2}$.
The constraint $\big|\big| \mathbf{X} \big|\big|^2 = 1$ is introduced by setting the scaling to $s=1$.

### Rayleigh Quotient

The above optimization problem can be solved by Rayleigh quotient.

$$
\min_{\mathbf{X}} \frac{\big|\big| A\mathbf{X} \big|\big|^2}{\big|\big| \mathbf{X} \big|\big|^2} =
\frac{\mathbf{X}^\top A^\top A \mathbf{X}}{\mathbf{X}^\top \mathbf{X}}
$$

Rayleigh quotient states that, the quotient reaches its minimum when $\mathbf{X}=\mathbf{v}_{min}$, where $\mathbf{v}_{min}$ is the eigenvector corresponding to the smallest eigenvalue $\sigma_{min}$ of $A^\top A$.
Similarly, it reaches maximum when $\mathbf{X}=\mathbf{v}_{max}$ corresponding to $\sigma_{max}$.

For the minimization problem, the optimum is $\mathbf{X}^*=\mathbf{v}_{min}$ computed from $A^\top A$.

$\mathbf{v}_{min} = [v_1\quad v_2\quad v_3\quad v_4]$ has four elements, and the fourth element relates to the scaling factor $s$. 
To take the scaling into account, the actual position of a 3d world point $(X, Y, Z)$ should be scaled by $s$, such that for depth, there is $Z=\frac{v_3}{v_4}$.
