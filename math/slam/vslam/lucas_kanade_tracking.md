# Lucas-Kanade Tracking

Lucas-Kanade in nature is a least squares problem that attempts to minimize the differences between two image $I_A$ and $I_B$ pixels (assumed image is of a size $n \times n$) by moving one image $I_B$ by $(dx, dy)$. By $\mathcal{L}_2$ norm error, there is
$$
\underset{dx, dy}{\min}
\sum_{x,y} 
\big|\big|
I_A(x,y) - I_B(x+dx,y+dy)
\big|\big|^2
$$

It can be used to find the offset between two chronologically sequential images, inside which moving objects' features are recorded. By these features, objects' poses can be estimated.

<div style="display: flex; justify-content: center;">
      <img src="imgs/sparse_optical_flow_example.png" width="40%" height="40%" alt="sparse_optical_flow_example" />
</div>
</br>

### Least Squares Retrospect

Given a residual function $\mathbf{r}(\mathbf{x})$ where $\mathbf{x} \in \mathbb{R}^n$ and output $\mathbf{r} \in \mathbb{R}^m$. Define an $\mathcal{L}_2$ norm cost function to be minimized:
$$
\underset{\mathbf{x}}{\min}\quad \mathbf{r}(\mathbf{x})^\top \mathbf{r}(\mathbf{x})
$$

By first order approximation, there is
$$
\mathbf{r}(\Delta \mathbf{x}) \approx
J(\mathbf{x}_0) \Delta \mathbf{x} + \mathbf{r}(\mathbf{x}_0)
$$

where $J$ is the Jacobian.

Take shorthand notes and write down $J_0=J(\mathbf{x}_0)$ and $\mathbf{r}_0=\mathbf{r}(\mathbf{x}_0)$, and take the Jacobian approximation into the cost function, there is
$$
\underset{\mathbf{x}}{\min}\quad \mathbf{r}(\mathbf{x})^\top \mathbf{r}(\mathbf{x})=
\Delta\mathbf{x}^\top J_0^\top J_0 \Delta\mathbf{x}+2 J_0^\top \mathbf{r}_0 \Delta\mathbf{x} +\mathbf{r}_0^\top \mathbf{r}_0
$$

The minima exists at $\frac{\partial\space \mathbf{r}^\top \mathbf{r}}{\partial \Delta\mathbf{x}}=0$, so that

$$
\begin{align*}
  &&
  2 J_0^\top J_0 \Delta\mathbf{x} &=
  2 J_0^\top \mathbf{r}_0
  \\\\ \Rightarrow &&
  \Delta\mathbf{x} &=
  \frac{J_0^\top \mathbf{r}_0}{J_0^\top J_0}
\end{align*}
$$

The optimal $\mathbf{x}^*$ can be determined by $\mathbf{x}^* = \mathbf{x}_0 + \Delta\mathbf{x}$

## Lucas-Kanade Problem Formulation

The Lucas-Kanade cost function shown as below is a least squares problem.

$$
\underset{dx, dy}{\min}
\sum_{x,y} 
\big|\big|
I_A(x,y) - I_B(x+dx,y+dy)
\big|\big|^2
$$
whose residuals with respects to each pixel are
$$
\mathbf{r}(dx,dy) = 
\begin{bmatrix}
    r_{x=1, y=1}(dx,dy) \\\\
    r_{x=1, y=2}(dx,dy) \\\\
    \vdots
    \\\\
    r_{x=i, y=j}(dx,dy) \\\\
    \vdots
    \\\\
    r_{x=n, y=n}(dx,dy) \\\\
\end{bmatrix}=
\begin{bmatrix}
    I_A(1,1) - I_B(1+dx, 1+dy) \\\\
    I_A(1,2) - I_B(1+dx, 2+dy) \\\\
    \vdots
    \\\\
    I_A(i,j) - I_B(i+dx, j+dy) \\\\
    \vdots
    \\\\
    I_A(n,n) - I_B(n+dx, n+dy) \\\\
\end{bmatrix}
$$

For a particular pixel $p(x\_i,y_j)$, the residual can be approximated by Jacobian

$$
\begin{align*}
    &&
    r_{x=i, y=j}(dx,dy)
    &=
    I_A(i,j) - I_B(i+dx, j+dy)
    \\\\ \Rightarrow &&
    r_{x=i, y=j}(\Delta dx,\Delta dy)
    &= 
    \underbrace{\begin{bmatrix}
        \frac{\partial r_{x=i, y=j} p(x\_i,y_j)}{\partial dx} &
        \frac{\partial r_{x=i, y=j} p(x\_i,y_j)}{\partial dy}
    \end{bmatrix}}_{\text{Jacobian}}
    \begin{bmatrix}
        \Delta dx \\\\
        \Delta dy
    \end{bmatrix}
    + r_{x=i, y=j}
\end{align*}
$$

To find the optimal $\begin{bmatrix}      \Delta dx & \Delta dy \end{bmatrix}^*$, its Jacobian should be computed.

## Lucas-Kanade Jacobian Formulation

Set the $dx=a$ and $dy=b$ as image offset. For the whole image's Jacobian, there is (only $x$-axis is shown, $y$-axis should the same expression).

$$
\begin{align*}
    \frac{\partial \mathbf{r}(dx,dy)}{\partial dx}
    \bigg|_{\begin{align*}
        \footnotesize{dx=a, dy=b} \\\\
        \footnotesize{x=1,2,...,n} \\\\
        \footnotesize{y=1,2,...,n} \\\\
    \end{align*}}
    &=
    \frac{\partial 
    \big(I_A(x,y)-I_B(x+dx, y+dy)\big)}{\partial dx}
    \bigg|_{dx=a, dy=b}
    \\\\ &=
    \frac{\partial 
    \big(-I_B(x+dx, y+dy)\big)}{\partial dx}
    \bigg|_{dx=a, dy=b}
\end{align*}
$$

This expression describes the image's $n \times n$ pixels' change rate along the $x$-axis.

By strict math definition, the Jacobian is not continuous for $dx, dy \in \mathbb{Z}^+$ and $0 \le dx, dy < n$. The Jacobian can be approximated by differential. An easy approximation is to set $\Delta=1$ since the smallest amount of image offset is one pixel.

$$
\begin{align*}
    \frac{\partial 
    \big(-I_B(x+dx, y+dy)\big)}{\partial dx}
    \bigg|_{dx=a, dy=b}
    &= 
    \lim_{\Delta \rightarrow 0} - 
    \frac{I_B(x+a-\Delta, y+b)-I_B(x+a+\Delta, y+b)}
    {2\Delta}
    \\\\ & \underset{\Delta=1}{\approx}
    - 
    \frac{I_B(x+a-1, y+b)-I_B(x+a+1, y+b)}
    {2}
\end{align*}
$$

Actually, the $\Delta=1$ approximation is exactly the operation of Sobel Derivatives:
$$
G_x=\begin{bmatrix}
    -1 & 0 & +1 \\\\
    -2 & 0 & +2 \\\\
    -1 & 0 & +1 \\\\
\end{bmatrix}
, \quad
G_y=\begin{bmatrix}
    -1 & -2 & -1 \\\\
    0 & 0 & 0 \\\\
    +1 & +2 & +1 \\\\
\end{bmatrix}
$$

## G2O Implementation