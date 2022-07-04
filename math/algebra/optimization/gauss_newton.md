# Gauss-Newton Method

Gauss–Newton algorithm is used to solve non-linear least squares problems, which is equivalent to minimizing a sum of squared function values.

## Definition

Given $m$ functions $\bold{r}=(r_1, r_2, ..., r_m)$ (aka residuals, often denoted as loss/cost function $\bold{e}=(e_1, e_2, ..., e_m)$) of $n$ variables $\bold{\beta}=(\beta_1, \beta_2, ..., \beta_n)$, with $m \ge n$.

In data fitting, where the goal is to find the parameters $\bold{\beta}$ to a known  model $\bold{f}(\bold{x}, \bold{\beta})$ that best fits observation data $(x_i, y_i)$, there is
$$
r_i = y_i - f(x_i, \bold{\beta})
$$ 

Gauss–Newton algorithm iteratively finds the value of the variables that minimize the sum of squares:
$$
arg \space \underset{\beta}{min} =\sum^m_{i=1}r_i(\beta)^2
$$

Iteration starts with an initial guess $\beta^{(0)}$, then $\beta^{(k)}$ update $\bold{\beta^{(k)}}$ towards local minima:
$$
\beta^{(k+1)}=\beta^{(k)}-(\bold{J}_r^T \bold{J}_r)^{-1} \bold{J}_r^T \bold{r}(\bold{\beta}^{(k)})
$$

where $\bold{J}_{\bold{r}}$ is Jacobian matrix, whose enrty is 
$$
(\bold{J}_{\bold{r}})_{i,j}=\frac{\partial \bold{r}_i (\bold{\beta}^{(k)})}{\partial \beta_j}
$$

Intuitively speaking, $(\bold{J}_r^T \bold{J}_r)^{-1} \bold{J}_r^T$ is a $\mathbb{R}^{n \times m}$ version of Newton method's $\frac{1}{f'(x)}$, and $\bold{r}(\bold{\beta}^{(k)})$ is a $\mathbb{R}^{m \times 1}$ version of Newton method's $f(x)$.

The iteration can be rewritten as
$$
\begin{align*}
\beta^{(k+1)} - \beta^{(k)}
&=
-(\bold{J}_r^T \bold{J}_r)^{-1} \bold{J}_r^T \bold{r}(\bold{\beta}^{(k)})
\\ 
\bold{J}_r^T \bold{J}_r (\beta^{(k+1)} - \beta^{(k)})
&=
-\bold{J}_r^T \bold{r}(\bold{\beta}^{(k)})
\end{align*}
$$

We want to compute the interation step $\Delta = \beta^{(k+1)} - \beta^{(k)}$. 

Now define $A=\bold{J}_r^T \bold{J}_r$, $\bold{x}=\Delta$ and $\bold{b}=-\bold{J}_r^T \bold{r}(\bold{\beta}^{(k)})$, iteration step $\bold{x}=\Delta$ can be found with 
$$
A\bold{x}=\bold{b}
$$
by methods such as QR Householder decomposition.
