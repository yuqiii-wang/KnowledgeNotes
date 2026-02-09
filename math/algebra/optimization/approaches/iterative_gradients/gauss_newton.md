# Gauss-Newton Method

Gauss–Newton algorithm is used to solve non-linear least squares problems, which is equivalent to minimizing a sum of squared function values.

## Definition

Given $m$ functions $\mathbf{r}=(r_1, r_2, ..., r_m)$ (aka residuals, often denoted as loss/cost function $\mathbf{e}=(e_1, e_2, ..., e_m)$) of $n$ variables $\mathbf{\beta}=(\beta_1, \beta_2, ..., \beta_n)$, with $m \ge n$.

In data fitting, where the goal is to find the parameters $\mathbf{\beta}$ to a known  model $\mathbf{f}(\mathbf{x}, \mathbf{\beta})$ that best fits observation data $(x\_i, y_i)$, there is
$$
r_i = y_i - f(x\_i, \mathbf{\beta})
$$ 

Gauss–Newton algorithm iteratively finds the value of the variables that minimize the sum of squares:
$$
arg \space \underset{\beta}{min} =\sum^m_{i=1}r_i(\beta)^2
$$

Iteration starts with an initial guess $\beta^{(0)}$, then $\beta^{(k)}$ update $\mathbf{\beta^{(k)}}$ towards local minima:
$$
\beta^{(k+1)}=\beta^{(k)}-(\mathbf{J}_r^T \mathbf{J}_r)^{-1} \mathbf{J}_r^T \mathbf{r}(\mathbf{\beta}^{(k)})
$$

where $\mathbf{J}_{\mathbf{r}}$ is Jacobian matrix, whose enrty is 
$$
(\mathbf{J}_{\mathbf{r}})_{i,j}=\frac{\partial \mathbf{r}\_i (\mathbf{\beta}^{(k)})}{\partial \beta_j}
$$

Intuitively speaking, $(\mathbf{J}_r^T \mathbf{J}_r)^{-1} \mathbf{J}_r^T$ is a $\mathbb{R}^{n \times m}$ version of Newton method's $\frac{1}{f'(x)}$, and $\mathbf{r}(\mathbf{\beta}^{(k)})$ is a $\mathbb{R}^{m \times 1}$ version of Newton method's $f(x)$.

The iteration can be rewritten as

$$
\begin{align*}
\beta^{(k+1)} - \beta^{(k)}&=
-(\mathbf{J}_r^T \mathbf{J}_r)^{-1} \mathbf{J}_r^T \mathbf{r}(\mathbf{\beta}^{(k)})
\\\\ 
\mathbf{J}_r^T \mathbf{J}_r (\beta^{(k+1)} - \beta^{(k)})&=
-\mathbf{J}_r^T \mathbf{r}(\mathbf{\beta}^{(k)})
\end{align*}
$$

We want to compute the interation step $\Delta = \beta^{(k+1)} - \beta^{(k)}$. 

Now define $A=\mathbf{J}_r^T \mathbf{J}_r$, $\mathbf{x}=\Delta$ and $\mathbf{b}=-\mathbf{J}_r^T \mathbf{r}(\mathbf{\beta}^{(k)})$, iteration step $\mathbf{x}=\Delta$ can be found with 
$$
A\mathbf{x}=\mathbf{b}
$$

by methods such as QR Householder decomposition.
