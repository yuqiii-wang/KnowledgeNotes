# Levenberg-Marquardt's Method

It is used to solve the least-squares curve fitting problem:
given a set of $m$ empirical pairs $(\mathbf{x}\_i, \mathbf{y}\_i)$, where $\mathbf{x}\_i$ is independent having $n$ dimensions/features such as $\mathbf{x}=[\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_m]$, of which $\mathbf{x}\_i=(x_{i,1}, x_{i,2}, ..., x_{i,n})$, while $\mathbf{y}\_i$ is dependent/observation, find the parameters $\beta$ of tthe model curve $f(x, \beta)$ so that the sum of the squares of the deviations $S(\beta)$ is minimized:

$$
\argmin_{\beta}
\sum^m_{i=1}[\mathbf{y}\_i - f(\mathbf{x}\_i, \beta)]^2
$$

Levenberg-Marquardt Method attempts to find solutions (minima/maxima) with an initial guess $\mathbf{\beta}_0$ not "far away" from the solution ("far away" means there is no more other stationary points in between the initial guess and the solution).

On each iteration, there is $\mathbf{\beta} \leftarrow \mathbf{\beta}+\mathbf{\sigma}$, computed by linear approximation:
$$
f(\mathbf{x}\_i, \mathbf{\beta}+\mathbf{\sigma})
\approx
f(\mathbf{x}\_i, \mathbf{\beta}) + \mathbf{J}\_i\mathbf{\sigma}\_i
$$
where $\mathbf{J}\_i$ is Jacobian matrix entry to partial derivative $\mathbf{\beta}$ such as
$$
\mathbf{J}\_i=\frac{\partial f(\mathbf{x}\_i, \mathbf{\beta})}{\partial \mathbf{\beta}}
$$

Hence,

$$
\begin{align*}
S(\mathbf{\beta}+\mathbf{\sigma}) &\approx
\sum_{i=1}^m [\mathbf{y}\_i - f(\mathbf{x}\_i,\mathbf{\beta})-\mathbf{J}\_i \sigma]^2 \\ &=
||\mathbf{y}-\mathbf{f}(\beta)-\mathbf{J}\mathbf{\sigma}||^2 \\ &=
[\mathbf{y}-\mathbf{f}(\beta)-\mathbf{J}\mathbf{\sigma}]^T[\mathbf{y}-\mathbf{f}(\beta)-\mathbf{J}\mathbf{\sigma}] \\ &=
[\mathbf{y}-\mathbf{f}(\beta)]^T[\mathbf{y}-\mathbf{f}(\beta)] -
[\mathbf{y}-\mathbf{f}(\beta)]^T \mathbf{J}\mathbf{\sigma} -
\mathbf{J}\mathbf{\sigma}^T [\mathbf{y}-\mathbf{f}(\beta)]+
\mathbf{J}^T \mathbf{\sigma}^T \mathbf{J}\mathbf{\sigma}
\end{align*}
$$

Compute $S(\mathbf{\beta}+\mathbf{\sigma})$'s derivative and set it to zero to find stationary points
$$
\begin{align*}
\frac{\partial \big(S(\mathbf{\beta}+\mathbf{\sigma}) \big)
}{\partial \mathbf{\sigma}}&=
\frac{\partial \space
    \big(-2[\mathbf{y}-\mathbf{f}(\beta)]^T \mathbf{J}\mathbf{\sigma}
    +
    \mathbf{J}^T \mathbf{\sigma}^T \mathbf{J}\mathbf{\sigma}
    \big)}
    {
        \partial \mathbf{\sigma}
    }
\\ &=
2[\mathbf{y}-\mathbf{f}(\beta)]^T \mathbf{J}
+
2 \mathbf{J}^T \mathbf{J} \mathbf{\sigma}
\end{align*}
$$

By setting the derivative to zero, there is
$$
\begin{align*}
0 &=
-2[\mathbf{y}-\mathbf{f}(\beta)]^T \mathbf{J}
+
2 \mathbf{J}^T \mathbf{J} \mathbf{\sigma}
\\ 
\mathbf{J}^T \mathbf{J} \mathbf{\sigma}&=
[\mathbf{y}-\mathbf{f}(\beta)]^T \mathbf{J}
\\
\mathbf{J}^T \mathbf{J} \mathbf{\sigma}&=
\mathbf{J}^T [\mathbf{y}-\mathbf{f}(\beta)]
\end{align*}
$$

Here comes the innovation of Levenberg-Marquardt's method derived from the above Gauss-Newton's method:

* Introduce a damping parameter $\lambda$ to the diagnol matrix $\text{diag}(\mathbf{J}^T \mathbf{J})$ such that
$$
[\mathbf{J}^T \mathbf{J} + \lambda \space \text{diag}(\mathbf{J}^T \mathbf{J})] \mathbf{\sigma}=
\mathbf{J}^T [\mathbf{y}-\mathbf{f}(\beta)]
$$

The motivation is that

* $\text{diag}(\mathbf{J}^T \mathbf{J})$ reflects the squares of each dimension of freedom's derivatives such as

$$
\text{diag}(\mathbf{J}^T \mathbf{J})=
\begin{bmatrix}
      J_1^2 & 0 & ... & 0 \\
      0 & J_2^2 & ... & 0 \\
      ... & ... & ... & ... \\
      0 & 0 & ... & J_n^2
\end{bmatrix}
$$ 

* $\lambda$ is a scaling factor to $\text{diag}(\mathbf{J}^T \mathbf{J})$

* $[\mathbf{J}^T \mathbf{J} + \lambda \space \text{diag}(\mathbf{J}^T \mathbf{J})]$ represents the "scaling" effect on iteration step $\mathbf{\sigma}$, resulting in fast convergence of $S(\mathbf{\beta})$.

Given the computed $\mathbf{\sigma}$, update $\mathbf{\beta}$ such as $\mathbf{\beta} \leftarrow \mathbf{\beta}+\mathbf{\sigma}$, until $S(\mathbf{\beta})$ is convergent.