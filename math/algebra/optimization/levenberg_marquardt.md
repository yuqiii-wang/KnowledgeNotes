# Levenberg-Marquardt's Method

It is used to solve the least-squares curve fitting problem:
given a set of $m$ empirical pairs $(x_i, y_i)$, where $x_i$ is independent having $n$ dimensions/features such as $\bold{x}=[x_1, x_2, ..., x_m]$, of which $x_i=(x_{i,1}, x_{i,2}, ..., x_{i,n})$, while $y_i$ is dependent/observation, find the parameters $\beta$ of tthe model curve $f(x, \beta)$ so that the sum of the squares of the deviations $S(\beta)$ is minimized:
$$
arg \space \underset{\beta}{min}
\sum^m_{i=1}[y_i - f(x_i, \beta)]^2
$$

Levenberg-Marquardt Method attempts to find solutions (minima/maxima) with an initial guess $\bold{\beta}_0$ not "far away" from the solution ("far away" means there is no more other stationary points in between the initial guess and the solution).

On each iteration, there is $\bold{\beta} \leftarrow \bold{\beta}+\bold{\sigma}$, computed by linear approximation:
$$
f(x_i, \bold{\beta}+\bold{\sigma})
\approx
f(x_i, \bold{\beta}) + \bold{J}_i\bold{\sigma}_i
$$
where $\bold{J}_i$ is Jacobian matrix entry to partial derivative $\bold{\beta}$ such as
$$
\bold{J}_i=\frac{\partial f(x_i, \bold{\beta})}{\partial \bold{\beta}}
$$

Hence, 
$$
\begin{align*}
S(\bold{\beta}+\bold{\sigma})
&\approx
\sum_{i=1}^m [y_i - f(x_i,\bold{\beta})-\bold{J}_i \sigma]^2
\\ &=
||\bold{y}-\bold{f}(\beta)-\bold{J}\bold{\sigma}||^2
\\ &=
[\bold{y}-\bold{f}(\beta)-\bold{J}\bold{\sigma}]^T[\bold{y}-\bold{f}(\beta)-\bold{J}\bold{\sigma}]
\\ &=
[\bold{y}-\bold{f}(\beta)]^T[\bold{y}-\bold{f}(\beta)]
-
[\bold{y}-\bold{f}(\beta)]^T \bold{J}\bold{\sigma}
-
\bold{J}\bold{\sigma}^T [\bold{y}-\bold{f}(\beta)]
+ 
\bold{J}^T \bold{\sigma}^T \bold{J}\bold{\sigma}
\end{align*}
$$

Compute $S(\bold{\beta}+\bold{\sigma})$'s derivative and set it to zero to find stationary points
$$
\begin{align*}
\frac{\partial \big(S(\bold{\beta}+\bold{\sigma}) \big)
}{\partial \bold{\sigma}}
&=
\frac{\partial \space
    \big(-2[\bold{y}-\bold{f}(\beta)]^T \bold{J}\bold{\sigma}
    +
    \bold{J}^T \bold{\sigma}^T \bold{J}\bold{\sigma}
    \big)}
    {
        \partial \bold{\sigma}
    }
\\ &=
2[\bold{y}-\bold{f}(\beta)]^T \bold{J}
+
2 \bold{J}^T \bold{J} \bold{\sigma}
\end{align*}
$$

By setting the derivative to zero, there is
$$
\begin{align*}
0 &=
-2[\bold{y}-\bold{f}(\beta)]^T \bold{J}
+
2 \bold{J}^T \bold{J} \bold{\sigma}
\\ 
\bold{J}^T \bold{J} \bold{\sigma}
&=
[\bold{y}-\bold{f}(\beta)]^T \bold{J}
\\
\bold{J}^T \bold{J} \bold{\sigma}
&=
\bold{J}^T [\bold{y}-\bold{f}(\beta)]
\end{align*}
$$

Here comes the innovation of Levenberg-Marquardt's method derived from the above Gauss-Newton's method:

* Introduce a damping parameter $\lambda$ to the diagnol matrix $diag(\bold{J}^T \bold{J})$ such that
$$
[\bold{J}^T \bold{J} + \lambda \space diag(\bold{J}^T \bold{J})] \bold{\sigma}
=
\bold{J}^T [\bold{y}-\bold{f}(\beta)]
$$

The motivation is that

* $diag(\bold{J}^T \bold{J})$ reflects the squares of each dimension of freedom's derivatives such as 
$$
diag(\bold{J}^T \bold{J})
=
\begin{bmatrix}
      J_1^2 & 0 & ... & 0 \\
      0 & J_2^2 & ... & 0 \\
      ... & ... & ... & ... \\
      0 & 0 & ... & J_n^2
\end{bmatrix}
$$ 

* $\lambda$ is a scaling factor to $diag(\bold{J}^T \bold{J})$

* $[\bold{J}^T \bold{J} + \lambda \space diag(\bold{J}^T \bold{J})]$ represents the "scaling" effect on iteration step $\bold{\sigma}$, resulting in fast convergence of $S(\bold{\beta})$.

Given the computed $\bold{\sigma}$, update $\bold{\beta}$ such as $\bold{\beta} \leftarrow \bold{\beta}+\bold{\sigma}$, until $S(\bold{\beta})$ is convergent.