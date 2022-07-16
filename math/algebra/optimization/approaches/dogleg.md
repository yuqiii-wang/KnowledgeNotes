# Powell's Dogleg method

## Trust Region

Trust region objective function fit/approximation is evaluated by comparing the ratio of expected improvement from the model approximation with the actual improvement observed in the objective function.

### Cauchy Point

Cauchy point is the point lying on the gradient controling the step being within the trust region. 

Typically, given an iterative step: $\triangledown f = f(x) - f(x+\Delta x)$, it can be computed by following the direction of gradient $\triangledown f$ finding the minima of the gradient, and $\delta_c = \Delta x$ is the Cauchy point.

$$
arg \space \underset{\Delta x}{min} \space \frac{\partial \triangledown f}{\partial \Delta x} = 0
$$

For a step $\Delta x \le \delta_c$, it is same as by Gauss-Newton's method; for a step $\Delta x \ge \delta_c$, Gauss-Newton's method might give a long step that approximation might be bad.

### Intuition

Given optimization via iterative step: $\triangledown f = f(x) - f(x+\Delta x)$, here introduces a scaling factor $\lambda$ to control the scale of the iterative step: $\triangledown f_{\lambda} = f(x) - f(x+\lambda\Delta x)$

If $\triangledown f_{\lambda}$ sees better convergence than $\triangledown f$, the introduced $\lambda$ is a good trust region over $\Delta x$; othyewise, $\lambda$ should be small.

* One salient example is Levenberg–Marquardt algorithm by using $\lambda \space diag(\bold{J})$ as a trust region.

## Dogleg Formulation

Given a least squares problem in the form
$$
arg \space \underset{\bold{x}}{min} \space
\frac{1}{2} ||\bold{f}(\bold{x})||^2
=
\frac{1}{2} \sum^m_{i=1} f_i (\bold{x})^2
$$

Solution to minimized squares can be appraoched iteratively updating $\bold{x}_k = \bold{x}_{k-1}+\delta_k$

Gauss-Newton step:
$$
\bold{\delta}_{gn}=-(\bold{J}^T \bold{J})^{-1}\bold{J}^T \bold{f}(\bold{x})
$$
where $\bold{J}$ is Jacobian matrix.

Gradient descent step is
$$
\bold{\delta}_{gd}=-\bold{J}^T \bold{f}(\bold{x})
$$

Add a scaling factor $t$ to Gradient descent step, the problem can be reformulated:
$$
\begin{align*}
\bold{f}(\bold{x}_{k-1}+\delta_k)
&\approx
\frac{1}{2} ||\bold{f}(\bold{x})+t\bold{J}(\bold{x})\bold{\delta_{gd}}||^2
\\ &=
\bold{f}^2(\bold{x}) + t \bold{\delta_{gd}}^T \bold{J}^T \bold{f}(\bold{x}) + \frac{1}{2}t^2||\bold{J}\bold{\delta_{gd}}||^2
\end{align*}
$$

Now Compute $t$ by setting $\frac{\partial \bold{f}(\bold{x}_{k-1}+\delta_k)}{\partial t}=0$, (Cauchy Point) there is
$$
\begin{align*}
\frac{\partial}{\partial t}
\big(\bold{f}^2(\bold{x}) + \bold{\delta_{gd}}^T \bold{J}^T \bold{f}(\bold{x}) + \frac{1}{2}t^2||\bold{J}\bold{\delta_{gd}}||^2 \big)
 &= 0
\\
\bold{\delta_{gd}}^T \bold{J}^T \bold{f}(\bold{x}) + t ||\bold{J}\bold{\delta_{gd}}||^2
&= 0
\\
t
&=
-\frac{\bold{\delta_{gd}}^T \bold{J}^T \bold{f}(\bold{x})}{||\bold{J}\bold{\delta_{gd}}||^2}
\\ t &=
\frac{||\bold{\delta_{gd}}||^2}{||\bold{J}\bold{\delta_{gd}}||^2}
\end{align*}
$$

Given a trust region $\Delta=t||\bold{\delta_{gd}}||$, Powell's Dogleg method choose update step $\delta_k$ by

* $\delta_k = \delta_{gn}$ when $||\delta_{gn}|| \le \Delta$

Same as directly applying $\delta_{gn}$ as the step.

* $\delta_k = \frac{\Delta}{||\bold{\delta_{gd}}||}\bold{\delta_{gd}}$ when both $||\delta_{gn}|| \ge \Delta, \space ||\delta_{gd}|| \ge \Delta$ 

Intuitvely, $\frac{\Delta}{||\bold{\delta_{gd}}||}\bold{\delta_{gd}}$ represents the ratio of $\Delta$ to $||\delta_{gd}||$ given the direction of $\delta_{gd}$.

* $\delta_k = t\bold{\delta_{gd}}+s(\bold{\delta}_{gn}-t\delta_{gd})$, where $s$ is customary such as $||\bold{\delta_{gd}}||=\Delta$, on when the condition  $||\delta_{gn}|| \ge \Delta, \space ||\delta_{gd}|| \le \Delta$ 
