# Powell's Dogleg Method

## Trust Region

To find the minimum of an objective function $\mathbf{f}$, trust region method first sets up a trust region $\Delta_k$ (trust region radius $\Delta_k \in \mathbb{R}+$ is a positive scalar) at the $k$-th step iteration's position $\mathbf{x}_k$, within this region then computes local minimum of the second order approximation (by Hessian $H$). 

If such a local minimum sees sufficient decrease (good convergence) in objective function value, then enlarge the trust region in the next step search iteration $k+1$; 
otherwise (bad convergence), shrinks the trust region and re-compute the local minimum until observed good convergence.

<div style="display: flex; justify-content: center;">
      <img src="imgs/trust_region.png" width="35%" height="35%" alt="trust_region" />
</div>
</br>

Consider the step at $\mathbf{x}_k$ for the $k$-th convergence iteration. Take $\mathbf{x}_k$ as the centroid drawing a circle $\Delta_k$ regarded as the trust region. Compute the second-order approximation denoted as $\mathbf{m}_k$:

$$
\mathbf{m}_k(\mathbf{p})=
\mathbf{f}_k + \nabla \mathbf{f}_k^\text{T} \mathbf{p} + 
\frac{1}{2} \mathbf{p}^\text{T} H_k \mathbf{p}
$$

where $H_k$ is a Hessian matrix of the objective function $\mathbf{f}_k$, and $\mathbf{p}$ is the variable to $\mathbf{m}_k$. Intuitively speaking, $\mathbf{p}=\Delta\mathbf{x}_k; \quad \mathbf{x}_{k+1}=\mathbf{x}_k+\Delta\mathbf{x}_k$ describes the possible next step. 

The gray area shows good approximation by $\mathbf{m}_k$ (the smaller the gray area, the better the approximation), where within the trust region, the contours of $\mathbf{f}$ and $\mathbf{m}_k$ are similar to each other having similar contour curvatures. However, outside the trust region, the contours of $\mathbf{f}$ and $\mathbf{m}_k$ are quite different, rendering bad approximation of $\mathbf{f}$ by $\mathbf{m}_k$. The line search method such as Newton's method would perform badly in this scenario.

The computation of trust region radius is shown as below

$$
\mathbf{\rho}_k = \frac{\mathbf{f}(\mathbf{x}_k)-\mathbf{f}(\mathbf{x}_k+\mathbf{p}_k)}{\mathbf{m}_k(0)-\mathbf{m}_k(\mathbf{p}_k)}
$$

that $\mathbf{m}_k(0)=\mathbf{f}(\mathbf{x}_k)$ is simply the objective function value at $\mathbf{x}_k$. 
Hence, as $\mathbf{p}_k$ changes, there should be $\mathbf{\rho}_k \rightarrow 1$ that indicates $\mathbf{m}_k$ is a good approximation to $\mathbf{f}$ at the $\mathbf{x}_k$.

Recall that $\mathbf{m}_k$ is a quadratic function. 
This means, before $\mathbf{m}_k$ reaching its extremum at $\mathbf{p}_k^*$, $\mathbf{m}_k$ should have the same movement direction as around $\mathbf{f}_k$; 
then after crossing over $\mathbf{p}_k^*$, the quadratic approximation $\mathbf{m}_k$ should see an opposite movement direction against $\mathbf{f}_k$. 
As a result, within the trust region $\Delta_k$, choosing $\mathbf{p}_k^*$ as the optimization step is a good idea.

## Cauchy Point

Cauchy point is a simple implementation of trust region to find $\mathbf{p}_k^*$ within a trust region radius $\Delta_k$.

Cauchy point step $\mathbf{p}_k^C$ is computed by a scaled step following the direction $-\nabla \mathbf{f}_k$ within $\Delta_k$:

$$
\mathbf{p}_k^C = 
-\tau_k \frac{\Delta_k}{\big|\big| \nabla \mathbf{f}_k \big|\big|} \nabla \mathbf{f}_k
$$

where
$$
\tau_k = \left\{
      \begin{matrix}
            1 & \text{if } \nabla\mathbf{f}_k^\text{T} H_k \nabla\mathbf{f}_k \le 0
            \\\\
            \min \big(
                  \frac{|| \nabla \mathbf{f}_k ||^3}{
                       \Delta_k \cdot \nabla\mathbf{f}_k^\text{T} H_k \nabla\mathbf{f}_k
                  }, 
                  1 \big)
            & \text{otherwise}
      \end{matrix}
\right.
$$

The explanation of $\tau_k$ can be illustrated by replacing $\mathbf{p}_k^C$ with its expression with $\tau_k$ into the second order approximation $\mathbf{m}_k$:

$$
\begin{align*}
\mathbf{m}_k \big(
      \mathbf{p}_k^C 
\big) &=
\mathbf{m}_k \bigg(
      -\tau_k \frac{\Delta_k}{\big|\big| \nabla \mathbf{f}_k \big|\big|} \nabla \mathbf{f}_k
\bigg)
\\\\ &=
\mathbf{f}_k + 
\nabla \mathbf{f}_k^\text{T} \bigg(
      -\tau_k \frac{\Delta_k}{\big|\big| \nabla \mathbf{f}_k \big|\big|} \nabla \mathbf{f}_k
\bigg) + 
\frac{1}{2} \bigg(
      -\tau_k \frac{\Delta_k}{\big|\big| \nabla \mathbf{f}_k \big|\big|} \nabla \mathbf{f}_k
\bigg)^\text{T} 
H_k 
\bigg(
      -\tau_k \frac{\Delta_k}{\big|\big| \nabla \mathbf{f}_k \big|\big|} \nabla \mathbf{f}_k
\bigg)
\\\\ &=
\mathbf{f}_k 
\underbrace{- \tau_k \frac{\Delta_k}{\big|\big| \nabla \mathbf{f}_k \big|\big|} \nabla \mathbf{f}_k^\text{T} \nabla\mathbf{f}_k }_{
      :=M_1}
+
\underbrace{\frac{1}{2} \tau_k^2 \frac{\Delta_k^2}{\big|\big| \nabla \mathbf{f}_k \big|\big|^2} 
\nabla \mathbf{f}_k^\text{T} H_k \nabla\mathbf{f}_k }_{
      := M_2}
\end{align*}
$$

If $\nabla \mathbf{f}_k^\text{T} H_k \nabla\mathbf{f}_k \le 0$, both $M_1$ and $M_2$ are monotonically decreasing as $\tau_k$ increases. So that set $\tau_k=1$ that $\mathbf{p}_k^C$'s length is equal to the trust region radius $\Delta_k$.

If $\nabla \mathbf{f}_k^\text{T} H_k \nabla\mathbf{f}_k > 0$ ($H_k$ is positive-definite), $\mathbf{m}_k \big( \mathbf{p}_k^C \big)$ is a quadratic function whose extremum takes place when $\nabla \mathbf{m}_k \big( \mathbf{p}_k^C \big)=0$, by which $\tau_k$ is set.

* Disadvantage

Cauchy point method is non-convergence in the whole trust region. 
It simply tracks the direction of $\nabla \mathbf{f}_k$ with the scalar $\tau_k$ as the step length. 
The objective function $\mathbf{f}_k$ within a trust region $\Delta_k$ might be very curvy that one simple direction $\nabla \mathbf{f}_k$ does not represent how $\mathbf{f}_k$ moves, rendering a bad approximation.

## Dogleg

*Dogleg* provides two steps to better approximate the objective function $\mathbf{f}$.

The first step $\mathbf{p}^U$ is same as Cauchy point's method following the direction of $-\nabla\mathbf{f}_k$ (denoted as $-\mathbf{g}$ as in the figure below). 

If $H_k$ is positive-definite, the extremum $\mathbf{p}^{C*}_k$ exists inside the trust region. Starting from $\mathbf{p}^{C*}_k$, compute the second step $\mathbf{p}^B=-H_k\nabla\mathbf{f}_k$

<div style="display: flex; justify-content: center;">
      <img src="imgs/dogleg.png" width="35%" height="35%" alt="dogleg" />
</div>
</br>
