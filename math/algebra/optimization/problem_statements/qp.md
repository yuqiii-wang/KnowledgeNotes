# Quadratic Programming

Quadratic Programming (QP) refers to optimization problems involving quadratic functions. 

Define
* an $n$-dimensional vector $\bold{x}$
* an $n \times n$-dimensional real symmetric matrix $Q$
* an $m \times n$-dimensional real matrix $A$
* an m-dimensional real vector $\bold{b}$

QP attempts to

$$
arg \space \underset{\bold{x}}{min} \space \frac{1}{2} \bold{x}^TQ\bold{x} + \bold{c}^T\bold{x}
$$
subject to
$$
A\bold{x} \le \bold{b}
$$

Optionally, there might be feasible constriaints for $\bold{x}$ such as
$$
\bold{x}_L \le \bold{x} \le \bold{x}_U
$$

### Least Squares

When $Q$ is symmetric positive-definite, the cost function reduces to least squares:

$$
arg \space \underset{\bold{x}}{min} \space \frac{1}{2} ||R\bold{x}-\bold{d}||^2
$$
subject to
$$
A\bold{x} \le \bold{b}
$$
where $Q=R^TR$ follows from the Cholesky decomposition of $Q$ and $\bold{c}=-R^T\bold{d}$.

## KKT Conditions

*Karush-Kuhn-Tucker* conditions are first derivative tests, for a solution in nonlinear programming to be optimal, provided that some regularity conditions are satisfied.

In other words, it is the precondition to establish a solution to be optimal in nonlinear programming.

### Langrange Multipliers

For a typical equality constraint optimization, there is
$$
min \quad f(\bold{x})
$$
subject to
$$
g(\bold{x}) = 0
$$

Define Lagrangian function:
$$
L(\bold{x}, \lambda) = f(\bold{x}) + \lambda g(\bold{x})
$$

Stationary points $\bold{x}^*$ are computed on the conditions when derivatives are zeros:
$$
\begin{align*}
\triangledown_x L &= \frac{\partial L}{\partial \bold{x}} = \triangledown f + \lambda \triangledown g = \bold{0}
\\
\triangledown_{\lambda} L &= \frac{\partial L}{\partial \lambda} = g(\bold{x}) = 0
\end{align*}
$$

### Inequality Constraint Optimization

KKT condition generalizes the use of Langrage Multipliers to inequality constraints to $\bold{x}$.

Here $g(\bold{x})$ has inequality constraints such as
$$
min \quad f(\bold{x})
$$
subject to
$$
g_j(\bold{x}) = 0 \quad j=1,2,...,m
\\
h_k(\bold{x}) \le 0 \quad k=1,2,...,p
$$

For feasible region $K=\bold{x} \in \mathbb{R}^n | g_j(\bold{x}) = 0, h(\bold{x}) \le 0$, denote $\bold{x}^*$ as the optimal solution, there is

* $h(\bold{x^*}) \lt 0$, $\bold{x^*}$ is named *interior solution*, that $\bold{x^*}$ resides inside feasible region $K$

$g(\bold{x^*})$ serves no constraints so that $\bold{x}^*$ can be computed via $\triangledown f = 0$ and $\lambda = 0$.

* $g(\bold{x^*}) = 0$ or $h(\bold{x^*}) = 0$, $\bold{x^*}$ is named *boundary solution*, that $\bold{x^*}$ resides on the edge of feasible region $K$

This draws similarity with Langrage Multipliers having equality constraints, so that
$$
\triangledown f = -\lambda \triangledown g
$$

Here defines Langrage function:
$$
L(\bold{x}, \{\lambda_j\}, \{\mu_k\})
=
f(\bold{x}) + \sum_{j=1}^m \lambda_j g_j(\bold{x}) + \sum_{k=1}^p \mu_k h_j(\bold{x})
$$

### KKT conditions

KKT conditions are defined as below having the four clauses:

* Stationarity
$$
\triangledown_x L = \frac{\partial L}{\partial \bold{x}} = \triangledown f + \lambda \triangledown g + \mu \triangledown h = \bold{0}
$$
* Primal feasibility
$$
g(\bold{x}) = 0
\\
h(\bold{x}) \le 0
$$
* Dual feasibility
$$
\mu \ge 0
$$
* Complementary slackness
$$
\mu h(\bold{x}) = 0
$$

## Optimization Solution

The typical solution techniques: 

When the objective function is strictly convex (having only one optimal point) and there are only equality constraints ($A\bold{x}=\bold{b}$), use *conjugate gradient method*. 

If there are inequality constraints ($A\bold{x} \le \bold{b}$), then *interior point* or *active set methods*.

If there are constraint ranges on $\bold{x}$ such as $\bold{x}_L \le \bold{x} \le \bold{x}_U$, use *trust-region method*.

## Example

