# Quadratic Programming

Quadratic Programming (QP) refers to optimization problems involving quadratic functions. 

Define
* an $n$-dimensional vector $\mathbf{x}$
* an $n \times n$-dimensional real symmetric matrix $Q$
* an $m \times n$-dimensional real matrix $A$
* an m-dimensional real vector $\mathbf{b}$

QP attempts to

$$
arg \space \underset{\mathbf{x}}{min} \space \frac{1}{2} \mathbf{x}^TQ\mathbf{x} + \mathbf{c}^T\mathbf{x}
$$

subject to
$$
A\mathbf{x} \le \mathbf{b}
$$

Optionally, there might be feasible constriaints for $\mathbf{x}$ such as
$$
\mathbf{x}_L \le \mathbf{x} \le \mathbf{x}_U
$$

### Least Squares

When $Q$ is symmetric positive-definite, the cost function reduces to least squares:

$$
arg \space \underset{\mathbf{x}}{min} \space \frac{1}{2} ||R\mathbf{x}-\mathbf{d}||^2
$$
subject to
$$
A\mathbf{x} \le \mathbf{b}
$$

where $Q=R^TR$ follows from the Cholesky decomposition of $Q$ and $\mathbf{c}=-R^\top \mathbf{d}$.

## KKT Conditions

*Karush-Kuhn-Tucker* conditions are first derivative tests, for a solution in nonlinear programming to be optimal, provided that some regularity conditions are satisfied.

In other words, it is the precondition to establish a solution to be optimal in nonlinear programming.

### Langrange Multipliers

For a typical equality constraint optimization, there is
$$
\min \quad f(\mathbf{x})
$$
subject to
$$
g(\mathbf{x}) = 0
$$

Define Lagrangian function:
$$
L(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda g(\mathbf{x})
$$

Stationary points $\mathbf{x}^*$ are computed on the conditions when derivatives are zeros:
$$
\begin{align*}
\nabla_x L &= \frac{\partial L}{\partial \mathbf{x}} = \nabla f + \lambda \nabla g = \mathbf{0} \\\\
\nabla_{\lambda} L &= \frac{\partial L}{\partial \lambda} = g(\mathbf{x}) = 0
\end{align*}
$$

### Inequality Constraint Optimization

KKT condition generalizes the use of Langrage Multipliers to inequality constraints to $\mathbf{x}$.

Here $g(\mathbf{x})$ has inequality constraints such as
$$
\min \quad f(\mathbf{x})
$$
subject to
$$
g_j(\mathbf{x}) = 0 \quad j=1,2,...,m \\\\
h_k(\mathbf{x}) \le 0 \quad k=1,2,...,p
$$

For feasible region $K=\mathbf{x} \in \mathbb{R}^n | g_j(\mathbf{x}) = 0, h(\mathbf{x}) \le 0$, denote $\mathbf{x}^*$ as the optimal solution, there is

* $h(\mathbf{x^*}) \lt 0$, $\mathbf{x^*}$ is named *interior solution*, that $\mathbf{x^*}$ resides inside feasible region $K$

$g(\mathbf{x^*})$ serves no constraints so that $\mathbf{x}^*$ can be computed via $\nabla f = 0$ and $\lambda = 0$.

* $g(\mathbf{x^*}) = 0$ or $h(\mathbf{x^*}) = 0$, $\mathbf{x^*}$ is named *boundary solution*, that $\mathbf{x^*}$ resides on the edge of feasible region $K$

This draws similarity with Langrage Multipliers having equality constraints, so that
$$
\nabla f = -\lambda \nabla g
$$

Here defines Langrage function:
$$
L(\mathbf{x}, \{\lambda_j\}, \{\mu_k\})=
f(\mathbf{x}) + \sum_{j=1}^m \lambda_j g_j(\mathbf{x}) + \sum_{k=1}^p \mu_k h_j(\mathbf{x})
$$

### KKT conditions

KKT conditions are defined as below having the four clauses:

* Stationarity
$$
\nabla_x L = \frac{\partial L}{\partial \mathbf{x}} = \nabla f + \lambda \nabla g + \mu \nabla h = \mathbf{0}
$$

This means that $L$ must have at least one optimal point where its derivative is zero.

* Primal feasibility
$$
g(\mathbf{x}) = 0 \\\\
h(\mathbf{x}) \le 0
$$

This means the constraints must be feasible 

* Dual feasibility
$$
\mu \ge 0
$$
* Complementary slackness
$$
\mu h(\mathbf{x}) = 0
$$

## Optimization Solution

The typical solution techniques: 

When the objective function is strictly convex (having only one optimal point) and there are only equality constraints ($A\mathbf{x}=\mathbf{b}$), use *conjugate gradient method*. 

If there are inequality constraints ($A\mathbf{x} \le \mathbf{b}$), then *interior point* or *active set methods*.

If there are constraint ranges on $\mathbf{x}$ such as $\mathbf{x}_L \le \mathbf{x} \le \mathbf{x}_U$, use *trust-region method*.

## Example

