# Primal-Dual Algorithm

The primal-dual algorithm is a method for solving linear programs inspired by the Ford–Fulkerson method.

Define a primal-dual pair:
$$
(\bold{P}) = \left\{
    \begin{matrix}
        \underset{\bold{x} \in \mathbb{R}^n}{\text{minimize}} 
        & \bold{c}^\top \bold{x} \\
        \text{subject to} 
        & A\bold{x} = \bold{b} \\
        & \bold{x} \ge 0
    \end{matrix}
\right.
, \quad
(\bold{D}) = \left\{
    \begin{matrix}
        \underset{\bold{u} \in \mathbb{R}^m}{\text{maximize}} 
        & \bold{u}^\top \bold{b} \\
        \text{subject to} 
        & \bold{u}^\top A \le \bold{c}^\top \\
        & \bold{u} \text{ unconstrained}
    \end{matrix}
\right.
$$

The primal-dual algorithm attempts to solve  a minimization problem $(\bold{P})$ of the unknown $\bold{x}$, and a maximization problem $(\bold{D})$ of the unknown $\bold{u}$, that share the same constraints $A$ and $\bold{b}, \bold{c}$.

