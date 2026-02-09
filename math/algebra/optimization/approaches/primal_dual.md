# Primal-Dual Algorithm

The primal-dual algorithm is a method for solving linear programs inspired by the Ford–Fulkerson method.

Define a primal-dual pair:

$$
(\mathbf{P}) = \left\{
    \begin{matrix}
        \underset{\mathbf{x} \in \mathbb{R}^n}{\text{minimize}} 
        & \mathbf{c}^\top \mathbf{x} \\\\
        \text{subject to} 
        & A\mathbf{x} = \mathbf{b} \\\\
        & \mathbf{x} \ge 0
    \end{matrix}
\right.
, \quad
(\mathbf{D}) = \left\{
    \begin{matrix}
        \underset{\mathbf{u} \in \mathbb{R}^m}{\text{maximize}} 
        & \mathbf{u}^\top \mathbf{b} \\\\
        \text{subject to} 
        & \mathbf{u}^\top A \le \mathbf{c}^\top \\\\
        & \mathbf{u} \text{ unconstrained}
    \end{matrix}
\right.
$$

The primal-dual algorithm attempts to solve  a minimization problem $(\mathbf{P})$ of the unknown $\mathbf{x}$, and a maximization problem $(\mathbf{D})$ of the unknown $\mathbf{u}$, that share the same constraints $A$ and $\mathbf{b}, \mathbf{c}$.

