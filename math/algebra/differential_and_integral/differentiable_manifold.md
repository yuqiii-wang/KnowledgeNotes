# Differentiable Manifold

## Flows of vector fields on manifolds

### Idea

Given a tangent vector field on a differentiable manifold $M$ then its flow is the group of diffeomorphisms of $M$ that lets the points of the manifold “flow along the vector field” hence which sends them along flow lines (integral curvs) that are tangent to the vector field.

### Definition

Let $M$ be a differentiable manifold, $T_p M$ denotes the tangent space of $p \in M$, $TM$ is the corresponding tangent bundle $TM = \sqcup_{p \in M} T_p M$, now define a smooth mapping
$$
f: \bold{R} \times M \rightarrow TM
$$
representing a time-dependent ($t \in \bold{R}$, $R$ is a time range) vector field on $M$, aka $f(t, p) \in T_p M$,

For a suitable time interval $I \subseteq \bold{R}$ containing $0$, the flow of $f$ is a function $\phi : I \times M \rightarrow M$ that satisfies
$$
\begin{align*}
\phi(0, x_0) = x_0
\space\space\space\space\space\space\space\space
\space\space\space\space\space\space\space\space
\space\space\space\space\space\space\space\space
\space\space\space\space\space\space\space\space
\space\space\space\space\space\space\space\space
\space\space
\forall x_0 \in M
\\
\frac{d}{dt}\big|_{t=t_0} \phi(t, x_0) = f(t_0, \phi(t_0, x_0))
\space\space\space\space\space\space\space\space\space
\forall x_0 \in M
,
\forall t \in I
\end{align*}
$$ 

That is, "flow" is $f$ instantiated by $\phi$'s derivative. $\phi$'s direvative indicates flow's direction.