# Residual neural network (ResNet)

## Forward and Back Propagation

### Forward Propagation

$$
\begin{align*}
H(\bold{x}) &= \bold{w}_2 {\sigma(\underbrace{\bold{w}_1 \bold{x} + \bold{b}_1}_{\bold{z}_1})} + \bold{b}_2 \\
\hat{\bold{y}} &= H(\bold{x}) + \bold{x}
\end{align*}
$$

### Backward propagation

$$
\begin{align*}
\frac{\partial L}{\partial \bold{w}_1} &=
\frac{\partial \frac{1}{2}||\hat{\bold{y}}-\bold{y}||^2}{\partial \bold{w}_1}
= (\hat{\bold{y}}-\bold{y}) \frac{\partial \hat{\bold{y}}}{\partial \bold{w}_1} && \text{Suppose loss by MSE} \\
&= (\hat{\bold{y}}-\bold{y}) \frac{\partial H}{\partial \bold{w}_1} + \bold{0} && \\
&= (\hat{\bold{y}}-\bold{y}) \bold{w}_2 \frac{\partial \bold{a}_1}{\partial \bold{w}_1} + \bold{0} && \text{For denotation } \bold{a}_1=\sigma(\bold{z}_1) \\
&= (\hat{\bold{y}}-\bold{y}) \bold{w}_2 \bold{a}_1(1-\bold{a}_1)\frac{\partial \bold{z}_1}{\partial \bold{w}_1} + \bold{0} && \text{Suppose activation is by sigmoid} \\
&= (\hat{\bold{y}}-\bold{y}) \bold{w}_2 \bold{a}_1(1-\bold{a}_1)\bold{x} + \bold{0} &&
\end{align*}
$$

### Error Distribution Analysis: Partial Derivative With Respect to Input $\bold{x}$

$\frac{\partial L}{\partial \bold{x}}$ determines how error signals are distributed across the network.

In the expression of $\frac{\partial L}{\partial \bold{x}}$, the original error term $\hat{\bold{y}}-\bold{y}$ is kept and passed from the final layer down to lower lower layer, thereby preserving a non-decaying error term same value as the $\frac{\partial L}{\partial \hat{\bold{y}}}$.
The additional $\hat{\bold{y}}-\bold{y}$ is generated from $+\bold{1}$ that is the derivative of the additional $+\bold{x}$ in feed forward pass.

$$
\begin{align*}
\frac{\partial L}{\partial \bold{x}} &=
\frac{\partial \frac{1}{2}||\hat{\bold{y}}-\bold{y}||^2}{\partial \bold{x}}
= (\hat{\bold{y}}-\bold{y}) \frac{\partial \hat{\bold{y}}}{\partial \bold{x}} && \text{Suppose loss by MSE} \\
&= (\hat{\bold{y}}-\bold{y}) (\frac{\partial H}{\partial \bold{x}} + \bold{1}) && \text{The } +\bold{1} \text{ is the derivative of the additional } +\bold{x} \text{ in feed forward} \\
&= (\hat{\bold{y}}-\bold{y}) \bold{w}_2 \bold{a}_1(1-\bold{a}_1)\bold{w}_1 +
\underbrace{(\hat{\bold{y}}-\bold{y})}_{\text{Original error}} &&
\end{align*}
$$

## Advantages

The good result by ResNet is attributed to its skipping layers that pass error from upper layers to lower layers.
This mechanism successfully retains error updating lower layer neurons, different from traditional neural networks that when going too deep, suffers from vanishing gradient issues (lower layer neurons only see very small gradient).

### Explained in Gradient Vanishing/Explosion

