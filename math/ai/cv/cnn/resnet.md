# Residual neural network (ResNet)

## Forward and Back Propagation

### Forward Propagation

$$
\begin{align*}
H(\mathbf{x}) &= \mathbf{w}_2 {\sigma(\underbrace{\mathbf{w}_1 \mathbf{x} + \mathbf{b}_1}\_{\mathbf{z}_1})} + \mathbf{b}_2 \\\\
\hat{\mathbf{y}} &= H(\mathbf{x}) + \mathbf{x}
\end{align*}
$$

### Backward propagation

$$
\begin{align*}
\frac{\partial L}{\partial \mathbf{w}_1} &=
\frac{\partial \frac{1}{2}||\hat{\mathbf{y}}-\mathbf{y}||^2}{\partial \mathbf{w}_1}
= (\hat{\mathbf{y}}-\mathbf{y}) \frac{\partial \hat{\mathbf{y}}}{\partial \mathbf{w}_1} && \text{Suppose loss by MSE} \\\\
&= (\hat{\mathbf{y}}-\mathbf{y}) \frac{\partial H}{\partial \mathbf{w}_1} + \mathbf{0} && \\\\
&= (\hat{\mathbf{y}}-\mathbf{y}) \mathbf{w}_2 \frac{\partial \mathbf{a}_1}{\partial \mathbf{w}_1} + \mathbf{0} && \text{For denotation } \mathbf{a}_1=\sigma(\mathbf{z}_1) \\\\
&= (\hat{\mathbf{y}}-\mathbf{y}) \mathbf{w}_2 \mathbf{a}_1(1-\mathbf{a}_1)\frac{\partial \mathbf{z}_1}{\partial \mathbf{w}_1} + \mathbf{0} && \text{Suppose activation is by sigmoid} \\\\
&= (\hat{\mathbf{y}}-\mathbf{y}) \mathbf{w}_2 \mathbf{a}_1(1-\mathbf{a}_1)\mathbf{x} + \mathbf{0} &&
\end{align*}
$$

### Error Distribution Analysis: Partial Derivative With Respect to Input $\mathbf{x}$

$\frac{\partial L}{\partial \mathbf{x}}$ determines how error signals are distributed across the network.

In the expression of $\frac{\partial L}{\partial \mathbf{x}}$, the original error term $\hat{\mathbf{y}}-\mathbf{y}$ is kept and passed from the final layer down to lower lower layer, thereby preserving a non-decaying error term same value as the $\frac{\partial L}{\partial \hat{\mathbf{y}}}$.
The additional $\hat{\mathbf{y}}-\mathbf{y}$ is generated from $+\mathbf{1}$ that is the derivative of the additional $+\mathbf{x}$ in feed forward pass.

$$
\begin{align*}
\frac{\partial L}{\partial \mathbf{x}} &=
\frac{\partial \frac{1}{2}||\hat{\mathbf{y}}-\mathbf{y}||^2}{\partial \mathbf{x}}
= (\hat{\mathbf{y}}-\mathbf{y}) \frac{\partial \hat{\mathbf{y}}}{\partial \mathbf{x}} && \text{Suppose loss by MSE} \\\\
&= (\hat{\mathbf{y}}-\mathbf{y}) (\frac{\partial H}{\partial \mathbf{x}} + \mathbf{1}) && \text{The } +\mathbf{1} \text{ is the derivative of the additional } +\mathbf{x} \text{ in feed forward} \\\\
&= (\hat{\mathbf{y}}-\mathbf{y}) \mathbf{w}_2 \mathbf{a}_1(1-\mathbf{a}_1)\mathbf{w}_1 +
\underbrace{(\hat{\mathbf{y}}-\mathbf{y})}\_{\text{Original error}} &&
\end{align*}
$$

## Advantages

The good result by ResNet is attributed to its skipping layers that pass error from upper layers to lower layers.
This mechanism successfully retains error updating lower layer neurons, different from traditional neural networks that when going too deep, suffers from vanishing gradient issues (lower layer neurons only see very small gradient).

### Explained in Gradient Vanishing/Explosion

## Typical Residual Block in Convolutional Network

### Basic Block

Used in smaller ResNet models (ResNet-18, ResNet-34)

* $3 \times 3$ Convolution Layers.
* Batch Normalization (BN) and ReLU activation.
* Shortcut (residual) connections to mitigate the vanishing gradient problem.

$$
F(x, W) = \underbrace{\text{ReLU}(\text{BN}(\text{Cov}\_{3 \times 3}}\_{\text{the }l_{+1}\text{-th layer output}}(
    \underbrace{\text{ReLU}(\text{BN}(\text{Cov}\_{3 \times 3}(x)))}\_{\text{the }l\text{-th layer output}}))) + x
$$

### Bottleneck Block

Used in deeper ResNet models (e.g., ResNet-50, ResNet-101, ResNet-152)

* $1 \times 1$ convolution for reducing channel dimensions (dimensionality reduction).
* $3 \times 3$ convolution for feature extraction.
* $1 \times 1$ convolution for restoring channel dimensions (dimensionality expansion).

$$
F(x, W) = \text{Cov}\_{1 \times 1}(\text{BN}(\text{Cov}\_{3 \times 3}(
    \text{BN}(\text{Cov}\_{1 \times 1}(x))))) + x
$$