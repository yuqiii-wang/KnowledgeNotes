# Normalization in Deep Learning

## Embedding Normalization

For example, given a vector $\bold{v}=[3,4]$, there are

* L1 Normalization (Manhattan Distance)

$$
\bold{v}_{\text{norm}} = \frac{[3,4]}{3+4} = [\frac{3}{7},\frac{4}{7}] \approx [0.4286, 0.5714]
$$

* L2 Normalization (Euclidean Distance)

$$
\bold{v}_{\text{norm}} = \frac{[3,4]}{\sqrt{3^2+4^2}} = \frac{[3,4]}{5} = [0.6, 0.8]
$$

* MAX Normalization (Chebyshev Distance)

$$
\bold{v}_{\text{norm}} = \frac{[3,4]}{\max(\{3, 4\})} = \frac{[3,4]}{4} = [0.75, 1]
$$

where L2 normalization is the most used for vector similarity computation, e.g., for cosine similarity.

### Layer Normalization

Layer normalization is applied over the feature dimensions of a single training example rather than over the batch.

Given $\mu=\frac{1}{d}\sum^{d}_{i=1}d_i$ and $\sigma^2=\frac{1}{d}\sum^d_{i=1}(x_i-\mu)^2$,
the layer norm is defined as

$$
\text{LayerNorm}(\bold{x})=
\bold{\gamma}\odot\frac{\bold{x}-\mu}{\sqrt{\sigma^2+\epsilon}}+\bold{b}
$$

where $\odot$ is an element-wise multiplication operator.

```py
import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    # Compute mean and variance along the last dimension
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    # Normalize
    normalized = (x - mean) / np.sqrt(var + eps)
    # Scale and shift
    return gamma * normalized + beta

# Example:
x = np.random.randn(2, 5)  # batch size of 2, 5 features each
gamma = np.ones((1, 5))    # Scale factor (broadcastable along batch)
beta = np.zeros((1, 5))    # Shift parameter (broadcastable along batch)

output_layernorm = layer_norm(x, gamma, beta)
print("LayerNorm output:\n", output_layernorm)
```

### RMS Layer Normalization

RMSNorm normalizes using the root mean square value only (i.e. without centering by the mean).

$$
\text{RMSNorm}(\bold{x})=\bold{\gamma}\odot\frac{\bold{x}}{\sqrt{\frac{1}{d}\sum^d_{i=1}x_i^2+\epsilon}}
$$

```py
def rms_norm(x, gamma, eps=1e-8):
    # Compute the mean of squares over the last dimension
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    normalized = x / rms
    # Scale using gamma
    return gamma * normalized

# Example:
x = np.random.randn(2, 5)  # batch of 2, 5 features
gamma_rms = np.ones((1, 5))  # scale parameter (broadcastable)
output_rmsnorm = rms_norm(x, gamma_rms)
print("RMSNorm output:\n", output_rmsnorm)
```
