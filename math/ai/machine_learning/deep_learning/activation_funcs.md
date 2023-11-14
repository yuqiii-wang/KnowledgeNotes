# Activation Functions

## sigmoid

$$
sigmoid \space x=
\frac{e^{x}}{e^x+1}
$$

$sigmoid$ maps input $(-\infty, +\infty)$ to output $(0,1)$. It is used to represent zero/full information flow. For example, in LSTM, it guards in/out/forget gates to permit data flow.

## Softmax

Defien a standard (unit) softmax function $\sigma: \mathbb{R}^K \rightarrow (0,1)^k$
$$
\sigma(\bold{z})_i=
\frac{e^{z_i}}{\sum^K_{j=1}e^{z_j}}
$$
for $i=1,2,...,K$ and $\bold{z}=(z_1, z_2, ..., z_K)\in \mathbb{R}^K$

Now define $\bold{z}=\bold{x}^\text{T}\bold{w}$, there is
$$
\text{softmax} \space (y=j | \bold{x})=
\frac{e^{\bold{x}^\text{T}\bold{w}_j}}{\sum^K_{k=1}e^{\bold{x}^\text{T}\bold{w}_k}}
$$

$softmax$ is often used in the final layer of a classifier network that outputs each class energy.

## ReLU

$$
\text{relu} \space x =
\max(0, x)
$$

$\text{relu}$ retains a constant gradient regardless of the input $x$. For $sigmoid$, gradeint approaches to zero when $x \rightarrow +\infty$, however, for $relu$, gradient remain constant.

$sigmoid$ generates positive gradients despite $x<0$, which might wrongfully encourage weight updates, while $relu$ simply puts it zero.

$\text{relu}$ is easy in computation.

### Gaussian error linear unit  (GeLU)

$$
GeLU(x) = \frac{x}{2} \Big( 1 + \text{erf}\big( \frac{x}{\sqrt{2}} \big) \Big)
$$

where $\text{erf}(z)=\frac{2}{\sqrt{\pi}}\int^z_0 e^{-t^2} dt$ is called *Gauss error function*.

Python implementtaion: `cdf` is a normal cummulative distribution function.

```py
import math

def gelu(x):
    return [0.5 * z * (1 + math.tanh(math.sqrt(2 / np.pi) * (z + 0.044715 * math.pow(z, 3)))) for z in x]
```


### SwiGLU (Swish GeLU)

$$
\text{silu}(x) = 
x \cdot \sigma(\beta x)
$$
where $\beta$ is a coefficient default to $\beta = 1$, and $\cdot$ is a element-wise dot multiplication.
$\sigma(x)=\frac{e^{x}}{e^x+1}$ is a sigmoid function.

```py
import numpy as np

def sigmoid(x_elem):
  return 1/(1 + np.exp(-x_elem))
def silu(x, theda = 1.0):
    return [x_elem * sigmoid(theda *x_elem) for x_elem in x]
```

### Discussions

In comparison to ReLU, GeLU and SwiGLU have continuous (differentiable) on-zero values for $x \in (-\infty, 0)$ that have derivatives ().

The global minimum, where the derivative is zero, serves as a "soft floor" on the weights that acts as an implicit regularizer that inhibits the learning of weights of large magnitudes.

<div style="display: flex; justify-content: center;">
      <img src="imgs/relu_and_variants.png" width="40%" height="40%" alt="relu_and_variants" />
</div>
</br>

The funcs and derivatives of ReLU and SwiGLU.

It can see that ReLU's derivative is a step function either 0 or 1, while SwiGLU has one zero point ($\frac{\partial \space\text{SwiGLU}}{\partial z}|_{z=z_0}=0$) at $z_0 \in (-2, 0)$.
For $z=0$, the derivative $\frac{\partial \space\text{SwiGLU}}{\partial z}|_{z=0}$ is differentiable that sees significant changes even overshooting when $z \in (-4, -2)$ and $z \in (2, 4)$.

This design by SwiGLU amplifies error feedback when switching back and forth positive/negative from back-propagation, and treats error far away from $z=0$ similar as that of by ReLU.

<div style="display: flex; justify-content: center;">
      <img src="imgs/relu_func_vs_derivative.png" width="40%" height="40%" alt="relu_func_vs_derivative" />
</div>
</br>

## tanh

$$
tanh \space x =
\frac{e^{2x-1}}{e^{2x+1}}
$$

$tanh$ maps input $(-\infty, +\infty)$ to output $(-1,1)$, that is good for features requiring both negative and positive gradients updating weights of neurons. 

$tanh$ is a good activation function to tackling vanishing gradient issues.

