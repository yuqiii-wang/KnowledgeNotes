# Activation Functions

## sigmoid

$$
\text{sigmoid} (x) =
\frac{e^{x}}{e^x+1} =
\frac{1}{1+e^{-x}}
$$

$\text{sigmoid}$ maps input $(-\infty, +\infty)$ to output $(0,1)$. It is used to represent zero/full information flow. For example, in LSTM, it guards in/out/forget gates to permit data flow.

### Sigmoid Derivative

$$
\begin{align*}
  \frac{d}{d x} \text{sigmoid}(x) &=
  \frac{d}{d x} \Big( \frac{1}{1+e^{-x}} \Big) 
\\ &=
  \frac{d}{d x} ( {1+e^{-x}} )^{-1}
\\ &=
  -( {1+e^{-x}} )^{-2} (-e^{-x})
\\ &=
  \frac{e^{-x}}{({1+e^{-x}})^2}
\\ &=
  \frac{e^{-x}}{{1+e^{-x}}}   \frac{1}{{1+e^{-x}}}
\\ &=
  \frac{(e^{-x}+1)-1}{{1+e^{-x}}}   \frac{1}{{1+e^{-x}}}
\\ &=
  \Big( \frac{e^{-x}+1}{{1+e^{-x}}} - \frac{1}{{1+e^{-x}}} \Big) \frac{1}{{1+e^{-x}}}
\\ &=
  \Big( 1 - \frac{1}{{1+e^{-x}}} \Big) \frac{1}{{1+e^{-x}}}
\\ &=
  \sigma(x) \cdot \big( 1-\sigma(x) \big)
\end{align*}
$$

## Softmax

Defien a standard (unit) softmax function $\sigma: \mathbb{R}^n \rightarrow (0,1)^n$

$$
\sigma(\bold{z})_i=
\frac{e^{z_i}}{\sum^n_{j=1}e^{z_j}}
$$

for $i=1,2,...,n$ and $\bold{z}=(z_1, z_2, ..., z_n)\in \mathbb{R}^n$

$softmax$ is often used in the final layer of a classifier network that outputs each class energy and in attention mechanism to normalize input energy.

Given one-hot encoded true labels $\bold{y}$, one forward pass of softmax is as below.

$$
\begin{align*}
    \hat{\bold{y}} &= \text{softmax}(\bold{z}) 
\\  \mathcal{L} &= - \bold{y}^{\top} \log \hat{\bold{y}}
\end{align*}
$$

### Softmax Derivative

Recall there is $\frac{\partial {\mathcal{L}}}{\partial \hat{y}_{i}} \frac{\partial \hat{y}_{i}}{\partial z_{i}} = \frac{1}{\hat{y}_{i}} \frac{\partial \hat{y}_{i}}{\partial z_{i}} = \frac{\partial }{\partial z_{t,j}} \Big( -\log \big( \underbrace{\text{softmax}({z}_{i})}_{\hat{y}_{i}} \big) \Big)$,
so that by simply moving $\hat{y}_{i}$ to the opposite side of the equal operator, there derives the derivative for $\text{softmax}$ such that

$$
\begin{align*}
  \frac{\partial \hat{y}_{i}}{\partial z_{i}} &=
  \hat{y}_{i} \frac{\partial }{\partial z_{t,j}} \Big( -\log \big( \underbrace{\text{softmax}({z}_{i})}_{\hat{y}_{i}} \big) \Big) 
\\ &=
  \hat{y}_{i} \Big( \underbrace{\frac{e^{z_{t,j}}}{\sum_i^n e^{z_{i}}}}_{\hat{y}_{j}} - \frac{\partial z_{i}}{\partial z_{t,j}} \Big)
\\ &=
    \left\{ \begin{array}{r}
        \hat{y}_{i} (\hat{y}_{j} - 1) \qquad i = j \\
        \hat{y}_{i}\hat{y}_{j} \qquad i \ne j
    \end{array}\right.
\end{align*}
$$

### Softmax In Practice

Given $Z$ such as below, apply Softmax along Axis $-1$ (Last Axis):

$$
Z=\begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9 \\
\end{bmatrix}
$$

For each row $i$:

$$
\text{softmax}(z_i)=\frac{e^{z_i}}{\sum^n_{j=1}e^{z_{ij}}}
$$

that gives

|Input|Output|Explained|
|-|-|-|
|$[1,2,3]$|$[0.0900, 0.2447, 0.6652]$|$\frac{\exp(1)}{\exp(1)+\exp(2)+\exp(3)},\frac{\exp(2)}{\exp(1)+\exp(2)+\exp(3)},\frac{\exp(3)}{\exp(1)+\exp(2)+\exp(3)}$|
|$[4,5,6]$|$[0.0900, 0.2447, 0.6652]$|$\frac{\exp(4)}{\exp(4)+\exp(5)+\exp(6)},\frac{\exp(5)}{\exp(4)+\exp(5)+\exp(6)},\frac{\exp(6)}{\exp(4)+\exp(5)+\exp(6)}$|
|$[7,8,9]$|$[0.0900, 0.2447, 0.6652]$|$\frac{\exp(7)}{\exp(7)+\exp(8)+\exp(9)},\frac{\exp(7)}{\exp(8)+\exp(8)+\exp(9)},\frac{\exp(9)}{\exp(7)+\exp(8)+\exp(9)}$|

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
\text{tanh} \space x =
\frac{e^{2x-1}}{e^{2x+1}} =
\frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

$\text{tanh}$ maps input $(-\infty, +\infty)$ to output $(-1,1)$, that is good for features requiring both negative and positive gradients updating weights of neurons. 

$\text{tanh}$ is a good activation function to tackling vanishing gradient issues.

### tanh Derivative

$$
\begin{align*}
  \frac{d}{d x} \text{tanh}(x) &=
  \frac{d}{d x} \Big( \frac{e^x - e^{-x}}{e^x + e^{-x}} \Big)
\\ &=
  \frac{(e^x + e^{-x})(e^x + e^{-x}) - (e^x - e^{-x})(e^x - e^{-x})}{(e^x + e^{-x})^2}
\\ &=
  1 - \frac{(e^x - e^{-x})^2}{(e^x + e^{-x})^2}
\\ &=
  1 - \text{tanh}^2(x)
\end{align*}
$$