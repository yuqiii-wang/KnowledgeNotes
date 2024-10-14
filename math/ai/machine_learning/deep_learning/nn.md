# Neural Network

## Forward and Back Propagation

### Forward Propagation

$$
\begin{align*}
\bold{z} &= \bold{w}^\top \bold{x} + \bold{b} \\
\hat{\bold{y}} &= \sigma(\bold{z})
\end{align*}
$$

where $\sigma(\bold{z})$ is an activation function, that for example, by sigmoid there is $\sigma(\bold{z})\frac{1}{1+e^{-\bold{z}}}$.

### Back Propagation

Define Mean Squared Error (MSE) as the loss function (as an example) $L=\frac{1}{2}||\hat{\bold{y}}-\bold{y}||^2=\frac{1}{n} \frac{1}{2}\sum^n_i ({y}_i-\hat{y}_i)^2$

* Gradient of Loss w.r.t Activation:

$$
\begin{align*}
\frac{\partial L}{\partial \hat{\bold{y}}} = \hat{\bold{y}}-\bold{y}
\end{align*}
$$

* Gradient of Activation w.r.t Weighted Sum:

$$
\begin{align*}
\frac{\partial \hat{\bold{y}}}{\partial {\bold{z}}} &= \sigma(\bold{z})\big(1-\sigma(\bold{z})\big) \\
&= \hat{\bold{y}}(1-\hat{\bold{y}})
\end{align*}
$$

* Gradient of Loss w.r.t Weighted Sum:

$$
\begin{align*}
\frac{\partial L}{\partial {\bold{z}}} &= \frac{\partial L}{\partial \hat{\bold{y}}} \frac{\partial \hat{\bold{y}}}{\partial {\bold{z}}} \\
&= (\hat{\bold{y}}-\bold{y})\hat{\bold{y}}(1-\hat{\bold{y}})
\end{align*}
$$

* Gradients of Loss w.r.t Weights and Bias:

$$
\begin{align*}
\frac{\partial L}{\partial {\bold{w}}} &= \frac{\partial L}{\partial \hat{\bold{y}}} \frac{\partial \hat{\bold{y}}}{\partial {\bold{z}}} \frac{\partial {\bold{z}}}{\partial {\bold{w}}} \\
&= \bold{x}^\top (\hat{\bold{y}}-\bold{y})\hat{\bold{y}}(1-\hat{\bold{y}}) \\
\frac{\partial L}{\partial {\bold{b}}} &= \frac{\partial L}{\partial \hat{\bold{y}}} \frac{\partial \hat{\bold{y}}}{\partial {\bold{z}}} \frac{\partial {\bold{z}}}{\partial {\bold{b}}} \\
&= (\hat{\bold{y}}-\bold{y})\hat{\bold{y}}(1-\hat{\bold{y}})
\end{align*}
$$

### Code

```py
import numpy as np

# Sigmoid function and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# Forward propagation
def forward_propagation(x, w, b):
    z = np.dot(x, w) + b
    y_hat = sigmoid(z)
    return y_hat, z

# Backward propagation
def backward_propagation(x, y, y_hat, z):
    m = x.shape[0] # num opf samples
    dz = (y_hat - y) * y_hat * (1-y_hat)
    dw = np.dot(x.T, dz)
    db = np.sum(dz, axis=0) / m
    return dw, db

# Example usage:
np.random.seed(42)

x = np.random.randn(5, 3)  # 5 samples, 3 features
y = np.random.randn(5, 2)  # 5 samples, 2 outputs

w = np.random.randn(3, 2)  # 3 features, 2 outputs
b = np.random.randn(2)     # 2 bias terms

learning_rate = 0.05

y_hat, z = forward_propagation(x, w, b)
loss = np.mean((y_hat - y) ** 2)

epoch = 0
while loss > 1e-5 and epoch < 1000:
    # Forward propagation
    y_hat, z = forward_propagation(x, w, b)

    # Compute loss (MSE)
    loss = np.mean((y_hat - y) ** 2)

    # Backward propagation
    dw, db = backward_propagation(x, y, y_hat, z)

    w -= learning_rate * dw
    b -= learning_rate * db

    epoch += 1
    if epoch % 100 == 0:
        print(f"epoch: {epoch}")
        print(f"w: {w}")
        print(f"b: {b}")
        print(f"Loss: {loss}")
```

## Model Parameter Initialization

Reference: https://www.deeplearning.ai/ai-notes/initialization/index.html
https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
https://stats.stackexchange.com/questions/390105/what-are-the-theoretical-practical-reasons-to-use-normal-distribution-to-initial
https://www.pinecone.io/learn/weight-initialization/

### Issues to Model Parameter Initialization

* Equal/zero parameter model initialization

Initializing all the weights with zeros leads the neurons to learn the same features during training.

For example, given an input having two features $(x_1, x_2)$, and associated wights $(w_1, w_2)$, by equal parameter initialization such that $w_1=w_2$,
both hidden units (learned feature $w_1 x_1+w_2 x_2$) will have identical influence on the cost, which will lead to identical gradients.

* A too-large/small initialization leads to exploding/vanishing gradients, respectively

Given $l\in\{1,2,...,L\}$ layers of a network, init weights to large values such that $|W^{[l]}|>1$.
After chained weight matrix multiplications $W^{[1]} W^{[2]} ... W^{[L]}$, the final result could be very large.
Choosing a right/dynamic learning rate $\eta$ is critical.

On the other side, init weights to very small values such that $|W^{[l]}| \approx 0$, there will be $W^{[1]} W^{[2]} ... W^{[L]} \approx \bold{0}$, besides having a learning rate that is often $\eta \ll 1$ contributing to gradients and weight updates.

### Reasons of Different Initialization Method Designs

To prevent the gradients of the network's activations from vanishing or exploding, here advises the following rules of thumb:

* The mean of the activations should be zero.
* The variance of the activations should stay the same across every layer.

Set $\bold{a}^{[l]}=\sigma(\bold{z}^{[l]})$ as the $l$-th layer neuron output, to keep gradient from vanishing/exploding, the pass-down neuron energy should be contained.
In detail, for different layer neuron outputs $\bold{a}^{[l]}$, should normalize them to the same normal distribution.

$$
\begin{align*}
    E(\bold{a}^{[l]}) &= E(\bold{a}^{[l-1]}) \\
    Var(\bold{a}^{[l]}) &= Var(\bold{a}^{[l-1]})
\end{align*}
$$

The scale of neuron output should stay constant such that $Var(\bold{a}^{[l]}) = Var(\bold{a}^{[l-1]})$.
Quotes from *Deep Learning* from Goodfellow et al:

```txt
We almost always initialize all the weights in the model to values drawn randomly from a Gaussian or uniform distribution.
The choice of Gaussian or uniform distribution does not seem to matter very much, but has not been exhaustively studied.
The scale of the initial distribution, however, does have a large effect on both the outcome of the optimization procedure and on the ability of the network to generalize.
```

For significance of scale:

* Initialization method should encourage large gradients for training.

#### Mean and Var of Uniform Distribution

Set $f(x)=\frac{1}{2a}, \forall x \in [-a, a]$ as probability density function (PDF) of $f(A)$ (indicative as height in the $f(A)$).

<div style="display: flex; justify-content: center;">
      <img src="imgs/uniform_dist.png" width="40%" height="20%" alt="uniform_dist" />
</div>
</br>

* Mean

$$
\begin{align*}
E(X) &= \int_{-a}^{a} x f(x) dx \\
&= \int_{-a}^{a} x \frac{1}{2a} dx \\
&= 0
\end{align*}
$$

* Variance

$$
\begin{align*}
Var(X) &= E(X^2) - \big(E(X)\big)^2 \\
&= \int_{-a}^{a} x^2 \frac{1}{2a} dx - 0 \\
&= \frac{1}{2a} \frac{x^3}{3}\bigg|^{a}_{-a} - 0 \\
&= \frac{1}{2a} \frac{2a^3}{3} - 0 \\
&= \frac{a^2}{3} \\
\end{align*}
$$

#### Parameter Initialization by Uniform Distribution Var

Set $n^{[l]}$ as the input dimension of the $l$-th layer, the lower and upper boundary of 

$$
\frac{a^2}{3}=\frac{1}{n^{[l]}}
\quad \Rightarrow \quad
a= \pm \sqrt{\frac{3}{n^{[l]}}}
$$

or

$$
\frac{a^2}{3}=\frac{2}{n^{[l]}+n^{[l-1]}}
\quad \Rightarrow \quad
a= \pm \sqrt{\frac{6}{n^{[l]}+n^{[l-1]}}}
$$

### Methods to Model Parameter Initialization

Generally speaking, good model parameter initialization should

* diversify error back-propagating to different weight parameters to learn different features
* be scale-invariant to prevent gradient exploding/vanishing
* encourage large gradients for training, prevent the scenarios where $\bold{z}^{[l]} \ll 0$ nor $\bold{z}^{[l]} \gg 0$, that activation function e.g., the gradients of sigmoid/tanh and relu, are $\approx 0$.

#### Uniform vs Normal Distribution Initialization

* Uniform Distribution $U$ Initialization: a uniform distribution has the equal probability of picking any number from a set of numbers.

Create diverse asymmetric parameter distribution with highest entropy, hence the uniform.
Have larger absolute weights the gradients will back-propagate better in a deep network.

$$
\begin{align*}
W^{[l]} &\sim \mathcal{U} \Big(-\frac{1}{\sqrt{n^{[l]}}}, \frac{1}{\sqrt{n^{[l]}}} \Big) \\
\bold{b}^{[l]} &= \bold{0}
\end{align*}
$$

* Normal Distribution $N$ Initialization: should have a mean of $\mu=0$ and a standard deviation of $\sigma=1/\sqrt{n^{[l]}}$

Effective in batch normalization to center distribution at $N(\mu=0, \sigma^2=1/n^{[l-1]})$.

$$
\begin{align*}
W^{[l]} &\sim \mathcal{N}(\mu=0, \sigma^2=1/n^{[l-1]}) \\
\bold{b}^{[l]} &= \bold{0}
\end{align*}
$$

#### Xavier vs He Weight Initialization

* Xavier Weight Initialization

Preferred for sigmoid/tanh.

**For normal distribution:**

$$
W^{[l]} \sim \mathcal{N}\left(0, \frac{2}{n^{[l-1]} + n^{[l]}}\right)
$$

**For uniform distribution:**

$$
W^{[l]} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n^{[l-1]} + n^{[l]}}}, \sqrt{\frac{6}{n^{[l-1]} + n^{[l]}}}\right)
$$

* He Weight Initialization

Preferred for ReLU and its variants.

**For normal distribution:**

$$
W^{[l]} \sim \mathcal{N}\left(0, \frac{2}{n^{[l-1]}}\right)
$$

**For uniform distribution:**

$$
W^{[l]} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n^{[l-1]}}}, \sqrt{\frac{6}{n^{[l-1]}}}\right)
$$

## Normalization

Normalization in statistics refers to adjustments to bring the entire probability distributions of adjusted values into alignment.

In machine learning where multiple dimensions are present, feature scaling should be conducted that typically takes data into the range $[0, 1]$.

The transform from a random variable $X$ to the scaled $X'$ is by this formula

$$
X'=\frac{X-X_{\min}}{X_{\max}-X_{\min}}
$$

### Example Batch Normalization (BN) vs Layer Normalization (LN)

For example, there are $N$ images each of the size $\bold{x}_i \in \mathbb{R}^{C \times H \times W}$, for $C$ channels/features of a window size $H \times W$.

* Batch Normalization (BN)

$$
\mu_i = \frac{1}{C \times H \times W}
\sum_{j=1}^C \sum_{h=1}^H \sum_{w=1}^W x_{i,j,h,w} \\
\sigma^2_i = \frac{1}{C \times H \times W}
\sum_{j=1}^C \sum_{h=1}^H \sum_{w=1}^W (x_{i,j,h,w} - \mu_i)^2
$$

* Layer Normalization (LN)

$$
\mu_j = \frac{1}{N \times H \times W}
\sum_{i=1}^N \sum_{h=1}^H \sum_{w=1}^W x_{i,j,h,w} \\
\sigma^2_j = \frac{1}{N \times H \times W}
\sum_{i=1}^N \sum_{h=1}^H \sum_{w=1}^W (x_{i,j,h,w} - \mu_j)^2
$$


### Batch Normalization (BN)

In batch normalization, instead by $X_{\max}$ and $X_{\min}$, mean $\mu_B$ and variance $\sigma^2_B$ of the batch are taken into scaling computation

$$
\begin{align*}
\text{Normalization } && \hat{x}_i &= \frac{x_i-\mu_B}{\sqrt{\sigma^2_B + \epsilon}} \\
\text{Scaling and Shifting } && {x}_i' &= \gamma \hat{x}_i + \beta
\end{align*}
$$

where $\gamma$ and $\beta$ are learned parameters, and $\epsilon$ is a small value (e.g., $\epsilon=10^{-5}$ in PyTorch) to prevent division by zero error.

For different batches there exist different means and variances, through which model learning can see increased robustness of prediction.

A too large batch size may produce too generalized means and variances, while a small size may contain noises.

The various batch sizes have implications on learning implemented by moving average.

* Exponential Moving Average (EMA) in Optimization

The most typical moving average in optimization is Exponential Moving Average (EMA), that can be used in parameter updates and gradient updates.

$$
\begin{align*}
\hat{\theta}_t &= \alpha\theta_t + (1-\alpha)\hat{\theta}_{t-1} \\
\hat{g}_t &= \alpha g_t + (1-\alpha) \hat{g}_{t-1}
\end{align*}
$$

where $\alpha$ is a momentum controlling parameter, e.g., for gradient update, $\alpha=0.9$ as the default for TF/Keras and $\alpha=0.99$ as the default for PyTorch.

A large momentum $\alpha$ gives a small step (slow convergence but small fluctuations), while a small momentum $\alpha$ gives a large step (fast convergence but significant fluctuations).

During inference, batch normalization uses the moving averages of the mean and variance (computed during training) to normalize the inputs.

* Scaling and Shifting

The normalized values (by $\hat{x}_i = \frac{x_i-\mu_B}{\sqrt{\sigma^2_B + \epsilon}}$) have zero mean (for assumed batch size of $m$, there is $\frac{1}{m}\sum^m_i \hat{x}_i-\mu_B=0$) and unit variance (for $\frac{1}{\sigma_B^2}\sum^m_i \frac{1}{m}(\hat{x}_i-\mu_B)^2=1$).
This is not good as zero mean and unit variance have removed scaling and shifting information.

The introduced learnable parameters $\gamma$ and $\beta$ restore the power of raw data feature representation.
Likely such real-world features are not zero-mean and unit-variance, hence the after-scaling-and-shifting distributions are more similar to the truth.

The batch normalization layer is typically added before/after activation layer.
Activation layer is non-linear, e.g., sigmoid and tanh, and easily gets saturated if deviated too much from central.
By constraining inputs/outputs to about the range $[-1, 1]$ (even having scaled and shifted, should drift away from the range), it prevents activation layer saturation.

### Layer Normalization (LN)

Normalization works on a layer of $d$ neurons that $\mu_L=\frac{1}{d}\sum_i^d x_i$ and $\sigma_L^2=\frac{1}{d}\sum_i^d (x_i-\mu_L)^2$.

$$
\begin{align*}
\text{Normalization } && \hat{x}_i &= \frac{x_i-\mu_L}{\sqrt{\sigma^2_L + \epsilon}} \\
\text{Scaling and Shifting } && {x}_i' &= \gamma \hat{x}_i + \beta
\end{align*}
$$

* Layer Normalization vs Batch Normalization

Reference:
https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d
https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

||CV|NLP|
|-|-|-|
|N|batch size|batch size|
|L|feature map height and width (H, W)|sequence length|
|H|channel (C)/Image Feature|embedding/hidden dimension|

Batch normalization computes on the L level.
For example, given a batch of two sentences $\bold{x}_1=$"I am a cat person" and $\bold{x}_2=$"You are a lovely dog person", a total of 11 tokens, and with assumed embedding dimension $d=768$, batch normalization would compute $\mu_{B,j}=\frac{1}{m}\sum_i^{m=11} x_{j,i}$, where $j=1,2,...,768$.

Layer normalization takes place on the L level.
For the same above example, the statistics are computed such that $\mu_{L,i}=\frac{1}{d}\sum_j^{d=768} x_{j,i}$, where $i=1,2,...,11$.

* Why Transformer uses LayerNorm

Layer Normalization has advantages in sequence and attention mechanism.

$\text{Attention Score} = Q K^{\top}$ follows a distribution that, according to the Central Limit Theorem, tends to a Gaussian distribution $\sim N(0,d)$ due to the sum of many independent random variables, where each variable is $\sim N(0,1)$.
Dividing by $\sqrt{d}$ makes the $\text{Attention Score}$ follow the same probability distributions $\sim N(0,1)$.

$$
\text{Attention}(Q,K,V) = \text{softmax} \Big( \frac{Q K^{\top}}{\sqrt{d}} \Big) V
$$

Having computed $\text{Attention}$, layer normalization standardizes the outputs of the $\text{Attention}$ to have a distribution $\sim N(0,1)$ across the feature dimension.

### Why BN good for images, LB good for NLP

In summary,

* Batch Normalization (BN) normalizes **cross-sample same-feature** data, that for images, visual features contain similar semantics across images
* Layer Normalization (LN) normalizes **same-sample cross-feature** data, that for text/token embeddings, two tokens have less common semantics but more likely each token embedding internally contain similar semantics
