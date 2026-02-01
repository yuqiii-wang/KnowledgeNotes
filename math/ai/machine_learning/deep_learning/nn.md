# Neural Network

## Forward and Back Propagation

### Forward Propagation

$$
\begin{align*}
\mathbf{z} &= \mathbf{w}^\top \mathbf{x} + \mathbf{b} \\
\hat{\mathbf{y}} &= \sigma(\mathbf{z})
\end{align*}
$$

where $\sigma(\mathbf{z})$ is an activation function, that for example, by sigmoid there is $\sigma(\mathbf{z})\frac{1}{1+e^{-\mathbf{z}}}$.

### Back Propagation

Define Mean Squared Error (MSE) as the loss function (as an example) $L=\frac{1}{2}||\hat{\mathbf{y}}-\mathbf{y}||^2=\frac{1}{n} \frac{1}{2}\sum^n_i ({y}\_i-\hat{y}\_i)^2$

* Gradient of Loss w.r.t Activation:

$$
\begin{align*}
\frac{\partial L}{\partial \hat{\mathbf{y}}} = \hat{\mathbf{y}}-\mathbf{y}
\end{align*}
$$

* Gradient of Activation w.r.t Weighted Sum:

$$
\begin{align*}
\frac{\partial \hat{\mathbf{y}}}{\partial {\mathbf{z}}} &= \sigma(\mathbf{z})\big(1-\sigma(\mathbf{z})\big) \\
&= \hat{\mathbf{y}}(1-\hat{\mathbf{y}})
\end{align*}
$$

* Gradient of Loss w.r.t Weighted Sum:

$$
\begin{align*}
\frac{\partial L}{\partial {\mathbf{z}}} &= \frac{\partial L}{\partial \hat{\mathbf{y}}} \frac{\partial \hat{\mathbf{y}}}{\partial {\mathbf{z}}} \\
&= (\hat{\mathbf{y}}-\mathbf{y})\hat{\mathbf{y}}(1-\hat{\mathbf{y}})
\end{align*}
$$

* Gradients of Loss w.r.t Weights and Bias:

$$
\begin{align*}
\frac{\partial L}{\partial {\mathbf{w}}} &= \frac{\partial L}{\partial \hat{\mathbf{y}}} \frac{\partial \hat{\mathbf{y}}}{\partial {\mathbf{z}}} \frac{\partial {\mathbf{z}}}{\partial {\mathbf{w}}} \\
&= \mathbf{x}^\top (\hat{\mathbf{y}}-\mathbf{y})\hat{\mathbf{y}}(1-\hat{\mathbf{y}}) \\
\frac{\partial L}{\partial {\mathbf{b}}} &= \frac{\partial L}{\partial \hat{\mathbf{y}}} \frac{\partial \hat{\mathbf{y}}}{\partial {\mathbf{z}}} \frac{\partial {\mathbf{z}}}{\partial {\mathbf{b}}} \\
&= (\hat{\mathbf{y}}-\mathbf{y})\hat{\mathbf{y}}(1-\hat{\mathbf{y}})
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

## Normalization

Normalization in statistics refers to adjustments to bring the entire probability distributions of adjusted values into alignment.

In machine learning where multiple dimensions are present, feature scaling should be conducted that typically takes data into the range $[0, 1]$.

The transform from a random variable $X$ to the scaled $X'$ is by this formula

$$
X'=\frac{X-X_{\min}}{X_{\max}-X_{\min}}
$$

### Example Batch Normalization (BN) vs Layer Normalization (LN)

For example, there are $N$ images each of the size $\mathbf{x}\_i \in \mathbb{R}^{C \times H \times W}$, for $C$ channels/features of a window size $H \times W$.

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
\text{Normalization } && \hat{x}\_i &= \frac{x\_i-\mu_B}{\sqrt{\sigma^2_B + \epsilon}} \\
\text{Scaling and Shifting } && {x}\_i' &= \gamma \hat{x}\_i + \beta
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
\hat{\theta}_t &= \alpha\theta_t + (1-\alpha)\hat{\theta}\_{t-1} \\
\hat{g}_t &= \alpha g_t + (1-\alpha) \hat{g}\_{t-1}
\end{align*}
$$

where $\alpha$ is a momentum controlling parameter, e.g., for gradient update, $\alpha=0.9$ as the default for TF/Keras and $\alpha=0.99$ as the default for PyTorch.

A large momentum $\alpha$ gives a small step (slow convergence but small fluctuations), while a small momentum $\alpha$ gives a large step (fast convergence but significant fluctuations).

During inference, batch normalization uses the moving averages of the mean and variance (computed during training) to normalize the inputs.

* Scaling and Shifting

The normalized values (by $\hat{x}\_i = \frac{x\_i-\mu_B}{\sqrt{\sigma^2_B + \epsilon}}$) have zero mean (for assumed batch size of $m$, there is $\frac{1}{m}\sum^m_i \hat{x}\_i-\mu_B=0$) and unit variance (for $\frac{1}{\sigma_B^2}\sum^m_i \frac{1}{m}(\hat{x}\_i-\mu_B)^2=1$).
This is not good as zero mean and unit variance have removed scaling and shifting information.

The introduced learnable parameters $\gamma$ and $\beta$ restore the power of raw data feature representation.
Likely such real-world features are not zero-mean and unit-variance, hence the after-scaling-and-shifting distributions are more similar to the truth.

The batch normalization layer is typically added before/after activation layer.
Activation layer is non-linear, e.g., sigmoid and tanh, and easily gets saturated if deviated too much from central.
By constraining inputs/outputs to about the range $[-1, 1]$ (even having scaled and shifted, should drift away from the range), it prevents activation layer saturation.

### Layer Normalization (LN)

Normalization works on a layer of $d$ neurons that $\mu_L=\frac{1}{d}\sum_i^d x\_i$ and $\sigma_L^2=\frac{1}{d}\sum_i^d (x\_i-\mu_L)^2$.

$$
\begin{align*}
\text{Normalization } && \hat{x}\_i &= \frac{x\_i-\mu_L}{\sqrt{\sigma^2_L + \epsilon}} \\
\text{Scaling and Shifting } && {x}\_i' &= \gamma \hat{x}\_i + \beta
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
For example, given a batch of two sentences $\mathbf{x}_1=$"I am a cat person" and $\mathbf{x}_2=$"You are a lovely dog person", a total of 11 tokens, and with assumed embedding dimension $d=768$, batch normalization would compute $\mu_{B,j}=\frac{1}{m}\sum_i^{m=11} x_{j,i}$, where $j=1,2,...,768$.

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
