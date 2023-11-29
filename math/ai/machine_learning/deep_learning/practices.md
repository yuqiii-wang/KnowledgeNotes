# Some Best Practice Discussions

## Loss Exploding

Training loss grows as iteration increases.

* Large learning rate

Large learning rate renders instability as iteration grows.

<div style="display: flex; justify-content: center;">
      <img src="imgs/training_loss_growth_for_large_learning_rate.png" width="50%" height="20%" alt="training_loss_growth_for_large_learning_rate" />
</div>
</br>

* ADAM Failed for large iteration $n$

For ADAM optimizer $\Delta W_{n+1} = \eta \frac{\hat{m}_{n+1}}{\sqrt{\hat{v}_{n+1}}+\epsilon}$ that takes into account the momentum, whose controlling parameters approach to zeros $\lim_{n \rightarrow +\infty} \beta_1^n = 0$ and $\lim_{n \rightarrow +\infty} \beta_2^n = 0$ when the iteration num $n$ goes very large.

* Instability of Gradients

$\frac{\partial \mathcal{L}}{\partial \bold{\theta}_k}$ can suddenly be very large due to gradient exploding.
This causes instability of the produced weights (totally large meaningless values) and the loss.

## Gradient Exploding/Vanishing

### Phenomena and Definitions

* Gradient Vanishing

By chain rule $\frac{\partial \mathcal{L}}{\partial \bold{\theta}_k}=\frac{\partial \mathcal{L}}{\partial \hat{y}}\frac{\partial \hat{y}}{\partial \bold{h}_l}\frac{\partial \bold{h}_l}{\partial \bold{h}_{l-1}}...\frac{\partial \bold{h}_{k+1}}{\partial \bold{h}_{k}} \frac{\partial \bold{h}_{k}}{\partial \bold{\theta}_k}$,
when a neural network is very deep, the gradient can be very small, and the weight/parameter/update $\bold{\theta}_k \leftarrow \eta\frac{\partial \mathcal{L}}{\partial \bold{\theta}_k} + \bold{\theta}_k$ is almost unchanged for $\frac{\partial \mathcal{L}}{\partial \bold{\theta}_k} \approx \bold{0}$.
Here, $\eta$ is learning rate.

A typical observation is that $\frac{\partial \mathcal{L}}{\partial \bold{h}_l}$ is large and $\frac{\partial \mathcal{L}}{\partial \bold{\theta}_k}$ is small.

* Gradient Exploding

The parameter update $\bold{\theta}_k \leftarrow \eta\frac{\partial \mathcal{L}}{\partial \bold{\theta}_{k}} + \bold{\theta}_k$ keeps increasing fast/is too large for `float32`/`double64` to hold full value that leads to overflow to `nan`.

There are many reasons.
It can be $\mathcal{L}(\bold{\theta})$ space being to mountainous containing abrupt cliffs that lead to sudden increase of derivatives.
It can be used optimizer having too strong accumulated momentum that rushes out/misses its nearby minimum.

Non-linearity introduces lots of abrupt cliffs.
In neural network, the activation function $\sigma(\space \cdot \space)$ is often non-linear (this is why one remediation solution is to use $\text{ReLU}$ as the activation function).
$$
\hat{\bold{y}} = \sigma \big( W_l \big( ... \sigma \big(W_2 \space \sigma(W_1 \bold{x} + \bold{b}_1) + \bold{b}_2) + ... + \big) + \bold{b}_l \big)
$$

For example, define $f(x)=-3.5 x^2 + 3.5x$.
Take $f(x)$'s output as another $f(x)$'s input, and keep stacking them, it can see the produced function space is very "mountainous".

<div style="display: flex; justify-content: center;">
      <img src="imgs/gradient_exploding_mountainous_space.png" width="40%" height="40%" alt="gradient_exploding_mountainous_space" />
</div>
</br>

### Reasons

For a typical forward pass $\bold{h}_{k+1}=\sigma(W_k \bold{h}_k + \bold{b}_k )$,
there exists derivative $\frac{\partial \bold{h}_{k+1}}{\partial \bold{h}_k}=\frac{\partial }{\partial \bold{h}_k} \Big(\sigma(W_k \bold{h}_k + \bold{b}_k )\Big)=W_k \space \sigma'(W_k \bold{h}_k + \bold{b}_k )$.

Diagonalizing a matrix $A$ is also equivalent to finding the matrix's eigenvalues $\lambda_i$, that comprise the diagonal values of $\Lambda$, whose rank is $\text{rank}(\Lambda)=r$.
Here, $P$ is the eigenvector-composed matrix.

$$
P^{-1} A P = D \Rightarrow
AP = PD =
\underbrace{
\begin{bmatrix}
  \bold{v}_1 & \bold{v}_2 & ... & \bold{v}_r
\end{bmatrix}}_{P}
\underbrace{
\begin{bmatrix}
  \lambda_1 & 0 & 0&  & 0\\
  0 & \lambda_2 & 0 &  & 0 \\
  0 & 0 & \lambda_3 &  & 0 \\
   &  &  & \ddots &  \\
  0 & 0 & 0 &  & \lambda_r
\end{bmatrix}}_{\Lambda}
$$

For spectral radius $||A\bold{x}|| \le \lambda_{max}||\bold{x}||$ that states that the max length stretching by $A$ is the max eigenvalue $\lambda_{max}$, here sets $A=\frac{\partial \bold{h}_{k+1}}{\partial \bold{h}_k}$.
$||A||$ describes the overall length/volume of a transform.
$||\frac{\partial \bold{h}_l}{\partial \bold{h}_{l-1}}||...||\frac{\partial \bold{h}_{k+1}}{\partial \bold{h}_{k}}|| \space ||\frac{\partial \bold{h}_{k}}{\partial \bold{\theta}_k}||$ can quantitatively describe the volume of change down to which layer of transform.

### Remediation for Both Vanishing and Exploding Gradient

* Batch Normalization

For gradient $\frac{\partial \mathcal{L}}{\partial \bold{\theta}_k}$ that works on sample input, take input $\bold{x}$ by large batch can resist extreme samples.

* Adam-Adjustable Learning Rate

ADAM updates any parameter with an individual learning rate.

### Remediation for Vanishing Gradient

* Use ResNet Bypass

* ReLU replacing non-linear activation function.

* Do not go too deep

By chain rule $\frac{\partial \mathcal{L}}{\partial \bold{\theta}_k}=\frac{\partial \mathcal{L}}{\partial \hat{y}}\frac{\partial \hat{y}}{\partial \bold{h}_l}\frac{\partial \bold{h}_l}{\partial \bold{h}_{l-1}}...\frac{\partial \bold{h}_{k+1}}{\partial \bold{h}_{k}} \frac{\partial \bold{h}_{k}}{\partial \bold{\theta}_k}$, 
when a neural network is very deep, the gradient can be very small.

* Weight Init

The parameter/weight init is to provide $W_0$ for $\mathcal{L}(\bold{\theta})$ to converge by updating $W$.
The initial parameters matter guiding where optimizer starts.

Typically, $W_0$ is init with standard normal distribution $W_0 \sim N(0, 1)$.

Large variances of initial parameters can render large gradient step, and $W_0 \sim N(0, \frac{1}{\sqrt{n}})$ is a remediation solution, where $n$ is the batch size.
Having assumed data sample is agreed to normal distribution, a large sample size/batch size should see small variances anyway.
Therefore, small variances $\sigma^2=\frac{1}{\sqrt{n}}$ can be used rather than $\sigma^2=1$.

### Remediation for Exploding Gradient

* Small learning rate

* Gradient Clipping

Simply set a threshold $t$ that if gradient $||\bold{g}||$ is too large, multiply gradient $\bold{g}$ with a small value $\epsilon$.

$$
\bold{g} \leftarrow \left\{
    \begin{array}{c}
        \epsilon \bold{g} & ||\bold{g}|| > t \\
        \bold{g} & ||\bold{g}|| \le t \\
    \end{array}
\right.
$$

* $\mathcal{L}^2$-norm regularization

Regularization can contain parameters from increasing too fast/to large values.

## Overfitting

When training loss curve vs validation loss curve diverge.
Should record the checkpoint model parameters and stop training.

<div style="display: flex; justify-content: center;">
      <img src="imgs/overfitting_plot.png" width="40%" height="40%" alt="overfitting_plot" />
</div>
</br>

### Use *Dropout* to enhance robustness of a network

Usually cut off $10\%$ neural connections of a dense layer (set some entries of a weight matrix to zeros).

### Regularization

Optimizer attempts to learn parameter $\bold{\theta} \in \mathbb{R}^d$.
Cost with added regularization can be defined as below.

$$
\min_{\bold{\theta}}
\mathcal{J}(\bold{\theta}) = 
\underbrace{\big( \bold{y} - \hat{\bold{y}} \big)^2}_{\text{traditional loss}} +
\underbrace{\lambda \sum_{i=1}^d \theta^p_i}_{\text{regularization}}
$$

where $\mathcal{L}_1$ penalty is $\lambda \sum^d_{i=1} |\theta_i|$ and $\mathcal{L}_2$ penalty is $\lambda \sum^d_{i=1} \theta^2_i$.

Explained (see the figure below for example): 

An optimizer attempts to learn the best $\bold{\theta} = \{\theta_1, \theta_2\}$ by $\min_{\bold{\theta}}\mathcal{J}(\bold{\theta})$.
Intuitively, the minimum (represented as the smallest blue inner circle) is located at somewhere $\bold{\theta} > \bold{0}$, and along the $\theta_1$-axis direction see steeper gradient (see contour lines, where contour intervals are small) than that of the $\theta_2$-axis' (indicating optimizer likely going along the $\theta_1$-axis' direction).

To regularize it, add $\lambda \sum_{i=1}^2 \theta^p_i$ (shown as orange contours).

The regularizer $\lambda \sum_{i=1}^d \theta^p_i$ increases cost when $\bold{\theta}$ stray away from the origin coordinate $(0, 0)$, hence, to reduce the overall cost $\mathcal{J}(\bold{\theta})$, $\theta_1 \rightarrow 0$ and $\theta_2 \rightarrow 0$ are contained close to the origin.
As a result, the best $\bold{\theta}$ are likely small values.

There are diffs between $\mathcal{L}_1$ (p=1) vs $\mathcal{L}_2$ (p=2),
that when converging $\mathcal{J}(\bold{\theta})$ by $|\theta_i|$ (the $\mathcal{L}_1$ scenario), individual $|\theta_1|$ would have more sway over $|\theta_2|$, dragging the new minimum (the white point in the figure) to the $\theta_1$-axis.
This results in totally missing out the $\theta_2$ info for the learned $\theta_2=0$ and $\theta_1 \gg \theta_2$.

In the $\mathcal{L}_2$ scenario, the regularizer is "rounded" that both $\theta_1 \ne 0$ and $\theta_2 \ne 0$ are learned.

<div style="display: flex; justify-content: center;">
      <img src="imgs/regularization_l1_vs_l2.png" width="40%" height="20%" alt="regularization_l1_vs_l2" />
</div>
</br>

### PyTorch Implementation

$\mathcal{L}_1$ penalty in python/pytorch is simply $\lambda|\bold{\theta|}$ by `torch.linalg.norm(param, p=1)`, where $\lambda=10^{-5}$.
The 

```python
l1_reg = torch.tensor(0., requires_grad=True)

for name, param in model.named_parameters():
    if 'weight' in name:
        l1_reg += torch.linalg.norm(param, p=1)

total_loss += 10e-5 * l1_reg
```

The $\lambda=10^{-5}$ is named `weight_decay` in pytroch.
It is often set to $\lambda=10^{-10} \eta$ proportional to learning rate $\eta$.

$\mathcal{L}_2$ penalty in python/pytorch is by `torch.linalg.norm(param, p=2)`.

```python
l2_reg = torch.tensor(0., requires_grad=True)

for param in model.parameters():
    if 'weight' in name:
        l2_reg += torch.linalg.norm(param, p=2)

loss += 10e-5 * l2_reg
```

By default, `torch.linalg.norm(...)` uses Frobenius norm if `p` is not set.

In PyTorch Adam optimizer, $\mathcal{L}_2$ regularizer (Decoupled Weight Decay Regularization) is added by default when `weight_decay` is not zero.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
```

### For CV

* Random Image Crop
* Random Flip/Rotation

### For NLP

## Warmup

## Model Compression: Pruning/Distillation and Quantization

### Pruning/Distillation

Pruning permanently drops certain weights.

In pytorch, there is `import torch.nn.utils.prune as prune`.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet().to(device=device)

parameters_to_prune = (
            (model.conv1, 'weight'),
            (model.conv2, 'weight'),
            (model.fc1, 'weight'),
            (model.fc2, 'weight'),
            (model.fc3, 'weight'))

## prune all layer parameters removing 20% of which by L1 unstructured pruning method (individual neuron removals by L1)
prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)
```

#### Best Practices and Lottery Ticket Hypothesis

*The Lottery Ticket Hypothesis: Training Pruned Neural Networks* proves that a subnet contains the same inference capability as a its parent large network.

*lottery ticket hypothesis*: dense, randomly-initialized, feed-forward networks contain subnetworks (winning tickets) that, when trained in isolation, reach test accuracy comparable to the original network in a similar number of iterations.

Good parameter initialization achieves more than $90\%$ compression rate.
Finding good parameter initialization is particularly important.

#### Unstructured vs Structured Pruning

* Unstructured Pruning: prune individual parameters
* Structured Pruning: removes entire structures of parameters (e.g., the whole layer of parameters)

#### Local vs Global Pruning

* Local: prune as per layer
* Global: prune as multiple layers, or all layers

Model's width per layer represents content of input;
model's depth represents non-linearity capability of transforming input.

Figure below shows that comprehensive pruning (width + depth) is better rather than just pruning on single layer neurons.

<div style="display: flex; justify-content: center;">
      <img src="imgs/pruning_effectiveness.png" width="30%" height="30%" alt="pruning_effectiveness" />
</div>
</br>

#### Pruning Methods: By L1, Ln Norms

The key idea is that, norm of a matrix $||W||_p$ describes the average info contained in one neuron.
Insignificant neurons should see low activation energy (low input value), hence the weights should have small $|w_i|_p$

For example, illustrated in the figure below, for a pruning rate of $50\%$ (remove half of parameters) and by $\mathcal{L}_1$ norm of $|W|$, the lowest $|w_i|$ are set to zeros.

<div style="display: flex; justify-content: center;">
      <img src="imgs/pruning_by_weight_magnitude.png" width="50%" height="20%" alt="pruning_by_weight_magnitude" />
</div>
</br>

The above $\mathcal{L}_1$ norm removal can be done in Pytorch such as below.
The original weight $W$ is renamed by prune as `weight_orig`, and the corresponding binary mask is `weight_mask` that sets lowest element to zeros, and retains full pass of original weights element-wise multiplying by $1$.

$$
W \odot M_{mask} = W_{pruned}, \qquad
\text{where }
M_{mask} = \begin{bmatrix}
    1 & 0 & 0 & 0 \\
    1 & 1 & 0 & 1 \\
    1 & 0 & 0 & 1 \\
\end{bmatrix}
$$

where $\odot$ is an element-wise multiplication operator.

```python
m = prune.l1_unstructured(nn.Linear(3, 4), 'weight', amount=0.5)
m.state_dict().keys()
# print
# odict_keys(['bias', 'weight_orig', 'weight_mask'])
```

* $\mathcal{L}_1$ Norm: remove the lowest $|w_i|$:

PyTorch implementation (remove the specified `amount` of (currently un-pruned) units with the lowest L1-norm) reference:
https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.l1_unstructured.html#torch.nn.utils.prune.l1_unstructured

```python
torch.nn.utils.prune.l1_unstructured(module, name, amount, importance_scores=None)
```

* $\mathcal{L}_2$ Norm: remove lowest $w_i^2$

PyTorch implementation (remove the specified `amount` of (currently un-pruned) channels along the specified `dim` with the lowest L`n=2`-norm) reference:
https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.ln_structured.html

```python
torch.nn.utils.prune.ln_structured(module, name, amount, n, dim, importance_scores=None)
```

### Quantization

*Quantization* refers to techniques for performing computations and storing tensors at lower bitwidths than floating point precision, such as `float32` $\rightarrow$ `int8`.

PyTorch supports INT8 quantization compared to typical FP32 models allowing for a 4x reduction in the model size as well as memory bandwidth.

In pytorch, simply add `<model_name>` and `dtype=torch.qint8` to `torch.ao.quantization.quantize_dynamic(...)`.

Reference: https://pytorch.org/docs/stable/quantization.html.

## Fine Tuning

Fine tuning is used for large model to adapt small sample data based on pre-trained parameters.

Parameter trainings:

* From scratch: totally from random parameters
* Full-parameter fine-tuning: all parameter fine-tuning
* Parameter-efficient fine-tuning: only less than $10\%$ of parameters are put in training

### LoRA: Low-Rank Adaptation of Large Language Models

For input $\bold{x} \in \mathbb{R}^{n \times d}$, where $d$ is for dimensionality, to fine tune an pretrained model, LoRA proposes below idea.
* $W_0 \in \mathbb{R}^{d \times d}$ is the pretrained parameters. 
* $A \sim N(0, \sigma^2) \in \mathbb{R}^{r \times d}$ is a weight matrix to be learned; it parameters are init by Gaussian distribution. $A$'s output reduces dimension to $r$
* $B = \bold{0} \in \mathbb{R}^{r \times d}$ is another weight matrix init to zeros. $B$'s output reset the dimension to $d$.

The training goes as below, that $W_0$ is kept unchanged/freezed; $B^{\top} A^{\top}$ are trainable parameter matrices. 

$r \ll d$ is referred to as *low-rank*.
$r=8$ is a typical implementation.
A small $r$ can help reduce computation maintaining small sizes for $A$ and $B$.

<div style="display: flex; justify-content: center;">
      <img src="imgs/lora.png" width="20%" height="20%" alt="lora" />
</div>
</br>


The new hidden layer matrix is computed as below.
$$
\bold{h} = W^{\top}_0\bold{x} + B^{\top} A^{\top} \bold{x}
$$

For intrinsic dimension (intrinsic dimension for a data set can be thought of as the number of variables needed in a minimal representation of the data), the number of neurons is small $r \ll d$ but can produce good approximation results.

###  Adapter: Parameter-Efficient Transfer Learning for NLP

Adapter adds new modules (called *adapter*) between layers of pre-trained transformer network.

Compared to LoRA, the adapter's Feed-forward down-project matrix is comparable to $A$, the up-project is comparable to $B$.
$m$ serves the same purposes as $r$'s reducing dimensions.

Adapter adds non-linearity between $A$ and $B$.

<div style="display: flex; justify-content: center;">
      <img src="imgs/petl_adapter.png" width="40%" height="50%" alt="petl_adapter" />
</div>
</br>

#### Improvement: Adapter Fusion

Adapter fusion adds attentions (value, key and query) that take adapter's output as value and key, and query from adapter's input.

Define the below parameter groups:
* Pretrained parameters $W_0$
* Adapter parameters $\Psi$
* Adapter Fusion parameters $\Phi$

The adapter fusion training goes as below:
1. fixed $W_0$, just train $\Psi$: there are multiple modules of adapters learning different knowledge
2. fixed $W_0$ and $\Psi$, train $\Phi$: attention serves as a filter that only task-specific knowledge is stored.

<div style="display: flex; justify-content: center;">
      <img src="imgs/adapter_fusion.png" width="20%" height="35%" alt="adapter_fusion" />
</div>
</br>

###  Prefix-tuning

