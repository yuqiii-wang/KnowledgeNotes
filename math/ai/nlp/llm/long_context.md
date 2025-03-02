# LLM Long Context Interpolation and Extrapolation

The maximal length of the sequences (the context window) is determined by its training processes, e.g., DeepSeek V2 context length is 4K when DeepSeek V2 is trained.

The pre-defined max context length is often not enough in practical use, and interpolation with little fine-tuning, or no fine-tuning should be investigated.
DeepSeek V2 managed to scale to 128K context length with YaRN implementation.

## Context Length and RoPE

RoPE (Rotary Position Embedding) uses a rotational transformation based on sine and cosine functions to encode the position information.

$\bold{q}_i$ and $\bold{k}_j$ are a query vector and key vector in the attention formula $\text{softmax} \big(\frac{\bold{q}_i^{\top} \bold{k}_j}{\sqrt{d_k}} \big) \bold{v}_i$.
The dimensions of $\bold{q}_i$ and $\bold{k}_j$ represent the sinusoidal-scaled position.

Let $\bold{q}_i=R_{i}\bold{q}_1$ and $\bold{k}_j=R_{j}\bold{k}_1$ so that their position info is represented via rotation matrices $R_{i}$ and $R_{j}$, there is

$$
\max \text{score}(\bold{q}_i, \bold{k}_j) =
(R_{i} \bold{q}_1)^{\top} (R_{j} \bold{k}_1) =
\bold{q}_1^{\top} R_{i}^{\top}  R_{j} \bold{k}_1 =
\bold{q}_1^{\top} R_{i-j} \bold{k}_1
$$

Now use and $\theta_i \in (10^{-4}, 1]$ such that $\theta_i=10000^{-\frac{2i}{\bold{d}}}$ to assign discrete values to $R_{i-j}$.

Let $D$ represent the dimension num of $\bold{v}_i \in \mathbb{R}^{1 \times D}$.
Let $R(\theta)$ be a rotation matrix for a vector $\bold{v}_i$, there is

$$
\cos(\theta) = \frac{\bold{v}_i \cdot \bold{v}_j}{||\bold{v}_i || \space || \bold{v}_j ||}
\qquad
R (\theta) = \begin{bmatrix}
      \cos \theta & -\sin \theta \\
      \sin \theta & \cos \theta \\
\end{bmatrix}
$$

Rotation relative info can be computed by $R_{\theta_{i}-\theta_{j}}=R_{\theta_{i}}^{\top}{R_{\theta_{j}}}$, there is

$$
R(\theta) = \begin{bmatrix}
    \cos \theta_1 & -\sin \theta_1 & 0 & 0 & & & 0 & 0 \\
    \sin \theta_1 & \cos \theta_1 & 0 & 0 & & & 0 & 0 \\
    0 & 0 & \cos \theta_2 & -\sin \theta_2 & & & 0 & 0 \\
    0 & 0 & \sin \theta_2 & \cos \theta_2 & & & 0 & 0 \\
    & & & & \ddots & \ddots & & & \\
    & & & & \ddots & \ddots & & & \\
    0 & 0 & 0 & 0 & & & \cos \theta_{D/2} & -\sin \theta_{D/2} \\
    0 & 0 & 0 & 0 & & & \sin \theta_{D/2} & \cos \theta_{D/2} \\
\end{bmatrix}
$$

If $\bold{v} \in \mathbb{R}^{n \times D}$, where $n$ is the num of tokens
Here sets $n=D$, there is

$$
R(\theta) \bold{v} =
\begin{bmatrix}
      \bold{v}_1 \\ \bold{v}_2 \\ \bold{v}_3 \\ \bold{v}_4 \\ \vdots \\ \bold{v}_{D-1} \\ \bold{v}_{D}
\end{bmatrix} \odot
\begin{bmatrix}
      \cos \theta_1 \\ \cos \theta_1  \\ \cos \theta_2 \\ \cos \theta_2 \\ \vdots \\ \cos \theta_{D/2} \\ \cos \theta_{D/2}
\end{bmatrix} +
\begin{bmatrix}
      \bold{v}_1 \\ \bold{v}_2 \\ \bold{v}_3 \\ \bold{v}_4 \\ \vdots \\ \bold{v}_{D-1} \\ \bold{v}_{D}
\end{bmatrix} \odot
\begin{bmatrix}
      -\sin \theta_1 \\ \sin \theta_1  \\ -\sin \theta_2 \\ \sin \theta_2 \\ \vdots \\ -\sin \theta_{D/2} \\ \sin \theta_{D/2}
\end{bmatrix}
$$

where $\odot$ is element-wise multiplication operator.

###

## NTK-Aware Context Length Extension

### Neural Tangent Kernel (NTK) Introduction

Neural Tangent Kernel (NTK) is a study of the **dynamics** of neural networks during training by **gradient descent** on the assumption that the network is **infinitely wide**.
It reveals what the core invariant behavior of the network is as parameters change/update in training process.

The NTK kernel is defined as

$$
\kappa(\bold{x}, \bold{x}') = \big(\nabla_{\bold{\theta}}f_{\bold{\theta}}(\bold{x})\big)^\top\big(\nabla_{\bold{\theta}}f_{\bold{\theta}}(\bold{x}')\big)
$$

where $\bold{x}, \bold{x}'\in\mathbb{R}^d$ are the input vectors, and $\bold{\theta}\in\mathbb{R}^d$ is the parameter vector for the neural network $f_{\bold{\theta}}(.)$.

NTK major conclusions are

* For an infinite-width network, if parameter $\bold{\theta}$ is initialized from a certain distribution (e.g., Gaussian distribution), then the kernel $\kappa_{\bold{\theta}}(\bold{x}, \bold{x}')$ is deterministic (invariant to individual parameter changes), and does not change as optimization progresses (indicated that Jacobian is invariant and equivalent to its initialization $\kappa_{\bold{\theta}}(\bold{x}, \bold{x}')=\kappa_{\bold{\theta}_0}(\bold{x}, \bold{x}')$).
* An infinite-width network is linear.
* Eigenvalues of NTK matrix determines effectiveness of learning rate $\eta$ setup in training a neural network.

NTK kernel progresses by covariance matrix multiplication.

$$
\begin{align*}
    \kappa^{(l)}(\bold{x}, \bold{x}') &= \dot{\Sigma}^{(l)}(\bold{x}, \bold{x}') \cdot
    \underbrace{E\Big(\big(\nabla_{\bold{\theta}}f^{(l-1)}_{\bold{\theta}}(\bold{x})\big)^\top\big(\nabla_{\bold{\theta}}f^{(l-1)}_{\bold{\theta}}(\bold{x}')\big)\Big)}_{\text{NTK }\kappa^{(l-1)}(\bold{x}, \bold{x}')} +
    \Sigma^{(l)}(\bold{x}, \bold{x}')
\end{align*}
$$

### Eigenvalue Spectrum and Frequency Learning

Given the derived $\kappa(\bold{x}, \bold{x}') = \big(\nabla_{\bold{\theta}}f_{\bold{\theta}}(\bold{x})\big)^\top\big(\nabla_{\bold{\theta}}f_{\bold{\theta}}(\bold{x})\big)$ approximated the gradient,
its eigenvalue spectrum determines which features (low/high-frequency) are learned efficiently:

* Low-frequency patterns (slow variation) align with flat directions in parameter space $\rightarrow$ larger eigenvalues (faster convergence).
* High-frequency patterns (rapid variation) align with sharp directions $\rightarrow$ smaller eigenvalues (slower convergence).

#### Proof of Learning in Respective Frequency

As already proved that $\nabla_{\bold{\theta}}f_{\bold{\theta}}(\bold{x})\approx\nabla_{\bold{\theta}}f_{\bold{\theta}_0}(\bold{x})$, each iterative step can be treated equally.
This means that the conclusion drawn from a single iterative step $\nabla_{\bold{\theta}_t}f_{\bold{\theta}_t}(\bold{x})$ is also valid for all the iterative steps.

Given a sequence of inputs $\bold{x}_1, \bold{x}_2, \cdots, \bold{x}_n$,
consider the convolution to $K(\bold{x},\bold{x}')$ by discrete Fourier transform $\mathcal{F}(\bold{f(.)})=\sum_{k\in\mathbb{Z}^+}c_k e^{ik\bold{f(.)}}$, where $k$ is the discrete frequency, below expression yields the spectrum of $K(\bold{x},\bold{x}')$.

$$
\begin{align*}
    \mathcal{F}(K(\bold{x},\bold{x}')) &= \mathcal{F}\Big(\big(\nabla_{\bold{\theta}}f_{\bold{\theta}}(\bold{x})\big)^\top\big(\nabla_{\bold{\theta}}f_{\bold{\theta}}(\bold{x})\big)\Big) \\
\end{align*}
$$

#### RoPE-based Context Length Extension by NTK-Aware Methods

## YaRN: Efficient Context Window Extension of Large Language Models

Reference:

* https://arxiv.org/pdf/2309.00071
