# Long Context Interpolation

The maximal length of the sequences (the context window) is determined by its training processes, e.g., DeepSeek V2 context length is 4K when DeepSeek V2 is trained.

The pre-defined max context length is often not enough in practical use, and interpolation with little fine-tuning, or no fine-tuning should be investigated.
DeepSeek V2 managed to scale to 128K context length with YaRN implementation.

## Context Length and RoPE

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


### Eigenvalue Spectrum and Frequency Learning

Given the derived $\kappa(\bold{x}, \bold{x}') = \big(\nabla_{\bold{\theta}}f_{\bold{\theta}}(\bold{x})\big)^\top\big(\nabla_{\bold{\theta}}f_{\bold{\theta}}(\bold{x})\big)$ approximated the gradient,
its eigenvalue spectrum determines which features (low/high-frequency) are learned efficiently:

* High-frequency components: Rapidly varying patterns (e.g., fine-grained details in sequences).
* Low-frequency components: Slowly varying patterns (e.g., global structure).

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
