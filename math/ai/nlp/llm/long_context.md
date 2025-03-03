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

### Long Distance

When two tokens ($\bold{q}_m$ positioned at $m$ and $\bold{k}_n$ positioned at $n$) are very distant $|n-m|=\Delta\rightarrow \infty$, the score $\langle \bold{q}_m, \bold{k}_n \rangle$ has multiple mappings hence the attention score cannot determine which query token be associated to which key token.
Consequently, in long distance, the attention mechanism fails.

Consider

$$
\begin{align*}
\langle \bold{q}_m, \bold{k}_n \rangle = \sum_{i=0}^{D/2-1} \Big(
    & \underbrace{\big(q_{2i}^m k_{2i}^{m+\Delta} + q_{2i+1}^m k_{2i+1}^{m+\Delta}\big)}_{\alpha_{\cos}} \cos(\Delta\theta_i) + \\
    & \underbrace{\big(q_{2i+1}^m k_{2i}^{m+\Delta} - q_{2i}^m k_{2i+1}^{m+\Delta}\big)}_{\alpha_{\sin}} \sin(\Delta\theta_i) \Big)
\end{align*}
$$

To study the series $\sum_{i=0}^{D/2-1}\big(\alpha_{\cos}\cos(\Delta\theta_i)+\alpha_{\sin}\sin(\Delta\theta_i)\big)$ as $\Delta\rightarrow\infty$,
first consider this expression $\cos(\Delta\theta_i)+\sin(\Delta\theta_i)$,

* $\cos(\Delta\theta_i)$ and $\sin(\Delta\theta_i)$ are oscillation function (does not converge to a limit but oscillate within a range)
* $\cos(\Delta\theta_i)$ and $\sin(\Delta\theta_i)$ linear combinations are also oscillating.

One can prove that

$$
\begin{align*}
    \max\big(\cos(\Delta\theta_i)+\sin(\Delta\theta_i)\big)&=\sqrt{2} \\
    \min\big(\cos(\Delta\theta_i)+\sin(\Delta\theta_i)\big)&=-\sqrt{2} \\
\end{align*}
$$

As a result, the convergence behavior of $\langle \bold{q}_m, \bold{k}_n \rangle$ is determined by its linear coefficients $\alpha_{\cos}$ and $\alpha_{\sin}$.
Further more, for $\alpha_{\cos}>\alpha_{\sin}$, the convergence behavior is dominated by the cosine term.

Recall that for cosine, the monotonic area is $[0, \pi)$, and $[\pi, 2\pi)$ area is the mirror of the $[0, \pi)$.
To ensure one-to-one mapping query-key computation to attention score, the theoretical context length is $[0, \pi)$.

For a very large $\Delta\rightarrow\infty$, the $\Delta\theta_i$ steps across multiple oscillation ranges $[[0, \pi), [\pi, 2\pi), [2\pi, 3\pi), ...]$,
so that the attention score cannot determine which query token be associated to which key token.

### Linear Naive RoPE Interpolation and Extrapolation

To extend context length, one can simply do scaling.

Given $\theta_i=10000^{-2i/d}$ for $d=1,2,...,D/2$, with token position gap $\Delta$ and the attention formula $\sum_{i=0}^{D/2-1}\big(\alpha_{\cos}\cos(\Delta\theta_i)+\alpha_{\sin}\sin(\Delta\theta_i)\big)$,
to scale context length, do $L'=sL$ for $s>1$ by $\Delta\rightarrow\Delta/s$.

Accordingly, the rotation angle is $\frac{1}{s}\Delta\theta_i$.

This approach is not good as it treats all frequencies indiscriminantly.

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

$$
\tilde{\theta}_i = \theta_i \cdot \left( \frac{s^{\alpha}}{s^{\alpha} - 1 + \theta_i^{\beta}} \right)^{\gamma}
$$

#### RoPE-based Context Length Extension by NTK-Aware Methods

## YaRN: Efficient Context Window Extension of Large Language Models

Reference:

* https://arxiv.org/pdf/2309.00071
