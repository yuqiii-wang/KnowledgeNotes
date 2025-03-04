# LLM Long Context Interpolation and Extrapolation

The maximal length of the sequences (the context window) is determined by its training processes, e.g., DeepSeek V2 context length is 4K when DeepSeek V2 is trained.

The pre-defined max context length is often not enough in practical use, and interpolation with little fine-tuning, or no fine-tuning should be investigated.
DeepSeek V2 managed to scale to 128K context length with YaRN implementation (empirical study shows only $0.1\%$ pre-training data can give good results).

## Context Length and Memory Consumption

Given $Q,K,V \in \mathbb{R}^{n \times d}$, where $n$ denotes token length and $d$ is a single attention head dimension, a standard attention can be written as

$$
\begin{align*}
    S &= Q K^{\top} \in \mathbb{R}^{n \times n} \\
    P &= \text{Softmax}(S) \\
    A &= PV \in \mathbb{R}^{n \times d}
\end{align*}
$$

Often, there is $n \gg d$ (e.g., for GPT2, $n=1024$ and $d=64$).
Attention score $S=Q K^{\top}$ has a large size if the row $n$ is large, or $S\sim\mathcal{O}(n^2)$.

## Context Length and RoPE

RoPE (Rotary Position Embedding) uses a rotational transformation based on sine and cosine functions to encode the position information.

$\bold{q}_m$ and $\bold{k}_n$ are a query vector and key vector in the attention formula $\text{softmax} \big(\frac{\bold{q}_m^{\top} \bold{k}_n}{\sqrt{d_k}} \big) \bold{v}_n$.
The dimensions of $\bold{q}_m$ and $\bold{k}_n$ represent the sinusoidal-scaled position.

Let $\bold{q}_m=R_{m}\bold{q}_1$ and $\bold{k}_n=R_{n}\bold{k}_1$ so that their position info is represented via rotation matrices $R_{m}$ and $R_{n}$, there is

$$
\max \text{score}(\bold{q}_m, \bold{k}_n) =
(R_{m} \bold{q}_1)^{\top} (R_{n} \bold{k}_1) =
\bold{q}_1^{\top} R_{m}^{\top}  R_{n} \bold{k}_1 =
\bold{q}_1^{\top} R_{n-m} \bold{k}_1
$$

The rotation angle can be represented by $\bold{w}_i = 10000^{-\frac{2i}{D}}$.

### Why RoPE Sees Limitation in Long Context

* RoPE angle $\theta_i$ is fixed in training, not capable of getting adapted in long context.
* Once reach over theoretical max token length, i.e., beyond $[0, \pi)$ range, the mappings will NOT be unique.

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

To extend context length $L$, one can simply do scaling such that $L'=sL$, where $s>1$ is a scaling factor.

Given $\theta_i=10000^{-2i/d}$ for $d=1,2,...,D/2$, with token position gap $\Delta$ and the attention formula $\sum_{i=0}^{D/2-1}\big(\alpha_{\cos}\cos(\Delta\theta_i)+\alpha_{\sin}\sin(\Delta\theta_i)\big)$,
to scale context length, do $L'=sL$ for $s>1$ by $\Delta\rightarrow\Delta/s$.

Accordingly, the rotation angle is $\frac{1}{s}\Delta\theta_i$.

This approach is not good as it treats all frequencies indiscriminantly and all frequencies drop, i.e., for $\Delta\theta_i>\frac{1}{s}\Delta\theta_i$, the scaled RoPE takes more rotation steps to complete a half wavelength, indicating frequency decrease across all frequency components.

In particular, higher-frequency components suffers more by the scaling $\frac{1}{s}\Delta\theta_i$ than lower-frequency ones.

## NTK-Aware Context Length Extension

### Neural Tangent Kernel (NTK) Introduction

Neural Tangent Kernel (NTK) is a study of the **dynamics** of neural networks during training by **gradient descent** on the assumption that the network is **infinitely wide**.
It reveals what the core invariant behavior of the network is as parameters change/update in training process.

The NTK kernel is defined as

$$
\kappa(\bold{x}, \bold{x}') = \big(\nabla_{\bold{w}}f_{\bold{w}}(\bold{x})\big)^\top\big(\nabla_{\bold{w}}f_{\bold{w}}(\bold{x}')\big)
$$

where $\bold{x}, \bold{x}'\in\mathbb{R}^d$ are the input vectors, and $\bold{w}\in\mathbb{R}^d$ is the parameter vector for the neural network $f_{\bold{w}}(.)$.

NTK major conclusions are

* For an infinite-width network, if parameter $\bold{w}$ is initialized from a certain distribution (e.g., Gaussian distribution), then the kernel $\kappa_{\bold{w}}(\bold{x}, \bold{x}')$ is deterministic (invariant to individual parameter changes), and does not change as optimization progresses (indicated that Jacobian is invariant and equivalent to its initialization $\kappa_{\bold{w}}(\bold{x}, \bold{x}')=\kappa_{\bold{w}_0}(\bold{x}, \bold{x}')$).
* An infinite-width network is linear.
* Eigenvalues of NTK matrix determines effectiveness of learning rate $\eta$ setup in training a neural network.

NTK kernel progresses by covariance matrix multiplication.

$$
\begin{align*}
    \kappa^{(l)}(\bold{x}, \bold{x}') &= \dot{\Sigma}^{(l)}(\bold{x}, \bold{x}') \cdot
    \underbrace{E\Big(\big(\nabla_{\bold{w}}f^{(l-1)}_{\bold{w}}(\bold{x})\big)^\top\big(\nabla_{\bold{w}}f^{(l-1)}_{\bold{w}}(\bold{x}')\big)\Big)}_{\text{NTK }\kappa^{(l-1)}(\bold{x}, \bold{x}')} +
    \Sigma^{(l)}(\bold{x}, \bold{x}')
\end{align*}
$$

### NTK-Aware Scaled RoPE

Rather than uniformly applying the scaling factor to all frequencies such that $\frac{1}{s}\Delta\theta_i$, the NTK-aware method **extrapolates high-frequency components** and conducts **interpolations in low-frequency components**.

Essentially, NTK-aware scaled RoPE is to replace the original base,
e.g., usually $10,000$, with $10000 \cdot k$.

$$
\theta'_i=\big(10000 \cdot k\big)^{-2\frac{i-1}{D}}
$$

As explained that higher-frequency components suffers more by the scaling $\frac{1}{s}\Delta\theta_i$ than lower-frequency ones,
the better scaling method should see mappings that

* for high-frequency low-dimension component, $s$ be dynamically set up so that $k\approx 1$, which gives $\theta'_i$ similar to $\theta_i$
* for low-frequency high-dimension component, $s$ be dynamically set up so that $\big(k \cdot 10000\big)^{-2\frac{i-1}{D}}$ be like linear interpolation

### NTK-Aware Scaled RoPE Derivation

Consider $\sum_{i=0}^{D/2-1}\big(\alpha_{\cos}\cos(\Delta\theta_i)+\alpha_{\sin}\sin(\Delta\theta_i)\big)$,
where $\bold{w}_i = 10000^{-\frac{2i}{D}}$.
Let $b=10000^{2/D}$ be the base, list the rotation angles

$$
[\Delta b^0, \Delta b^{-1}, ..., \Delta b^{-(D/2-1)}]
$$

The context length expansion goal is to increase the base $b$'s granularity to hold more encodings.
Here introduces a scaling factor $k>1$ to base, and $k \cdot b$ be the new scaled base.

For example, for $D=16$ and $b=10000^{2/D}$,
and let $\lambda_i=\frac{2\pi}{\theta_i}=2\pi \cdot 10000^{2i/16}$ be wavelength,
and let the half wavelength (for cosine in $[0, \pi)$ is monotonically decreasing) of the highest dimension be the theoretical max token length, i.e., $\lambda_i/2=9934.6$.

Then consider the new scaled base: for $k \cdot b$ as the new base, let $k=5$, there is
$\lambda'_i=\frac{2\pi}{\theta'_i}=2\pi \cdot 50000^{2i/16}$,
then the theoretical max token length is
$\lambda'_i/2=40620.8$.

#### Low-Frequency Components

For the highest dimension $\Delta b^{-(D/2-1)}$ (low-frequency components), having done scaling, there is $\Delta (k \cdot b)^{-(D/2-1)}$.

For low-frequency there should be interpolation, consider

$$
\frac{\Delta}{(k \cdot b)^{D/2-1}}=
\frac{\Delta/s}{b^{D/2-1}}
$$

Solve the above equation, there is $k=s^{2/(D-2)}$ or $s=k^{D/2-1}$.

To the original angle $\Delta b^{-(D/2-1)}$, here applies the scaling $k$, there is
$\Delta (k \cdot b)^{-(D/2-1)}$ that acts as linear interpolation.

#### High-Frequency Components

For high-frequency components (large $\theta_i$), the lowest dimension component is $\Delta b^{-1}$.
Again scale the base by $k$, there is $\Delta(k\cdot b)^{-1}$.

For usually $D\gg 0$ is large, there is $b\rightarrow 1$ makes no much change to high-frequency components,
and the expression $\Delta k \cdot 10000^{-2/D}\rightarrow \Delta$ is basically extrapolation with a step of $\Delta$.

### NTK-Aware Scaled RoPE Proof For NTK Kernel Invariance

Essentially, NTK-Aware method is to adjust the rotation angle such that $\theta'=\theta\cdot s^{D/(D-2)}$,
this is identical to dynamically scale per each dimension $\theta'_i=\theta_i\cdot s^{-2i/(D-2)}$

Take $\bold{w}$ as parameter set to a neural network $f_{\bold{w}}$,
for query $\bold{q}_m^{i}=\bold{w}_q^{(i)}\cdot e^{im\theta_i}$ and key $\bold{k}_n^{i}=\bold{w}_k^{(i)}\cdot e^{in\theta_i}$

Attention score is computed by $z_{m,n}=\sum^{D/2-1}_{i=0}(\bold{q}_m^{i})^{\top}\bold{k}_n^{i}$

Let the exterior product $\kappa_{\bold{w}}=E(\nabla_{\bold{w}}f \otimes \nabla_{\bold{w}}f)$ be the kernel.

#### Gradient Change With Respect To Angle Scaling

$$
\nabla_{\bold{w}}f'=\frac{\partial f}{\partial \bold{w}_q}\frac{\partial \bold{w}_q}{\partial \theta'_i}\frac{\partial \theta'_i}{\partial \theta_i}+
\frac{\partial f}{\partial \bold{w}_k}\frac{\partial \bold{w}_k}{\partial \theta'_i}\frac{\partial \theta'_i}{\partial \theta_i}
$$

where $\frac{\partial \theta'_i}{\partial \theta_i}=s^{-2i/(D-2)}$.

Construct Jacobian for angle scaling

$$
\begin{align*}
    J_{s}&=\text{diag}(s^{2i/(D-2)}) \in \mathbb{R}^{\frac{D}{2}\times\frac{D}{2}} \\
    &= \begin{bmatrix}
        s^{0} & 0 & 0 & & 0 \\
        0 & s^{2/(D-2)} & 0 & & 0 \\
        0 & 0 & s^{4/(D-2)} & & 0 \\
        & & & \ddots & \\
        0 & 0 & 0 & & s^{D/(D-2)} \\
    \end{bmatrix}
\end{align*}
$$

Then rewrite the gradient to

$$
\nabla_{\bold{w}}f'=J^{-1}_{s}\nabla_{\bold{w}}f
$$

#### Proof of NTK Kernel Invariance

Derive the new NTK kernel

$$
\begin{align*}
    \kappa'_{\bold{w}}&=E(J^{-1}_{s}\nabla_{\bold{w}}f \otimes J^{-1}_{s}\nabla_{\bold{w}}f) \\
    &= J^{-1}_{s} \kappa_{\bold{w}} J^{-1}_{s}
\end{align*}
$$

To prove $\kappa'_{\bold{w}}=\kappa_{\bold{w}}$, need to show $J^{-1}_s=J_s^{\top}$ that $J_s$ is an orthogonal matrix.
Recall that $J_s$ is a diagonal matrix, hence only its diagonal elements $s^{4i/(D-2)}$ needs to be computed for its determinant.

$$
\frac{1}{D}\text{det}(J_s)=
\lim_{D\rightarrow\infty} \frac{1}{D}\sum_{i=1}^{D} s^{4i/(D-2)}\approx
\int^{1}_{0}s^{-4x}dx=
\frac{1-s^{-4}}{4\ln s}=1
$$

Since $J_s$ is a diagonal matrix, $\frac{1}{D}\text{det}(J_s)=1$ can be considered that statistically speaking, each diagonal element is average to $1$, i.e., $E(J_s)=I$ is an identity matrix.

So that $J^{-1}_{s} \kappa_{\bold{w}} J^{\top}_{s}=\kappa_{\bold{w}}$.

##### Proof of Calculation $\lim_{D\rightarrow\infty} \frac{1}{D}\sum_{i=1}^{D} s^{4i/(D-2)}\approx 1$

* Riemann sum approximation

$$
\int^b_a f(x)dx=\lim_{n\rightarrow\infty}\sum^n_{i=1} f(x^*_i)\Delta x
$$

where $\Delta x=\frac{b-a}{n}$, and $x^*_i\in[x_{i-1}, x_i]$ is a sampling point within the interval $[x_{i-1}, x_i]$.
When there are sufficient sampling points, the sum can approximate integral.

Let $x=\frac{i}{D}$, use Riemann sum approximation

$$
\lim_{D\rightarrow\infty} \frac{1}{D}\sum_{i=1}^{D} s^{4i/(D-2)}\approx
\frac{1}{D}\int^{D}_{0}s^{-4x}dx
$$

* Compute the integral

$$
\int^{1}_{0}s^{-4x}dx=
\frac{s^{-4x}}{-4\ln s}\bigg|^{1}_{0}=
\frac{s^{-4}-1}{-4\ln s}
$$

This expression $\frac{1-s^{-4}}{4\ln s}=1$ means the scaling factor $s$ needs to satisfy (overall statistically) this equation as prerequisite for $J^{-1}_{s} \kappa_{\bold{w}} J^{\top}_{s}=\kappa_{\bold{w}}$ to hold true.

##### Proof of $J^{-1}_{s} \kappa_{\bold{w}} J^{\top}_{s}=\kappa_{\bold{w}}$

Simplify the question to prove $A^{-1}BA^{\top}=B$, where

* $A$ is statistically considered an identity matrix, i.e., $E(A)=I$
* $B$ is a positive symmetric matrix

Then, $E(A^{-1}BA^{\top})=E(A)^{-1}BE(A)^{\top}=B$.

## YaRN: Efficient Context Window Extension of Large Language Models

Reference:

* https://arxiv.org/pdf/2309.00071

### NTK-by-parts

### Yarn: NTK-by-parts + Temperature Tuning
