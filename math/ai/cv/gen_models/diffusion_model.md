# Diffusion Model

Image data is added Gaussian noises multiple $T$ times and eventually becomes absolute Gaussian noise.
Diffusion model learns this behavior reversely so that the model knows how pixel should get updated by what patterns between two chronologically related image frames.

References:

* https://zhuanlan.zhihu.com/p/525106459
* https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

## Diffusion Forward

Provided an image $\mathbf{x}_0\sim q(\mathbf{x})$, Gaussian noise is added to derive $\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T$ image frames.
This process is named $q$ process.

Let $\{\beta_t \in (0,1)\}^T_{t=1}$ be Gaussian variances.
For each $t$ moment the forward is only related to the previous $t-1$ moment, the process is a Markov process:

$$
\begin{align*}
    q(\mathbf{x}_t|\mathbf{x}_{t-1}) &= \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I}) \\
    q(\mathbf{x}_{1:T}|\mathbf{x}_{0}) &= \prod^T_{i=1} q(\mathbf{x}_t|\mathbf{x}_{t-1})
 \end{align*}
$$

Apparently, as $t$ increases, $\mathbf{x}_t$ approaches to pure Gaussian noise.

### Reparameterization Trick

Differentiation in sampling from a distribution is not applicable.
Reparameterization trick comes in help to make the process differentiable by an introduced independent variable $\epsilon$.

For example, to sample from a Gaussian distribution $z\sim N(z;\mu_\theta, \sigma_\theta^2 I)$, it can be written as

$$
z=\mu_\theta+\sigma_\theta \odot\epsilon,\quad \epsilon\sim N(0,I)
$$

where $\mu_\theta, \sigma_\theta$ are mean and variance of a Gaussian distribution that is determined by a neural network parameterized by $\theta$.
The differentiation is on $\epsilon$.

### Why Used ${1-\beta_t}$ as Variance in Progress

The primary role of $\beta_t$ is to control the variance, or amount, of Gaussian noise that is added to an image at each timestep t in the forward process.
Then, define $\alpha_t = 1 - \beta_t$ that represents the proportion of the signal (the previous image $\mathbf{x}_{t-1 }$)that is preserved at each step.

While $\beta_t$ controls the noise, $\alpha_t$ controls how much of the original image's characteristics are kept.
A high $\alpha_t$ means less noise is added, and more of the signal is preserved.

Recall that $q(\mathbf{x})$ needs to maintain every $\mathbf{x}_t$ converge to $N(0, I)$ for $t=1,2,...,T$.
In any arbitrary step $t$, the progress can be expressed as

$$
\begin{align*}
\mathbf{x}_t &= \sqrt{\alpha_t}\mathbf{x}_{t-1}+\sqrt{1-\alpha_t}\space\mathbf{z}_{1} \\
&= \sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}\mathbf{x}_{t-2}+\sqrt{1-\alpha_{t-1}}\mathbf{z}_{2}\right)+\sqrt{1-\alpha_t}\space\mathbf{z}_{1} \\
&= \sqrt{\alpha_t \alpha_{t-1}}\mathbf{x}_{t-2}+\left(\sqrt{\alpha_t}\sqrt{1-\alpha_{t-1}}\mathbf{z}_{2}+\sqrt{1-\alpha_t}\space\mathbf{z}_{1} \right) \\
&= \sqrt{\alpha_t \alpha_{t-1} \alpha_{t-2}}\mathbf{x}_{t-3}+\left(\sqrt{\alpha_{t-1}}\sqrt{1-\alpha_{t-2}}\mathbf{z}_{3}+\sqrt{\alpha_t}\sqrt{1-\alpha_{t-1}}\mathbf{z}_{2}+\sqrt{1-\alpha_t}\space\mathbf{z}_{1} \right) \\
&= ... \\
\end{align*}
$$

where $\mathbf{z}_{1}, \mathbf{z}_{2}, ... \sim N(0,I)$.

Given Gaussian distribution independence, there is $N(0,\sigma_1^2 I)+N(0,\sigma_2^2 I)\sim N\left(0,(\sigma_1^2+\sigma_2^2)I\right)$, so that

$$
\begin{align*}
\sqrt{\alpha_{t}}\sqrt{1-\alpha_{t-1}}\mathbf{z}_{2}+\sqrt{1-\alpha_t}\space\mathbf{z}_{1} &\sim
N\left(0, ({\alpha_{t}}({1-\alpha_{t-1}})+{1-\alpha_t})I\right) \\
&= N\left(0, (1-\alpha_t\alpha_{t-1})I\right) \\
\end{align*}
$$

Let $\overline{\alpha}_t=\prod^t_{\tau=1}\alpha_{\tau}$ be the chained product, the above $\mathbf{x}_t$ can be expressed as

$$
\begin{align*}
\mathbf{x}_t &= \sqrt{\alpha_t \alpha_{t-1} \alpha_{t-2}}\mathbf{x}_{t-3}+\left(\sqrt{\alpha_{t-1}}\sqrt{1-\alpha_{t-2}}\mathbf{z}_{3}+\sqrt{\alpha_t}\sqrt{1-\alpha_{t-1}}\mathbf{z}_{2}+\sqrt{1-\alpha_t}\space\mathbf{z}_{1} \right) \\
&= ... \\
&= \sqrt{\overline{\alpha}_t}\space\mathbf{x}_{0}+\sqrt{1-\overline{\alpha}_t}\space\overline{\mathbf{z}_{t}}
\end{align*}
$$

## Reverse Diffusion Process

The reverse process is to remove noise such that $q(\mathbf{x}_{t-1}|\mathbf{x}_{t})$.
