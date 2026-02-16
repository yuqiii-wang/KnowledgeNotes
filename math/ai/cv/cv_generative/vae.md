
# Variational Autoencoder (VAE)

An autoencoder is a neural network trained to reconstruct its input through a low-dimensional bottleneck (the latent code). It comprises an encoder and a decoder:

- Encoder: maps input $\mathbf{x}$ to latent code $\mathbf{c}$.
- Decoder: maps latent code $\mathbf{c}$ back to a reconstruction $\hat{\mathbf{x}}$.

<div style="display: flex; justify-content: center;">
      <img src="imgs/vae.png" width="70%" height="30%" alt="vae" />
</div>
</br>

The latent code $\mathbf{c}$ can be used to compress data into a "compression distribution" where desired features are extracted.
For example, put VAE with a loss function attentive to image texture features, one can then use Stable Diffusion (SD) as a guide to coarse content generation, then use VAE to restore texture details.

## Autoencoder (AE) vs Variational Autoencoder (VAE)

The fundamental diff is **Deterministic vs. Probabilistic**:

* Autoencoder (AE): is deterministic.
    * If fed an input, e.g., the same image of a cat twice, AE produces the exact same latent code vector $\mathbf{c}$
    * It treats the latent code as a **fixed coordinate**.
    * It is **good for compression**
* Variational Autoencoder (VAE): is probabilistic/stochastic.
    * If fed the same input, e.g., again the same image of a cat but twice, VAE produces the two similar latent code vectors $\mathbf{c}_1$ and $\mathbf{c}_2$
    * It treats the latent code as a **distribution** (typically, Gussian).
    * It is **good for generation**

They have diff loss functions

* Autoencoder (AE):
    * Reconstruction loss only: e.g., $|x-\hat{x}|^2$
* Variational Autoencoder (VAE):
    * Reconstruction loss  + KL Divergence: e.g., $|x-\hat{x}|^2+D_{KL}(q|p)$

## How to Know Latent Code $\mathbf{c}$

Latent code $\mathbf{c}$ are part of the model, but not observable, also not part of the dataset.

$$
p_{\theta}(\mathbf{x})=
\int_{\mathbf{c}} p_{\theta}(\mathbf{x}, \mathbf{c}) d\mathbf{c}
$$

However, this integration is intractable for it is impossible to go through all $\mathbf{c}$ for two reasons:

* $\mathbf{c}$ typically sit in a high-dimensional continuous space which is too large to search given relatively small size of input $\mathbf{x}$, e.g., a cat or dog image
* There is not known true posterior $p_{\theta}(\mathbf{c}|\mathbf{x})$ by which a supervised training can guide.

Here introduces a (typically, Gaussian) distribution $q_\phi(\mathbf{c} | \mathbf{x})\sim \mathcal{N}(\mathbf{c}; \mathbf{\mu}, \mathbf{\sigma}^2I)$ parameterized by $\phi$ as the guide to learn the latent variables $\mathbf{c}$.

### Prerequisite: Variational Inference

* In calculus, the purpose is to find a simple value $x$ that minimizes a function $f(x)$, use standard derivatives.
* In Calculus of Variations, the purpose is to find an entire function (or distribution) $q(x)$ that minimizes a "functional" (a function of functions).

### Prerequisite: Why Assume A Gaussian Distribution for The Latent Code

It is assumed the prior $p(\mathbf{c})\sim\mathcal{N}(\mathbf{0}, \mathbf{1})$ is a standard Gaussian and the posterior is another Gaussian distribution $q_\phi(\mathbf{c} | \mathbf{x})\sim\mathcal{N}(\mathbf{\mu}, \mathbf{\sigma}^2)$ for

* $p(\mathbf{c})\sim\mathcal{N}(\mathbf{0}, \mathbf{1})$ represents all possible codes without seeing any specific input. It can serve as the goal of learning.
* $q_\phi(\mathbf{c} | \mathbf{x})\sim\mathcal{N}(\mathbf{\mu}, \mathbf{\sigma}^2)$ is about under the influence of $\mathbf{x}$. As a result, the learned $\mathbf{c}$ is biased.

### Integration With $q_\phi(\mathbf{c} | \mathbf{x})$

Having introduced the $q_\phi(\mathbf{c} | \mathbf{x})$, the above intractable integration becomes solvable for now there is a guide of what the latent distribution should look like.

To maximize $\log p_{\theta}(\mathbf{x})$ with $q_\phi(\mathbf{c} | \mathbf{x})$, there is

$$
\begin{align*}
\log p_{\theta}(\mathbf{x}) &=
\log \int_{\mathbf{c}} p_{\theta}(\mathbf{x}, \mathbf{c}) d\mathbf{c} \\\\
&= \log \int_{\mathbf{c}} p_{\theta}(\mathbf{x}, \mathbf{c}) \frac{q_\phi(\mathbf{c} | \mathbf{x})}{q_\phi(\mathbf{c} | \mathbf{x})} d\mathbf{c} \\\\
&= \log E_{\mathbf{c}\sim q_\phi(\mathbf{c} | \mathbf{x})} \left(\frac{p_{\theta}(\mathbf{x}, \mathbf{c})}{q_\phi(\mathbf{c} | \mathbf{x})}\right)
\end{align*}
$$

### ELBO (Evidence Lower Bound)

ELBO is used in optimization/minimization utilizing an easy-to-solve lower bound function/distribution to replace the original complex function.

By Jensen inequality, there is

$$
\log E_{\mathbf{c}\sim q_\phi(\mathbf{c} | \mathbf{x})} \left(\frac{p_{\theta}(\mathbf{x}, \mathbf{c})}{q_\phi(\mathbf{c} | \mathbf{x})}\right) \ge
E_{\mathbf{c}\sim q_\phi(\mathbf{c} | \mathbf{x})} \left(\log \frac{p_{\theta}(\mathbf{x}, \mathbf{c})}{q_\phi(\mathbf{c} | \mathbf{x})}\right)
$$

The right hand side is ELBO (Evidence Lower Bound).

The term inside the expectation can be expanded using conditional probability $p_{\theta}(\mathbf{x}, \mathbf{c}) = p_{\theta}(\mathbf{x} | \mathbf{c}) p(\mathbf{c})$:

$$
\begin{align*}
\text{ELBO} &= E_{\mathbf{c}\sim q_\phi(\mathbf{c} | \mathbf{x})} \left(\log \frac{p_{\theta}(\mathbf{x} | \mathbf{c}) p(\mathbf{c})}{q_\phi(\mathbf{c} | \mathbf{x})}\right) \\\\
&= E_{\mathbf{c}\sim q_\phi(\mathbf{c} | \mathbf{x})} \left(\log p_{\theta}(\mathbf{x} | \mathbf{c}) + \log \frac{p(\mathbf{c})}{q_\phi(\mathbf{c} | \mathbf{x})}\right) \\\\
&= \underbrace{E_{\mathbf{c}\sim q_\phi(\mathbf{c} | \mathbf{x})} \bigg(\log p_{\theta}(\mathbf{x} | \mathbf{c})\bigg)}_{\text{Reconstruction Term}} - \underbrace{D_{KL}\bigg(q_\phi(\mathbf{c} | \mathbf{x}) \| p(\mathbf{c})\bigg)}_{\text{KL Divergence}}
\end{align*}
$$

Loss is designed as negative EBLO such that $-\text{ELBO}$, as a result,

* A small reconstruction loss reflects that latent code as decoded should recover the original dataset input $\mathbf{x}$
* The KL divergence describes the information lost in $p(\mathbf{c})$ when distribution $q_\phi(\mathbf{c} | \mathbf{x})$ is used to approximate distribution. In optimization, this divergence effectively forces $q_{\phi}$ to approximate the standard normal distribution $p(\mathbf{c})\sim\mathcal{N}(\mathbf{0}, I)$. Reflected in vision it is effectively blurring image.

### Reparameterization Trick

In the VAE loss function, there needs to estimate the expectation $E_{\mathbf{c}\sim q_\phi(\mathbf{c} | \mathbf{x})}$. This is usually done by sampling $\mathbf{c}$ from the distribution $q_\phi(\mathbf{c} | \mathbf{x})$.

However, **sampling operation is non-differentiable**, i.e., by directly sampling $\mathbf{c} \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\sigma}^2I)$, the randomness is embedded in the node $\mathbf{c}$, and therefore failed to compute gradients or backpropagate errors through to the encoder parameters $\phi$ (which predict $\mu$ and $\sigma$).

#### Why Sampling Breaks Backpropagation

Let sampling process be explicitly expressed as a function:
$\mathbf{c} = \text{sample}(\mathbf{\mu}, \mathbf{\sigma})$

This function is a "black box" of randomness. To calculate how a change in $\mathbf{\mu}$ or $\mathbf{\sigma}$ affects the loss, there are:

1.  The gradients propagate backward from the loss to $\mathbf{c}$.
2.  However, at node $\mathbf{c}$, the connection to $\mathbf{\mu}$ and $\mathbf{\sigma}$ is probabilistic, not functional.
3.  The derivative $\frac{\partial \mathbf{c}}{\partial \mathbf{\mu}}$ and $\frac{\partial \mathbf{c}}{\partial \mathbf{\sigma}}$ are undefined or zero because $c$ is technically just a fixed number drawn from a distribution during the forward pass. The randomness "blocks" the gradient flow.

#### Reparameterization Implementation

Instead of sampling $\mathbf{c}$ directly, this trick expresses it as a deterministic transformation of the parameters $(\mu, \sigma)$ and a random noise variable $\epsilon$ by a pairwise multiplication operator $\odot$.

1.  Sample a parameter-free noise $\mathbf{\epsilon} \sim \mathcal{N}(0, I)$.
2.  Apply a deterministic transformation to obtain $\mathbf{c}$:

In other words, the reparameterization process is essentially a **Monte Carlo** estimator that is differentiable with respect to the encoder parameters:

$$
\mathbf{c} = \mu + \sigma \odot \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, I)
$$

$\mu$ and $\sigma$ are now treated as deterministic constants/variables in the formula, while the randomness is entirely contained in $\mathbf{\epsilon}$.

#### Backpropagation

1. Calculate gradients of loss $\mathcal{L}$ with respect to value weights in Encoder ($\phi$) and Decoder ($\theta$).
    *   Gradient flows through $\hat{\mathbf{x}} \to \text{Decoder} \to \mathbf{c}$.
    *   Gradient flows through $\mathbf{c} \to \mathbf{\mu}, \mathbf{\sigma} \to \text{Encoder}$.
2.  Update: Adjust weights of the Encoder ($\phi$) and Decoder ($\theta$) using an optimizer (e.g., Adam, SGD).

## How VAE Assists Stable Disffusion (SD)

Abstract: instead of guiding SD to learn everything from an image by denosing, SD is set up to learn from the latent code $\mathbf{c}$.

Revisit the ELBO expression,

$$
\text{ELBO}=\underbrace{E_{\mathbf{c}\sim q_\phi(\mathbf{c} | \mathbf{x})} \bigg(\log p_{\theta}(\mathbf{x} | \mathbf{c})\bigg)}_{\text{Reconstruction Term}} - \underbrace{D_{KL}\bigg(q_\phi(\mathbf{c} | \mathbf{x}) \| p(\mathbf{c})\bigg)}_{\text{KL Divergence}}
$$

The reconstruction term $\log p_{\theta}(\mathbf{x} | \mathbf{c})$ in a standard VAE usually corresponds to a pixel-wise Mean Squared Error (MSE) loss, assuming the decoder output follows a Gaussian distribution around the true pixel values:

$$
\mathcal{L}_{MSE} = || \mathbf{x} - \hat{\mathbf{x}} ||^2
$$

where $\hat{\mathbf{x}}=\text{Decoder}(\mathbf{c})$.

MSE minimizes the average error, which often results in blurry predictions (high-frequency details are averaged out).

To address this issue, Stable Diffusion (SD) uses a **VQ-regularized** or **KL-regularized** autoencoder (specifically `AutoencoderKL`) that modifies the training objective to prioritize **Perceptual Compression**.

### SD Modified Loss Function

To ensure high-fidelity reconstruction, SD replaces the simple reconstruction term with a composite loss function involving a **Perceptual Loss** and a **PatchGAN Adversarial Loss**:

$$
\mathcal{L}_{Autoencoder} = \mathcal{L}_{rec}(\mathbf{x}, \text{Decoder}(\mathbf{c})) + \lambda_{KL} \mathcal{L}_{KL} + \lambda_{adv} \mathcal{L}_{GAN}
$$

Where:
*   $\mathcal{L}_{rec}$: Is perceptual loss (LPIPS), calculated in the feature space of a pre-trained VGG network, not just pixel space.
    $$ \mathcal{L}_{rec} \approx || \phi_{VGG}(\mathbf{x}) - \phi_{VGG}(\text{Decoder}(\mathbf{c})) ||_2 $$
*   $\mathcal{L}_{GAN}$: Is the discriminator loss from a PatchGAN, forcing the decoder to generate realistic high-frequency textures.
*   $\lambda_{KL}$: A small weight factor for the KL divergence to prevent the latent space from degrading too much while avoiding over-regularization (which causes blur).


### Two-Stage Training (Latent Diffusion)

Stable Diffusion (SD) separates the training into two distinct phases: one for VAE and one for diffusion model (U-Net/DM).

#### Stage A: Perceptual Compression (Training the VAE)

The VAE is trained first to find an optimal low-dimensional space $\mathcal{C}$.
*   Input: High-dimensional Pixel Space $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$.
*   Output: Low-dimensional Latent Space $\mathbf{c} \in \mathbb{R}^{h \times w \times 4}$.
*   Compression Factor: typically $f = H/h = 8$.

$$
\mathbf{c} = \text{Encoder}(\mathbf{x}), \quad \hat{\mathbf{x}} = \text{Decoder}(\mathbf{c}) \approx \mathbf{x}
$$

#### Stage B: Latent Diffusion (Training the DM)

Having encoder and decoder frozen, take a clean latent code $\mathbf{c}_0 = \text{Encoder}(\mathbf{x})$ and progressively add Gaussian noise to it over $T$ timesteps.
At any timestep $t$, the noisy latent $\mathbf{c}_t$ can be expressed as:

$$
\mathbf{c}_t = \sqrt{\bar{\alpha}_t}\mathbf{c}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

where $\bar{\alpha}_t$ is a noise schedule parameter. As $t \to T$, $\mathbf{c}_t$ becomes pure Gaussian noise.

In reverse/denoising, the training objective (Loss Function) is to minimize the difference between the **real noise** added and the **predicted noise**:

$$
\min \big|\big| \epsilon - \epsilon_\theta(\mathbf{c}_t, t, \tau_{cond}) \big|\big|^2_2 
$$

where

* $\mathbf{c}=\text{Encoder}(\mathbf{x})$: The frozen clean latent code as the ground truth to learn from.
* $\epsilon \sim \mathcal{N}(0,1)$: The actual noise sample added to $\mathbf{c}$.
* $t$: The diffusion timestep, uniformly sampled from $t \in \{1, \dots, T\}$.
*   $\mathbf{c}_t$: The noisy latent input. It is a linear combination of the clean latent code and noise: $\mathbf{c}_t = \sqrt{\bar{\alpha}_t}\mathbf{c} + \sqrt{1-\bar{\alpha}_t}\epsilon$.
*   $\tau_{cond}$: The conditioning vector (e.g., text embedding from CLIP) that guides the denoising process.
*   $\epsilon_\theta(\mathbf{v}_t, t, \tau_{cond})$: The **predicted noise**. The U-Net takes the noisy latent $\mathbf{c}_t$, the time $t$, and the prompt $\tau$ and outputs a tensor of the same shape as $\mathbf{c}$ representing the estimated noise.
