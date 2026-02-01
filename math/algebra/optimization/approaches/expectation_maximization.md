# Expectation Maximization 

Expectation–maximization (EM) algorithm is an iterative method to find (local) maximum likelihood or maximum a posteriori (MAP) estimates of parameters in statistical models.

## Definition

Define $\mathbf{X}$ as observation data, $\mathbf{Z}$ as latent/unobserved data, and a vector of unknown vector $\mathbf{\theta}$ with a likelihood function $L(\mathbf{\theta}; \mathbf{X}, \mathbf{Z})=p(\mathbf{X}, \mathbf{Z} | \mathbf{\theta})$.

Semantically speaking, $\mathbf{Z}$ usually refers to as labels/classes of observed data $\mathbf{X}.$

The maximum likelihood estimate (MLE) of the unknown parameters $\mathbf{\theta}$ is determined by maximizing the marginal likelihood of the observed data $\mathbf{X}$.
$$
\begin{align*}
L(\mathbf{\theta}; \mathbf{X})&=
p(\mathbf{X} | \mathbf{\theta})
\\\\ &=
\int p(\mathbf{X}, \mathbf{Z} | \mathbf{\theta}) d \mathbf{Z}
\\\\ &=
\int p(\mathbf{X} | \mathbf{Z}, \mathbf{\theta}) p(\mathbf{Z} | \mathbf{\theta}) d \mathbf{Z}
\end{align*}
$$

There are two alternative steps updating $\mathbf{\theta}$:

1. Expectation step (E step):

Define $Q(\theta | \theta^{(t)})$ as the expected value of the log likelihood function, in which $\theta^{(t)}$ is the current estimate of $\theta$:
$$
Q(\theta | \theta^{(t)})=
E_{\mathbf{Z}|\mathbf{X}, \mathbf{\theta}^{(t)}}[log \space L(\mathbf{\theta}; \mathbf{X}, \mathbf{Z})]
$$

2. Maximization step (M step):

Find the parameters that maximize this quantity:
$$
\mathbf{\theta}^{(t+1)}=
arg \space \underset{\mathbf{\theta}}{max} \space Q(\theta | \theta^{(t)})
$$

## Application in Gaussian Mixture Models

Expected to use two Gaussian distributions $N_1(\mu_1, \sigma_1^2)$ and $N_2(\mu_2, \sigma_2^2)$ to represent a set of $n$ data points $\mathbf{x}=[x_1, x_2, ..., x_n]$.

Define $\mathbf{\pi}=[\pi_1, \pi_2]$ which is the mixing probability for the two Gaussian distributions, subject to $1 = \sum_i \pi_i$.

The model parameters $\mathbf{\theta}$ are
$$
\mathbf{\theta}=(\mathbf{\pi}, \mu_1, \sigma_1^2, \mu_2, \sigma_2^2)
$$

the probability density function (PDF) $p$ of the mixture model is
$$
p(\mathbf{x} | \mathbf{\theta})=
\pi_1 \cdot g_1(\mathbf{x} | \mu_1, \sigma_1^2)
+
\pi_2 \cdot g_2(\mathbf{x} | \mu_2, \sigma_2^2)
$$
where $g_1, g_2$ are PDFs for the two aforementioned Gaussian distributions $N_1$ and $N_2$.

The probability (likelihood) of observing our entire dataset of $n$ points is:
$$
L(\mathbf{\theta};\mathbf{x})=
\prod_{i=1}^n p(x\_i; \mathbf{\theta})
$$

The log representation is

$$
\begin{align*}
log \space L(\mathbf{\theta};\mathbf{x})&=
\sum_{i=1}^n log \space p(x\_i; \mathbf{\theta})
\\\\ &=
\sum_{i=1}^n log \space 
[
    \pi_1 \cdot g_1(\mathbf{x} | \mu_1, \sigma_1^2)
    +
    \pi_2 \cdot g_2(\mathbf{x} | \mu_2, \sigma_2^2)
]
\end{align*}
$$

Define $\mathbf{\gamma} = [\gamma_{1,i}, \gamma_{2,i}]$ as the $x\_i$ 's label probability, and $x\_i$ 's label can be determined by $max(\gamma_{1,i}, \gamma_{2,i})$.

Alternatively run Expectation and Maximization:

* Expectation:

$$
\gamma_{1,i} = 
\frac
{\pi_1 \cdot g_1(x\_i | \mu_1, \sigma_1^2)}
{\pi_1 \cdot g_1(x\_i | \mu_1, \sigma_1^2)
+
\pi_2 \cdot g_2(x\_i | \mu_2, \sigma_2^2)}\\\\
\space \\\\

\gamma_{2,i} = 
\frac
{\pi_2 \cdot g_2(x\_i | \mu_2, \sigma_2^2)}
{\pi_1 \cdot g_1(x\_i | \mu_1, \sigma_1^2)
+
\pi_2 \cdot g_2(x\_i | \mu_2, \sigma_2^2)}
$$

subject to
$$
1 = \sum_k \pi_k
$$

* Maximization

$$
\mu_1=\frac{\sum_i \gamma_{1,i} x\_i}{\sum_i \gamma_{1,i}}
\quad
\mu_2=\frac{\sum_i \gamma_{2,i} x\_i}{\sum_i \gamma_{2,i}}\\\\
\space \\\\

\sigma_1 = \frac{\sum_i \gamma_{1,i} (x\_i-\mu_1)^2}{\sum_i \gamma_{1,i} }
\quad
\sigma_2 = \frac{\sum_i \gamma_{2,i} (x\_i-\mu_2)^2}{\sum_i \gamma_{2,i} }\\\\
\quad \\\\

\pi_1 = \frac{1}{n} \sum_i \gamma_{1,i}
\quad
\pi_2 = \frac{1}{n} \sum_i \gamma_{2,i}
$$

Convergence occurs when the change of $\mathbf{\gamma}$ is small such as $\Delta \mathbf{\gamma} < \mathbf{\epsilon}$.