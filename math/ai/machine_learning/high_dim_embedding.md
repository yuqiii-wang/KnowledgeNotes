# High Dimension Embedding

Generally speaking, the embedding is vector representation of input aims to encode semantic info by training on an objective.

## Intrinsic Dimension

Minimum dimensions required for representing an entity.

## Curse of Dimensionality

When num of dimensions goes large, it is generally hard to train and vector representation is sparse.

### Random Unit Vector Inner Product Orthogonality

Let $\mathbf{x}, \mathbf{y}\in\mathbb{R}^n$ constrained on $||\mathbf{x}||=||\mathbf{y}||=1$ be random unit vectors, here to show

$$
\mathbf{x}\cdot\mathbf{y} \rightarrow 0, \quad\text{as } n\rightarrow\infty
$$

Assume each dimension of $\mathbf{x}, \mathbf{y}$ follows Gaussian distribution

$$
\begin{align*}
\mathbf{x}&=\frac{\mathbf{g}}{||\mathbf{g}||}, \text{ where } \mathbf{g}=[g_1, g_2, ..., g_n] \text{ with } g_i\sim\mathcal{N}(0,1) \\\\
\mathbf{y}&=\frac{\mathbf{h}}{||\mathbf{h}||}, \text{ where } \mathbf{h}=[h_1, h_2, ..., h_n] \text{ with } h_i\sim\mathcal{N}(0,1)
\end{align*}
$$

So that, the inner product can be written as

$$
\mathbf{x}\cdot\mathbf{y}=
\frac{\mathbf{g}\cdot\mathbf{h}}{||\mathbf{g}||\space||\mathbf{h}||}
$$

where the numerator and denominator follow

* $\mathbf{g}\cdot\mathbf{h}=\sum^n_{i=1}g_i h_i$. Since $g_i, h_i\sim\mathcal{N}(0,1)$, and the product of two standard normal distribution random variables also follows a standard normal distribution, i.e., $g_i h_i\sim\mathcal{N}(0,1)$, by the Central Limit Theorem (CLT) (Let $X_i$ be a random variable; as sample size $n$ gets larger, there is ${\sqrt{n}}({\overline{X}}_{n}-\mu) \rightarrow \mathcal{N}(0, \sigma^2)$), there is $\mathbf{g}\cdot\mathbf{h}\sim\mathcal{N}(0,n)$
* The Law of Large Numbers states that $||\mathbf{g}||$ and $||\mathbf{h}||$ approach their truth means as $n\rightarrow\infty$; the truth means are $||\mathbf{g}||\approx \sqrt{n}$ and $||\mathbf{h}||\approx \sqrt{n}$

As a result for a very large $n\rightarrow\infty$,
the inner product goes to $\rightarrow \mathcal{N}(0,0)$, therefore totally orthogonal.

$$
\mathbf{x}\cdot\mathbf{y}\approx\frac{\mathcal{N}(0,n)}{n}=
\mathcal{N}(0,\frac{1}{n})
$$

### High Dimensionality Geometric Explanation by HyperSphere

A hypersphere in $n$-dimensional space is defined by all points at a fixed radius $R$ from the origin $\mathbf{0}$.
Denote its volume as $V_n(R)$ and surface area as $S_n(R)$:

$$
\begin{align*}
    V_n(R) &=\frac{\pi^{n/2}}{\Gamma(\frac{n}{2}+1)}R^n \\\\
    S_n(R) &=\frac{d}{dR}V_n(R)=\frac{2\pi^{n/2}}{\Gamma(\frac{n}{2})}R^{n-1}
\end{align*}
$$

where $\Gamma(z)$ is gamma function such that

* $\Gamma(k)=(k-1)!$ for $k\in\mathbb{Z}^+$
* $\Gamma(1/2)=\sqrt{\pi}$

$\mathbf{x}, \mathbf{y}$ can be said be drawn from the surface of a unit hypersphere $S_{n}(1)$.
For $\mathbf{x}\cdot\mathbf{y}\sim\mathcal{N}(0,\frac{1}{n})$, as $n\rightarrow\infty$, the two vectors $\mathbf{x}, \mathbf{y}$ are likely irrelevant/orthogonal.

The surface area of a hypersphere scales as $S_{n}(1)\propto n^{n/2}$.
Consequently, the number of points needed to "cover" the surface grows exponentially.

#### Example of HyperSphere Surface Point Sampling and Density/Sparsity Intuition

Define two vector $\mathbf{x}, \mathbf{y}$ that are dense/relevant/close to each other if they happen to fall in the same $\pi/2$ segment of a hypersphere (for $\text{cos}(\mathbf{x}, \mathbf{y})\in[0, 1]$ it can be said that the two vectors are positively related).

* For $n=2$ (a circle), there are $4$ segments
* For $n=3$ (a sphere), there are $8$ segments
* For $n=4$ (a hyper-sphere), there are $16$ segments

For $n$ is very large such as $n=10000$, there are $2^{10000}$ vectors from which on average only one vector is considered close to a vector existed in an arbitrary $\pi/2$ hypersphere segment.
It is impractical to collect such a large size sample hence the sample feature space is sparse.

## t-SNE for Embedding Visualization

t-distributed stochastic neighbor embedding (t-SNE) is a statistical method for compressing high-dimensional data by normal distribution mapping.

In other words, for high-dimensional input $\mathbf{x}\in\mathbb{R}^{d}$ (usually $d \gg 3$), t-SNE aims to produce $\hat{\mathbf{y}}\in\{\mathbb{R}^{2}, \mathbb{R}^{3}\}$ so that the compressed $\hat{\mathbf{y}}$ can be visually presented.

### t-SNE Derivation

#### The P Matrix by Exponentiated Vector Distance

The similarity of datapoint $\mathbf{x}_{j}$ to datapoint $\mathbf{x}_{i}$ is the conditional probability $\mathbf{x}_{j|i}$ that
$\mathbf{x}_{i}$ would pick $\mathbf{x}_{j}$ as its neighbor if neighbors were picked in proportion to their probability density under a Gaussian centered at $\mathbf{x}_{i}$.

This can be expressed (for $i\ne j$)

$$
p_{j|i} = \frac{\exp\big(-\frac{1}{2\sigma^2}||\mathbf{x}_i-\mathbf{x}_j||^2\big)
    }{\sum_{k\ne i}\exp\big(-\frac{1}{2\sigma^2}||\mathbf{x}_i-\mathbf{x}_k||^2\big)}
$$

that the similarity probability $p_{j|i}$ is defined as vector distance mapped normal distribution ratio.

For singularity point $i=j$, define $p_{i|i}=0$.

Assume there are $N$ samples, and each sample sampling $p_i$ is treated indiscriminantly such that $1/N$.
Besides, $i$ and $j$ are interchangeable in sampling; as a result, the probability of selecting both $p_i$ and $p_j$ are

$$
p_{ij}=\frac{p_{i|j}+p_{j|i}}{2N}
$$

The full vector distance $P$ matrix is hence given as below.
For $p_{ij}=p_{ji}$, the $P$ is symmetric.

$$
P= \begin{bmatrix}
    0 & p_{12} & p_{13} & \\\\
    p_{21} & 0 & p_{23} & \\\\
    p_{31} & p_{32} & 0 & \\\\
     &  &  & \ddots \\\\
\end{bmatrix}
$$

##### Intrinsic Dimensionality Estimation for Normal Distribution Bandwdith

The scalar $\frac{1}{2\sigma^2}$ adjusts the gap $\mathbf{x}_i-\mathbf{x}_j$ that a large $\frac{1}{2\sigma^2}$ tunes the gap to small, and vice versa.

Ideally, $\frac{1}{2\sigma^2}$ should be set such that distinctly different groups of sample points remain distant, spatially close ones are grouped together.

Intrinsic dimensionality refers to the minimum number of latent variables needed to capture the essential structure of a dataset without significant information loss.

There are many methods for estimation, two popular ones are

* Principal Component Analysis (PCA):

Analyze the variance explained by each principal component. The number of components required to retain a high fraction (e.g., 95%) of total variance indicates the intrinsic dimension.

* Maximum Likelihood Estimation (MLE):

Uses nearest-neighbor distances to estimate dimension based on data density.

#### The Q Matrix to Approximate the P Matrix by Student's t-Distribution (1 degree of Freedom)

The $P$ Matrix is approximated by a Student's t-Distribution (1 degree of Freedom) $Q$.

The Student's t-Distribution is defined and set degree of freedom $v=1$.

$$
\begin{align*}
    & f_v(t) &&= \frac{\Gamma\big(\frac{v+1}{2}\big)}{\sqrt{\pi v}\space\Gamma\big(\frac{v}{2}\big)}\bigg(1+\frac{t^2}{v}\bigg)^{-(v+1)/2} \\\\
    \text{set } v=1 \Rightarrow\quad & &&= \frac{1}{\sqrt{\pi}\cdot\sqrt{\pi}}\bigg(1+t^2\bigg)^{-1} \\\\
    &&&= \frac{1}{\pi}\bigg(1+t^2\bigg)^{-1}
\end{align*}
$$

where Gamma function is $\Gamma(v)=\int_0^{\infty}t^{v-1}e^{-t} dt$.
In particular for $v\in\mathbb{Z}^{+}$, there is $\Gamma(v)=(v-1)!$.

The t statistic is $t=\frac{Z}{\sqrt{Z^2_1/1}}=\frac{Z}{|Z_1|}$.

The entry of $Q$ is

$$
q_{ij}=\frac{\big(1+||\mathbf{y}_i-\mathbf{y}_j||^2\big)^{-1}}{\sum_{k \ne l}\big(1+||\mathbf{y}_k-\mathbf{y}_l||^2\big)^{-1}}
$$

Recall that the t statistic in Gaussian distribution is defined as $t=\frac{\overline{X}-\mu}{s/\sqrt{n}}$, that in comparison to the $q_{ij}$, the $\overline{X}-\mu$ is analogously the gap $\mathbf{y}_i-\mathbf{y}_j$.

The $\sum_{k \ne l}(...)$ is the normalization term.

##### Benefits of Implementing Cauchy Distribution for $Q$

The "heavy tail" property of Cauchy distribution indicates that the gap $\mathbf{y}_i-\mathbf{y}_j$ is more tolerant against outlier sample points compared to a typical Gaussian distribution.

#### Training and Cost Function (Kullback-Leibler Divergence)

The optimal $\hat{\mathbf{y}}$ is trained with the objective to minimize a cost function: Kullback-Leibler divergence that the $P$ and $Q$ should be as similar as possible.

$$
\hat{\mathbf{y}}=\argmin_{\mathbf{y}} \text{KL}(P||Q) =
\sum_{i\ne j}\log p_{ij}\frac{p_{ij}}{q_{ij}}
$$

#### Perplexity Setup and Desired Probability Distribution Distance

Intuitively speaking, perplexity is the num of neighbors of $x_i$ to include in t-SNE computation for desired probability distribution distance.
The normal distribution bandwdith $\frac{1}{2\sigma^2}$ is dynamically computed to keep perplexity at constant per **manually** defined usually set between $5$ to $50$.

Perplexity in its nature is defined as exponentiation of summed Shannon entropy.

$$
\text{Perplexity}(p)=2^{H(p)}=
2^{-\sum_{x}p(x)\log_2 p(x)}
$$

For example, below results show that as the prediction uncertainty increases, perplexity value grows.

||Event Scenario|Perplexity|Perplexity Inverse|
|-|-|-|-|
|Scenario 1|$p_{x_1}=1.0$|$1.0=2^{-1.0\log_2 1.0}$|$1.0\approx 1/1.0$|
|Scenario 2|$p_{x_1}=0.1$, $p_{x_2}=0.9$|$1.38\approx 2^{-0.1\log_2 0.1 - 0.9\log_2 0.9}$|$0.72\approx 1/1.38$|
|Scenario 3|$p_{x_1}=p_{x_2}=0.5$|$1.617\approx 2^{-2\times 0.5\log_2 0.5}$|$0.618\approx 1/1.617$|
|Scenario 4|$p_{x_1}=p_{x_2}=p_{x_3}=0.333, \sum_{x_i\notin\{x_1, x_2, x_3\}}p_{x_i}=0.001$|$3.137\approx 2^{-3\times 0.333\log_2 0.333-0.001\times\log_2 0.001}$|$0.319\approx 1/3.137$|

In t-SNE, it is defined ${H(p_i)}=2^{-\sum_{j}p_{j|i}\log_2 p_{j|i}}$.
Perplexity can be interpreted as a smooth measure of the effective number of neighbors.
For example in the scenario 4, the insignificant neighbors of $x_i$ represented by $\sum_{x_j\notin\{x_1, x_2, x_3\}}p_{x_j}=0.001$ see $0\approx{-0.001\times\log_2 0.001}$, and $\{x_1, x_2, x_3\}$ are mostly related to the three nearest neighbors indicative by $3.137$.
