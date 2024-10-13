# Attention Mechanism Further Explained

## Understand Attention

Given the attention formula, for $n$ tokens each of $d$ dimensions, there are $Q \in \mathbb{R}^{n \times d}$ and $K^{\top} \in \mathbb{R}^{d \times n}$, and finally $V \in \mathbb{R}^{n \times d}$, (below see $d$ and $d_k$ interchangeably).

$$
\text{Attention}(Q,K,V) = \text{softmax} \big(\frac{Q K^{\top}}{\sqrt{d_k}} \big) V
$$

$Q K^{\top} \in \mathbb{R}^{n \times n}$ gives an "attention score matrix" what tokens be correlated to what next/previous tokens by what scores.
The resulted size $n \times n$ captures the context tokens.

The division by $\sqrt{d_k}$ is to scale down/flat the results of $Q K^{\top}$, so that by $\text{softmax}$ non-linearity, large values of $Q K^{\top}$ will be reduced, and the diffs between small values of $Q K^{\top}$ are amplified.
The flattened values help back-propagation leading to more stable training compared to non-$\sqrt{d_k}$ division.

Define softmax: for $i=1,2,...,n$ and $\bold{z}=(z_1, z_2, ..., z_n)\in \mathbb{R}^n$,

$$
\sigma(\bold{z})_i=
\frac{e^{z_i}}{\sum^n_{j=1}e^{z_j}}
$$

For $e^{z_i}$ grows exponentially, large inputs $z_i$ has significant influence over the activation energy.
This justifies the normalization by division of $\sqrt{d_k}$.

Given $Q K^{\top} \in \mathbb{R}^{n \times n}$, the multiplication by $V \in \mathbb{R}^{n \times d}$ yields the attention of $\mathbb{R}^{n \times d}$ that converts the $n \times n$ token context space back to $n \times d$ feature dimension space for the $n$ tokens.

```py
import numpy as np

Q = X @ W_Q  # Query
K = X @ W_K  # Key
V = X @ W_V  # Value

# Calculate attention scores
scores = Q @ K.T / np.sqrt(d_k)  # Scale the dot product
attention_weights = softmax(scores)  # Apply softmax to get attention weights
output = attention_weights @ V  # Weighted sum of values
```

### Attention Score by $Q K^{\top}$

For example, given $Q \in \mathbb{R}^{5 \times 6}$ and $K^{\top} \in \mathbb{R}^{6 \times 5}$,

```py
import numpy as np

Q = np.array(
[[0,0.9,0.1,0,0,0],
 [0,0,1,0,0,0],
 [0,0,1,0,0,0],
 [0,0,1,0,0,0],
 [0,0,1,0,0,0]])

K = np.array(
[[0,1,0,0,0,0],
 [0,1,0,0,0,0],
 [0,0.1,0.9,0,0,0],
 [0,1,0,0,0,0],
 [0,1,0,0,0,0]])

S = Q @ K.T

print(S)
```

that prints

```txt
[[0.9  0.9  0.18 0.9  0.9 ]
 [0.   0.   0.9  0.   0.  ]
 [0.   0.   0.9  0.   0.  ]
 [0.   0.   0.9  0.   0.  ]
 [0.   0.   0.9  0.   0.  ]]
```

where the 1st row and 3rd col see large vals.
The result derives from $0.9$ in the (1st row, 2nd col) entry in $Q$ correlating to 2nd col in $K$, another $0.9$ at the 3rd col in $Q$ correlating to the (3rd row, 3rd col) in $K$.

This shows that the first token in query $Q$ is correlated to all tokens in key $K$ (except for the 3rd token);
and the 3rd token in key $K$ is correlated to all tokens in query $Q$ (also except for the 3rd token).

### Why Divided by $\sqrt{d_k}$

One sentence explained: for normal distribution random vectors, the dot product tends to have a mean of $d$ and a variance that also scales with $d$.

Let $\bold{q}$ and $\bold{k}$ be a row of $Q$ and $K$.
Assume features present as standard normal distribution, so that each entry of $\bold{q}$ and $\bold{k}$ is of $q_i, k_i \sim N(0,1)$.

Each entry/score of $Q K^{\top}$ is $s=\bold{q} \bold{k} = \sum_{i=1}^d q_i k_i$.

Remember, here to prove $s \in Q K^{\top}$ that there is $s \sim N(0, d)$, not individual for $q_i k_i$.
$s=\bold{q} \bold{k}$ is an entry result of matrix multiplication, not to get confused with normalization by $\frac{1}{d}$, as $s \sim N(0, d)$ is not concerned of $q_i k_i$.

#### Expected Value (Mean)

For $q_i, k_i \sim N(0,1)$, and $q_i$ and $k_i$ are independent to each other, there is

$$
E[q_i k_i] = E[q_i] E[k_i] = 0
$$

So that,

$$
E[s] = E\Bigg[ \sum_{i=1}^d q_i k_i \Bigg] =
 \sum_{i=1}^d E[q_i k_i] = 0
$$

#### The Variance

By definition, there is

$$
\begin{align*}
\text{Var}(s) = \sum_{i=1}^d \big( q_i k_i - \underbrace{E[q_i k_i]}_{=0} \big)^2
= E[s^2]
\end{align*}
$$

Hence, only $E[s^2]$ needs to be computed.

$$
\begin{align*}
&&& \quad s^2 = \Bigg( \sum_{i=1}^d q_i k_i \Bigg)^2
    = \sum_{i=1}^d (q_i k_i)^2 + \sum_{i \ne j} q_i k_i q_j k_j \\
\Rightarrow &&& E(s^2) 
    = \sum_{i=1}^d E\Big[ (q_i k_i)^2 \Big] + \sum_{i \ne j} E\Big[q_i k_i q_j k_j \Big]
\end{align*}
$$

where, for $q_i, k_i \sim N(0,1)$, and $q_i$ and $k_i$ are assumed independent to each other, there are

$$
\begin{align*}
E\Big[ (q_i k_i)^2 \Big] &= E[q_i^2] E[k_i^2] = 1 \cdot 1 = 1 \\
E\Big[q_i k_i q_j k_j \Big] &= E[q_i] E[k_i] E[q_j] E[k_j] = 0
\end{align*}
$$

So that

$$
\text{Var}(s) = E[s^2] =
\sum_{i=1}^d 1 = d
$$

The standard deviation $\sqrt{d}$ describes the mean variance that $\frac{1}{\sqrt{d}}$ makes every score back to standard normal distribution $s \sim N(0, 1)$.

#### Code Example

```py
import numpy as np

rand_size = 10000

mu = 0
sigma = 1
rands_1 = np.random.normal(mu, sigma, rand_size)
rands_2 = np.random.normal(mu, sigma, rand_size)
s = rands_1 * rands_2

s_sum =  np.sum(s)
print(f"sum: {s_sum}")
s_mean =  np.mean(s)
print(f"mean: {s_mean}")
var_s2 = np.sum([x**2 for x in s]) - np.mean(s)**2
print(f"var: {var_s2}")
```

that shows

```txt
sum: 29.899064924872384
mean: 0.0029899064924872386
var: 10317.40248994817
```

### Multi-Head Attention

Define an attention head $\text{head}_i=\text{Attention}(Q_i,K_i,V_i)$, for multi-head, there is

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, ..., \text{head}_n) W^O
$$

where $W^O$ is a learned weight matrix.

#### Why Multi-Head

Basically, the multi-head transform is a linear transform by $W^O$.

* Project heads into multiple lower-dimensional spaces
* "Summarize" different head attentions, respectively

#### Design of $W^O$

The linear projection by $W^O$ should yield a result matrix same size of input, i.e., `[batch_size, seq_length, d_model]`.

For example for BERT-base with $d_{\text{model}}=768$ and $n_{\text{head}}=12$ heads, there is $d_k=d_v=768/12=64$.
Each attention head gives `[batch_size, seq_length, 64]`.

To retain the shape to `[batch_size, seq_length, d_model]`, there is $W^O \in \mathbb{R}^{768 \times 768}$ derived from `[n_head * d_v, d_model]`

### Self-Attention vs Cross-Attention

Mathematically speaking, cross-attention has two inputs: $X$ for query and $Y$ for key and value (query results), while self-attention uses the same sequence input $X$.

#### Self-attention

$$
Q = X W_Q,
\qquad
K = X W_K,
\qquad
V = X W_V,
\qquad
$$

* Self-attention (as encoder) focuses on contextualized representations of words by considering **all** parts of the source input.
* In decoder, produce a target sequence by ensuring that each word in the target sequence can attend to all previously generated words.

#### Cross-attention

$$
Q = X W_Q,
\qquad
K = Y W_K,
\qquad
V = Y W_V,
\qquad
$$

* Cross-attention ensures that the decoder focuses on the **most relevant** parts of the source input when generating each word in the target sequence.
* For example, in English-to-Japanese translation, the verb usually comes at the end of the sentence in Japanese, while in English it often comes earlier. Cross-attention helps the model retain the correct verb and align it appropriately across the language pairs.

## Encoder and Decoder

