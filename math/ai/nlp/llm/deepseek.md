# DeepSeek

Generally speaking, most tech and algo innovations are in DeepSeek V2; DeepSeek V3 scales out to host much larger parameter LLMs; DeepSeek R1 added reasoning capability.

References:

* DeepSeek V2: https://arxiv.org/pdf/2405.04434
* DeepSeek V3: https://arxiv.org/pdf/2412.19437
* DeepSeek R1: https://arxiv.org/pdf/2501.12948

## DeepSeek V2

DeepSeek V2 major innovation contributions:

* Multi-Head Latent Attention (MLA)
* DeepSeekMoE Architecture

### Multi-Head Latent Attention (MLA)

Multi-Head Latent Attention (MLA) proposes low-rank key-value joint compression to reduce KV Cache memory.
DeepSeekV2 and DeepSeekV3 uses MLA.

Notice $K$ and $V$ memory consumption grows as context length grows.
The MLA attempts to compress the cached $K$ and $V$ with a low-rank joint compression matrix $C$.

#### Derive the compression matrix $C$

Let $\bold{h}_t\in\mathbb{R}^{d}$ be the input to an attention layer, where $d=n_h\times d_h$ is the embedding dimension in which $n_h$ is the number of attention heads, and $d_h$ is the dimension per head.

##### Preliminaries: Standard Multi-Head Attention

For standard multi-head attention, $\bold{q}_t, \bold{k}, \bold{v}_t$ are computed by linear projection from $\bold{h}_t$, and sliced into $n_h$ heads/blocks.

$$
\begin{align*}
    [\bold{q}_{t,1};\bold{q}_{t,2};...;\bold{q}_{t,n_h}]=\bold{q}_t=W^{Q}\bold{h}_t \\
    [\bold{k}_{t,1};\bold{k}_{t,2};...;\bold{k}_{t,n_h}]=\bold{k}_t=W^{K}\bold{h}_t \\
    [\bold{v}_{t,1};\bold{v}_{t,2};...;\bold{v}_{t,n_h}]=\bold{v}_t=W^{V}\bold{h}_t \\
\end{align*}
$$

The sliced $\bold{q}_t, \bold{k}, \bold{v}_t$ are used for the multi-head attention computation.

$$
\begin{align*}
    \bold{o}_{t,i} &= \sum_{j=1}^{t} \text{softmax}_j\Big(\frac{\bold{q}^{\top}_{t,i}\bold{k}_{j,i}}{\sqrt{d_h}}\Big)\bold{v}_{j,i} \\
    \bold{o}_{t} &= W^{O}[\bold{o}_{t,1};\bold{o}_{t,2};...;\bold{o}_{t,n_h}]
\end{align*}
$$

where $[...]$ is a concatenation operator.

##### Add Compression Cache Matrices

Add a down-projection matrix $W^{\text{Down-}KV}$ to generate the KV cache $\bold{c}_t^{KV}$, by which add two up-projection matrices to restore $K$ by $W^{\text{up-}K}$ and $V$ by $W^{\text{up-}V}$ to full multi-head dimension: $\bold{k}_t^{C},\bold{v}_t^{C}\in\mathbb{R}^{n_h d_h}$.

During inference, MLA only needs to cache $\bold{c}_t^{KV}$.

$$
\begin{align*}
    \bold{c}_t^{KV} &= W^{\text{Down-}KV}\bold{h}_t \\
    \bold{k}_t^{C} &= W^{\text{Up-}K}\bold{c}_t^{KV} \\
    \bold{v}_t^{C} &= W^{\text{Up-}V}\bold{c}_t^{KV} \\
\end{align*}
$$

where $\bold{c}_t^{KV}\in\mathbb{R}^{d_c}$ is the compressed latent vector for keys and values such that $d_c\ll d_h n_h$.
This shows that the token cache $\bold{c}_t^{KV}$ compresses the token's multi-head vectors into a small encoding.

$W^{\text{Up-}K}, W^{\text{Up-}V} \in\mathbb{R}^{d_h n_h \times d_c}$ restore the key $K$ and value $V$ to full dimension $d=d_h \times n_h$.

Also perform low-rank compression for the queries (this is for training):

$$
\begin{align*}
    \bold{c}_t^{Q} &= W^{\text{Down-}Q}\bold{h}_t \\
    \bold{q}_t^{C} &= W^{\text{Up-}Q}\bold{c}_t^{Q} \\
\end{align*}
$$

#### Decoupled Rotary Position Embedding (RoPE) for KV Restoration

Empirical study by DeepSeek found high importance of positional info, considered $\text{RoPE}$ be introduced.

RoPE is position-sensitive for both keys and queries, that only $Q$ and $K$ are applied RoPE.

$$
\begin{align*}
    [\bold{q}_{t,1}^{\text{Ro}};\bold{q}_{t,2}^{\text{Ro}};...;\bold{q}_{t,n_h}^{\text{Ro}}]=\bold{q}_{t}^{\text{Ro}}=\text{RoPE}(W^{\text{Ro-}Q}\bold{c}_t^Q) \\
    \bold{k}_{t}^{\text{Ro}}=\text{RoPE}(W^{\text{Ro-}K}\bold{h}_t) \\
\end{align*}
$$

Accordingly, the $Q$ and $K$ are

$$
\begin{align*}
    \bold{q}_{t,i}=[\bold{q}_{t,i}^{\text{C}};\bold{q}_{t,i}^{\text{Ro}}] \\
    \bold{k}_{t,i}=[\bold{k}_{t,i}^{\text{C}};\bold{k}_{t}^{\text{Ro}}] \\
\end{align*}
$$

Notice here $\bold{k}_{t,i}=[\bold{k}_{t,i}^{\text{C}};\bold{k}_{t}^{\text{Ro}}]$ for each token key head $\bold{k}_{t,i}$ share the same key $\bold{k}_{t}^{\text{Ro}}$.

##### Motivation: The non-commutative RoPE

Recall the $QK^{\top}$ definition that for the attention score of the token $t$, it can be decomposed into $\Big(W^{Q}\bold{h}_t\Big)\Big(W^{K}\bold{h}_t\Big)^{\top}$.

Then introduce compression, there is $\Big(W^{Q}\bold{h}_t\Big)\Big(W^{\text{Up-}KV}W^{\text{Down-}KV}\bold{h}_t\Big)^{\top}$.
Recall that $\bold{c}_t^{KV}=W^{\text{Down-}KV}\bold{h}_t\in\mathbb{R}^{d_c}$ is quite small in dimension length compared to the full dimension multiplication $W^{\text{Up-}KV}W^{\text{Down-}KV}\bold{h}_t\in\mathbb{R}^{d}$, it can be arranged that $W^{Q}{(W^{\text{Up-}KV})}^{\top}\bold{h}_t$ be absorbed together in matrix multiplication to reduce memory footprint.

$$
\underbrace{\Big(W^{Q}\bold{h}_t\Big)}_{\bold{q}_t\in\mathbb{R}^{d}}\Big(W^{\text{Up-}KV}W^{\text{Down-}KV}\bold{h}_t\Big)^{\top}
\quad\Rightarrow\quad \underbrace{\Big(W^{Q}{(W^{\text{Up-}KV})}^{\top}\bold{h}_t\Big)}_{\bold{q}_t\in\mathbb{R}^{d_c}} \Big(W^{\text{Down-}KV}\bold{h}_t\Big)^{\top}
$$

However, if added RoPE, the above linear matrix multiplication does not hold for matrix multiplication does not follow commutative rules.

Introduce RoPE to keys: $\Big(W^{Q}\bold{h}_t\Big)\Big(\text{RoPE}\big(W^{\text{Up-}KV}W^{\text{Down-}KV}\bold{h}_t\big)\Big)^{\top}$.

But RoPE cannot commute with $W^{\text{Up-}KV}$:

$$
\Big(W^{Q}\bold{h}_t\Big)\Big(\text{RoPE}\big(W^{\text{Up-}KV}...\big)\Big)^{\top}
\quad\not\Rightarrow\quad \Big(W^{Q}\big(\text{RoPE} \cdot W^{\text{Up-}KV}\big)^{\top}\bold{h}_t\Big)\Big(...\Big)^{\top}
$$

##### Solution: Decoupled RoPE to query and key

The solution is to decouple RoPE by adding additional multi-head queries $\bold{q}_{t,i}^{\text{Ro}}\in\mathbb{R}^{d^{\text{Ro}}_h}$ and a shared key $\bold{k}_{t}^{\text{Ro}}\in\mathbb{R}^{d^{\text{Ro}}_h}$ to carry RoPE.

Introduce $W^{\text{Ro-}Q}\in\mathbb{R}^{d^{\text{Ro}}_hn_h\times d_c^Q}$ and $W^{\text{Ro-}K}\in\mathbb{R}^{d^{\text{Ro}}_h\times d}$

$$
\begin{align*}
    [\bold{q}_{t,1}^{\text{Ro}};\bold{q}_{t,2}^{\text{Ro}};...;\bold{q}_{t,n_h}^{\text{Ro}}]=\bold{q}_{t}^{\text{Ro}}=\text{RoPE}(W^{\text{Ro-}Q}\bold{c}_t^Q) \\
    \bold{k}_{t}^{\text{Ro}}=\text{RoPE}(W^{\text{Ro-}K}\bold{h}_t) \\
\end{align*}
$$

Accordingly, the $Q$ and $K$ are

$$
\begin{align*}
    \bold{q}_{t,i}=[\bold{q}_{t,i}^{\text{C}};\bold{q}_{t,i}^{\text{Ro}}] \\
    \bold{k}_{t,i}=[\bold{k}_{t,i}^{\text{C}};\bold{k}_{t}^{\text{Ro}}] \\
\end{align*}
$$

Let $l$ be history output token number, MLA requires a total KV cache containing $(d_c+d^{\text{Ro}}_h)l$ elements.

#### Final: Combine the Cache and RoPE

For each token, the attention is

$$
\begin{align*}
    \bold{q}_{t,i} &=[\bold{q}_{t,i}^{\text{C}};\bold{q}_{t,i}^{\text{Ro}}] \\
    \bold{k}_{t,i} &=[\bold{k}_{t,i}^{\text{C}};\bold{k}_{t}^{\text{Ro}}] \\
    \bold{o}_{t,i} &= \sum_{j=1}^{t} \text{softmax}_j\Big(\frac{\bold{q}^{\top}_{t,i}\bold{k}_{j,i}}{\sqrt{d_h+d^{\text{Ro}}_h}}\Big)\bold{v}_{j,i}^C \\
    \bold{o}_{t} &= W^{O}[\bold{o}_{t,1};\bold{o}_{t,2};...;\bold{o}_{t,n_h}]
\end{align*}
$$

DeepSeek sets

* number of attention heads $n_h=128$
* per-head dimension $d_h=128$
* KV compression dimension $\bold{c}_t^{KV}\in\mathbb{R}^{512}$, or $d_c=4d_h$
* query compression dimension $\bold{c}_t^{Q}\in\mathbb{R}^{1536}$
* decoupled query and key  per-head dimension $\bold{q}_{t,i}^{\text{Ro}},\bold{k}_{t,i}^{\text{Ro}}\in\mathbb{R}^{64}$, or $d^{\text{Ro}}_h=\frac{1}{2}d_h$

### DeepSeekMoE Architecture

In comparison to traditional MoEs,
DeepSeek employs two types of experts: shared $\text{FNN}_i^{(s)}$ and routed experts $\text{FNN}_i^{(r)}$,
where the routed experts are chosen by $g_{i,t}$ that is decided by top-K most-similar input $\bold{u}_t$ to the expert representative vector/centroid $\bold{e}_i$.

Let $\bold{u}_t$ be the $L$-th layer input, the next layer output $\bold{h}_t^{(L+1)}$ is computed by below:
residual + $N_s$ shared experts and $N_r$ routed experts.

$$
\begin{align*}
    \bold{h}_t^{(L+1)} &= \bold{u}_t+\sum^{N_s}_{i=1} \text{FNN}_i^{(s)}(\bold{u}_t)+\sum^{N_r}_ig_{i,t} \text{FNN}_i^{(r)}(\bold{u}_t) \\
    g_{i,t} &= \begin{cases}
        s_{i,t} & s_{i,t} \in \text{TopK}\big(\{s_{j,t} | 1 \le j \le N_r\}, K_r \big) \\
        0 & \text{otherwise}
    \end{cases} \\
    s_{i,t} &= \text{Softmax}_i(\bold{u}_t^{\top} \bold{e}_i)
\end{align*}
$$

where $K_r$ is the number of activated routed experts that $s_{i,t}$ could retain non-zero values only if they are top $K_r$ by $\text{Softmax}_i(\bold{u}_t^{\top} \bold{e}_i)$.
$\bold{e}_i$ is the learned centroid of the $i$-th routed expert in this $L$ layer.

DeepSeek V2 sets

* $\text{FFN}_i$ has a hidden dimension of $1536$.
* each MoE layer consisted of $2$ shared experts and $160$ routed experts

#### Device-Limited Routing

DeepSeekMoE ensures that the target experts of each token will be distributed on at most $M$ devices,
and the top-K selections happen only among experts on these $M$ devices.

Empirical study by DeepSeek said $M\ge 3$ is enough to give good results.
There are $D=8$ devices/groups $\{\mathcal{E}_1, \mathcal{E}_2, ..., \mathcal{E}_D\}$ for each layer, and the routed experts are uniformly deployed.

#### Auxiliary Loss for Expert/Device Load Balance

DeepSeek V2 proposes three kinds of auxiliary loss to learn routing strategies (how expert $\bold{e}_i$ be learned):

$$
\min_{\bold{e}_i} \big(
\mathcal{L}_{\text{expert-balance}}+\mathcal{L}_{\text{device-balance}}+\mathcal{L}_{\text{communication-balance}} \big)
$$

Let $\alpha_1, \alpha_2, \alpha_3$ be loss coefficients, $T$ be the number of tokens in a sequence.
Deepseek V2 sets $\alpha_1=0.003, \alpha_2=0.05, \alpha_3=0.02$.

##### Motivation: Load Imbalance

In traditional MoE models, homogeneous expert sizes can lead to skewed routing, where popular experts are overloaded.

A general remediation solution is penalizing experts overly received the routed tokens.
Recall Cauchy-Schwarz inequality that $\mathcal{L}_{\text{balance}}$ reaches it minimum when the two random variables $f_i$ and $p_i$ are equally distributed, indicating that each expert $\bold{e}_i$ has the same probability receiving the same number of tokens.

$$
\mathcal{L}_{\text{balance}}=\alpha\sum_{i=1}^{N_r} \big(f_i \cdot p_i \big)
$$

where

* $f_i$ Fraction of tokens routed to expert $\bold{e}_i$
* $p_i$ Average router probability for expert $\bold{e}_i$
* $\alpha$ Hyperparameter controlling the penalty strength

DeepSeek V2 applies/extends this concept to expert, device and communication levels.

##### Expert-Level Balance Loss

$\mathcal{L}_{\text{expert-balance}}$ ensures that each expert $\bold{e}_i$ receives the same amount of tokens.

$$
\mathcal{L}_{\text{expert-balance}}=
\alpha_1\sum_{i=1}^{N_r}\Big(
    \underbrace{\frac{N_r}{K_r T} \sum_{t=1}^{T}\mathcal{1}(t\text{ if selected expert } \bold{e}_i)}_{f_i}\Big) \cdot \Big(
    \underbrace{\frac{1}{T}\sum_{t=1}^{T}s_{i,t}}_{p_i} \Big)
$$

where $\mathcal{1}(\text{condition})=\begin{cases} 1 & \text{condition is true} \\ 0 & \text{condition is false} \end{cases}$ denotes the indicator function.

##### Device-Level Balance Loss

$\mathcal{L}_{\text{device-balance}}$ ensures balanced computation across different devices.

To reach the goal, DeepSeek V2 partitions all routed experts into $D$ groups $\{\mathcal{E}_1, \mathcal{E}_2, ..., \mathcal{E}_D\}$, and each group is assigned a computation device.
$\frac{1}{|\mathcal{E}_i|}\sum_{j\in\mathcal{E}_i} (.)$ is the average group/device load.

$$
\mathcal{L}_{\text{device-balance}}=
\alpha_2\sum_{i=1}^{D}\Big(\frac{1}{|\mathcal{E}_i|}\sum_{j\in\mathcal{E}_i}f_i \Big)
\cdot \Big( \frac{1}{|\mathcal{E}_i|}\sum_{j\in\mathcal{E}_i} p_i \Big)
$$

##### Communication Balance Loss

$\mathcal{L}_{\text{communication-balance}}$ added $\mathcal{1}(t\text{ if selected device }\mathcal{E}_i)$ that encourages equally-spread computation across different devices $\mathcal{E}_i$.

$$
\mathcal{L}_{\text{communication-balance}}=
\alpha_3\sum_{i=1}^{D}\Big(
    \frac{D}{M T} \sum_{t=1}^{T}\mathcal{1}(t\text{ if selected device }\mathcal{E}_i)\Big) \cdot \Big(
    \sum_{j\in\mathcal{E}_i} p_i \Big)
$$

### Training Aggrangment

#### Data Preparation and Tokenization

DeepSeek V2 uses Byte-level Byte-Pair Encoding (BBPE) algorithm and has a vocabulary size of 100K.
The tokenized pretraining corpus contains 8.1T tokens, where Chinese tokens are approximately $12\%$ more than English ones.

#### Model Hyper-Parameters

* Number of Transformer Layers: $60$
* Transformer FFN Dimension: $5120$

#### Model Training Config

* AdamW optimizer: $\beta_1=0.9$ and $\beta_2=0.95$
* learning rate scheduling: warmup-and-step-decay strategy
* Batch size: adaptively set between $2304$ and $9216$
* Maximum sequence length: 4K

##### Warmup-and-Step-Decay strategy in Detail

1. Learning rate from $0$ to the maximum value $=2.4\times 10^{-4}$ during the first 2K steps
2. Keep the maximum learning rate till having trained $60\%$ of tokens
3. The learning rate is multiplied by $0.316$ after training about $60\%$ of tokens
4. The learning rate is again multiplied by $0.316$ after training about $90\%$ of tokens
