# Contrastive Learning

Contrastive learning is a technique where a model learns to distinguish between similar and dissimilar examples by comparing them in pairs or triplets.

Provided a batch of $N$ positive sentence/document pairs $\{(s_i, s_i^+)\}^{N}_{i=1}$, where for each anchor $s_i$, the positives are $s_i^+$ (the one another sentence/document most similar to the anchor),
and the negatives are all other $s_j^+$ in the batch for $j\ne i$,
the loss for a single pair $(s_i, s_i^+)$ is

$$
\mathcal{L}_i=-\log\frac{\exp\big(\frac{1}{\tau}\text{sim}(s_i, s_i^+)\big)}{\sum^N_{j=1}\exp\big(\frac{1}{\tau}\text{sim}(s_i, s_j^+)\big)}
$$

where $\tau$ is a temperature hyperparameter to scale similarity.

$\text{sim}(s_i, s_j)$ denotes cosine similarity between embeddings

$$
\text{sim}(s_i, s_j) =
\frac{\text{emb}(s_i) \cdot \text{emb}(s_j)}{|| \text{emb}(s_i) \cdot \text{emb}(s_j) ||}
$$

The total loss over the batch is:

$$
\mathcal{L}=\frac{1}{N}\sum^N_{i=1}\mathcal{L}_i
$$

## Obtain Sentence/Document Embedding from Token Embeddings

In LLM, embedding is usually applied per token.
To use ONE embedding to represent a whole sentence/document, need to aggregate/consolidate the all sentence/document tokens/embeddings.

Let $\text{emb}(t_i)\in\mathbb{R}^d$ be the embedding of token $\text{emb}(t_i)\in\text{emb}(\bold{t})\in\mathbb{R}^{n\times d}$ from an $n$ length sentence/document $\bold{t}\in\mathbb{R}^n$.

### Embedding Aggregation

One can simply compute max/mean of token embeddings to represent a sentence/document.
The pooling is per-DIMENSION/normalized by sequence length $n$, i.e., $\mathbb{R}^{n\times d}\rightarrow\mathbb{R}^{1\times d}$.

#### Max-Pooling Embedding Aggregation

$$
\text{emb}_{\max}(s)=\max\text{emb}(\bold{t})=
\begin{bmatrix}
    \max(t_{1,1}, t_{2,1}, ..., t_{n,1}) \\
    \max(t_{1,2}, t_{2,2}, ..., t_{n,2}) \\
    \vdots \\
    \max(t_{1,d}, t_{2,d}, ..., t_{n,d}) \\
\end{bmatrix}
$$

#### Mean-Pooling Embedding Aggregation

$$
\text{emb}_{\mu}(s)=\frac{1}{n}\sum_{i=1}^n\text{emb}(\bold{t})=
\frac{1}{n}\begin{bmatrix}
    t_{1,1} + t_{2,1} + ... + t_{n,1} \\
    t_{1,2} + t_{2,2} + ... + t_{n,2} \\
    \vdots \\
    t_{1,d} + t_{2,d} + ... + t_{n,d} \\
\end{bmatrix}
$$

### The `[CLS]` Token

In transformer-based LLMs such as BERT, the `[CLS]` token is explicitly added at the beginning of the input such that

$$
\text{[CLS]}, t_1, t_2, t_3, ...
$$

During pretraining, BERT includes the Next Sentence Prediction (NSP) and classification tasks that train the model to encode meaningful information in the `[CLS]` token.

Let $\bold{h}^{(l)}_i$ be the $i$-th hidden state at the $l$-th layer, together there is

$$
\bold{h}^{(l)}=[\bold{h}^{(l)}_{\text{[CLS]}}, \bold{h}^{(l)}_{1}, \bold{h}^{(l)}_{2}, ...,  \bold{h}^{(l)}_{n}]
$$

Take self-attention as an example, a naive transformer is

$$
\begin{align*}
    \bold{h}^{(l+1)}_{i} &=\text{FFN}\big(\text{SelfAttention}(\bold{h}^{(l)}, \bold{h}^{(l)}_i)\big) \\
    &=\text{FFN}\big(\text{Softmax}\big(\frac{\bold{q}_iK^{\top}}{\sqrt{d}}\big)\bold{v}_i\big)
\end{align*}
$$

Let $i=\text{\text{[CLS]}}$, the $l+1$_th layer hidden state for `[CLS]` is implicitly trained to summarize the entire $l$-th layer sequence.

## How All-MiniLM-L6-v2 is Trained

The all-MiniLM-L6-v2 model is a sentence embedding model built on top of a distilled transformer called MiniLM.

* Backbone: MiniLM-L6 (6-layer transformer with fewer parameters than BERT)
* Training Objective: Contrastive loss (Multiple Negatives Ranking Loss)
