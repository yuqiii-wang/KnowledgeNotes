# LLM Inference Practices

## Structure

* Encoder-Only

BERT (Bidirectional Encoder Representations from Transformers)

* Prefix-Decoder (Non-Causal Decoder)

T5 (Text-To-Text Transfer Transformer)

* Causal Decoder

GPT (Generative Pre-trained Transformer)

* Encoder-Decoder

## LLM Memory Consumption During Inference

There are

* Model Parameters
* Key Value Caches
* Temporary Computation Results

### Model Parameters

Take BERT-base as an example.
In total, there are $108,369,656$ parameters.
By FP16, the model memory consumption is $108,369,656 \times 2\text{bytes} = 216.7 \text{MB}$.

#### Embedding Layers

* Token Embeddings $30,000 \times 768 = 23,040,000$
* Position Embeddings: $512 \times 768 = 393,216$ for a maximum sequence length (commonly 512)
* Token Type Embeddings: $2 \times 768 = 1,536$ for 2 token types (for sentence A and sentence B)

#### Transformer Layer Components

For each of $12$ transformer layers, there are

* Query, Key, and Value weights: $3 \times 768\times 768 = 1,769,472$
* Attention Output Linear Projection: $768\times 768=589,824$
* Feed-Forward: $768\times 3072 + 3072 \times 768 = 4,718,592$

### Key Value Caches

The key $K$ and value $V$ of the $\text{Attention}(Q,K,V)$ are stored for previous tokens for next token prediction.

For example, assumed model has already processed $128$ tokens, base on which to predict the $129$-th token.

For ONE layer of BERT-base, there is

$$
K\in\mathbb{R}^{\text{numHeads}\times\text{seqLen}\times\text{headDim}}=\mathbb{R}^{12\times 128\times 64} \\
V\in\mathbb{R}^{\text{numHeads}\times\text{seqLen}\times\text{headDim}}=\mathbb{R}^{12\times 128\times 64}
$$

These caches are maintained per layer. Thus, there are $12$ independent pairs of caches (one pair per layer).

For 4k context length with FP16, there is KV cache $2 \times 12 \times 12\times 4096\times 64 \times 2\text{bytes}=144\text{MB}$.

### Temporary Intermediate Computation Results

Temporary computation results are used only on the current layer, and on the next layer the intermediate values are re-computed.

Again take BERT-base as example, for each head $h=1,2,...,12$ in ONE layer, given $128$ already predicted tokens, there is

* Raw Attention Score

$$
S_{h}=\frac{Q_hK_h^{\top}}{\sqrt{64}}\in\mathbb{R}^{128}
$$

* Attention Score Softmax Normalization

$$
a_{h,i}=\frac{\exp(S_{h,i})}{\sum_{i=1}^{128}\exp(S_{h,i})},
\qquad a_{h}\in\mathbb{R}^{128}
$$

* Weighted Sum over Values

$$
O_h = \sum^{128}_{i=1}a_{h,i}V_{h,i} \in\mathbb{R}^{64}
$$

* Output Concatenation for all $12$ heads

$$
O=\text{Concat}(O_1, O_2, ..., O_{12})\in\mathbb{R}^{12\times 64}
$$

* Compute the new $K$ and $V$ for the $129$-th token for all $12$ heads, and add them to KV Caches

$$
K_{i=129} \in\mathbb{R}^{12\times 64} \\
V_{i=129} \in\mathbb{R}^{12\times 64}
$$

## The Problem of Token Repetition

The probability of a token $t_{i+1}$ being selected takes into consideration of all previous tokens $P(t_{i+1}|t_1,t_2,...,t_i)$.

* Contextual Bias: If the model has seen similar patterns during training (e.g., repetitive phrases like "hello hello hello"), it may overestimate the probability of repeating certain tokens.
* Overconfidence in Token Selection: The model might select the most probable token repeatedly, leading to a loop.

### Non-Training Mitigation Solutions

Such options are helpful in mitigating the repeated token generation.

```py
from transformers import GPT2LMHeadModel

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")

outputs = model.generate(input_ids, 
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=2.0)
```

#### Temperature Scaling

Temperature scaling is a common method for controlling randomness in predictions.
Given temperature $T$, for token prediction by softmax, there is

$$
t_i=\frac{\exp(\frac{\text{logit}_i}{T})}{\sum_{j=1}^n\exp(\frac{\text{logit}_j}{T})}
$$

* High Temperature $T > 1$: Increases randomness by flattening the distribution. The logits are scaled down, causing the difference between the probabilities of different tokens to become smaller. The results are more diverse.
* Low Temperature $T < 1$: Increases determinism by sharpening the distribution. The logits are amplified, causing higher-probability tokens to become more dominant. This results in more predictable, conservative outputs.
* Temperature $T = 1$: The distribution remains unchanged, as it represents the default probability scale from the model.
* Temperature $T = 0$: $\frac{\text{logit}_i}{T}$ becomes extremely large for the highest logit, and the other logits become negligible. The model will produce the same output every time for a given input, as it always selects the most probable token.

#### Penalty for Repetition

Introduce a hyperparameter $\lambda$ that controls the strength of the penalty to adjust logit.

For next token $t_{i+1}$ prediction, let $\hat{t}_{i+1}$ be the $\text{logit}_{i+1}$ supposed corresponding prediction by softmax.
After having adjusted as per $\hat{\text{logit}}_{i+1}$, this new logit might predict a new token different from the old one.

$$
\hat{\text{logit}}_{i+1}=\text{logit}_{i+1}(\hat{t}_{i+1})-\lambda\cdot 1(\hat{t}_{i+1}=t_{1}, \hat{t}_{i+1}=t_{2}, ..., \hat{t}_{i+1}=t_{i})
$$

where

* $1(.)$ is the indicator function that checks if the token has already appeared in the sequence.

#### Top-k and Top-p Sampling

Top-k sampling restricts the selection of the next token to the top $k$ tokens with the highest probabilities.

$$
P_{\text{top-k}}(t_{i+1}|t_1,t_2,...,t_i)=\begin{cases}
    \frac{\exp(\frac{\text{logit}_i}{T})}{\sum_{j=1}^n\exp(\frac{\text{logit}_j}{T})} & \text{if } t_{i+1} \in \text{top-k} \\
    0 & \text{otherwise}
\end{cases}
$$

Top-p sampling restricts the selection by a defined cut-off threshold.

$$
P_{\text{top-p}}(t_{i+1}|t_1,t_2,...,t_i) > p_{\text{threshold}}
$$

#### Prompt Engineering

For example, include this in prompt to affect prior distribution.

* "Provide a response where each word is unique and does not repeat."
