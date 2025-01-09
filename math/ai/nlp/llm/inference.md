# LLM Inference Practices

## Structure

* Encoder-Only

BERT (Bidirectional Encoder Representations from Transformers)

* Prefix-Decoder (Non-Causal Decoder)

T5 (Text-To-Text Transfer Transformer)

* Causal Decoder

GPT (Generative Pre-trained Transformer)

* Encoder-Decoder

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
