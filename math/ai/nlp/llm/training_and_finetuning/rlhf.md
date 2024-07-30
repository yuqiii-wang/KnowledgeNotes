# Reinforcement Learning from Human Feedback (RLHF)

Use human responses/ratings to LLM generated texts as feedback, and design a reward/loss function to transform such feedback to fine-tune LLMs.

One most popular approach is to produce a few LLM responses, then let user select the best answers.
The best answers are labelled with high reward value, while others are zeros.
LLM is fine-tuned periodically (e.g., everyday) by the best answers.

## Formulation

Given input token sequence $\bold{x}$ (e.g., long text to summarize) and LLM produced answers $\bold{y}_1, \bold{y}_2, ..., \bold{y}_n$ (e.g., short summary), assume a human user selected a choice $\bold{y}_{i=c}$ as his/her considered best answer.

Define a reward model $r$ that takes input prompt $\bold{x}$ and LLM output $\bold{y}_i \in Y$.
The objective is to minimize the loss $\mathcal{L}(r)$ by fine-tuning the LLM.

$$
\min \mathcal{L}(r)=
\min \mathbb{E} \Big( -\log \frac{e^{r(\bold{x}, \bold{y}_{i=c})}}{\sum_i e^{r(\bold{x}, \bold{y}_i)}} \Big)
$$

where the reward $r$ can be simply defined as exact match of human best rating answer $\bold{y}_{i=c}$ vs other LLM outputs $\bold{y}_i$.
In addition, the reward $r$ can consider other aspects such as encouraging producing "positive" tokens that make a response concise and straightforward.

## LLM Alignment

For an LLM to be used in business, after having done pretraining and fine-tuning, LLM should go through RLHF to align its outputs to human preferences.
