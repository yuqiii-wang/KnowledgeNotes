# LLM Evaluation Enhancement Practices

## LLM Evaluation Common Practices

Generally speaking, to evaluate an LLM without human intervention, there are

* Lexical stats
* Convert into a regression or classification problem
* Check embedding similarity between truths vs predictions

### Industry Fast-Path Practices for Data Preparation

Manually craft dataset (prepare source texts, manually propose questions/challenges, and answers/labelling) can be time-consuming,
below shows a fast path how to evaluate a fine-tuned LLM with provided source texts.

1. Given source texts, ask LLM to generate a few questions and answers. These are proposed as the training set.
2. By tasks, can further ask LLM to propose challenges and label the challenges, e.g., entailment tasks.
3. Given the generated questions/challenges, ask LLM to paraphrase the questions/challenges, and these paraphrased questions/challenges should be semantically identical to the original questions/challenges.
4. Before fine-tuning an LLM, use the raw LLM to answer the paraphrased questions/challenges, served as the benchmark.
5. Fine-tuning the LLM with the generated trained dataset.
6. Use the fine-tuned LLM to answer the paraphrased questions/challenges, and compared with the benchmark results.

## Hallucination

Hallucination is inevitable for in the attention mechanism the $Q$ and $K$ yield a high attention score if tokens' embedding positions are strongly correlated.

$$
\text{Attention}(Q,K,V) = \text{softmax} \big(\frac{Q K^{\top}}{\sqrt{d_k}} \big) V
$$

### Hallucination Mitigation

#### Hallucination Mitigation By Prompts

The goal is from a long and large prompt to increase the percentage of desired input/output.

* Chain-of-Thought (CoT) Prompting: add explanation
* Few-Shot Prompting: repeat correct patterns
* Clear Instructions: remove noisy history instructions

RAG is a prominent example of this philosophy that from a large corpus of private knowledge to retrieve relevant data only, and feed it in prompt.

#### Hallucination Mitigation By Training

* Fine-tuning includes negative samples that is quite similar to true answers with only small wording differences that give totally wrong semantics
