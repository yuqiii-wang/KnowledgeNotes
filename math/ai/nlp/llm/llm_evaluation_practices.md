# LLM Evaluation Practices

Generally speaking, to evaluate an LLM without human intervention, there are 

* Lexical stats
* Convert into a regression or classification problem
* Check embedding similarity between truths vs predictions

## Industry Fast-Path Practices for Data Preparation

Manually craft dataset (prepare source texts, manually propose questions/challenges, and answers/labelling) can be time-consuming,
below shows a fast path how to evaluate a fine-tuned LLM with provided source texts.

1. Given source texts, ask LLM to generate a few questions and answers. These are proposed as the training set.
2. By tasks, can further ask LLM to propose challenges and label the challenges, e.g., entailment tasks.
3. Given the generated questions/challenges, ask LLM to paraphrase the questions/challenges, and these paraphrased questions/challenges should be semantically identical to the original questions/challenges.
4. Before fine-tuning an LLM, use the raw LLM to answer the paraphrased questions/challenges, served as the benchmark.
5. Fine-tuning the LLM with the generated trained dataset.
6. Use the fine-tuned LLM to answer the paraphrased questions/challenges, and compared with the benchmark results.

## Security Checking

### Common Checking Items

#### Hallucination

### Open Source Libs

#### LangKit

#### LangCheck

#### Llama Index

#### LangChain