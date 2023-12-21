# LLM Evaluations

Simple accuracy tests against exact matches between prediction tokens vs label tokens at the exact sequence position.

$$
\text{Accuracy}=
\frac{\text{TruePositive}+\text{TrueNegative}}{\text{TrueNegative}+\text{FalsePositive}+\text{FalseNegative}+\text{TruePositive}}
$$

However, in human language there are synonyms, and by certain grammar arrangements tokens at different sequence/sentence positions may give the same semantic/linguistic meanings.

There are a few evaluation solutions:

* Use advanced models such as ChatGPT4 to evaluate small fine-tuned models
* 

In LangChain, below evaluation aspects are proposed.

```python
# Langchain Eval types
EVAL_TYPES={
    "hallucination": True,
    "conciseness": True,
    "relevance": True,
    "coherence": True,
    "harmfulness": True,
    "maliciousness": True,
    "helpfulness": True,
    "controversiality": True,
    "misogyny": True,
    "criminality": True,
    "insensitivity": True
}
```

## GLUE

GLUE, the General Language Understanding Evaluation benchmark is a collection of resources for training, evaluating, and analyzing natural language understanding systems.

