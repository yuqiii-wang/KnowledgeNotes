# LLM Evaluations

Simple accuracy tests against exact matches between prediction tokens vs label tokens at the exact sequence position.

$$
\text{Accuracy}=
\frac{\text{TruePositive}+\text{TrueNegative}}{\text{TrueNegative}+\text{FalsePositive}+\text{FalseNegative}+\text{TruePositive}}
$$

However, in human language there are synonyms, and by certain grammar arrangements tokens at different sequence/sentence positions may give the same semantic/linguistic meanings.

An LLM base/pretrained model learns from common knowledge from tasks such as Masked Language Modeling (MLM) from texts such as  Wikipedia and academic publications, and forms the "consciousness" of how to "chain" the vocabularies.
The final layer output from LLM/transformer only represents the "consciousness" of such knowledge, and does not produce accurate results with respects to different NLP tasks, and finetuning is required.

In evaluation, models are finetuned for subtasks for particular metric evaluation.
For example, to test if a sentence is acceptable in grammar, prepare input sentences and corresponding human-annotated binary labels where $1$ is grammar-good, while $0$ is grammar-wrong;
add a classifier/pooler to the base/pretrained model, then finetune the model;
evaluate the finetuned model by binary classification metrics such as Matthews correlation coefficient (MCC).

There are many evaluation benchmarks: *GLUE*, *BLUE*, etc.

## GLUE

GLUE, the General Language Understanding Evaluation benchmark is a collection of resources for training, evaluating, and analyzing natural language understanding systems.

