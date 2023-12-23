# LLM Evaluations

Simple accuracy tests against exact matches between prediction tokens vs label tokens at the exact sequence position.

$$
\text{Accuracy}=
\frac{\text{TruePositive}+\text{TrueNegative}}{\text{TrueNegative}+\text{FalsePositive}+\text{FalseNegative}+\text{TruePositive}}
$$

However, in human language there are synonyms, and by certain grammar arrangements tokens at different sequence/sentence positions may give the same semantic/linguistic meanings.

An LLM base/pretrained model learns from common knowledge from tasks such as Masked Language Modeling (MLM) from texts such as  Wikipedia and academic publications, and forms the "consciousness" of how to "chain" the vocabularies.
The final layer output from LLM/transformer only represents the "consciousness" of such knowledge, and does not produce 

## GLUE

GLUE, the General Language Understanding Evaluation benchmark is a collection of resources for training, evaluating, and analyzing natural language understanding systems.

