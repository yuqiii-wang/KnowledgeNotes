# LLM Evaluations

Simple accuracy tests against exact matches between prediction tokens vs label tokens at the exact sequence position.

$$
\text{Accuracy}=
\frac{\text{TruePositive}+\text{TrueNegative}}{\text{TrueNegative}+\text{FalsePositive}+\text{FalseNegative}+\text{TruePositive}}
$$

However, in human language there are synonyms, and by certain grammar arrangements tokens at different sequence/sentence positions may give the same semantic/linguistic meanings.

An LLM base/pretrained model learns from common knowledge from tasks such as Masked Language Modeling (MLM) from texts such as  Wikipedia and academic publications, and forms the "consciousness" of how to "chain" the vocabularies.
The final layer output from LLM/transformer only represents the "consciousness" of such knowledge, and does not produce accurate results with respects to different NLP tasks, and finetuning is required.

In evaluation, models are finetuned for sub-tasks for particular metric evaluation.
For example, to evaluate if model's output sentences are considered grammatically correct,

1. prepare input sentences and corresponding human-annotated binary labels where $1$ is grammar-good, while $0$ is grammar-wrong;
2. add a classifier/pooler appended to the base/pretrained model;
3. then finetune the model by the prepared sentences and labels;
4. evaluate the finetuned model by binary classification metrics such as Matthews correlation coefficient (MCC).

There are many evaluation benchmarks: *GLUE*, *BLUE*, etc., that use a diverse NLP tasks with corresponding datasets for testing.

* GLUE

GLUE, the General Language Understanding Evaluation benchmark is a collection of resources for training, evaluating, and analyzing natural language understanding systems.

It is consisted of multiple datasets with labels that converts typical NLP tasks to classification problems. 

* BLEU

BLEU (Bilingual Evaluation Understudy) and BLEU-derived metrics are most often used for machine translation.

It proposes clipped *n-grams* percentage $p_n$:

$$
p_n =
\frac{\sum_{C \in \text{Candidates}} \sum_{\text{n-gram} \in C} Count_{clip}(\text{n-gram})}
{\sum_{C' \in \text{Candidates}} \sum_{\text{n-gram}' \in C'} Count(\text{n-gram}')}
$$

where n-gram refers to n-sequence tokens present both in two texts.
The $Count(\text{n-gram})$ is the count of the contained tokens,
and $Count_{clip}(\text{n-gram})=\max\big(Count(\text{n-gram}), maxCount \big)$ simply clips the count by setting a max threshold, that if an n-gram repeats for too many times, 

For example, there are $37$ words in the candidate prediction, and by setting $maxCount=2$, for 2-gram, there are "It is a guide to action" x 1, "ensures that the military" x 1, "the party" x 3, "absolute control" x 1, "the military" x 1.
The 2-gram token count is $20$. However, having set the threshold $maxCount=2$, the "the party" is only counted twice instead of three times.
Finally, the result is $p_2=\frac{18}{37}$.

|Candidate Prediction|Reference Truth|$p_n$|
|-|-|-|
|It is a guide to action which ensures that the military always obeys the commands of the party. The party should hold absolute control over the military, and no one can resist the order of the party.|It is a guide to action that ensures that the military will forever heed the party's commands. The party demands absolute control of the military, and nobody can disobey the party.|for bi-gram: $p_2 = \frac{18}{37}$|

BLEU adds *brevity penalty* to $p_n$ to penalize long candidate prediction sentences.
This is for model may over-elaborate semantics with repeated n-gram tokens, such as "It is a guide to action and a guide to ensure, also a guide to enforce that ...", to increase $p_n$.

$$
\text{BrevityPenalty} = \left\{
    \begin{matrix}
        1 && c > r \\
        e^{1-\frac{r}{c}} && c \le r
    \end{matrix}
\right.
$$

where $c$ is the length/token count of a candidate prediction sentence, $r$ is the average/normalized length/token count of reference sentences from an article/document.
The penalty is exponential that sees increasingly heavy penalty as prediction sentences grow too long.

## Common NLP Evaluations

### Grammar Test

If sentences are grammatically correct judged by native speakers with annotations.

|Example Sentences|Labels|Meaning|
|-|-|-|
|We yelled ourselves hoarse.|1|Grammar-good|
|We yelled ourselves.|0|Grammar-wrong|

* cola (Corpus of Linguistic Acceptability): from books and journal articles on linguistic theory.

### Entailment

Entailment: If the meaning of the hypothesis can be inferred or logically deduced from the premise, it is considered an entailment. For example:

|Premise|Hypothesis|Labels|Meaning|
|-|-|-|-|
|The cat is sitting on the mat.|The cat is on a piece of furniture.|0|Entailment|
|The cat is sitting on the mat.|The dog is playing with a rubber ball.|1|Neutral|
|The cat is sitting on the mat.|The cat is sitting on a chair.|2|Contradiction|

* mnli (Multi-Genre Natural Language Inference Corpus): from ten different sources, including transcribed speech, fiction, and government reports. 
* snli (Stanford Natural Language Inference): a collection of 570k human-written English sentence pairs manually labeled for balanced classification with the labels "entailment", "contradiction", and "neutral".
* rte (Recognizing Textual Entailment): Examples are constructed based on news and Wikipedia text.

### Equivalence

If two sentences are semantically identical.

|Sentence 1|Sentence 2|Labels|Meaning|
|-|-|-|-|
|Please give me a recipe of making pasta.|Please hand me a recipe of how to make pasta.|1|Equivalent|
|Please give me a recipe of making pasta.|Please give me a recipe of preparing vegan soup.|0|Not equivalent|

* mrpc (Microsoft Research Paraphrase Corpus): from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent.
* qqp: (Quora Question Pairs2): a collection of question pairs from the community question-answering website Quora. The task is to determine whether a pair of questions are semantically equivalent.

#### Semantic Similarity

If two sentences are semantically similar.

|Sentence 1|Sentence 2|Labels|Meaning|
|-|-|-|-|
|A man is playing a flute.|A man is playing a bamboo flute.|3.867|$\text{score} \in [0,5]$|
|A man is playing a flute.|A man is playing a flute with his fingers.|4.5|$\text{score} \in [0,5]$|
|A man is playing a flute.|A man is playing a piano.|1|$\text{score} \in [0,5]$|
|A man is playing a flute.|A cat is sitting on a mat.|0|$\text{score} \in [0,5]$|

* stsb (Semantic Textual Similarity Benchmark): sentence pairs drawn from news headlines, video and image captions, and natural language inference data. Each pair is human-annotated with a similarity score from 1 to 5.

### Question & Answer

#### Answer Entailment

The task is to determine whether the context sentence contains the answer to the question.

|Question|Answer|Labels|Meaning|
|-|-|-|-|
|How is Nirvana achieved?|In Theravada Buddhism, the ultimate goal is the attainment of the sublime state of Nirvana, achieved by practicing the Noble Eightfold Path (also known as the Middle Way), thus escaping what is seen as a cycle of suffering and rebirth.|1|Entailed|
|How is Nirvana achieved?|Nirvana in the Gita is a Buddhist term adopted by the Hindus. In the Buddhist tradition, nirvana is described as the extinguishing of the fires that cause rebirths and associated suffering.|0|Not entailed|

* qnli (Stanford Question Answering Dataset): question-paragraph pairs, where one of the sentences in the paragraph (drawn from Wikipedia) contains the answer to the corresponding question (written by an annotator).

#### Answer Retrieval



### Sentiment

Describes sentiment of texts.

Binary classification typically says $1$ for positive and $0$ for negative;
Multi-class classification can include granular level sentiments such as "anger", "happy", "hate", "thrilled", etc.

|Example Sentences|Labels|Meaning|
|-|-|-|
|The movie is just boring|0|Negative|
|I have never been so excited waiting for this release, and it did not disappoint me.|1|Positive|

* sst2 (Stanford Sentiment Treebank): from movie reviews and human annotations of their sentiment.

### Comprehension

#### Pronoun Reference Comprehension

Pronoun reference tests that, given a paragraph consisted of multiple pronouns referring to different people/objects and behaviors accordingly, if LLM can correctly link action vs people/objects by de-referencing pronouns.

|Paragraph|Pronoun/Action Reference|Labels|Meaning|
|-|-|-|-|
|Jack is a fan of Elon Musk, a legendary entrepreneur who would host an AI tech exhibit next month. Jack's best friend Jason had been closely following this event's news, and he was happy to join, but did not have the entry ticket. He then borrowed one ticket from Jack who happened to have many entry tickets. Besides giving one to Jason, Jack gave one to his wife Jasmine, but Jasmine declined this invitation for she needs to take care of their newborn baby John. Finally, Jack decided driving to the AI exhibit with Jason.|Jack needs to take care of John.|0|Not entailed|
|(same text as above)|Jason needs to take care of John.|0|Not entailed|
|(same text as above)|Jasmine needs to take care of John.|1|Entailed|
|(same text as above)|Elon Musk needs to take care of John.|0|Not entailed|

* wnli (Winograd Schema Challenge): let LLM choose from a list of ambiguous pronoun/action pairs (multi-class classification) where there is only one true answer; examples are derived from fiction books with manual .

#### Summarization

A good summary covers main info of a paragraph.

* ROUGE (Recall-Oriented Understudy for Gisting Evaluation): gauges the overlap of words between a generated output and a reference text, provided tools are ROUGE-N (n-grams), ROUGE-L (longest common sub-sequence), ROUGE-W (weighted longest common sub-sequence), and ROUGE-S (skip-bi-gram).