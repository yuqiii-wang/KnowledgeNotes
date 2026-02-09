# LLM Evaluations

Simple accuracy tests against exact matches between prediction tokens vs label tokens at the exact sequence position.

$$
\text{Accuracy}=
\frac{\text{TruePositive}+\text{TrueNegative}}{\text{TrueNegative}+\text{FalsePositive}+\text{FalseNegative}+\text{TruePositive}}
$$

However, in human language there are synonyms, and by certain grammar arrangements tokens at different sequence/sentence positions may give the same semantic/linguistic meanings.

## Evaluation Aspects

### Embedding Evaluation

Good embedding representations of words should best approximate vocabulary distribution in texts composed by humans.

Reference: https://openai.com/blog/new-and-improved-embedding-model

* Good Geometry

Geometry refers to how embedding vectors are distributed in a vector space.

Generally speaking, a smaller set of more frequent, unrelated words should be evenly distributed throughout the space while a larger set of rare words should cluster around frequent words.

* Word Similarity

Semantically similar words should have high similarity score.
Typically cosine similarity is used.

$$
\text{similarity}_{\cos}(\mathbf{v}\_i, \mathbf{v}_j) = \cos(\theta) = \frac{\mathbf{v}\_i \cdot \mathbf{v}_j}{||\mathbf{v}\_i || \space || \mathbf{v}_j ||}
$$

* Concept Categorization

Words belonged to the same concept category should be similar (measured by $\text{similarity}_{\cos}(\mathbf{v}\_i, \mathbf{v}_j)$).

For example, "mountain", "river", "plain" should be more similar than "cat", "dog", "rabbit", etc.

* LLM Prediction vs Truth

For LLM prediction accuracy measurement, one can make embedding on the prediction and compare with truth's embedding.

### Lexical Overlap

Aim to compare if two sentences are the same in terms of token sequences by exact token match.

* Clipped *n-grams* percentage $p_n$:

$$
p_n =
\frac{\sum_{C \in \text{Candidates}} \sum_{\text{n-gram} \in C} \text{Count}_{clip}(\text{n-gram})}
{\sum_{C' \in \text{Candidates}} \sum_{\text{n-gram}' \in C'} \text{Count}(\text{n-gram}')}
$$

where n-gram refers to n-sequence tokens present both in two texts.
The $\text{Count}(\text{n-gram})$ is the count of the contained tokens,
and $\text{Count}_{clip}(\text{n-gram})=\max\big(\text{Count}(\text{n-gram}), \text{maxCount} \big)$ simply clips the count by setting a max threshold, that if an n-gram repeats for too many times, 

For example, there are $37$ words in the candidate prediction, and by setting $\text{maxCount}=2$, for 2-gram, there are "It is a guide to action" x 1, "ensures that the military" x 1, "the party" x 3, "absolute control" x 1, "the military" x 1.
The 2-gram token count is $20$. However, having set the threshold $\text{maxCount}=2$, the "the party" is only counted twice instead of three times.
Finally, the result is $p_2=\frac{18}{37}$.

|Candidate Prediction|Reference Truth|$p_n$|
|-|-|-|
|It is a guide to action which ensures that the military always obeys the commands of the party. The party should hold absolute control over the military, and no one can resist the order of the party.|It is a guide to action that ensures that the military will forever heed the party's commands. The party demands absolute control of the military, and nobody can disobey the party.|for bi-gram: $p_2 = \frac{18}{37}$|

* Longest Common Sub-Sequence (LCS)

Find the longest common sub-sequence of two texts $\mathbf{v}_A$ and $\mathbf{v}_B$.
The precision $P_{lcs}$ and recall $R_{lcs}$ are computed as below, by which F score is derived.

$$
\begin{align*}
&& R_{lcs} &= \frac{LCS(\mathbf{v}_A, \mathbf{v}_B)}{\text{len}(\mathbf{v}_A)}
&&
P_{lcs} &= \frac{LCS(\mathbf{v}_A, \mathbf{v}_B)}{\text{len}(\mathbf{v}_B)} \\\\
\Rightarrow && F_{lcs} &= \frac{(1+\beta^2)R_{lcs}P_{lcs}}{R_{lcs}+\beta^2 P_{lcs}}
\end{align*}
$$

where $\beta$ is the coefficient controlling the relative importance of $P_{lcs}$ and $R_{lcs}$, such that $\lim_{\beta \rightarrow 0} F_{lcs} = P_{lcs}$ and $\lim_{\beta \rightarrow +\infty} F_{lcs} = R_{lcs}$.
$\text{len}(\mathbf{v})$ is the count of tokens in the vector $\mathbf{v}$.

* Perplexity

Perplexity can be thought of as an evaluation of the model's ability to predict uniformly among the set of specified tokens in a corpus.

For a sequence $\mathbf{x}$ of $T$ tokens, the perplexity is computed as

$$
\text{Perplexity}(\mathbf{x}) =
\exp \bigg( -\frac{1}{T} \sum_{t=1}^T \log p_{\theta} (x_t | \mathbf{x}_{1:t-1}) \bigg)
$$

where $p_{\theta}(...) \in [0,1]$. Negative log likelihood $-\log p_{\theta}(...) \in [0, +\infty)$ sees $-\log p_{\theta}(1) = 0$.
This means when the prediction of $x_t$ is almost certain, $\text{Perplexity}(\mathbf{x})$ is very small.

Cross entropy $\text{H}$ measures how close  two distributions $P$ and $Q$ are.
Set $P$ as the label truth token sequence, and $Q$ as the LLM prediction token sequence, so that predictions vs labels can be measured in cross entropy.

$$
\begin{align*}
\text{H}(P, Q) &= E_P \big( -\log Q \big) \\\\
    &= -\sum_{x_t \in \mathbf{x}} P(x_t) \log Q(x_t) \\\\
    &= -\sum_{x_t \in \mathbf{x}} P(x_t) \big(\log P(x_t) +  \log Q(x_t) -\log P(x_t) \big) \\\\
    &= -\sum_{x_t \in \mathbf{x}} P(x_t) \log P(x_t) - \sum_{x_t \in \mathbf{x}} P(x_t) \log \frac{Q(x_t)}{P(x_t)} \\\\
    &= \text{H}(P) + D_{KL}(P || Q)
\end{align*}
$$

where $D_{KL}(P || Q) > 0$ is Kullback-Leibler (KL) divergence describing how far between $P$ and $Q$. 

### Transform The Evaluation Problem Into A Classification/Regression Problem

An LLM base/pretrained model learns from common knowledge from tasks such as Masked Language Modeling (MLM) from texts such as  Wikipedia and academic publications, and forms the "consciousness" of how to "chain" the vocabularies.
The final layer output from LLM/transformer only represents the "consciousness" of such knowledge, and does not produce accurate results with respects to different NLP tasks, and finetuning is required.

In evaluation, models are finetuned for sub-tasks for particular metric evaluation.
For example, to evaluate if model's output sentences are considered grammatically correct,

1. prepare input sentences and corresponding human-annotated binary labels where $1$ is grammar-good, while $0$ is grammar-wrong;
2. add a classifier/pooler appended to the base/pretrained model;
3. then finetune the model by the prepared sentences and labels;
4. evaluate the finetuned model by binary classification metrics such as Matthews correlation coefficient (MCC).

### Popular Benchmarks

There are many evaluation benchmarks: *GLUE*, *BLUE*, etc., that use a diverse NLP tasks with corresponding datasets for testing.

* GLUE

GLUE, the General Language Understanding Evaluation benchmark is a collection of resources for training, evaluating, and analyzing natural language understanding systems.

It is consisted of multiple datasets with labels that converts typical NLP tasks to classification problems. 

* BLEU

BLEU (Bilingual Evaluation Understudy) and BLEU-derived metrics are most often used for machine translation.

It proposes clipped *n-grams* percentage $p_n$:

$$
p_n =
\frac{\sum_{C \in \text{Candidates}} \sum_{\text{n-gram} \in C} \text{Count}_{clip}(\text{n-gram})}
{\sum_{C' \in \text{Candidates}} \sum_{\text{n-gram}' \in C'} \text{Count}(\text{n-gram}')}
$$

BLEU adds *brevity penalty* to $p_n$ to penalize long candidate prediction sentences.
This is for model may over-elaborate semantics with repeated n-gram tokens, such as "It is a guide to action and a guide to ensure, also a guide to enforce that ...", to increase $p_n$.

$$
\text{BrevityPenalty} = \left\{
    \begin{matrix}
        1 && c > r \\\\
        e^{1-\frac{r}{c}} && c \le r
    \end{matrix}
\right.
$$

where $c$ is the length/token count of a candidate prediction sentence, $r$ is the average/normalized length/token count of reference sentences from an article/document.
The penalty is exponential that sees increasingly heavy penalty as prediction sentences grow too long.

## Semantic Comprehension Evaluations

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

#### Answer Retrieval from Source Texts

This task assumes that, the source context text has exact token sequence as the answer to a proposed question.
The answer usually has two regression values: start token position in source text and answer length/end token position.

The benchmark for this task can be loss value of how much the start token position deviates from the truth position, plus length/end token position.
An alternative can be entropy/accuracy of exact token match.

|Context|Question|Answer|Answer Start Token Position|
|-|-|-|-|
|The College of Engineering was established in 1920, however, early courses in civil and mechanical engineering were a part of the College of Science since the 1870s. Today the college, housed in the Fitzpatrick, Cushing, and Stinson-Remick Halls of Engineering, includes five departments of study – aerospace and mechanical engineering, chemical and biomolecular engineering, civil engineering and geological sciences, computer science and engineering, and electrical engineering – with eight B.S. degrees offered. Additionally, the college offers five-year dual degree programs with the Colleges of Arts and Letters and of Business awarding additional B.A. and Master of Business Administration (MBA) degrees, respectively.|Before the creation of the College of Engineering similar studies were carried out at which Notre Dame college?|the College of Science|126|

* squad (Stanford Question Answering Dataset) 

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

The evaluation dataset is consisted of paragraphs and summaries.
An LLM reads a paragraph and outputs a summary.

Given the produced summary, one can perform lexical similarity between prediction vs reference, or use a very large generic SOTA LLM to do semantic equivalence test.

|Paragraph|Summary|
|-|-|
|Hannah: Hey, do you have Betty's number?\nAmanda: Lemme check\nHannah: <file_gif>\nAmanda: Sorry, can't find it.\nAmanda: Ask Larry\nAmanda: He called her last time we were at the park together\nHannah: I don't know him well\nHannah: <file_gif>\nAmanda: Don't be shy, he's very nice\nHannah: If you say so..\nHannah: I'd rather you texted him\nAmanda: Just text him 🙂\nHannah: Urgh.. Alright\nHannah: Bye\nAmanda: Bye bye|Hannah needs Betty's number but Amanda doesn't have it. She needs to contact Larry.|


* ROUGE (Recall-Oriented Understudy for Gisting Evaluation): gauges the overlap of words between a generated output and a reference text, provided tools are ROUGE-N (n-grams), ROUGE-L (longest common sub-sequence), ROUGE-W (weighted longest common sub-sequence), and ROUGE-S (skip-bi-gram).
* samsum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization

## Language and Content Qualitative Evaluations

### Hallucination

Hallucination refers to the generated content that is nonsensical or unfaithful to the provided source content.

By definition, there are

* Faithfulness: LLM outputs agree with input content, chat history, user provided documents

* Factuality: LLM outputs agree with to common sense

Accordingly, there are

* Intrinsic Hallucination: conflicts found within existing documents/chats

* Extrinsic Hallucination: generated contents are out of "imagination"/nonsensical

#### Reasons and Remediation

* Corpus have conflicts of info, need to filter out false info from training set
* Top-p text generation scheme and randomness introduce uncertainty of generating contents, use less scholastic methods
* Training data enhancement by providing few-shot intermediate inputs, so that LLMs know how to deduct the logic
* Use Retrieval augmented generation (RAG)

#### Dataset

* truthful_qa: comprises 817 questions that span 38 categories, including health, law, finance and politics. Questions are crafted so that some humans would answer falsely due to a false belief or misconception.

The "Best Answer" is used to perform equivalence test with LLM output.

The "Correct Answer" and "Wrong Answer" are concatenated as a list for multiple-choice task.

|Question|Best Answer|Correct Answer|Wrong Answer|Document reference for RAG|
|-|-|-|-|-|
|Who first started Tesla Motors?|Martin Eberhard and Marc Tarpenning first started Tesla Motors|[ "Martin Eberhard and Marc Tarpenning first started Tesla Motors", "Martin Eberhard and Marc Tarpenning", "Eberhard and Tarpenning" ]|[ "Elon Musk, Martin Eberhard and Marc Tarpenning first started Tesla Motors", "Elon Musk first started Tesla Motors", "Elon Musk", "Elon Musk founded Tesla" ]|https://en.wikipedia.org/wiki/Tesla,_Inc.|

### Conciseness

Conciseness is a writing principle of removing redundant information in text.

Conciseness can be considered a variant of summarization, that LLM input by such as `Summarize the text '[INPUT_TEXTS]'` can be simply replaced to `Rewrite to more concise for the text '[INPUT_TEXTS]'`, and LLM output can be measured by equivalence test against labelled summarized texts.

Info density can be measured by
$\text{Inverse Document Frequency}_{i} = \log \frac{\text{Total no. of documents}}{\text{No. of documents containing term i}}$.

### Bias and Fairness

LLMs inherit stereotypes, misrepresentations, derogatory and exclusionary language, and other denigrating behaviors. These harms are forms of *social bias*, a subjective and normative term we broadly use to refer to disparate treatment or outcomes between social groups that arise from historical and structural power asymmetries.
