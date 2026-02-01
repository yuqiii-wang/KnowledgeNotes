# Text Similarity/Overlap

*Text similarity* measures how close two pieces of text are with respect to either their use of words or characters (lexical similarity) or in terms of meaning (semantic similarity).

## Embedding Vector Similarity: Cosine Similarity

Vocab semantics can be represented by embeddings derived by training language model.

*Cosine similarity* between two vector $\mathbf{v}\_i, \mathbf{v}_j$ is define as

$$
\text{similarity}\_{\cos}(\mathbf{v}\_i, \mathbf{v}_j) = \cos(\theta) = \frac{\mathbf{v}\_i \cdot \mathbf{v}_j}{||\mathbf{v}\_i || \space || \mathbf{v}_j ||}
$$

There is $\cos(\theta) \in [-1, 1]$, where $-1$ means being exactly opposite, $-1$ means being exactly the same, $0$ means orthogonality (being totally different).

*Cosine distance* is simply $1 - \cos(\theta)$.

Cosine similarity can be used for two embeddings' comparison.
If predicted embeddings are very similar to an existing token's embedding, such embeddings can be said this token's representation.

## Example

Below cosine similarity implementation uses word co-occurrences (lexical overlap) as embeddings such that the two sentence vectors are $\text{sentence}_1 \in \mathbb{R}^{11 \times 1}$ and $\text{sentence}_2 \in \mathbb{R}^{7 \times 1}$.

```python
# cosine similarity
import math
import re
from collections import Counter as Count

word = re.compile(r"\w+")
 
sentence_1 = "A well has been in this village for many many years."
sentence_2 = "The well dries up in summer season."

def cosine_similarity(vector_1, vector_2):
    inter = set(vector_1.keys()) & set(vector_1.keys())
    numer = sum([vector_1[i] * vector_2[i] for i in inter])

    s_1 = sum([vector_1[i] ** 2 for i in list(vector_1.keys())])
    s_2 = sum([vector_2[i] ** 2 for i in list(vector_2.keys())])
    deno = math.sqrt(s_1) * math.sqrt(s_2)

    if not deno:
        return 0.0
    else:
        return float(numer) / deno

def generate_vectors(sent):
    w = word.findall(sent)
    return Count(w)

vec_1 = generate_vectors(sentence_1)
vec_2 = generate_vectors(sentence_2)
# Counter({'many': 2, 'A': 1, 'well': 1, 'has': 1, 'been': 1, 'in': 1, 'this': 1, 'village': 1, 'for': 1, 'years': 1})# Counter({'The': 1, 'well': 1, 'dries': 1, 'up': 1, 'in': 1, 'summer': 1, 'season': 1})

sim = cosine_similarity(vec_1, vec_2)
# Similarity(cosine): 0.20965696734438366
```

## Lexical Similarity

*Lexical similarity* is measured by describing identical token/word presence in two texts $\mathbf{v}_A$ and $\mathbf{v}_B$.

*Jaccard similarity* is the simplest way of computing such description by counting token/word presence.
The result is the percentage of the intersection set $\mathbf{v}_A \bigcap \mathbf{v}_B$ (same tokens/words present in both texts) over the union set $\mathbf{v}_A \bigcup \mathbf{v}_B$ (all unique tokens/words).

$$
\text{JaccardSimilarity}(\mathbf{v}_A, \mathbf{v}_B) =
\frac{\mathbf{v}_A \bigcap \mathbf{v}_B}{\mathbf{v}_A \bigcup \mathbf{v}_B} = 
\frac{\mathbf{v}_A \bigcap \mathbf{v}_B}{|\mathbf{v}_A| + |\mathbf{v}_B| - \mathbf{v}_A \bigcap \mathbf{v}_B}
$$

## BM25

BM25 (BM represents *Best Matching*) is a bag-of-words (bag-of-words embedding does not contain positional information) retrieval function that ranks and retrieves most similar documents against a query.

Given a query $Q$ composed of a sequence of tokens $\{q_1, q_2, ..., q_n\}$, the BM25 score of a document $D \in \mathbf{D}$ select from a set of documents $\mathbf{D}$ matching this query $Q$ is

$$
\text{score}(D,Q) =
\sum^n\_{i=1} \text{IDF}(q_i)
\frac{(k+1) \cdot f(q_i, D)}{f(q_i, D)+k\big(1-b+\frac{|D|}{\mu(\mathbf{D})}b\big)}
$$

where

* $f(q_i, D)$ is the number of times that the keyword $q_i$ occurs in the document $D$
* $|D|$ is the length (total token count) of document $D$
* $\mu(\mathbf{D})$ is the average length of a document in the set $\mathbf{D}$
* $k \in [1.2, 2.0]$ and $b=0.75$ are config parameters, that large $k$ increases importance of $f(q_i, D)$ and large $b$ promotes importance of $\frac{|D|}{\mu(\mathbf{D})}$ (reduce the importance of $f(q_i, D)$).
* $\text{IDF}(q_i)$ is the IDF (inverse document frequency) weight of the query term $q_i$.
Given $N$ as the total number of documents, $n(q_i)$ is the number of documents containing $q_i$, there is

$$
\text{IDF}(q_i) =
\ln \Big( \frac{N-n(q_i)+0.5}{n(q_i)+0.5}+1 \Big)
$$
