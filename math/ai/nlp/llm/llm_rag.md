# Retrieval Augmented Generation (RAG)

*Retrieval augmented generation* (RAG) is output enhancement to generative LLM by referencing user provided documents.

* user provided documents are domain knowledge intensive and are vectorized as embeddings stored in vector DB before any query
* user query is accompanied with an instruction/prompt
* user query is represented as vectorized embeddings to match existing domain knowledge in vector BD by methods such as *cosine similarity* or retrieval texts trained by an LLM
* vector DB returns relevant texts as context, thereby $\{ \text{context}, \text{prompt}, \text{query} \}$ is formed as input to LLM
* evaluation of RAG outputs are performed by comparing between open-book answer (LLM output enhanced with existing domain knowledge) and close-book answer (direct LLM output)

Below is a general data flow fo RAG.

<div style="display: flex; justify-content: center;">
      <img src="imgs/rag.jpg" width="70%" height="50%" alt="rag" />
</div>
</br>

## Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

Reference: https://arxiv.org/pdf/2005.11401.pdf

The RAG can be formularized as given input token sequence $\bold{x}$, retrieve text documents $\bold{z}$ and concatenate them as new input $\{ \bold{x}, \bold{z} \}$ to LLM to generate an answer ${\bold{y}}$.

### Retriever and Generator

* a retriever $p_{\eta}(\bold{z} | \bold{x})$ with parameters $\eta$ that returns top-K truncated distributions $\bold{z}$ over stored domain knowledge texts

$$
p_{\eta} (\bold{z} | \bold{x}) \propto \exp \big( \bold{d}^{\top}_{\bold{z}} \space \bold{q}_{\bold{x}} \big)
\qquad
\bold{d}_{\bold{z}} = \text{LLM}_\bold{d}(\bold{z}), \quad \bold{q}_{\bold{x}} = \text{LLM}_{\bold{q}}(\bold{x})
$$

where $\bold{d}_{\bold{z}}$ is a dense/embedding representation of a document produced by an LLM such as BERT-base, named *document encoder*, and $\bold{q}_{\bold{x}}$ is the representation produced by an LLM also such as BERT-base, named *query encoder*.

$\bold{d}^{\top}_{\bold{z}} \space \bold{q}_{\bold{x}}$ is the inner product of the two vectors, that if the product result is a large value, $\bold{d}^{\top}_{\bold{z}}$ and $\bold{q}_{\bold{x}}$ are similar, otherwise orthogonal/different.

* a generator $p_{\theta}({y}_i | \bold{x}, \bold{z}, \bold{y}_{1:i-1})$ parametrized by $\theta$ to generate $y_i$ token by token.

A generator is usually an encoder-decoder LLM such as BART for a seq2seq task.

### RAG-Sequence and RAG-Token

There are two alternatives to generate answer sequence $\bold{y}$ by chaining the retriever and the generator:

* RAG-Sequence

For each retrieved context $\bold{z} \in Z_{\text{top-K}}$, predict the whole length of $\bold{y}$.

$$
p_{seq}(\bold{y} | \bold{x}) \approx
\sum_{\bold{z} \in Z_{\text{top-K}}} p_{\eta}(\bold{z} | \bold{x}) p_{\theta}(\bold{y} | \bold{x}, \bold{z} ) =
\sum_{\bold{z} \in Z_{\text{top-K}}} p_{\eta}(\bold{z} | \bold{x}) \prod^N_{i=1} p_{\theta}({y}_i | \bold{x}, \bold{z}, \bold{y}_{1:i-1})
$$

* RAG-Token

For each token $y_i$, sum all probabilities of multiple retrieved contexts $\bold{z} \in Z_{\text{top-K}}$, then chain together individual token $y_i$ by multiplication.

$$
p_{token}(\bold{y} | \bold{x}) \approx
\prod^N_{i=1} \bigg( \sum_{\bold{z} \in Z_{\text{top-K}}} p_{\eta}(\bold{z} | \bold{x}) p_{\theta}({y}_i | \bold{x}, \bold{z}, \bold{y}_{1:i-1}) \bigg)
$$

### Training

### Evaluation by Open-domain Question Answering (Open-QA)

## REALM: Retrieval-Augmented Language Model Pre-Training

Reference: https://arxiv.org/pdf/2002.08909.pdf

This topic describes how to train a retriever $p_{\eta}(\bold{z} | \bold{x})$

*knowledge retriever* is a concept of how to extract relevant info/documents from large corpus/vector DB.

$$
p(\bold{z} | \bold{x}) = \frac{\exp \Big( f(\bold{x}, \bold{z}) \Big) }{\sum_{\bold{z} \in Z} \exp \Big( f(\bold{x}, \bold{z}) \Big)}
, \qquad
f(\bold{x}, \bold{z}) = \bold{d}^{\top}_{\bold{z}} \space \bold{q}_{\bold{x}}
$$

where $Z$ is the set of all stored domain knowledge documents.
$f(\bold{x}, \bold{z})$ is called *relevance score*.

REALM proposes $W_{\bold{q}}$ and $W_{\bold{d}}$ as weights to project output from LLMs to two equal-length vectors $\bold{q}_{\bold{x}}$ and $\bold{d}_{\bold{z}}$.

$$
\bold{q}_{\bold{x}} = W_{\bold{q}} \space \text{LLM}(\bold{x})
\qquad
\bold{d}_{\bold{z}} = W_{\bold{d}} \space \text{LLM}(\bold{z})
$$

### Training

* Pretraining

To let LLM form the "consciousness" of how to link a query $\bold{x}$ with relevant document $\bold{z}$, train the LLM by mask language modeling (MLM) for a total of $J_{\bold{z}}$ masked tokens in $\bold{z}$.

$$
\sum_{j=1}^{J_{\bold{z}}} p_{\eta} (y_j) =
\sum_{j=1}^{J_{\bold{z}}} \text{LLM}(\bold{x}, \bold{z}, \bold{y}_{1:j-1})
$$

* Finetuning

For open-book QA finetuning, assume that the target answer is a contiguous sequence of $n$ tokens that can be found in the provided document $\bold{y} \in \bold{z}$ by exact match,
so that there are $y_1 = z_{start}$ and $y_n = z_{start+n}$.

$$
\begin{align*}
&&
z_{start} &= \text{LLM}(\bold{x}, \bold{z}, \bold{y})
&&
z_{start+n} &= \text{LLM}(\bold{x}, \bold{z}, \bold{y}) \\
\Rightarrow && 
    p_{\theta} (\bold{y} | \bold{x}, \bold{z}) &\propto
    \sum_{\bold{y} \in \bold{z}} \exp \big( \text{MLP}(y_1, y_n) \big)
\end{align*}
$$

## Vector DB

Chroma DB is an open-source vector storage system (vector database) designed for the storing and retrieving vector embeddings.
By default, Chroma uses the Sentence Transformers (a Python framework for state-of-the-art sentence, text and image embeddings) `all-MiniLM-L6-v2` model to create embeddings.

Chroma is passed a list of documents, it will automatically tokenize and embed them.

Reference: https://docs.trychroma.com/getting-started

```python
import chromadb
chroma_client = chromadb.Client(host='localhost', port=8765)

## chromadb is of document-structure
collection = chroma_client.create_collection(name="my_collection")

## by default use all-MiniLM-L6-v2 for embeddings
collection.add(
    documents=["This is a document", "This is another document"],
    metadatas=[{"source": "my_source"}, {"source": "my_source"}],
    ids=["id1", "id2"]
)

## user can directly provide embeddings
collection.add(
    embeddings=[[1.2, 2.3, 4.5], [6.7, 8.2, 9.2]],
    documents=["This is a document", "This is another document"],
    metadatas=[{"source": "my_source"}, {"source": "my_source"}],
    ids=["id1", "id2"]
)

## fine two most similar documents
results = collection.query(
    query_texts=["This is a query document"],
    n_results=2
)
```

The similarity measurement can be one of the three options "l2", "ip, "or "cosine" (by year 2023) to `hnsw:space`.
The default is "l2" which is the squared L2 norm.

```python
collection = client.create_collection(
    name="collection_name",
    metadata={"hnsw:space": "cosine"} # l2 is the default
)
```

|Distance|parameter|Equation|
|-|-|-|
|Squared L2|'l2'|$d = \sum\left(A_i-B_i\right)^2$|
|Inner product|'ip'|$d = 1.0 - \sum\left(A_i \times B_i\right)$|
|Cosine similarity|'cosine'|$d = 1.0 - \frac{\sum\left(A_i \times B_i\right)}{\sqrt{\sum\left(A_i^2\right)} \cdot \sqrt{\sum\left(B_i^2\right)}}$|

## Hallucination Issues

Hallucination refers to LLM generated content having no truth from stored documents but from its own "imagination".

* Faithfulness: fit to input content, user provided documents
* Factualness: fit to common sense