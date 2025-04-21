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


## Document Embedding and Match

Document embedding is usually learned by *contrastive learning* that is a technique where a model learns to distinguish between similar and dissimilar examples by comparing them in pairs or triplets.

Token embeddings are "aggregated" such as by max/mean-pooling to ONE document embedding.
This is problematic that the aggregated embeddings may not reflect in semantics how similar two documents are.
For example, in domain knowledge rich texts, informational significance is strongly related to some key terminologies while most other/remaining texts look alike.
However, it is questionable if contrastive learning can pick up such key terminologies as factors for document discrimination.

### Embedding for Time Series Data

Time series data refers to msgs identifiable by a timestamp plus contents.
Timestamp itself contains rich semantics but for embeddings are learned per token then aggregated to represent the document, the semantic significance might not be reflected.

```txt
[2025-04-12 12:33:13:331] Yuqi->Derek, "Hiii"
[2025-04-12 12:33:23:522] Yuqi->Derek, "Could you gv your bid"
[2025-04-12 12:33:28:842] Yuqi->Derek, "US395028121"
[2025-04-12 12:33:42:791] Derek->Yuqi, "Hii, yes, wait"
```

### Embedding for Domain Knowledge Data

Domain knowledge language is characterized by containing rich domain knowledge terminologies/wording formatted in a particular way that only industry professionals could understand.

For example, to trade a bond in human language say

```txt
Hi Jason, could you give your bid to this bond (TENCENT with coupon yield 4.45% matures in Auguster 2035) with price and quantity.
```

This in bond trader language would be

```txt
gv bid TENCENT 4.45 08/35, incl price and amt
```

## Vector DB

Vector DB is used to store vector data that are searched/indexed by vector similarity search.

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
|Squared L2|'l2'|$d = \sum\left(\bold{v}_i-\bold{v}_j\right)^2$|
|Inner product|'ip'|$d = 1.0 - \sum\left(\bold{v}_i \times \bold{v}_j\right)$|
|Cosine similarity|'cosine'|$d = 1.0 - \frac{\sum\left(\bold{v}_i \times \bold{v}_j\right)}{\sqrt{\sum\left(\bold{v}_i^2\right)} \cdot \sqrt{\sum\left(\bold{v}_j^2\right)}}$|
