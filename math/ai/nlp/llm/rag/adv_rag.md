# Advanced RAG

## Graph RAG

### Knowledge Graph RAG

## RAG Fusion

## RAG Ranking and Re-Ranking

## Tricks to Improve RAG

### HYDE (Hypothetical Document Embeddings)

Improve retrieval by generating a hypothetical answer to the query and retrieving documents based on that instead of the original query.

#### Example of HYDE RAG

Query: "What are the benefits of vitamin D?"

LLM generates a doc: "Vitamin D supports immune function, bone health, and reduces inflammation..."

That doc is embedded and used to retrieve similar documents from the knowledge base.

### FLARE (Feedback-Augmented Retrieval)

Improve RAG by letting the LLM interact with the retriever iteratively, asking for better documents if the current ones are not helpful.

#### Example of FLARE RAG

Query: "Explain quantum entanglement."

First docs are general physics intros → LLM says "I need more on Bell's Theorem."

Follow-up query: "Bell's Theorem in quantum entanglement"

Retrieves better documents → generates a more accurate answer.
