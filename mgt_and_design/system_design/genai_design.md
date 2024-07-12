# Enterprise Generative AI (NLP) Architecture

<div style="display: flex; justify-content: center;">
      <img src="imgs/genai_design.drawio.svg" width="80%" height="80%" alt="genai_design.drawio" />
</div>
</br>

* LLM servicing functions

`fastchat` deployed multiple LLMs.

One can consider wrapping LLMs as dockers to load/remove on/from GPU if GPU resource is limited.
However, LLM loading can be slow, hence one can have a persistent generic LLM loaded on GPU, while not frequently queried domain knowledge finetuned LLMs can be dynamically loaded.

* Functional tools (may not or may chained with UI)

PDF parser to HTML/text: `PyPDF`, `PyPDFMiner`
OCR: `paddlepaddle`

Tools can be made with agent, e.g., `langraph`.

* RAG and vector DB, be high-availability and -resilience

`llama-index` with `Elasticsearch-vector-store` or `FAISS`

* AI Guardian services: LLM response performance and security, blocked highly inaccurate/dangerous/harmful response.

`DeepEval` or `langcheck`

* Evaluation service: record human ratings/comments on LLm response (e.g., provide thumb up/down button on UI, and let user label/comment what they thought as wrong), also provided quantitative evaluation,

In details, evaluation metrics include: cosine similarity scores of LLM response vs context/prompts, n-grams hit rates, LLM graded response scores $\in [0, 1]$, quantified user feedbacks, added performance and security check metrics.

* On-going fine-tuning: automate fine-tuning with human feedback (RLHF)
* Tracking and monitor: track request/response and perform analytics

`langfuse`

* Caching services: various other services need a middleman for shared stateful data, e.g., user session bounded data, e.g., requested different rag strategies and re-embeddings.

For example, an end user submits a long-time processing request (e.g., re-embedding many documents with a different chunk-split strategy) from an agent device, e.g., a browser window; then closed this browser window.
This end user expects an async response that will get checked later from another agent device.
This needs a cache service usually by redis or postgresql.

* API services:

`eureka` as URL management server, `swagger` as API documentation

* Authorization Management: OAuth2 with LDAP
* Drag and Drop Low Code Platform

`flowise` or `langflow`

* Admin:

An admin should be able to read/write all DBs, enables/disables LLM servicing, and triggers LLM fine-tuning on demand.
