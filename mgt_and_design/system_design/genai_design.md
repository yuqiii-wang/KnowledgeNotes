# Generative AI Architecture

* LLM servicing functions: e.g., fastchat deployed multiple LLMs, image recognition exposed with APIs
* Core AI backend that distributes requests to various components
* Functional tools (may not or may chained with UI): such as PDF parser to HTML/text, chat UI with rag, OCR to (tabular) text.
* Vector DB, be high-availability and -resilience
* AI Guardian services: LLM response performance and security, blocked highly inaccurate/dangerous/harmful response.
* Evaluation service: record human ratings/comments on LLm response (e.g., provide thumb up/down button on UI, and let user label/comment what they thought as wrong), also provided quantitative evaluation,

In details, evaluation metrics include: cosine similarity scores of LLM response vs context/prompts, n-grams hit rates, LLM graded response scores $\in [0, 1]$, quantified user feedbacks, added performance and security check metrics.

* On-going fine-tuning: automate fine-tuning with human feedback (RLHF)
* Tracking and monitor: track request/response and perform analytics, e.g., by langfuse
* Caching services: various other services need a middleman for shared stateful data, e.g., user session bounded data, e.g., requested different rag strategies and re-embeddings.

For example, an end user submits a long-time processing request (e.g., re-embedding many documents with a different chunk-split strategy) from an agent device, e.g., a browser window; then closed this browser window.
This end user expects an async response that will get checked later from another agent device.
This needs a cache service usually by redis or postgresql.

* General web backend services: such as eureka as URL management server, swagger as API documentation
