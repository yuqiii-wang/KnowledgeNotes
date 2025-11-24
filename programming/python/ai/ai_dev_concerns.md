# Python for AI Server Development

AI use scenarios typically contains multi-media interactions, e.g., file upload, and agentic AI requests/responses in which AI takes time to respond.
Consequently, besides typical IT backend arch design considerations, this gives additional concerns of

Tech side

* Python as backend needs Python-centric tuning, e.g., python GC and monitoring
* Design file IO as independent services, and should be monitored separately
* Cache design, diff from typical cache retrievable by hash key, user msg, e.g., "good morning" vs "good afternoon", if questions were timing-concerned, they should be treated as two requests, otherwise categorized as greeting. Embedding with question context should be considered.

Biz side:

* Due to agentic LLM behavior, user might need to wait for a long time for a final response. Be good to decompose and give responses from intermediate steps to keep user be aware of the progress.
* Hallucination evaluation, warning, mitigation and auto-improvement: from frontend UI shall engage reference proof and user feedback, e.g., by recording user time on staying on a page to evaluate hallucination; use evaluation metrics to ensure content quality
