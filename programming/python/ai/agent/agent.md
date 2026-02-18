# AI Agent Development Beginner Guide

This guide covers fundamental concepts in modern AI application development, focusing on Large Language Models (LLMs) and Agentic workflows.

## Core Concepts

### 1. LLM (Large Language Model)

The foundation of modern AI applications. Models like GPT-4, Claude 3, and Gemini are trained on vast amounts of text data to predict the next token in a sequence.
- **Context Window**: The limit on the amount of text (measured in tokens) the model can process at once (input + output).
- **Tokens**: The basic unit of text for an LLM (roughly 0.75 words in English).
- **Temperature**: A parameter controlling randomness. Low (0.0) is deterministic; high (0.8+) is creative.

### 2. Prompts & Prompt Engineering

The art of crafting inputs to guide the model's output.
- **System Prompt**: The initial instructions that define the AI's persona, constraints, and behavior (e.g., "You act as a senior Python developer").
- **Zero-shot vs. Few-shot**: Asking the model to perform a task with no examples vs. providing a few examples of input/output pairs in the prompt.
- **Chain of Thought (CoT)**: Encouraging the model to "think step-by-step" to improve reasoning for complex tasks.

### 3. Embeddings & Similarity (Emb Sim)

- **Embeddings**: Converting text (or images) into a list of numbers (vectors) that represent semantic meaning. "Dog" and "Puppy" will have vectors that are numerically close.
- **Vector Database**: A specialized database (e.g., Pinecone, Chroma, pgvector) optimized for storing and querying these high-dimensional vectors.
- **Cosine Similarity**: A metric used to measure how similar two embedding vectors are. Used to find the most relevant documents for a user's query.

### 4. RAG (Retrieval-Augmented Generation)

A technique to ground LLM responses in specific, external data that the model wasn't trained on.
1.  **Retrieve**: User query is converted to an embedding ~> database finds relevant chunks of text.
2.  **Augment**: These chunks are added to the prompt as context.
3.  **Generate**: The LLM answers the user's question using the provided context.

### 5. Agents

An AI system that uses an LLM as a "brain" to reason, plan, and execute actions to achieve a goal. unlike a simple chatbot (input -> output), an agent operates in a loop:
1.  **Thought**: Analyze the request.
2.  **Plan**: Decide which tools to use.
3.  **Action**: Call a tool (function).
4.  **Observation**: Read the tool output.
5.  **Repeat**: Until the task is done.

### 6. Tools & Skills

- **Tools (Function Calling)**: Capabilities given to an Agent. These are functions the LLM can "call" by generating structured JSON (e.g., `get_weather(city="London")`). The application runs the code and returns the result to the LLM.
- **Skills**: A higher-level grouping of tools or a specific competency of an agent (e.g., "Web Research Skill" or "Python Coding Skill").

### 7. MCP (Model Context Protocol)

An open standard that enables AI models to connect to external data and tools uniformly. Instead of writing custom integrations for every data source (GitHub, Google Drive, Slack), MCP provides a common protocol.

#### Architecture

The MCP architecture consists of three main components:

- **MCP Host**: The application where the AI model lives and interacts with the user (e.g., Claude Desktop, VS Code, Cursor). The Host manages connections and discovers available servers.
- **MCP Client**: The protocol implementation within the Host that speaks the "MCP language". It maintains a 1:1 connection with a server.
- **MCP Server**: A lightweight service that exposes specific data or capabilities. It can provide three main things:
    - **Resources**: File-like data (logs, code, database schemas) that can be read by clients.
    - **Prompts**: Pre-written templates (e.g., "Analyze this error log").
    - **Tools**: Executable functions (e.g., `execute_sql_query`, `fetch_webpage`).

#### Example: Analyzing Database Performance

Imagine you want your AI assistant to help debug a slow query in your local database.
1.  **Server**: You run a `postgres-mcp-server` locally that has tools like `list_tables()` and `run_query()`.
2.  **Host**: You configure your AI editor (Host) to connect to this server.
3.  **Interaction**: You ask the AI, "Why is the users table query slow?"
    - The AI (via the Client) calls the `get_schema("users")` tool on the Server.
    - The Server returns the schema (indexes, types).
    - The AI analyzes it and suggests adding an index.

### 8. Fine-tuning

The process of taking a pre-trained base model and training it further on a specific dataset to specialize it for a particular task or tone, rather than relying solely on prompting.

## AI toC Products (Consumer Agents)

AI "toC" (to Consumer) products are applications designed for end-users that leverage agentic workflows to perform tasks autonomously, moving beyond simple chat interfaces.

### 1. Manus (manus.im)

A general-purpose AI agent designed to execute complex workflows. Instead of just giving advice, it can operate a browser to book flights, research topics, or use its creative suite to generate slides and designs. It represents the shift from "chatbots" to "do-bots".

### 2. OpenClaw

An open-source personal AI agent that focuses on local execution and privacy. It connects to personal tools (calendar, email, Slack) to perform actions on the user's behalf. It is known for its "skills" marketplace where users can download new capabilities for their agent.

### 3. Other Notable Products

- **Operator (OpenAI)**: An agent capable of browsing the web and performing multi-step tasks.
- **Computer Use (Anthropic)**: A capability allowing Claude to control a computer cursor and keyboard to accomplish tasks previously requiring human intervention.

