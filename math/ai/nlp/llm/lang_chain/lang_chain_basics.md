# LangChain Basics

LangChain is an LLM framework. It provides the three main services (by year 2023).

#### Model I/O:

Interface with language models

E.g., generate prompts and send to LLM for text output

```python
from langchain.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)
resp = llm.invoke("Hello, how are you?")
```

#### Retrieval:

Interface with domain/application-specific data;
store user-provided documents, and perform tokenization and embedding on the documents, and retrieve the documents' contents as response to a query.

E.g., `Chroma` is a vector DB storing embeddings of tokens extracted from the document `state_of_the_union.txt`.
A query `"What did the president say about Ketanji Brown Jackson"` is sent to this DB, that extracts source texts from the document `state_of_the_union.txt`.
The extracted texts are performed `similarity_search` to select the most matched texts as the final output.

```python
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('../../../state_of_the_union.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, OpenAIEmbeddings())

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print(docs[0].page_content)
```

#### Agents:

Interface with chaining a series of actions for a specific task with help of tools (tools are customized functions such as triggering external google search);
models should be first fine-tuned to using tools (modern OpenAI GPTs have done such fine-tuning).

E.g., set up an agent that answers char statistics on a user input vocab.

* In tools, define functions that are trained together with LLMs
* In prompting, stack various texts as input: history chat text, role/context pair, user input, intermediate steps for few shots
* In agent invocation, put history chat texts as input there by constructing a series of business logic

```python
################ Tools
from langchain.agents import tool

@tool
def getUniqueCharSetLength(word: str) -> int:
    return len(set(word))

@tool
def getWordLength(word: str) -> int:
    return len(word)

tools = [getUniqueCharSetLength, getWordLength]

################ Bind tools to LLMs
from langchain.tools.render import format_tool_to_openai_function

llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

################ Prompts
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "system",
            "You are an assistant good at vocab char analysis.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

################ Define agent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

################ Invoke this agent
input1 = "how many letters in the word educa?"
result = agent_executor.invoke({"input": input1, "chat_history": chat_history})
chat_history = []
chat_history.extend(
    [
        HumanMessage(content=input1),
        AIMessage(content=result["output"]),
    ]
)
agent_executor.invoke({"input": "is that a real word?", "chat_history": chat_history})
```

## Terminologies

* LLMs vs Chat models

Chat models are often backed by LLMs but tuned specifically for having conversations.

* String vs Message

Strings are pure texts serving as inputs/output to/from LLMs.
Messages are structured data defined in LangChain.

The parent class of message is `BaseMessage` by which complex messages are derived conditioned on specific NLP tasks. The `BaseMessage` has two attributes:

`content`: The content of the message. Usually a string.
`role`: The entity from which the BaseMessage is coming.

For example, in the code below an `llm` returns a string, while an `chat_model` returns an object.

```python
from langchain.schema import HumanMessage
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

llm = OpenAI()
chat_model = ChatOpenAI()

text = "What would be a good company name for a company that makes colorful socks?"
messages = [HumanMessage(content=text)]

llm.invoke(text)
# >> Feetful of Fun

chat_model.invoke(messages)
# >> AIMessage(content="Socks O'Color")
```

* Prompt Templates

Prompts are important guiding LLM in what domain knowledge it should be based on to provide an answer.

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
prompt.format(product="colorful socks")

print(prompt)
## >>> input_variables=['product'] template='What is a good name for a company that makes {product}?'
```

* Retrieval Augmented Generation (RAG)

Many LLM applications require user-specific data that is not part of the model's training set.
The primary way of accomplishing this is through *Retrieval Augmented Generation* (RAG).
