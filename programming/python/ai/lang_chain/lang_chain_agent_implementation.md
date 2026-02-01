# LangChain Agent Implementation

A good AI agent should consider the below aspects:

* Ecosystem by plugins and fine-tuned base models
* Manage chat history (store/read/delete a chain of history chats)
* Provide rich prompt templates
* RAG (Retrieval Augmented Generation)
* Interaction with SQL/NoSQL DB
* Online search for the latest info
* OCR (Optical Character Recognition)

## Ecosystem by plugins and fine-tuned base models

Users are asked to choose a plugin and a base model prior to starting a query.

Fine-tuned base models target particular domain knowledge areas.

Plugins provide dedicated instruction execution to business logics and load RAG for relevant document search.

```python
from Flask import Flask, request

class ChatbotConfigManager:
    def __init__(self):
        self.plugins["stock"] = stock_market_handler
        self.plugins["bond"] = bond_market_handler
        self.plugins["gold"] = gold_market_handler
        self.plugins["silver"] = silver_market_handler
        self.base_models["stock"] = finetuned_stock_model
        self.base_models["bond"] = finetuned_bond_model
        self.base_models["precious_metal"] = finetuned_precious_metal_model

    def get_plugin(self, plugin_name:str):
        return self.plugins[plugin_name]

    def get_model(self, model_name:str):
        return self.base_models[model_name]

chatbotConfigManager = ChatbotConfigManager()

@app.route(rule="user_config", methods=["POST"])
def config_chatbot():
    config_input = request.json
    plugin = chatbotConfigManager.get_plugin(config_input["plugin_name"])
    base_model = chatbotConfigManager.get_model(config_input["model_name"])
```

## Manage chat history

* `ConversationalRetrievalChain`

This chain takes in chat history (a list of messages) and new questions, and then returns an answer to that question.

* `ConversationalBufferMemory` and `ConversationalBufferWindowMemory`

This memory allows for storing messages and then extracts the messages in a variable.
The buffer window means only retaining a limited number of chats.

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context({"input": "hi"}, {"output": "whats up"})

memory.load_memory_variables({})
# {'history': 'Human: hi\nAI: whats up'}
```

Together, they are as below.
`from_llm` is a convenience method to load chain from LLM and retriever.

```python
mem = ConversationalBufferMemory(
    memory_key="chat_history",
    return_message=True,
    output_key="answer"
)

conv_qa_chain = ConversationalRetrievalChain.from_llm(
    llm=base_model,
    retriever=retriever,
    memory=mem,
    get_chat_history=lambda h: h,
    return_source_documents=True
)

answer = conv_qa_chain({"question": query})
```

## Provide rich prompt templates

An agent should provide prompt templates for different tasks.
Users' inputs/queries are embedded into prompts that are treated as the final total input to LLM.

```python
from langchain.prompts import PromptTemplate

class PromptManager:

    def __init__(self):
        self.summary_prompt = """Please use the document and summarize the content.
        content: {doc}
        """

        self.sensitivity_test_prompt = """Please use the document and analyze the content to check if the content contains any sensitive information.
        The sensitive information includes politics, military, sexism, race, hate.
        content: {doc}
        """

    def generate_query_for_summary(self, doc:str):
        return self.summary_prompt.format(doc=doc)

    def generate_query_for_sensitivity_test(self, doc: str):
        return self.sensitivity_test_prompt.format(doc=doc)
```

## RAG

### Knowledge DB: Indexing and Retriever

```python
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, Doc2txtLoader
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from typing import List


class CustomRetriever(BaseRetriever):
    
    # implementation of `get_relevant_documents`
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return [Document(page_content=query)]


class KnowledgeBaseManager(CustomRetriever):

    # embedding_name: embedding_name
    # kb_path: knowledge base vector db
    def __init__(self, embedding_name: str, kb_path="/path/to/vector/db"):
        embeddings = OpenAIEmbeddings()
        self.db = FAISS.load_local(kb_path, embeddings)
        self.doc_size = doc_size
        self.splitter = CharacterTextSplitter(separator=["\n", "\n\n", ""], chunk_size=1000, chunk_overlap=100)
        self.retriever = CustomRetriever()
        retriever = db.as_retriever(search_kwargs={"k": 1})

    def store_doc_file(self, file_path:str):
        loader = loader = DirectoryLoader(file_path, glob="**/*.pdf", loader_cls=TextLoader)
        doc_ls = loader.load()
        for doc in doc_ls:
            self.store_doc_txt(doc.page_content)

    def store_doc_txt(self, txt):
        source_chunks = []
        for chunk in self.splitter.split_text(txt):
            source_chunks.append(chunk)
        self.db.add_document(source_chunks)

    def retrieve_doc(self, query):
        return retriever.get_relevant_documents(query)
```

### Embeddings and LLAMA Indexing

`text-embedding-ada-002` is an OpenAI developed embedding model in 2023.

Azure stores this embedding model on cloud.

```python
azureEmbs = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
    api_version="2023-05-15",
    api_key="your AzureOpenAI key",
    azure_endpoint="https://<your-endpoint>.openai.azure.com/"
)
```

LlamaIndex is an LLM RAG data framework.

#### LLAMA Indexing Preprocessing Texts

By performing similarity search against whole text embedding directly vs a very short query such as "What is blah blah", similarity search could fail retrieving most relevant documents.

For example, a sub-title may appear only once in a long text, query to this sub-title may not see llama indexing returning relevant docs.

Enhancements from texts to documents are

* Data cleansing
* Tabular data augmentation: add "This is the first row cell_1_1, cell_1_2, ..., this is the first column cell_1_1, cell_2_1, ..." rather than directly serializing "cell_1_1, cell_1_2, ..."
* Extract keywords: for example for HTML, extract `<h1>`, `<h2>`, ..., `<strong>`, `<li>`, etc, and add the keywords as metadata to documents
* Good splitter

Format text by the below separators.

```python
SentenceSplitter.from_defaults(
    separator: str = ' ', 
    chunk_size: int = 1024, 
    chunk_overlap: int = 200,
    paragraph_separator: str = '\n\n\n',
)
```

* Rich metadata

Below processes texts and add additional metadata to a document.

```python
from llama_index.extractors.entity import EntityExtractor

transformations = [
    SentenceSplitter(),
    TitleExtractor(nodes=5),
    QuestionsAnsweredExtractor(questions=3),
    SummaryExtractor(summaries=["prev", "self"]),
    KeywordExtractor(keywords=10),
    EntityExtractor(prediction_threshold=0.5),
]

from llama_index.core.ingestion import IngestionPipeline

pipeline = IngestionPipeline(transformations=transformations)
nodes = pipeline.run(documents=documents)
```

#### LLAMA Indexing As Retriever to LangChain

A LLAMA Indexing retriever can have the below definition.

```python
from langchain_core.documents import Document as langDocument
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, \
                        StorageContext, load_index_from_storage
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.schema import NodeWithScore
from llama_index.vector_stores import FaissVectorStore

from pydantic import Extra
from typing import List

class LlamaIndexDB:

    def __init__(self, db_path:str):
        pass

    def create_db(from_text_dir:str, to_faiss_db_dir:str):
        llamaindex_reader = SimpleDirectoryReader(text_dir, file_metadate=LlamaIndexDB.set_doc_metadata,
                            required_exts=[".txt"], recursive=True)
        all_docs = []
        for docs in llamaindex_reader.iter_data():
            for doc in docs:
                doc.text = doc.text.split("\n\n")[-1]
                all_docs.append(doc)
        service_context = LlamaIndexDB.get_llamaindex_cfg_service_context()
        dim = 1536 # for "text-embedding-ada-002"
        faiss_index = faiss.IndexFlatL2(dim)
        faiss_vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=faiss_vector_store,
                                                        persist_dir=to_faiss_db_dir)
        index = VectorStoreIndex.from_documents(all_docs, service_context=service_context,
                                                storage_context=storage_context,
                                                show_progress=True)
        index.storage_context.persist(to_faiss_db_dir)

    def insert_docs(self, from_text_dir:str, to_faiss_db_dir:str):
        pass

    def update_docs(self, title:str):
        pass

    def delete_docs(self, title:str):
        pass

    @classmethod
    def get_llamaindex_cfg_service_context():
        llamaindex_embs = AzureOpenAIEmbedding(
                            model="text-embedding-ada-002",
                            deployment_name="text-embedding-ada-002",
                            api_version="2023-05-15",
                            api_key="xxx",
                            azure_endpoint="https://xxx.openai.azure.com/")
        service_context = ServiceContext.from_defaults(llm=None, embed_model=llamaindex_embs)
        return service_context

    @classmethod
    def get_llamaindex_cfg_storage_context(dp_path:str):
        assert os.path.exists(db_path)
        faiss_vector_store = FaissVectorStore.from_persist_dir(db_path)
        storage_context = StorageContext.from_defaults(vector_store=faiss_vector_store,
                                                        persist_dir=db_path)
        return storage_context

    # read from a text and set accordingly metadata
    # custom text format, such as <title>\n<url>\n<keywords>\n\n<contents>
    @classmethod
    def set_doc_metadata(one_text_path_filename:str):
        with open(one_text_path_filename, "r", encoding="UTF-8") as filehandle:
            txt = filehandle.read()
        title, url, keywords = txt.split("\n\n")[0].split("\n")
        return {"title":title, "url":url, "keywords":keywords}

class LlamaIndexDBRetriever(BaseRetriever):
    class Config:
        extra = Extra.allow

    def __init__(self, db_path:str):
        super(LlamaIndexDBRetriever, self).__init__()
        self._llamaindex_index = self._init_llamaindex_index(db_path)
        self._llamaindex_retirever = self._llamaindex_index.as_retriever(
            similarity_top=5
        )

    def _init_llamaindex_index(self, db_path:str):
        service_context = LlamaIndexDB.get_llamaindex_cfg_service_context()
        storage_context = LlamaIndexDB.get_llamaindex_cfg_storage_context()
        vector_store_index = load_index_from_storage(service_context=service_context,
                                                    storage_context=storage_context)
        return vector_store_index

    # _get_relevant_documents is an abstract method of BaseRetriever
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) \
    -> List[langDocument] :
        llamaindex_docs: List[NodeWithScore] = self._llamaindex_retirever.retrieve(query)
        lang_docs = [doc for doc in map(lambda node: langDocument(page_content=node.text, metadata={'score':node.score, 'title':node.metadata['title'],
                                                                                        'url':node.metadata['url'], 'keywords':node.metadata['keywords']})
                                                    if "url" in node.metadata and "title" in node.metadata and "keywords" in node.metadata
                                                    else langDocument(page_content=node.text, metadata={'score':node.score}))]
        return lang_docs
```

#### LLAMA Multi-Indexing

* VectorStoreIndex
* Summary Index
* Tree Index
* Keyword Table Index
* Knowledge Graph Index

### `FAISS`

*Facebook AI Similarity Search* (Faiss) is a library for efficient similarity search and clustering of dense vectors.

#### Similarity Search and Optimization in FAISS

Example: given $\text{Sentences} \in \mathbb{R}^{14504 \times 100}$ of the size $\text{numSentences} \times \text{numTokensPerSentence}$.

```python
print(sentences)
# [['Jack', 'is', 'play', '##ing', 'with', 'his', 'cat', '[PAD]', '[PAD]', ...],
#  ['Jason', 'is', 'driv', '##ing', 'his', 'toy', 'car', '[PAD]', '[PAD]', ...],
#  ['Margret', 'is', 'dress', '##ing', 'up', 'her', 'Barbie', 'toy', 'girl', '[PAD]', ...],
#  ...]
```

First provide embeddings as "compressed" representations of sentences by sentence transformer.
For example, a BERT-base gives $\text{SentenceEmbeddings} \in \mathbb{R}^{14504 \times 768}$.

```python
from sentence_transformers import SentenceTransformer

# initialize sentence transformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# create sentence embeddings
sentence_embeddings = model.encode(sentences)
sentence_embeddings.shape
# [14504, 768]
```

* $\mathcal{L}_2$ by `IndexFlatL2`: for every $\mathbf{v}\_i \in \mathbb{R}$

$$
d = \sum\left(\mathbf{v}\_i-\mathbf{v}_j\right)^2
$$

```python
dim = sentence_embeddings.shape[1] # 768
index = faiss.IndexFlatL2(dim)     # init indexing that each vec is of 768 length
index.add(sentence_embeddings)   # add all 14504 sentence embeddings
k = 2                            # top k search
xq = model.encode(["Someone is driving a car"])
doc, idx_sentence = index.search(xq, k)  # do search, `idx_sentence` is the idx of the most similar vectors
```

* Optimization by `IndexIVFFlat` (Inverted File (IVF)) and multiple probes

`IndexIVFFlat` refers to indexing with partitioned areas (*Voronoi cells*), where most similar vectors are clustered into one cell.
The similarity is measured by typical vector distance measurements such as $\mathcal{L}_2$ and Cosine.

The search starts at finding the most similar centroid then exhaustively searching all vectors in the centroid-corresponding cell.

<div style="display: flex; justify-content: center;">
      <img src="imgs/voronoi_cells.png" width="30%" height="15%" alt="voronoi_cells" />
</div>

`index.nprobe` is the number of Voronoi cells to search.
For example, `index.nprobe=10` means the search will be conducted in the current cell and the 9 neighbor/nearest cells.

```python
nlist = 50                          # how many Voronoi cells/partition areas in a cluster,
                                    # where search will be confined in that partition area
dim = sentence_embeddings.shape[1]  # 768
index_FlatL2 = faiss.IndexFlatL2(dim) # distance measurement
index = faiss.IndexIVFFlat(index_FlatL2, dim, nlist)
index.nprobe = 10

index.train(sentence_embeddings)    # vector clustering
index.is_trained                    # check if index is now trained

index.add(sentence_embeddings)   # add all 14504 sentence embeddings

xq = model.encode(["Someone is driving a car"])
doc, idx_sentence = index.search(xq, k)  # do search, `idx_sentence` is the idx of the most similar vectors
```

* Optimization by `IndexIVFPQ` (Product Quantization (PQ))

*Product Quantization* (PQ) is a quantization method of `IndexIVF` splitting vectors into sub-vectors then do clustering and replacing full precision centroids with quantized ones.

```python
m = 8  # number of centroid IDs in final compressed vectors
bits = 8 # number of bits in each centroid
dim = sentence_embeddings.shape[1]  # 768
index_FlatL2 = faiss.IndexFlatL2(dim)  # we keep the same L2 distance flat index
index = faiss.IndexIVFPQ(index_FlatL2, dim, nlist, m, bits)

index.train(sentence_embeddings)    # vector clustering
index.is_trained                    # check if index is now trained

index.add(sentence_embeddings)   # add all 14504 sentence embeddings
```

## Interaction with SQL/NoSQL DB

## Online search for the latest info

## OCR (Optical Character Recognition)
