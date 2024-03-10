from pydantic import Extra
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document as langDocument
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

from llama_index.core.schema import NodeWithScore
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core import Document as llamaIndexDoc

from llm_tools import set_metadata

from typing import List

def create_llamaindex_db(input_txt_path:str, db_path:str):

    llamaindex_docs = SimpleDirectoryReader(input_txt_path, file_metadata=set_metadata,
                                required_exts=".txt").load_data()
    for doc in llamaindex_docs:
        doc.metadata["title"] = doc.metadata["keywords"][0]
        doc.doc_id = doc.metadata["title"]

    # dimensions of BAAI/bge-small-en-v1.5s
    d = 384
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        llamaindex_docs, storage_context=storage_context,
        show_progress=True
    )

    index.storage_context.persist("./storage")
    
class LlamaIndexDBRetriever(BaseRetriever):
    class Config:
        extra = Extra.allow

    def __init__(self, db_path:str):
        super(LlamaIndexDBRetriever, self).__init__()
        self._llamaindex_index = self._init_llamaindex_index(db_path)
        self._llamaindex_retirever = self._llamaindex_index.as_retriever(
            similarity_top=2
        )

    def _init_llamaindex_index(self, db_path:str):
        faiss_vector_store = FaissVectorStore.from_persist_dir(db_path)
        storage_context = StorageContext.from_defaults(vector_store=faiss_vector_store,
                                                        persist_dir=db_path)
        vector_store_index = load_index_from_storage(storage_context=storage_context)
        return vector_store_index

    # _get_relevant_documents is an abstract method of BaseRetriever
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) \
    -> List[langDocument] :
        llamaindex_docs: List[NodeWithScore] = self._llamaindex_retirever.retrieve(query)
        lang_docs = [doc for doc in map(lambda node: langDocument(page_content=node.text, metadata={'score':node.score, 'title':node.metadata['title'],
                                                                                        'url':node.metadata['url'], 'keywords':node.metadata['keywords']})
                                                    if "url" in node.metadata and "title" in node.metadata and "keywords" in node.metadata
                                                    else langDocument(page_content=node.text, metadata={'score':node.score}), llamaindex_docs)]
        return lang_docs
    
    def insert_docs(self, filename:str):
        with open(filename, "r", encoding="UTF-8") as filehandle:
            text=filehandle.read()
        insert_doc = llamaIndexDoc(text=text)
        insert_doc.metadata = set_metadata(filename)
        insert_doc.metadata["title"] = insert_doc.metadata["keywords"][0]
        insert_doc.doc_id = insert_doc.metadata["title"]
        self._llamaindex_index.insert(insert_doc)

    def update_docs(self, filename:str):
        with open(filename, "r", encoding="UTF-8") as filehandle:
            text=filehandle.read()
        update_doc = llamaIndexDoc(text=text)
        update_doc.metadata = set_metadata(filename)
        update_doc.metadata["title"] = update_doc.metadata["keywords"][0]
        update_doc.doc_id = update_doc.metadata["title"]
        self._llamaindex_index.update(update_doc)

    def delete_docs(self, doc_id:str):
        self._llamaindex_index.delete_ref_doc(doc_id)