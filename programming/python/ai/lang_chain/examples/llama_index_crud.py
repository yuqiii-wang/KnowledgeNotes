import glob
import logging
import sys
import os

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

from LlamaIndexDB import create_llamaindex_db, LlamaIndexDBRetriever 

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# loads BAAI/bge-small-en-v1.5
# from https://huggingface.co/BAAI/bge-small-en-v1.5
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.embed_model = embed_model

##### Create DB
create_llamaindex_db("o_henry_stories/", "./storage")

##### Read DB
llamaIndexDBRetriever = LlamaIndexDBRetriever("./storage")
docs = llamaIndexDBRetriever.get_relevant_documents("what happened in Denver")

##### update docs
llamaIndexDBRetriever.update_docs("o_henry_stories/story_1.txt")
print("Done update_docs")

##### Delete docs
llamaIndexDBRetriever.delete_docs("After Twenty Years")
print("Done delete_docs")

##### Insert docs
llamaIndexDBRetriever.insert_docs("o_henry_stories/story_1.txt")
print("Done insert_docs")
