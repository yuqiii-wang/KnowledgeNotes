import os

from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama

from LlamaIndexDB import create_llamaindex_db, LlamaIndexDBRetriever 
from llm_config import huggingface_config

os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_config["access_key"]

# Define your system instruction
system_instruction = "The assistant should provide detailed explanations."

# Define your template with the system instruction
template = (
    f"{system_instruction} "
    "Combine the chat history and follow up question into "
    "a standalone question. Chat History: {chat_history}"
    "Follow up question: {question}"
)

# Create the prompt template
condense_question_prompt = PromptTemplate.from_template(template)

llm = 

# Now you can pass this prompt to the from_llm method
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=LlamaIndexDBRetriever("./storage"),
    condense_question_prompt=condense_question_prompt,
    chain_type="stuff",
)