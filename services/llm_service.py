
from typing import List
from langchain.docstore.document import Document
from llama_index import VectorStoreIndex
from pymilvus import connections, db, CollectionSchema, FieldSchema, DataType, Collection, utility
from message.inbound_http_request import ChatRequest
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import OllamaEmbeddings
from backports import configparser
# from langchain_community.vectorstores.chroma import Chroma
# from langchain_community.vectorstores.milvus import Milvus
import json


class InternalLlmService:
    _langchain_llm: Ollama
    _embeddings: OllamaEmbeddings

    def __init__(self):
        parser = configparser.ConfigParser()
        parser.read("config.conf")
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    """You are a helpful assistant use {docs} this is my system data. Find comany information from id."""),
                HumanMessagePromptTemplate.from_template("{question}?"),
            ]
        )
        llm = Ollama(
            model=parser["model"]["name"],
            callback_manager=CallbackManager([
                StreamingStdOutCallbackHandler()
            ])
        )
        self._langchain_llm = LLMChain(llm=llm, prompt=prompt)
        self._embeddings = OllamaEmbeddings(model=parser["model"]["name"])

    def llama_chat(self, request: ChatRequest):
        return {}

    async def lc_chat(self, request: ChatRequest):
        question_vector: List[float] = await self._embeddings.aembed_query(request.prompt)
        connections.connect(host="127.0.0.1", port=19530)
        db.using_database("invoice")
        collection = Collection("company")
        collection.load()
        results = collection.search(
            data=[question_vector],
            limit=5,
            anns_field="vector",
            param={},
            output_fields=["company_id", "company_name", "company_address"],
            consistency_level="Strong"
        )
        print(results)
        return question_vector
