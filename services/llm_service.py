
from typing import List
from langchain.docstore.document import Document
from llama_index import VectorStoreIndex
from message.inbound_http_request import ChatRequest
from langchain.document_loaders.mongodb import MongodbLoader
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import OllamaEmbeddings
from pymongo import MongoClient
from langchain_community.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from backports import configparser

import json


class InternalLlmService:
    _mongo_loader: MongodbLoader
    _langchain_llm: Ollama
    _embeddings: OllamaEmbeddings
    _mongo_client: MongoClient
    _parser: configparser.ConfigParser
    _vectorstore: ElasticsearchStore

    def __init__(self, mongo_loader: MongodbLoader, parser: configparser.ConfigParser):
        self._mongo_loader = mongo_loader
        self._parser = parser
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    """You are a helpful assistant use {docs} that find comany information from id"""),
                HumanMessagePromptTemplate.from_template("{question}?"),
            ]
        )

        llm = Ollama(
            model=parser["model"]["name"],
            callback_manager=CallbackManager(
                [StreamingStdOutCallbackHandler()])
        )

        self._langchain_llm = LLMChain(llm=llm, prompt=prompt)
        self._embeddings = OllamaEmbeddings(model=parser["model"]["name"])
        self._vectorstore = ElasticsearchStore(
            es_url=self._parser["es"]["url"],
            index_name=self._parser["es"]["index_name"],
            embedding=self._embeddings,
            es_user=self._parser["es"]["user"],
            es_password=self._parser["es"]["password"]
        )

    def llama_chat(self, request: ChatRequest):
        return {}

    async def lc_chat(self, request: ChatRequest):
        question_vector: List[float] = await self._embeddings.aembed_query(request.prompt)

        docs = self._vectorstore.similarity_search(
            query=request.prompt,
            fields=["company_id", "company_name", "company_address"]
        )

        result = self._langchain_llm(
            {"question": request.prompt, "docs": docs})
        return {"answer": result["text"], "question_vector": question_vector}
