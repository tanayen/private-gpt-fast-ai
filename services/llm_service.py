
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
from langchain_community.embeddings import OpenAIEmbeddings
from pymongo import MongoClient
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from backports import configparser
from getpass import getpass

import json


class InternalLlmService:
    _mongo_loader: MongodbLoader
    _langchain_llm: Ollama
    _embeddings_by_ollama: OllamaEmbeddings
    _embeddings_by_open_ai: OllamaEmbeddings
    _mongo_client: MongoClient
    _parser: configparser.ConfigParser

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
        self._embeddings_by_ollama = OllamaEmbeddings(model=parser["model"]["name"])
        openai_key = getpass("Please input OpenAI Key : ") 
        self._embeddings_by_open_ai = OpenAIEmbeddings(openai_api_key=openai_key)

    def llama_chat(self, request: ChatRequest):
        return {}

    async def lc_chat(self, request: ChatRequest):
        # docs = await self._mongo_loader.aload()
        embed_model = self._embeddings_by_ollama
        index_name = "company"
        if(request.embed_model == "open_ai") :
            embed_model = self._embeddings_by_open_ai
            index_name = "company_openai"
        # Embed & Store data
        vectorstore: ElasticsearchStore = ElasticsearchStore(
            es_url="http://10.101.3.19:9200",
            index_name=index_name,
            embedding=embed_model,
            es_user="cashgrow",
            es_password="uHzXk73jCCEdNMu"
        )
        # ElasticVectorSearch(elasticsearch_url="http://cashgrow:uHzXk73jCCEdNMu@10.101.3.19:9200", index_name="company", embedding=self._embeddings)
        question_vector: List[float] = await embed_model.aembed_query(request.prompt)
        docs = vectorstore.similarity_search(query=request.prompt, fields=[
                                             "company_id", "company_name", "company_address"])
        print(docs)
        result = self._langchain_llm(
            {"question": request.prompt, "docs": docs})
        return {"answer": result["text"], "question_vector": question_vector}
