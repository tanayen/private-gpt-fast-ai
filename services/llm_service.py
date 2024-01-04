
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
import json


class InternalLlmService:
    _mongo_loader: MongodbLoader
    _langchain_llm: Ollama
    _embeddings: OllamaEmbeddings
    _mongo_client: MongoClient

    def __init__(self, mongo_loader: MongodbLoader, mongo_client: MongoClient, model: str):
        self._mongo_loader = mongo_loader
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    """You are a helpful assistant use {docs} that find comany information from id"""),
                HumanMessagePromptTemplate.from_template("{question}?"),
            ]
        )
        llm = Ollama(
            model=model,
            callback_manager=CallbackManager(
                [StreamingStdOutCallbackHandler()])
        )
        self._langchain_llm = LLMChain(llm=llm, prompt=prompt)
        self._mongo_client = mongo_client
        self._embeddings = OllamaEmbeddings(model=model)

    def llama_chat(self, request: ChatRequest):
        return {}

    async def lc_chat(self, request: ChatRequest):
        # docs = await self._mongo_loader.aload()

        # Embed & Store data
        vectorstore: ElasticVectorSearch = ElasticVectorSearch(elasticsearch_url="http://cashgrow:uHzXk73jCCEdNMu@10.101.3.19:9200", index_name="company", embedding=self._embeddings)
        question_vector: List[float] = await self._embeddings.aembed_query(request.prompt)
        docs = vectorstore.similarity_search(question_vector)
        print(docs)
        return {"vector": question_vector}
