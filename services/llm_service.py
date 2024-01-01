
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
from langchain_community.vectorstores.chroma import Chroma
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

        docs = []
        # Load data from mongo
        for company in self._mongo_client.get_database("fenrir-invoice").get_collection("company_info_gen_ai").find({}).limit(300):
            # Transform data
            raw_info = json.loads(company["minvoice_data"])
            object_content = {
                "id": company["_id"],
                "company_name": company["name"],
                "company_address": company["address"],
                "town": raw_info["dctshuyenten"],
                "district": raw_info["dctsxaten"],
            }
            doc = Document(page_content=json.dumps(
                object_content), metadata={"source": "mongo", "id" : company["_id"]})
            docs.append(doc)

        # Embed & Store data
        vectorstore: Chroma = Chroma.from_documents(docs, self._embeddings)
        question_vector: List[float] = await self._embeddings.aembed_query(request.prompt)
        docs = vectorstore.similarity_search_by_vector_with_relevance_scores(
            question_vector)
        result = self._langchain_llm(
            {"question": request.prompt, "docs": docs})
        return {"answer": result["text"]}
