
from message.inbound_http_request import ChatRequest
from message.outbound_http_response import ChatResponse
from langchain.document_loaders.mongodb import MongodbLoader
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import (PromptTemplate, ChatPromptTemplate, AIMessagePromptTemplate,
                               HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class InternalLlmService:
    mongo_loader: MongodbLoader
    langchain_llm: Ollama

    def __init__(self,  mongo_loader: MongodbLoader) -> None:
        self.mongo_loader = mongo_loader
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    """You are a helpful assistant, your name is Flex will use {docs} to answer user question in Vietnamese"""),
                HumanMessagePromptTemplate.from_template("{question}?"),
            ]
        )
        llm = Ollama(
            model="llama2",
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
        self.langchain_llm = LLMChain(llm=llm, prompt=prompt)

    @classmethod
    def chat(self, request: ChatRequest) -> ChatResponse:
        return