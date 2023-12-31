from fastapi import FastAPI, File, UploadFile, Request
import logging
import time
import json
from backports import configparser

from fastapi.concurrency import asynccontextmanager
from pymongo import MongoClient
import uvicorn

from message.inbound_http_request import ChatRequest
from message.outbound_http_response import ChatResponse
from langchain.document_loaders.mongodb import MongodbLoader

from services.llm_service import InternalLlmService

app = FastAPI(title="FastApi - PrivateGpt")

llm_service: InternalLlmService = any

parser: configparser.ConfigParser = any

@app.on_event("startup")
async def startup():
    global llm_service
    global parser
    parser = configparser.ConfigParser()
    parser.read("config.conf")
    loader = MongodbLoader(
        connection_string=parser["mongo"]["url"],
        db_name=parser["mongo"]["database"],
        collection_name=parser["mongo"]["collection"]
    )
    mongo_client = MongoClient(parser["mongo"]["url"])
    llm_service = InternalLlmService(
        mongo_loader=loader, mongo_client=mongo_client, model=parser["model"]["name"])


@app.post("/llama-inddex/conversation/chat", response_model=ChatResponse)
async def llama_conversation_chat(chat_request: ChatRequest):
    return llm_service.llama_chat(chat_request=chat_request)


@app.post("/langchain/conversation/chat", response_model={})
async def lc_conversation_chat(chat_request: ChatRequest):
    return await llm_service.lc_chat(request=chat_request)

if __name__ == '__main__':
    uvicorn.run(f'main:app', host='localhost', port=8000)
