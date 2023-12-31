from fastapi import FastAPI, File, UploadFile, Request
import logging
import time
import json

from fastapi.concurrency import asynccontextmanager
import uvicorn

from message.inbound_http_request import ChatRequest
from message.outbound_http_response import ChatResponse
from langchain.document_loaders.mongodb import MongodbLoader

from services.llm_service import InternalLlmService

app = FastAPI(title="FastApi - PrivateGpt")

llm_service:InternalLlmService = any

@asynccontextmanager
async def startup():
    global counter
    loader = MongodbLoader(
        connection_string="mongodb://localhost:27017/",
        db_name="sample_restaurants",
        collection_name="restaurants",
        filter_criteria={"borough": "Bronx", "cuisine": "Bakery"},
    )
    llm_service = InternalLlmService(mongo_loader=loader)

@app.post("/conversation/chat", response_model=ChatResponse)
async def conversation_chat(chat_request: ChatRequest):
    return llm_service.chat(chat_request=chat_request)

if __name__ == '__main__':
    uvicorn.run(f'main:app', host='localhost', port=8000)