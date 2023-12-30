from fastapi import FastAPI, File, UploadFile, Request
import logging
import time
import json

app = FastAPI()

@app.post("/conversation/chat")
async def upload_file(file: UploadFile):




@app.post("/uploadfile")
async def upload_file(file: UploadFile):
    logging.info("Receive file : " + file.filename)
    file_bytes = file.file.read()
    return {"content": file.filename}