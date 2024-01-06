import json
from backports import configparser
from elasticsearch import Elasticsearch, helpers
from pymongo import MongoClient
from langchain.embeddings import OllamaEmbeddings
from getpass import getpass
import sys
import requests

parser = configparser.ConfigParser()
parser.read("../config.conf")
es = Elasticsearch(
    ['http://10.101.3.19:9200'],
    basic_auth=('cashgrow', 'uHzXk73jCCEdNMu')
)

type = getpass("Choose Model Type 1:Ollama - 2:ChatGPT : ")

model_type = int(type)

if (model_type not in [1, 2]):
    sys.exit()

embedding = any
index_name = "company_ollama"
openai_key = ""
dims = 4096
if (model_type == 1):
    embedding = OllamaEmbeddings(model=parser["model"]["name"])

if (model_type == 2):
    openai_key = getpass("Please input OpenAI Key : ")
    index_name = "company_openai"
    dims = 1536

if (model_type == 2 and openai_key == ""):
    sys.exit()

try:
    es.options(ignore_status=[400, 404]).indices.delete(index=index_name)
    es.indices.create(
        index=index_name,
        body={
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "dense_vector",
                        "dims": dims,
                        "similarity": "max_inner_product"
                    },
                    "company_name": {
                        "type": "text"
                    },
                    "company_id": {
                        "type": "text"
                    },
                    "company_address": {
                        "type": "text"
                    }
                }
            }
        }
    )
except:
    print("An exception occurred")

def openai_embedding_text(text: str) -> list[float]:
    global openai_key
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai_key,
    }
    data = {
        "input": text,
        "model": "text-embedding-ada-002"
    }
    rsp = requests.request(
        "POST", 
        "https://api.openai.com/v1/embeddings", 
        headers=headers, 
        json=data
    )    
    return rsp.json()["data"][0]["embedding"]


def embedding_text(text: str) -> list[float]:
    global model_type
    if (model_type == 1):
        global embedding
        return embedding.embed_documents([text])[0]
    return openai_embedding_text(text=text)

mongo_client = MongoClient(parser["mongo"]["url"])

docs = []

index = 0

for company in mongo_client.get_database("fenrir-invoice").get_collection("company_info").find({}):
    # Transform data
    id = company["_id"]
    print("[" + str(index) + "] - Process company : " + id)
    # print(len(embedding.embed_documents([id + " " + company["name"]])[0]))
    es.index(index=index_name, body={
        'company_id': id,
        'company_name': company["name"],
        'company_address': company["address"],
        'vector': embedding_text(id + " " + company["name"])
    })
    index = index + 1
