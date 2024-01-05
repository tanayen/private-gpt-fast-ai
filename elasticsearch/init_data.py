from backports import configparser
from elasticsearch import Elasticsearch, helpers
from pymongo import MongoClient
from langchain.embeddings import OllamaEmbeddings

index_name = "company"

parser = configparser.ConfigParser()
parser.read("../config.conf")
es = Elasticsearch(['http://10.101.3.19:9200'],
                   basic_auth=('cashgrow', 'uHzXk73jCCEdNMu'))
mongo_client = MongoClient(parser["mongo"]["url"])
embedding = OllamaEmbeddings(model=parser["model"]["name"])

docs = []
index = 0
for company in mongo_client.get_database("fenrir-invoice").get_collection("company_info").find({}):
    # Transform data
    id = company["_id"]
    print("[" + str(index) + "] - Process company : " + id)
    # print(len(embedding.embed_documents([id + " " + company["name"]])[0]))
    es.index(index="company", body = {
        'company_id': id,
        'company_name': company["name"],
        'company_address': company["address"],
        'vector': embedding.embed_documents([id + " " + company["name"]])[0]
        
    })
    index = index + 1
#     doc = 
#     docs.append(doc)
#     index = index + 1
#     if (index in [300, 600, 900, 1200, 1500, 1800, 2100, 2400]):
#         helpers.bulk(es, docs, index='company')
#         document_list = []

# if (len(docs) > 0):
#     helpers.bulk(es, docs, index='company')
#     document_list = []
