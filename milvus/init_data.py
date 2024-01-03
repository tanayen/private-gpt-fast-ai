from backports import configparser
from pymilvus import connections, db, CollectionSchema, FieldSchema, DataType, Collection, utility
from pymongo import MongoClient
from langchain.embeddings import OllamaEmbeddings
import random
milvus_collection_name = "company"
milvus_db_name = "invoice"

# Create DB
# Document : https://milvus.io/docs/manage_databases.md
conn = connections.connect(host="127.0.0.1", port=19530)
try:
    db.using_database(milvus_db_name)
    utility.drop_collection(milvus_collection_name)
    db.drop_database(milvus_db_name)
except:
    print("An exception occurred")

database = db.create_database(milvus_db_name)
db.using_database(milvus_db_name)

# Create Collection
# Document : https://milvus.io/docs/create_collection.md
id = FieldSchema(
    name="id", dtype=DataType.INT64, is_primary=True, auto_id=True
)

company_id = FieldSchema(
    name="company_id", dtype=DataType.VARCHAR, max_length=200, default_value="Unknown"
)

company_name = FieldSchema(
    name="company_name", dtype=DataType.VARCHAR, max_length=200, default_value="Unknown"
)

company_address = FieldSchema(
    name="company_address", dtype=DataType.VARCHAR, max_length=255, default_value="Unknown"
)

vector = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=4096)

schema = CollectionSchema(
    fields=[id,
            company_id,
            company_name,
            company_address,
            vector],
    description="Company search",
    enable_dynamic_field=True
)

collection = Collection(name=milvus_collection_name,
                        schema=schema, consistency_level="Strong")

# Create Index Collection
# Document : https://milvus.io/docs/create_collection.md
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}

collection.create_index(
    field_name="vector",
    index_params=index_params
)

utility.index_building_progress(milvus_collection_name)

# Insert data invoice
# Document : https://milvus.io/docs/insert_data.md
# Incase we load data company info and insert to milvus

parser = configparser.ConfigParser()
parser.read("../config.conf")

# Load data from mongo
mongo_client = MongoClient(parser["mongo"]["url"])
embedding = OllamaEmbeddings(model=parser["model"]["name"])

index = 1

data_id = []
data_company_id = []
data_company_name = []
data_company_address = []
data_company_vector = []

for company in mongo_client.get_database("fenrir-invoice").get_collection("company_info").find({}):
    # Transform data
    id = company["_id"]
    print("[" + str(index) + "] - Process company : " + id)
    data_company_id.append(id)
    data_company_name.append(company["name"])
    data_company_address.append(company["address"])
    vector = embedding.embed_documents([id + "-" + company["name"]])[0]
    data_company_vector.append(vector)
    index = index+1
    if (index in [300, 600, 900, 1200, 1500, 1800, 2100, 2400]):
        collection.insert([
            data_company_id,
            data_company_name,
            data_company_address,
            data_company_vector
        ])
        data_company_id.clear()
        data_company_name.clear()
        data_company_address.clear()
        data_company_vector.clear()

if (len(data_company_id) > 0):
    collection.insert([
        data_company_id,
        data_company_name,
        data_company_address,
        data_company_vector
    ])

collection.flush()