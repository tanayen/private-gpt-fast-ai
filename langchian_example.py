from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import (PromptTemplate, ChatPromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import HumanMessage, SystemMessage

llm = Ollama(
    model="llama2:7b-chat-q5_0",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

loader = WebBaseLoader("https://blog.flexmoney.vn/?p=1382")
data = loader.load()
print("Retrieve data from https://blog.flexmoney.vn/?p=1382")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
print("Embeding using GPT4AllEmbeddings and store data in Chroma vector database")

# # Prompt
# prompt = PromptTemplate.from_template(
#     "Summarize the main themes in these retrieved docs: {docs}"
# )

# # Chain
# llm_chain = LLMChain(llm=llm, prompt=prompt)

# # Run
# question = "What are the approaches to Task Decomposition?"
# docs = vectorstore.similarity_search(question)
# result = llm_chain(docs)

# # Output
# result["text"]


prompt = ChatPromptTemplate(
    messages = [
        SystemMessagePromptTemplate.from_template("""
            You are a helpful assistant, your name is Flex will use {docs} to answer user question in Vietnamese
        """),
        # SystemMessagePromptTemplate.from_template("""
        #     Bạn là trợ lý Flex sẽ dùng thông tin {docs} để trả lời câu hỏi của tôi bằng Tiếng Việt
        # """),
        # HumanMessagePromptTemplate.from_template("Who are you?"),
        # AIMessagePromptTemplate.from_template("I am Flex the assitant"),
        # HumanMessagePromptTemplate.from_template("Can you use vietnamese from now on?"),
        # AIMessagePromptTemplate.from_template("Được chứ từ giờ tôi sẽ dùng Tiếng Việt"),
        HumanMessagePromptTemplate.from_template("{question}?"),
    ]
)

question = 'Cách sử dụng tính năng “Yêu cầu thanh toán"'
# question = input("Hỏi tui đi: ")
print(question)
docs = vectorstore.similarity_search(question)

llm_chain = LLMChain(llm=llm, prompt=prompt)
result = llm_chain({"question": question, "docs": docs})
result["text"]