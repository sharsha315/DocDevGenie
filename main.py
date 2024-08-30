# Chat Retrieval from Single Page url

import os
import bs4
from groq import Groq
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.document_loaders.base import Document

# Load environment variables from .env file
load_dotenv()

# 1.1 - Indexing - Load

# Only keep post title, headers, and content from the full HTML.
#bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

url = "https://react.dev/reference/react/apis"
loader = WebBaseLoader(url)
docs = loader.load()

# 1.2 - Indexing - Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# 1.3 - Indexing - Store

# Embeddings
modelPath = "intfloat/e5-large-unsupervised"
embeddings = HuggingFaceEmbeddings(
  model_name = modelPath, 
  #encode_kwargs={'normalize_embeddings':False},
)

# vectorstore
vectorstore = Chroma.from_documents(
    documents=all_splits, 
    embedding=embeddings
)

# 2.1 - Retreival and Generation - Retrieve
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


# 2.2 - Generation

# Initializing GROQ API KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize ChatGroq
llm = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    api_key=GROQ_API_KEY
)

prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use one or two sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:"""

prompt = PromptTemplate.from_template(prompt_template)

# Retrieval Chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

#Query
query = "How to use act api"
result = qa_chain.invoke({"query": query})

print("answer:", result['result'])


