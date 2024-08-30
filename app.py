import os
from groq import Groq
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from apify_client import ApifyClient
from langchain_community.document_loaders import ApifyDatasetLoader
from langchain_core.document_loaders.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Initializing APIFY API TOKEN
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")

url = "https://react.dev/reference/react"

apify = ApifyClient(APIFY_API_TOKEN)

apify_actor = apify.actor('apify/website-content-crawler').call(
    run_input = {"startUrls": [{"url": url}]}
)

# Save data to vector database
loader = ApifyDatasetLoader(
    dataset_id=apify_actor['defaultDatasetId'],
    dataset_mapping_function=lambda item: Document(
        page_content=item['text'] or '', metadata={'source': item['url']}
    ),
)

# Fetch data from an existing Apify dataset.
# loader = ApifyDatasetLoader(
#     dataset_id="your datasetID", # datasetID is obtained from Apify console after running Apify actor
#     dataset_mapping_function=lambda item: Document(
#         page_content=item["Text"] or "", metadata={"source": item["Link"]}
#     ),
# )

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Embeddings
modelPath = "intfloat/e5-large-unsupervised"
embeddings = HuggingFaceEmbeddings(
  model_name = modelPath, 
  encode_kwargs={'normalize_embeddings':False},
)

# vectorstore
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings,
    #persist_directory='db2',
)
#vectorstore.persist()

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

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

# Main function
def main():
    os.system("clear")
    print("Welcome!!!")
    print("I am DocDevGenie - Always-Updated API Assistance for Developers")
    print("\nAsk your question (or type 'exit' to quit): ")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit":
            print("Goodbye!!!")
            break
        else:
            result = qa_chain.invoke({"query": user_input})
            print("ChatBot:", result['result'])

if __name__ == "__main__":
    main()