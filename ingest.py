import os
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    WebBaseLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceInferenceEmbeddings
from langchain_astradb.vectorstores import AstraDBVectorStore

load_dotenv()

# ------------------------
# Load Documents
# ------------------------
pdf_docs = PyPDFDirectoryLoader("data").load()

web_docs = WebBaseLoader(
    web_paths=[
        "https://example.com",
        "https://en.wikipedia.org/wiki/Python_(programming_language)"
    ]
).load()

for d in pdf_docs:
    d.metadata["source_type"] = "pdf"

for d in web_docs:
    d.metadata["source_type"] = "web"

all_docs = pdf_docs + web_docs

# ------------------------
# Split
# ------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(all_docs)

# ------------------------
# Vector Store (HF API)
# ------------------------
embeddings = HuggingFaceInferenceEmbeddings(
    api_key=os.environ["HF_API_KEY"],
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

vector_store = AstraDBVectorStore(
    embedding=embeddings,
    collection_name="rag_chunks",
)

vector_store.add_documents(chunks)

print("Ingestion complete.")
