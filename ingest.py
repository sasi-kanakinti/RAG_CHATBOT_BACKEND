from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    WebBaseLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_astradb.vectorstores import AstraDBVectorStore
from dotenv import load_dotenv

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
# Vector Store
# ------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = AstraDBVectorStore(
    embedding=embeddings,
    collection_name="rag_chunks",
)

vector_store.add_documents(chunks)

print("Ingestion complete.")