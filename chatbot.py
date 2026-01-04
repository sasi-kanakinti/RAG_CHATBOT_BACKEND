import os
from dotenv import load_dotenv
from typing import List

from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.docstore.document import Document
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

# ------------------------
# Embeddings (HuggingFace)
# ------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------------
# AstraDB Vector Store
# ------------------------
vector_store = AstraDBVectorStore(
    embedding=embeddings,
    collection_name="rag_chunks",
)

def hybrid_retriever(query: str):
    return vector_store.similarity_search(
        query,
        k=4,
        search_type="hybrid"
    )

# ------------------------
# Wikipedia Retriever
# ------------------------
wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=2,
        max_summary_chars=1500
    )
)

def wiki_retriever(query: str):
    text = wiki.run(query)
    return [
        Document(
            page_content=text,
            metadata={"source": "wikipedia"}
        )
    ]

# ------------------------
# LLM (Groq)
# ------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

# ------------------------
# Prompts
# ------------------------
RAG_PROMPT = ChatPromptTemplate.from_template(
"""
You are a factual assistant.

Conversation history:
{history}

Answer the question using ONLY the context below.
If the answer is not present, say exactly:
"I do not know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
)

WIKI_PROMPT = ChatPromptTemplate.from_template(
"""
Conversation history:
{history}

Answer using the Wikipedia context below.
Be concise and factual.

Context:
{context}

Question:
{question}

Answer:
"""
)

GENERAL_PROMPT = ChatPromptTemplate.from_template(
"""
Conversation history:
{history}

You are a helpful general-purpose assistant.

Question:
{question}

Answer:
"""
)

CONDENSE_PROMPT = ChatPromptTemplate.from_template(
"""
Given the conversation history and the latest user question,
rewrite the question so it is standalone.

Conversation history:
{history}

User question:
{question}

Standalone question:
"""
)

# ------------------------
# Helpers
# ------------------------
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(history: list) -> str:
    return "\n".join(
        f"{msg['role'].upper()}: {msg['text']}"
        for msg in history
    )

def condense_query(query: str, history_text: str) -> str:
    if not history_text:
        return query

    condensed = llm.invoke(
        CONDENSE_PROMPT.format(
            history=history_text,
            question=query
        )
    )
    return condensed.content.strip()

# ------------------------
# Answer Paths
# ------------------------
def rag_answer(query: str, history_text: str) -> str:
    standalone = condense_query(query, history_text)
    docs = hybrid_retriever(standalone)
    context = format_docs(docs)

    resp = llm.invoke(
        RAG_PROMPT.format(
            history=history_text,
            context=context,
            question=query
        )
    )
    return resp.content.strip()

def wiki_answer(query: str, history_text: str) -> str:
    standalone = condense_query(query, history_text)
    docs = wiki_retriever(standalone)
    context = format_docs(docs)

    resp = llm.invoke(
        WIKI_PROMPT.format(
            history=history_text,
            context=context,
            question=query
        )
    )
    return resp.content.strip()

def general_answer(query: str, history_text: str) -> str:
    resp = llm.invoke(
        GENERAL_PROMPT.format(
            history=history_text,
            question=query
        )
    )
    return resp.content.strip()

# ------------------------
# Router (MAIN ENTRY)
# ------------------------
def run_chatbot(query: str, history: list) -> str:
    history_text = format_history(history)

    rag_resp = rag_answer(query, history_text)
    if "I do not know based on the provided documents." not in rag_resp:
        return rag_resp

    wiki_resp = wiki_answer(query, history_text)
    if wiki_resp and len(wiki_resp.split()) > 50:
        return wiki_resp

    return general_answer(query, history_text)
