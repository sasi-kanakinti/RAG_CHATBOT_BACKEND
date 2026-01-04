import os
from dotenv import load_dotenv

from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceInferenceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.docstore.document import Document
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

# ==============================
# Lazy-loaded globals
# ==============================
_embeddings = None
_vector_store = None
_llm = None
_wiki = None

# ==============================
# Initializers (SAFE for Railway)
# ==============================
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceInferenceEmbeddings(
            api_key=os.environ["HF_API_KEY"],
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
    return _embeddings

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = AstraDBVectorStore(
            embedding=get_embeddings(),
            collection_name="rag_chunks",
        )
    return _vector_store

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
        )
    return _llm

def get_wiki():
    global _wiki
    if _wiki is None:
        _wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(
                top_k_results=2,
                max_summary_chars=1500,
            )
        )
    return _wiki

# ==============================
# Prompts
# ==============================
RAG_PROMPT = ChatPromptTemplate.from_template(
"""
You are a factual assistant.

Conversation history:
{history}

Answer the question using ONLY the context below.
If the answer is not present, say:
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

# ==============================
# Helpers
# ==============================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(history):
    return "\n".join(
        f"{msg['role'].upper()}: {msg['text']}"
        for msg in history
    )

def condense_query(query, history_text):
    if not history_text:
        return query

    llm = get_llm()
    condensed = llm.invoke(
        CONDENSE_PROMPT.format(
            history=history_text,
            question=query
        )
    )
    return condensed.content.strip()

# ==============================
# Answer Paths
# ==============================
def rag_answer(query, history_text):
    retriever = get_vector_store()
    llm = get_llm()

    standalone = condense_query(query, history_text)
    docs = retriever.similarity_search(standalone, k=4)
    context = format_docs(docs)

    resp = llm.invoke(
        RAG_PROMPT.format(
            history=history_text,
            context=context,
            question=query
        )
    )
    return resp.content.strip()

def wiki_answer(query, history_text):
    wiki = get_wiki()
    llm = get_llm()

    standalone = condense_query(query, history_text)
    text = wiki.run(standalone)

    docs = [Document(page_content=text)]
    context = format_docs(docs)

    resp = llm.invoke(
        WIKI_PROMPT.format(
            history=history_text,
            context=context,
            question=query
        )
    )
    return resp.content.strip()

def general_answer(query, history_text):
    llm = get_llm()
    resp = llm.invoke(
        GENERAL_PROMPT.format(
            history=history_text,
            question=query
        )
    )
    return resp.content.strip()

# ==============================
# Router
# ==============================
def run_chatbot(query: str, history: list) -> str:
    history_text = format_history(history)

    rag_resp = rag_answer(query, history_text)
    if "I do not know based on the provided documents." not in rag_resp:
        return rag_resp

    wiki_resp = wiki_answer(query, history_text)
    if wiki_resp and len(wiki_resp.split()) > 50:
        return wiki_resp

    return general_answer(query, history_text)
