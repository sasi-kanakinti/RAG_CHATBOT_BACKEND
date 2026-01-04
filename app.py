from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

from chatbot import run_chatbot

app = FastAPI(
    title="RAG ChatBot API",
    description="An API for the RAG ChatBot application",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow Vercel + local
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    answer = run_chatbot(
        query=request.query,
        history=request.history
    )
    return ChatResponse(response=answer)

@app.get("/health")
def health():
    return {"status": "ok"}