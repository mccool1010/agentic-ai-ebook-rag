from fastapi import FastAPI
from pydantic import BaseModel, Field
from graph import rag_graph

app = FastAPI(title="Agentic AI Ebook RAG")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)


class Chunk(BaseModel):
    text: str
    page_start: int
    page_end: int
    heading: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    chunks: list[Chunk]
    confidence: float


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = rag_graph.invoke({"question": request.question})
    chunks = result["chunks"]
    confidence = sum(chunk["score"] for chunk in chunks) / len(chunks) if chunks else 0.0
    return ChatResponse(answer=result["answer"], chunks=chunks, confidence=round(confidence, 3))
