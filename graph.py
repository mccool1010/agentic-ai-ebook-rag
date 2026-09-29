"""Phase 2: LangGraph pipeline - retrieve, grade, generate, check grounding."""

import json
import os
import sys
from typing import TypedDict

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, START, StateGraph
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from ingest import EMBED_MODEL, INDEX_NAME, NAMESPACE

load_dotenv()

LLM_MODEL = "openai/gpt-oss-120b"

TOP_K = 5
# Minimum cosine similarity for a chunk to count as relevant. If no retrieved chunk
# reaches it, the graph refuses without calling the LLM. In evaluation, an answerable
# synthesis question scored only 0.464 at the top, and chunks it needed scored
# 0.347-0.376, while the unanswerable question scored 0.416. Similarity cannot
# separate the two, so the threshold is kept low as an off-topic filter only;
# on-topic questions the document cannot answer are refused by the LLM layers.


RELEVANCE_THRESHOLD = 0.30

REFUSAL_MESSAGE = "I can't answer that from the provided document."

GENERATE_PROMPT = f"""You answer questions about an ebook on agentic AI using only the context passages provided.
Rules:
- Use only facts stated in the context. Do not use outside knowledge, even if you know the answer.
- If the context does not contain the answer, reply with exactly this sentence and nothing else: {REFUSAL_MESSAGE}
- If the context answers only part of the question, answer that part and state plainly which part the document does not cover.
- Be concise."""

GROUNDING_PROMPT = """You check whether an answer is fully supported by context passages.
An answer is grounded only if every factual claim in it is stated in, or directly follows from, the context.
Statements that the document does not cover something are acceptable.
Respond with JSON: {"grounded": true} or {"grounded": false}."""

embedder = SentenceTransformer(EMBED_MODEL)
index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(INDEX_NAME)
llm = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=5)


class RAGState(TypedDict):
    question: str
    chunks: list[dict]
    relevant_chunks: list[dict]
    answer: str
    grounded: bool


def retrieve(state):
    vector = embedder.encode(state["question"], normalize_embeddings=True).tolist()
    results = index.query(vector=vector, top_k=TOP_K, include_metadata=True, namespace=NAMESPACE)
    chunks = [
        {
            "text": match.metadata["text"],
            "page_start": int(match.metadata["page_start"]),
            "page_end": int(match.metadata["page_end"]),
            "heading": match.metadata["heading"],
            "score": match.score,
        }
        for match in results.matches
    ]
    return {"chunks": chunks}


def grade(state):
    relevant = [chunk for chunk in state["chunks"] if chunk["score"] >= RELEVANCE_THRESHOLD]
    return {"relevant_chunks": relevant}


def route_after_grade(state):
    return "generate" if state["relevant_chunks"] else "refuse"


def format_context(chunks):
    return "\n\n".join(
        f"[Source {i}, pages {c['page_start']}-{c['page_end']}, section: {c['heading']}]\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )


def generate(state):
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": GENERATE_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{format_context(state['relevant_chunks'])}\n\nQuestion: {state['question']}",
            },
        ],
    )
    return {"answer": response.choices[0].message.content.strip()}


def check_grounding(state):
    # A refusal makes no claims, so there is nothing to check.
    if state["answer"] == REFUSAL_MESSAGE:
        return {"grounded": True}
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": GROUNDING_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{format_context(state['relevant_chunks'])}\n\nAnswer:\n{state['answer']}",
            },
        ],
    )
    verdict = json.loads(response.choices[0].message.content)
    return {"grounded": verdict.get("grounded") is True}


def route_after_grounding(state):
    return "end" if state["grounded"] else "refuse"


def refuse(state):
    return {"answer": REFUSAL_MESSAGE}


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("generate", generate)
    graph.add_node("check_grounding", check_grounding)
    graph.add_node("refuse", refuse)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {"generate": "generate", "refuse": "refuse"})
    graph.add_edge("generate", "check_grounding")
    graph.add_conditional_edges("check_grounding", route_after_grounding, {"end": END, "refuse": "refuse"})
    graph.add_edge("refuse", END)
    return graph.compile()


rag_graph = build_graph()


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is agentic AI?"
    result = rag_graph.invoke({"question": question})
    print(result["answer"])
    print()
    for chunk in result["chunks"]:
        print(f"{chunk['score']:.3f}  pages {chunk['page_start']}-{chunk['page_end']}  {chunk['heading']}")


