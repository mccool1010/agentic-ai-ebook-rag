

from datetime import datetime
from pathlib import Path

from api import ChatRequest, chat
from graph import REFUSAL_MESSAGE, RELEVANCE_THRESHOLD, TOP_K

OUTPUT_PATH = Path("eval_results.md")

QUERIES = [
    ("Straightforward", "What is agentic AI?"),
    ("Straightforward", "What are the core components of an AI agent?"),
    ("Straightforward", "What business benefits of agentic AI does the ebook describe?"),
    (
        "Synthesis across sections",
        "How do the atomic agent types described in section 2.5 work together in the supply chain scenario in section 3.2?",
    ),
    (
        "Partly covered",
        "What key steps does the ebook recommend for starting with agentic AI, and what budget should a company allocate to each step?",
    ),
    ("Not answerable", "What was Konverge.ai's annual revenue last year?"),
]



def snippet(text, length=200):
    flat = " ".join(text.split()).replace("|", "\\|")
    return flat[:length] + ("..." if len(flat) > length else "")


def main():
    results = []
    for category, question in QUERIES:
        print(f"Running: {question}")
        results.append((category, question, chat(ChatRequest(question=question))))

    lines = [
        "# Evaluation results",
        "",
        f"Run at {datetime.now():%Y-%m-%d %H:%M}. Relevance threshold {RELEVANCE_THRESHOLD}, top-k {TOP_K}.",
        "",
        "| # | Category | Question | Refused | Confidence | Top score |",
        "|---|---|---|---|---|---|",
    ]
    for i, (category, question, response) in enumerate(results, start=1):
        refused = "yes" if response.answer == REFUSAL_MESSAGE else "no"
        top_score = max((chunk.score for chunk in response.chunks), default=0.0)
        lines.append(
            f"| {i} | {category} | {snippet(question, 120)} | {refused} | {response.confidence:.3f} | {top_score:.3f} |"
        )

    for i, (category, question, response) in enumerate(results, start=1):
        lines += [
            "",
            f"## {i}. {category}",
            "",
            f"**Question:** {question}",
            "",
            "**Answer:**",
            "",
            response.answer,
            "",
            f"**Confidence:** {response.confidence:.3f}",
            "",
            "| Rank | Score | Pages | Heading | Text (start) |",
            "|---|---|---|---|---|",
        ]
        for rank, chunk in enumerate(response.chunks, start=1):
            lines.append(
                f"| {rank} | {chunk.score:.3f} | {chunk.page_start}-{chunk.page_end} "
                f"| {snippet(chunk.heading, 60)} | {snippet(chunk.text)} |"
            )

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {OUTPUT_PATH}")

    if results[-1][2].answer != REFUSAL_MESSAGE:
        print("WARNING: the unanswerable query was not refused")


if __name__ == "__main__":
    main()
