# Agentic AI Ebook — RAG Chatbot

Answers questions strictly from the Konverge.ai ebook *Agentic AI*
(https://konverge.ai/pdf/Ebook-Agentic-AI.pdf). Every response includes the
retrieved source chunks with page numbers and similarity scores, and a confidence value.

Stack: LangGraph (pipeline), Pinecone (vector database), FastAPI (API),
sentence-transformers `all-MiniLM-L6-v2` (embeddings), Groq `llama-3.3-70b-versatile` (generation).

## Setup

Requires Python 3.10+, a Pinecone API key, and a Groq API key (free tiers work).

1. Create a virtual environment and install dependencies:

       python -m venv .venv
       .venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
       pip install requests pypdf sentence-transformers pinecone python-dotenv langgraph groq fastapi uvicorn

2. Create `.env` in the project root:

       PINECONE_API_KEY=...
       GROQ_API_KEY=...

3. Ingest the PDF (downloads it, chunks, embeds, and upserts to Pinecone, then prints a test query):

       python ingest.py

4. Start the API:

       uvicorn api:app

   Then POST to `http://127.0.0.1:8000/chat`, or use the interactive docs at `/docs`:

       Request:  {"question": "What is agentic AI?"}
       Response: {
         "answer": "...",
         "chunks": [{"text": "...", "page_start": 3, "page_end": 4, "heading": "...", "score": 0.61}, ...],
         "confidence": 0.52
       }

5. Run the evaluation (writes `eval_results.md`):

       python evaluate.py

## Graph structure

    START → retrieve → grade ──(no chunk ≥ threshold)──────────────────────→ refuse → END
                         └──(≥1 relevant chunk)→ generate → check_grounding ─(not grounded)→ refuse
                                                                  └──(grounded)→ END

- **retrieve** — embeds the question with MiniLM and fetches the top 5 chunks from Pinecone by cosine similarity.
- **grade** — keeps chunks whose similarity is at least `RELEVANCE_THRESHOLD`. If none pass, the graph
  refuses without calling the LLM.
- **generate** — Llama 3.3 70B answers using only the relevant chunks, at temperature 0. It is told to
  reply with a fixed refusal sentence when the context lacks the answer, and to say which part is not
  covered when the context answers only part of the question.
- **check_grounding** — a second LLM call judges whether every claim in the answer is supported by the
  chunks (JSON output). Unsupported answers are replaced with the refusal.
- **refuse** — returns the fixed refusal message.

Refusal can therefore come from three layers: the similarity gate (off-topic questions), the generation
prompt (on-topic questions whose specific answer is absent), and the grounding check (hallucinations
that got past both).

## Why 250-token chunks with 25-token overlap

Tokens are counted with the embedding model's own WordPiece tokenizer. Chunks run continuously across
page boundaries and record the page range they span and the nearest preceding section heading
(detected by font size).

The first version used 500-token chunks with 50-token overlap. `all-MiniLM-L6-v2` reads at most 256
tokens and silently truncates longer input, so only about the first half of each chunk determined its
embedding. Text in the second half could only be retrieved if the first half also matched the query.

250 tokens fits under that limit, including the two special tokens the model adds, so every chunk is
embedded in full. The 25-token (10%) overlap keeps the same ratio: a sentence split at a boundary still
appears whole in one of the two chunks.

Comparing `eval_results_500.md` with `eval_results.md`, retrieval improved. The chunk introducing the
core components of agentic AI moved to rank 1 for that question, and author biographies dropped out of
the top 5 for every on-topic query. The table of contents was excluded from ingestion in the same run
(`SKIP_PAGES`), so the improvement comes from both changes together.

Trade-off: five chunks now supply about 1,250 tokens of context instead of about 2,500, and lists or
arguments longer than one chunk can be split so that only part of them is retrieved.


## What confidence does and doesn't measure

`confidence` is the mean cosine similarity between the question and the 5 retrieved chunks.

It **does** measure how closely the best-matching passages in the ebook relate to the question in the
embedding space. A high value means the document contains text on this topic.

It **does not** measure:
- **Correctness.** Retrieval can be close while the answer is still incomplete or wrong.
- **Grounding.** The grounding check is separate and is not reflected in this number.
- **Probability.** 0.6 does not mean a 60% chance of being right. The scores are uncalibrated and
  their typical range depends on the embedding model.
- **Coverage of the specific fact asked for.** An on-topic question about a detail the ebook omits
  can score moderately and still be refused.

It is also diluted by weak chunks, because all 5 are averaged including those below the relevance
threshold. And it reflects only the first 256 tokens of each chunk (see the caveat above).
In the evaluation, the partly covered question scored 0.643 even though half of it was unanswerable,
and the fully answered synthesis question scored 0.388.

## Limitations

- **Embedding truncation:** 500-token chunks exceed MiniLM's 256-token input limit (see above).
- **No OCR:** pages whose text exists only as images have no text layer and are invisible to the
  system. `ingest.py` prints how many pages had extractable text.
- **Heading detection is heuristic:** a line counts as a heading if its font is at least 1.2× the body
  size. Unusual layouts can mislabel headings. A chunk is labelled with the heading in effect at its
  first character, which may be the previous section when the chunk crosses a section boundary.
- **Page numbers are PDF page positions** (1-based), which may differ from numbers printed on the pages.
- **The relevance threshold is a single global value** set from a small evaluation set. It is not
  calibrated per query type.
- **The grounding check is self-evaluation:** the same model judges its own answer and may share its
  blind spots. It adds latency and token cost to every answered question.
- **Chunks can split mid-sentence.** Overlap reduces but does not remove this.
- **No conversation memory:** each question is answered independently.
- **Free-tier rate limits:** Groq's per-minute and per-day token limits slow down or block bursts of
  requests. The client retries on HTTP 429.
- **Single document, manual re-ingestion:** re-running `ingest.py` replaces the index contents.
- **No authentication** on the API.
- **The relevance threshold is only an off-topic filter (0.30).** In evaluation, an answerable synthesis
  question scored 0.464 at the top while an unanswerable one scored 0.416, and chunks the synthesis
  answer needed scored as low as 0.347. Raising the threshold would have refused or starved legitimate
  questions, so on-topic questions the document cannot answer are refused by the generator and
  grounding check instead.
- **Multi-section questions can be under-retrieved.** Five 250-token chunks cannot cover two whole
  sections. The synthesis question in the evaluation retrieved only part of section 2.5, and the
  answer omits the worker and utility agent types.
