import bisect
import math
import os
import time
from collections import Counter
from pathlib import Path

import requests
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


load_dotenv()


PDF_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
PDF_PATH = Path("data/Ebook-Agentic-AI.pdf")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 
EMBED_DIM = 384


CHUNK_TOKENS = 250 
OVERLAP_TOKENS = 25


SKIP_PAGES = {5}


HEADING_SIZE_RATIO = 1.2
MAX_HEADING_CHARS = 120


INDEX_NAME = "agentic-ai-ebook"
NAMESPACE = "ebook"
UPSERT_BATCH = 100


TEST_QUERY = "What is agentic AI and how does it differ from traditional AI?"

def download_pdf():
    if PDF_PATH.exists():
        print(f"Using cached PDF at {PDF_PATH}")
        return
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(PDF_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError("Downloaded file is not a PDF")
    PDF_PATH.write_bytes(response.content)
    print(f"Downloaded {len(response.content) / 1e6:.1f} MB to {PDF_PATH}")

def extract_lines(reader):
    lines = []
    for page_number, page in enumerate(reader.pages, start=1):
        runs = []
        if page_number in SKIP_PAGES:
            continue

        def visitor(text, cm, tm, font_dict, font_size):
            size = font_size * math.hypot(tm[2], tm[3]) * math.hypot(cm[2], cm[3])
            runs.append((text, size))

        page.extract_text(visitor_text=visitor)

        page_lines = []
        line_parts, line_size = [], 0.0
        for text, size in runs:
            for i, piece in enumerate(text.split("\n")):
                if i > 0:
                    page_lines.append(("".join(line_parts), line_size))
                    line_parts, line_size = [], 0.0
                line_parts.append(piece)
                if piece.strip():
                    line_size = max(line_size, size)
        page_lines.append(("".join(line_parts), line_size))

        for raw, size in page_lines:
            text = " ".join(raw.split())
            if text:
                lines.append({"page": page_number, "text": text, "size": round(size, 1)})
    return lines
def label_headings(lines):
    size_weights = Counter()
    for line in lines:
        size_weights[line["size"]] += len(line["text"])
    body_size = size_weights.most_common(1)[0][0]

    current_heading = "(no heading)"
    previous = None
    for line in lines:
        is_heading = (
            line["size"] >= body_size * HEADING_SIZE_RATIO
            and len(line["text"]) <= MAX_HEADING_CHARS
            and any(c.isalpha() for c in line["text"])
        )
        if is_heading:
            merged = f"{current_heading} {line['text']}"
            continues_previous = (
                previous is not None
                and previous["is_heading"]
                and previous["page"] == line["page"]
                and len(merged) <= MAX_HEADING_CHARS
            )
            current_heading = merged if continues_previous else line["text"]
        line["heading"] = current_heading
        line["is_heading"] = is_heading
        previous = line
    return body_size

def build_document(lines):
    parts, line_starts = [], []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        parts.append(line["text"])
        offset += len(line["text"]) + 1  # +1 for the joining newline
    return "\n".join(parts), line_starts

def chunk_document(document, line_starts, lines, tokenizer):
    encoding = tokenizer(
        document, add_special_tokens=False, return_offsets_mapping=True, verbose=False
    )
    offsets = encoding["offset_mapping"]
    step = CHUNK_TOKENS - OVERLAP_TOKENS

    chunks = []
    for start in range(0, len(offsets), step):
        end = min(start + CHUNK_TOKENS, len(offsets))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        first_line = lines[bisect.bisect_right(line_starts, char_start) - 1]
        last_line = lines[bisect.bisect_right(line_starts, char_end - 1) - 1]
        chunks.append({
            "text": document[char_start:char_end],
            "page_start": first_line["page"],
            "page_end": last_line["page"],
            "heading": first_line["heading"],
            "token_count": end - start,
        })
        if end == len(offsets):
            break
    return chunks

def get_index(pc):
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(INDEX_NAME)

def namespace_count(index):
    summary = index.describe_index_stats().namespaces.get(NAMESPACE)
    return summary.vector_count if summary else 0


def wait_for_count(index, expected, timeout=60):
    deadline = time.time() + timeout
    while namespace_count(index) != expected:
        if time.time() > deadline:
            raise TimeoutError(f"Namespace did not reach {expected} vectors within {timeout}s")
        time.sleep(2)


def upsert_chunks(index, chunks, embeddings):
    if NAMESPACE in index.describe_index_stats().namespaces:
        index.delete(delete_all=True, namespace=NAMESPACE)
        wait_for_count(index, 0)

    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"chunk-{i:04d}",
            "values": embedding.tolist(),
            "metadata": {
                "text": chunk["text"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "heading": chunk["heading"],
            },
        })

    for i in range(0, len(vectors), UPSERT_BATCH):
        index.upsert(vectors=vectors[i:i + UPSERT_BATCH], namespace=NAMESPACE)
    wait_for_count(index, len(vectors))
    print(f"Upserted {len(vectors)} vectors to {INDEX_NAME}/{NAMESPACE}")


def test_query(index, model):
    query_vector = model.encode(TEST_QUERY, normalize_embeddings=True).tolist()
    results = index.query(
        vector=query_vector, top_k=5, include_metadata=True, namespace=NAMESPACE
    )
    print(f"\nTest query: {TEST_QUERY}\n")
    for rank, match in enumerate(results.matches, start=1):
        meta = match.metadata
        print(
            f"#{rank}  score={match.score:.3f}  "
            f"pages {int(meta['page_start'])}-{int(meta['page_end'])}  "
            f"heading: {meta['heading']}"
        )
        print(meta["text"])
        print("-" * 80)


def main():
    download_pdf()
    reader = PdfReader(PDF_PATH)

    lines = extract_lines(reader)
    if not lines:
        raise ValueError("No extractable text found; the PDF may be scanned images")
    pages_with_text = {line["page"] for line in lines}
    print(f"{len(reader.pages)} pages, {len(pages_with_text)} with extractable text")

    body_size = label_headings(lines)
    headings = list(dict.fromkeys(line["heading"] for line in lines))
    print(f"Body font size {body_size}; {len(headings)} headings detected:")
    for heading in headings:
        print(f"  - {heading}")

    model = SentenceTransformer(EMBED_MODEL)
    document, line_starts = build_document(lines)
    chunks = chunk_document(document, line_starts, lines, model.tokenizer)
    print(f"{len(chunks)} chunks")

    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = get_index(pc)
    upsert_chunks(index, chunks, embeddings)
    test_query(index, model)


if __name__ == "__main__":
    main()