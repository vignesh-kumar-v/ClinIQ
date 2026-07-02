"""
Embed mtsamples chunks from SQLite into persistent ChromaDB using Ollama qwen3-embedding.

Usage:
    python embed_chunks.py [--model qwen3-embedding:4b] [--batch-size 32] [--reset]

Prerequisites:
    pip install chromadb ollama tqdm
    ollama pull qwen3-embedding:4b
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import chromadb
import ollama
from tqdm import tqdm
from normalize_sections import normalize_section

DB_PATH = Path(__file__).parent / "data" / "mtsamples_staging.db"
CHROMA_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "mtsamples_chunks"
MAX_CHARS = 3000  # ~750 tokens — safe under qwen3-embedding 4096 ctx limit


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-embedding:4b")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--reset", action="store_true", help="Drop and recreate collection")
    return p.parse_args()


def load_chunks(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(
        "SELECT chunk_id, text, specialty, section, keywords, source, row_index FROM chunks"
    )
    rows = []
    for row in cur.fetchall():
        chunk_id, text, specialty, section, keywords_raw, source, row_index = row
        try:
            kw_list = json.loads(keywords_raw) if keywords_raw else []
            keywords_str = ", ".join(str(k) for k in kw_list)
        except (json.JSONDecodeError, TypeError):
            keywords_str = str(keywords_raw or "")
        rows.append({
            "chunk_id": chunk_id,
            "text": text or "",
            "specialty": specialty or "",
            "section": normalize_section(section or ""),
            "keywords": keywords_str,
            "source": source or "mtsamples",
            "row_index": int(row_index or 0),
        })
    return rows


def get_existing_ids(collection: chromadb.Collection) -> set[str]:
    result = collection.get(include=[])
    return set(result["ids"])


def clean(text: str) -> str:
    text = text.lstrip(",").lstrip(" ")
    return text[:MAX_CHARS] if len(text) > MAX_CHARS else text


def embed_batch(model: str, texts: list[str]) -> list[list[float]]:
    response = ollama.embed(model=model, input=[clean(t) for t in texts])
    return response["embeddings"]


def run(args):
    # verify Ollama reachable and model available
    try:
        models = [m.model for m in ollama.list().models]
    except Exception as e:
        print(f"ERROR: Cannot reach Ollama — {e}")
        print("Start Ollama: `ollama serve`")
        sys.exit(1)

    if args.model not in models:
        print(f"ERROR: Model '{args.model}' not found in Ollama.")
        print(f"Available: {models}")
        print(f"Pull with: ollama pull {args.model}")
        sys.exit(1)

    # connect SQLite
    conn = sqlite3.connect(DB_PATH)
    print(f"Loading chunks from {DB_PATH} ...")
    chunks = load_chunks(conn)
    conn.close()
    print(f"  {len(chunks):,} chunks loaded")

    # setup ChromaDB
    CHROMA_PATH.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    if args.reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Dropped collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # skip already-embedded chunks (resume support)
    existing_ids = get_existing_ids(collection)
    if existing_ids:
        print(f"  {len(existing_ids):,} already embedded — skipping")
    pending = [c for c in chunks if c["chunk_id"] not in existing_ids]
    print(f"  {len(pending):,} chunks to embed")

    if not pending:
        print("Nothing to do. Use --reset to re-embed.")
        return

    # embed and upsert in batches
    bs = args.batch_size
    total_batches = (len(pending) + bs - 1) // bs

    with tqdm(total=len(pending), unit="chunk", desc="Embedding") as bar:
        for i in range(0, len(pending), bs):
            batch = pending[i : i + bs]
            texts = [c["text"] for c in batch]

            try:
                embeddings = embed_batch(args.model, texts)
            except Exception as e:
                print(f"\nBatch {i//bs}/{total_batches} failed ({e}), falling back to 1-by-1 ...")
                embeddings = []
                for t in texts:
                    try:
                        embeddings.extend(embed_batch(args.model, [t]))
                    except Exception as e2:
                        print(f"  Skipping chunk (embed failed): {e2}")
                        embeddings.append(None)
                # filter out failed chunks
                valid = [(c, emb) for c, emb in zip(batch, embeddings) if emb is not None]
                if not valid:
                    bar.update(len(batch))
                    continue
                batch, embeddings = zip(*valid)
                batch = list(batch)
                embeddings = list(embeddings)
                texts = [c["text"] for c in batch]

            collection.upsert(
                ids=[c["chunk_id"] for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "specialty": c["specialty"],
                        "section": c["section"],
                        "keywords": c["keywords"],
                        "source": c["source"],
                        "row_index": c["row_index"],
                    }
                    for c in batch
                ],
            )
            bar.update(len(batch))

    final_count = collection.count()
    print(f"\nDone. Collection '{COLLECTION_NAME}' has {final_count:,} vectors.")
    print(f"ChromaDB path: {CHROMA_PATH.resolve()}")


if __name__ == "__main__":
    run(parse_args())
