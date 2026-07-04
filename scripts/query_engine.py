#!/usr/bin/env python3
"""
ClinicalRecall RAG Query Engine.

Two modes:
  - Patient-scoped: /patient <name>  → searches synthea_structured + mtsamples_knowledge
  - General:        any query        → searches mtsamples_knowledge only

Uses Ollama for both embedding (qwen3-embedding:0.6b) and generation (qwen3.5:9b-mlx).
"""

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Optional

import chromadb
import requests

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
CHAT_MODEL = "qwen3.5:9b-mlx"
CHROMA_PATH = Path("chroma_db")
TOP_K_PATIENT = 20
TOP_K_KNOWLEDGE = 10

chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

SYSTEM_PROMPT = textwrap.dedent("""\
    You are ClinicalRecall, an AI clinical decision support assistant.
    Answer the clinician's question using ONLY the provided context below.
    If the context does not contain enough information, say so clearly.
    Cite specific data from the context (lab values, dates, medication names).
    Be concise and clinically precise. Do not hallucinate.
    Format your answer with clear sections when appropriate.
""")

PATIENT_SYSTEM_PROMPT = textwrap.dedent("""\
    You are ClinicalRecall, an AI clinical decision support assistant.
    Answer the clinician's question about the specified patient using ONLY the
    provided context below. The context includes the patient's medical records
    (demographics, conditions, medications, lab results, encounters, clinician notes)
    and relevant general medical knowledge.

    If the context does not contain enough information, say so clearly.
    Cite specific data from the context (lab values, dates, medication names).
    Be concise and clinically precise. Do not hallucinate.
    Format your answer with clear sections when appropriate.
""")


def embed_query(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": [text]},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def generate(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def find_patient(name_query: str) -> Optional[dict]:
    """Fuzzy search patient_index by name. Returns best match or None."""
    col = chroma_client.get_collection("patient_index")
    embedding = embed_query(name_query)
    results = col.query(query_embeddings=[embedding], n_results=5)
    if not results["ids"][0]:
        return None
    best = results["metadatas"][0][0]
    return {
        "patient_id": best["patient_id"],
        "patient_name": best["patient_name"],
        "conditions_summary": best.get("conditions_summary", ""),
        "last_visit_date": best.get("last_visit_date", ""),
    }


def retrieve_patient_context(patient_id: str, query: str) -> list[dict]:
    """Search synthea_structured filtered by patient_id."""
    col = chroma_client.get_collection("synthea_structured")
    embedding = embed_query(query)
    results = col.query(
        query_embeddings=[embedding],
        n_results=TOP_K_PATIENT,
        where={"patient_id": patient_id},
    )
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        chunks.append({"text": doc, "metadata": meta})
    return chunks


def retrieve_knowledge(query: str, k: int = TOP_K_KNOWLEDGE) -> list[dict]:
    """Search mtsamples_knowledge for general clinical knowledge."""
    col = chroma_client.get_collection("mtsamples_knowledge")
    embedding = embed_query(query)
    results = col.query(query_embeddings=[embedding], n_results=k)
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        chunks.append({"text": doc, "metadata": meta})
    return chunks


def format_context(chunks: list[dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        source = meta.get("source", "unknown")
        dtype = meta.get("data_type", meta.get("section", ""))
        date = meta.get("date", "")
        lines.append(f"[{i}] [{source}/{dtype}] {date} | {chunk['text']}")
    return "\n".join(lines)


def build_patient_prompt(patient: dict, query: str, context: str) -> str:
    return textwrap.dedent(f"""\
        Patient: {patient['patient_name']} (ID: {patient['patient_id']})
        Known conditions: {patient['conditions_summary'][:300]}
        Last visit: {patient['last_visit_date']}

        Context from patient records and medical knowledge:
        {context}

        Clinician's question: {query}
    """)


def build_general_prompt(query: str, context: str) -> str:
    return textwrap.dedent(f"""\
        Context from medical knowledge base:
        {context}

        Clinician's question: {query}
    """)


def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          ClinicalRecall — RAG Query Engine              ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Commands:                                              ║")
    print("║    /patient <name>  —  switch to patient-scoped mode    ║")
    print("║    /clear            —  switch to general mode          ║")
    print("║    /quit             —  exit                            ║")
    print("║    /help             —  show this help                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def main():
    print_banner()

    current_patient: Optional[dict] = None

    while True:
        try:
            if current_patient:
                prompt_prefix = f"[{current_patient['patient_name']}]"
            else:
                prompt_prefix = "[general]"
            user_input = input(f"\n{prompt_prefix} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit":
                print("Goodbye.")
                break
            elif cmd == "/help":
                print_banner()
            elif cmd == "/clear":
                current_patient = None
                print("  Switched to general mode.")
            elif cmd == "/patient":
                if not arg:
                    print("  Usage: /patient <name>")
                    continue
                print(f"  Searching for patient: {arg}...")
                patient = find_patient(arg)
                if patient:
                    current_patient = patient
                    print(f"  Found: {patient['patient_name']}")
                    print(f"  Conditions: {patient['conditions_summary'][:200]}...")
                    print(f"  Last visit: {patient['last_visit_date']}")
                else:
                    print("  No matching patient found.")
            else:
                print(f"  Unknown command: {cmd}")
            continue

        print("  Retrieving context...", end="", flush=True)

        if current_patient:
            patient_chunks = retrieve_patient_context(
                current_patient["patient_id"], user_input
            )
            knowledge_chunks = retrieve_knowledge(user_input, k=5)
            all_chunks = patient_chunks + knowledge_chunks
        else:
            all_chunks = retrieve_knowledge(user_input, k=TOP_K_KNOWLEDGE)

        context = format_context(all_chunks)
        print(f" {len(all_chunks)} chunks found.")

        print("  Generating answer...", end="", flush=True)

        if current_patient:
            prompt = build_patient_prompt(current_patient, user_input, context)
            system = PATIENT_SYSTEM_PROMPT
        else:
            prompt = build_general_prompt(user_input, context)
            system = SYSTEM_PROMPT

        answer = generate(prompt, system=system)
        print(" done.\n")
        print(answer)


if __name__ == "__main__":
    main()
