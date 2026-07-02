#!/usr/bin/env python3
"""
Complete ChromaDB ingestion pipeline for ClinicalRecall.

Ingests all three data sources into ChromaDB:
  1. Synthea FHIR bundles → synthea_structured
  2. Clinician notes → synthea_structured (same namespace)
  3. MTSamples transcriptions → mtsamples_knowledge
  4. Builds patient_index for sidebar

Uses Ollama local embedding with batch processing and checkpoint resume.
"""

import json
import os
import re
import sqlite3
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
import requests

# ── Configuration ──────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = "qwen3-embedding:8b"
BATCH_SIZE = 100
CHROMA_PATH = Path("chroma_db")
PROGRESS_DIR = Path("chroma_ingestion_progress")
FHIR_DIR = Path("data/fhir")
NOTES_DIR = Path("data/clinician_notes")
MTSAMPLES_DB = Path("data/mtsamples_staging.db")

# ── Initialize ────────────────────────────────────────────────────────────

chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)


def get_collection(name: str) -> chromadb.Collection:
    return chroma_client.get_or_create_collection(name)


def load_progress(name: str) -> set:
    path = PROGRESS_DIR / f"{name}.txt"
    if path.exists():
        with open(path) as f:
            return set(line.strip() for line in f)
    return set()


def save_progress(name: str, key: str):
    path = PROGRESS_DIR / f"{name}.txt"
    with open(path, "a") as f:
        f.write(f"{key}\n")


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via Ollama API with retry logic."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": EMBEDDING_MODEL, "input": texts},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["embeddings"]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                raise


def flush_batch(collection: chromadb.Collection, ids: list, docs: list, metas: list, embeddings: list):
    if not ids:
        return
    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)


# ── Progress Bar ───────────────────────────────────────────────────────────

class ProgressBar:
    """Renders a progress bar with percentage, count, elapsed time, and ETA."""

    def __init__(self, total: int, label: str = "", width: int = 40):
        self.total = total
        self.label = label
        self.width = width
        self.start_time = time.time()
        self.current = 0
        self._last_render = 0

    def update(self, n: int = 1):
        self.current += n
        now = time.time()
        if now - self._last_render < 0.5 and self.current < self.total:
            return
        self._last_render = now
        self._render()

    def _render(self):
        pct = min(self.current / self.total, 1.0) if self.total > 0 else 1.0
        filled = int(self.width * pct)
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = time.time() - self.start_time

        if self.current > 0 and pct < 1.0:
            eta_seconds = (elapsed / self.current) * (self.total - self.current)
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "0s"

        elapsed_str = self._format_time(elapsed)
        line = (
            f"\r  {self.label} [{bar}] {pct * 100:5.1f}% "
            f"({self.current}/{self.total}) "
            f"[{elapsed_str}<{eta_str}]"
        )
        sys.stdout.write(line)
        sys.stdout.flush()

    def done(self):
        self.current = self.total
        self._render()
        sys.stdout.write("\n")
        sys.stdout.flush()

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"{m}m{s}s"
        else:
            h, rem = divmod(int(seconds), 3600)
            m, s = divmod(rem, 60)
            return f"{h}h{m}m{s}s"


# ── 1. Synthea FHIR Ingestion ─────────────────────────────────────────────

def parse_synthea_bundle(filepath: Path) -> tuple[Optional[str], list[dict]]:
    with open(filepath) as f:
        bundle = json.load(f)

    entries = bundle.get("entry", [])
    patient_id = None
    patient_name = ""
    patient_dob = ""
    patient_gender = ""
    chunks = []

    for entry in entries:
        resource = entry.get("resource", {})
        rt = resource.get("resourceType")

        if rt == "Patient":
            patient_id = resource["id"]
            name = resource.get("name", [{}])[0]
            patient_name = f"{name.get('given', [''])[0]} {name.get('family', '')}"
            patient_dob = resource.get("birthDate", "")
            patient_gender = resource.get("gender", "")
            chunks.append({
                "text": f"Patient {patient_name}, DOB {patient_dob}, Gender {patient_gender}",
                "metadata": {
                    "patient_id": patient_id,
                    "patient_name": patient_name,
                    "data_type": "demographics",
                    "source": "synthea",
                    "date": patient_dob,
                    "chunk_id": str(uuid.uuid4()),
                },
            })

    if not patient_id:
        return None, []

    for entry in entries:
        resource = entry.get("resource", {})
        rt = resource.get("resourceType")

        if rt == "Condition":
            code = resource.get("code", {}).get("coding", [{}])[0].get("display", "")
            onset = resource.get("onsetDateTime", "")
            clinical_status = resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "")
            if code:
                chunks.append({
                    "text": f"Condition: {code}, onset {onset}, status {clinical_status}",
                    "metadata": {
                        "patient_id": patient_id,
                        "patient_name": patient_name,
                        "data_type": "condition",
                        "source": "synthea",
                        "date": onset,
                        "chunk_id": str(uuid.uuid4()),
                    },
                })

        elif rt == "MedicationRequest":
            med = resource.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("display", "")
            authored = resource.get("authoredOn", "")
            dosage = ""
            if resource.get("dosageInstruction"):
                dosage = resource["dosageInstruction"][0].get("text", "")
            status = resource.get("status", "")
            if med:
                chunks.append({
                    "text": f"Medication: {med}, dosage: {dosage}, prescribed {authored}, status {status}",
                    "metadata": {
                        "patient_id": patient_id,
                        "patient_name": patient_name,
                        "data_type": "medication",
                        "source": "synthea",
                        "date": authored,
                        "chunk_id": str(uuid.uuid4()),
                    },
                })

        elif rt == "Observation":
            code = resource.get("code", {}).get("coding", [{}])[0].get("display", "")
            vq = resource.get("valueQuantity", {})
            value = vq.get("value")
            unit = vq.get("unit", "")
            effective = resource.get("effectiveDateTime", "")
            if code and value is not None:
                chunks.append({
                    "text": f"Lab: {code} = {value} {unit} on {effective}",
                    "metadata": {
                        "patient_id": patient_id,
                        "patient_name": patient_name,
                        "data_type": "observation",
                        "source": "synthea",
                        "date": effective,
                        "chunk_id": str(uuid.uuid4()),
                    },
                })

        elif rt == "Encounter":
            enc_type = resource.get("type", [{}])[0].get("coding", [{}])[0].get("display", "")
            period = resource.get("period", {})
            start = period.get("start", "")
            reason = ""
            if resource.get("reasonCode"):
                reason = resource["reasonCode"][0].get("coding", [{}])[0].get("display", "")
            if enc_type:
                chunks.append({
                    "text": f"Encounter: {enc_type}, reason: {reason}, date {start}",
                    "metadata": {
                        "patient_id": patient_id,
                        "patient_name": patient_name,
                        "data_type": "encounter",
                        "source": "synthea",
                        "date": start,
                        "chunk_id": str(uuid.uuid4()),
                    },
                })

    return patient_id, chunks


def ingest_synthea():
    print("\n" + "=" * 60)
    print("INGESTING SYNTHEA FHIR BUNDLES")
    print("=" * 60)

    collection = get_collection("synthea_structured")
    completed = load_progress("synthea")
    fhir_files = sorted(FHIR_DIR.glob("*.json"))
    fhir_files = [f for f in fhir_files if "hospitalInformation" not in f.name
                  and "practitionerInformation" not in f.name]

    total = len(fhir_files)
    remaining = total - len(completed)
    print(f"Files: {total} total, {len(completed)} done, {remaining} remaining")

    batch_ids, batch_docs, batch_metas = [], [], []
    total_chunks = 0
    patient_index_entries = []
    errors = 0

    progress = ProgressBar(total, label="Parsing & embedding")

    for i, fpath in enumerate(fhir_files):
        patient_id_key = fpath.stem
        if patient_id_key in completed:
            progress.update()
            continue

        try:
            patient_id, chunks = parse_synthea_bundle(fpath)
            if not patient_id or not chunks:
                completed.add(patient_id_key)
                save_progress("synthea", patient_id_key)
                progress.update()
                continue

            conditions = [c for c in chunks if c["metadata"]["data_type"] == "condition"]
            encounters = [c for c in chunks if c["metadata"]["data_type"] == "encounter"]
            last_visit = max((c["metadata"]["date"] for c in encounters), default="")
            condition_summary = ", ".join(
                c["text"].replace("Condition: ", "").split(",")[0]
                for c in conditions[:5]
            )

            patient_index_entries.append({
                "patient_id": patient_id,
                "patient_name": chunks[0]["metadata"]["patient_name"],
                "conditions_summary": condition_summary,
                "last_visit_date": last_visit,
                "chunk_count": len(chunks),
            })

            for chunk in chunks:
                batch_ids.append(chunk["metadata"]["chunk_id"])
                batch_docs.append(chunk["text"])
                batch_metas.append(chunk["metadata"])

                if len(batch_ids) >= BATCH_SIZE:
                    embeddings = embed_batch(batch_docs)
                    flush_batch(collection, batch_ids, batch_docs, batch_metas, embeddings)
                    total_chunks += len(batch_ids)
                    batch_ids, batch_docs, batch_metas = [], [], []

            completed.add(patient_id_key)
            save_progress("synthea", patient_id_key)

        except Exception as e:
            errors += 1

        progress.update()

    progress.done()

    if batch_ids:
        embeddings = embed_batch(batch_docs)
        flush_batch(collection, batch_ids, batch_docs, batch_metas, embeddings)
        total_chunks += len(batch_ids)

    print(f"  Chunks embedded: {total_chunks}")
    if errors:
        print(f"  Errors: {errors}")

    if patient_index_entries:
        print("\n  Building patient_index...")
        idx_collection = get_collection("patient_index")
        idx_ids, idx_docs, idx_metas = [], [], []
        idx_progress = ProgressBar(len(patient_index_entries), label="Indexing patients")

        for entry in patient_index_entries:
            idx_ids.append(entry["patient_id"])
            idx_docs.append(
                f"Patient {entry['patient_name']}. "
                f"Conditions: {entry['conditions_summary']}. "
                f"Last visit: {entry['last_visit_date']}. "
                f"Records: {entry['chunk_count']} chunks."
            )
            idx_metas.append({
                "patient_id": entry["patient_id"],
                "patient_name": entry["patient_name"],
                "conditions_summary": entry["conditions_summary"],
                "last_visit_date": entry["last_visit_date"],
                "chunk_count": entry["chunk_count"],
            })

            if len(idx_ids) >= BATCH_SIZE:
                idx_embeddings = embed_batch(idx_docs)
                flush_batch(idx_collection, idx_ids, idx_docs, idx_metas, idx_embeddings)
                idx_progress.update(len(idx_ids))
                idx_ids, idx_docs, idx_metas = [], [], []

        if idx_ids:
            idx_embeddings = embed_batch(idx_docs)
            flush_batch(idx_collection, idx_ids, idx_docs, idx_metas, idx_embeddings)
            idx_progress.update(len(idx_ids))

        idx_progress.done()
        print(f"  {len(patient_index_entries)} patients indexed")


# ── 2. Clinician Notes Ingestion ──────────────────────────────────────────

def ingest_clinician_notes():
    print("\n" + "=" * 60)
    print("INGESTING CLINICIAN NOTES")
    print("=" * 60)

    collection = get_collection("synthea_structured")
    completed = load_progress("clinician_notes")
    notes_files = sorted(NOTES_DIR.glob("*.json"))

    total = len(notes_files)
    remaining = total - len(completed)
    print(f"Files: {total} total, {len(completed)} done, {remaining} remaining")

    batch_ids, batch_docs, batch_metas = [], [], []
    total_chunks = 0
    errors = 0

    progress = ProgressBar(total, label="Embedding notes")

    for i, fpath in enumerate(notes_files):
        patient_id = fpath.stem
        if patient_id in completed:
            progress.update()
            continue

        try:
            with open(fpath) as f:
                notes = json.load(f)

            for note in notes:
                chunk_id = note.get("note_id", str(uuid.uuid4()))
                batch_ids.append(chunk_id)
                batch_docs.append(note["note_text"])
                batch_metas.append({
                    "patient_id": note["patient_id"],
                    "patient_name": note.get("patient_name", ""),
                    "data_type": "clinician_note",
                    "note_type": note.get("note_type", ""),
                    "source": "clinician_note",
                    "date": note.get("timestamp", ""),
                    "chunk_id": chunk_id,
                })

                if len(batch_ids) >= BATCH_SIZE:
                    embeddings = embed_batch(batch_docs)
                    flush_batch(collection, batch_ids, batch_docs, batch_metas, embeddings)
                    total_chunks += len(batch_ids)
                    batch_ids, batch_docs, batch_metas = [], [], []

            completed.add(patient_id)
            save_progress("clinician_notes", patient_id)

        except Exception as e:
            errors += 1

        progress.update()

    progress.done()

    if batch_ids:
        embeddings = embed_batch(batch_docs)
        flush_batch(collection, batch_ids, batch_docs, batch_metas, embeddings)
        total_chunks += len(batch_ids)

    print(f"  Notes embedded: {total_chunks}")
    if errors:
        print(f"  Errors: {errors}")


# ── 3. MTSamples Ingestion ────────────────────────────────────────────────

def ingest_mtsamples():
    print("\n" + "=" * 60)
    print("INGESTING MTSAMPLES")
    print("=" * 60)

    if not MTSAMPLES_DB.exists():
        print(f"  ERROR: {MTSAMPLES_DB} not found. Run ingest_mtsamples.py first.")
        return

    collection = get_collection("mtsamples_knowledge")
    completed = load_progress("mtsamples_embed")

    conn = sqlite3.connect(str(MTSAMPLES_DB))
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    remaining = total - len(completed)
    print(f"Chunks: {total} total, {len(completed)} done, {remaining} remaining")

    batch_ids, batch_docs, batch_metas = [], [], []
    total_embedded = 0

    progress = ProgressBar(total, label="Embedding chunks")

    offset = 0
    limit = 500
    while offset < total:
        rows = conn.execute(
            "SELECT chunk_id, text, specialty, section, keywords, row_index FROM chunks "
            "ORDER BY row_index LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

        for row in rows:
            chunk_id, text, specialty, section, keywords_json, row_index = row
            if chunk_id in completed:
                progress.update()
                continue

            text = re.sub(r"^,\s*", "", text.strip())
            if not text:
                completed.add(chunk_id)
                save_progress("mtsamples_embed", chunk_id)
                progress.update()
                continue

            keywords = json.loads(keywords_json) if keywords_json else []

            batch_ids.append(chunk_id)
            batch_docs.append(text)
            batch_metas.append({
                "specialty": specialty,
                "section": section,
                "keywords": ", ".join(keywords[:10]),
                "source": "mtsamples",
                "chunk_id": chunk_id,
            })

            if len(batch_ids) >= BATCH_SIZE:
                embeddings = embed_batch(batch_docs)
                flush_batch(collection, batch_ids, batch_docs, batch_metas, embeddings)
                for cid in batch_ids:
                    completed.add(cid)
                    save_progress("mtsamples_embed", cid)
                total_embedded += len(batch_ids)
                progress.update(len(batch_ids))
                batch_ids, batch_docs, batch_metas = [], [], []

        offset += limit

    if batch_ids:
        embeddings = embed_batch(batch_docs)
        flush_batch(collection, batch_ids, batch_docs, batch_metas, embeddings)
        for cid in batch_ids:
            completed.add(cid)
            save_progress("mtsamples_embed", cid)
        total_embedded += len(batch_ids)
        progress.update(len(batch_ids))

    progress.done()
    conn.close()
    print(f"  Chunks embedded: {total_embedded}")


# ── 4. Create empty utility collections ────────────────────────────────────

def create_utility_collections():
    print("\n" + "=" * 60)
    print("CREATING UTILITY COLLECTIONS")
    print("=" * 60)
    get_collection("activity_log")
    get_collection("audit_log")
    print("  activity_log: ready")
    print("  audit_log: ready")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     ClinicalRecall — ChromaDB Ingestion Pipeline         ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Embedding model : {EMBEDDING_MODEL:<36}║")
    print(f"║  Batch size      : {BATCH_SIZE:<36}║")
    print(f"║  Ollama host     : {OLLAMA_BASE_URL:<36}║")
    print("╚══════════════════════════════════════════════════════════╝")

    overall_start = time.time()

    ingest_synthea()
    ingest_clinician_notes()
    ingest_mtsamples()
    create_utility_collections()

    overall_elapsed = time.time() - overall_start
    m, s = divmod(int(overall_elapsed), 60)
    h, m = divmod(m, 60)

    print(f"\n{'=' * 60}")
    print(f"INGESTION COMPLETE in {h}h {m}m {s}s")
    print(f"{'=' * 60}")

    for name in ["synthea_structured", "mtsamples_knowledge", "patient_index",
                 "activity_log", "audit_log"]:
        try:
            col = chroma_client.get_collection(name)
            print(f"  {name}: {col.count():,} chunks")
        except Exception:
            print(f"  {name}: NOT FOUND")


if __name__ == "__main__":
    main()
