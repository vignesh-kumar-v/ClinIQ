"""
Ingest Synthea FHIR bundles + clinician notes into ChromaDB.

Creates two collections:
  - synthea_structured : per-patient clinical chunks (conditions, meds, encounters, notes)
  - patient_index      : one summary document per patient for fast patient lookup

Usage:
    python embed_patients.py [--model qwen3-embedding:8b] [--batch-size 16] [--reset]

Prerequisites:
    pip install chromadb ollama tqdm
    ollama pull qwen3-embedding:8b
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import chromadb
import ollama
from tqdm import tqdm

FHIR_PATH = Path(__file__).parent / "data" / "fhir"
NOTES_PATH = Path(__file__).parent / "data" / "clinician_notes"
CHROMA_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_STRUCTURED = "synthea_structured"
COLLECTION_INDEX = "patient_index"
MAX_CHARS = 3000


# ── helpers ──────────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    text = text.lstrip(",").lstrip(" ")
    return text[:MAX_CHARS] if len(text) > MAX_CHARS else text


def get_name(patient: dict) -> str:
    names = patient.get("name", [])
    if not names:
        return "Unknown"
    n = names[0]
    given = " ".join(n.get("given", []))
    family = n.get("family", "")
    return f"{given} {family}".strip()


def calc_age(dob: str, deceased: str | None) -> int:
    if not dob:
        return 0
    from datetime import date
    birth = date.fromisoformat(dob)
    end = date.fromisoformat(deceased[:10]) if deceased else date.today()
    return (end - birth).days // 365


def decode_docref(resource: dict) -> str:
    try:
        b64 = resource["content"][0]["attachment"].get("data", "")
        return base64.b64decode(b64).decode("utf-8") if b64 else ""
    except Exception:
        return ""


# ── FHIR extraction ───────────────────────────────────────────────────────────

def extract_patient_chunks(fhir_file: Path) -> tuple[list[dict], dict | None]:
    """
    Returns:
        chunks  : list of {id, text, metadata}
        summary : dict for patient_index (or None on error)
    """
    try:
        data = json.loads(fhir_file.read_text())
    except Exception:
        return [], None

    resources_by_type: dict[str, list] = {}
    for entry in data.get("entry", []):
        res = entry.get("resource", {})
        rt = res.get("resourceType", "")
        resources_by_type.setdefault(rt, []).append(res)

    patient_list = resources_by_type.get("Patient", [])
    if not patient_list:
        return [], None
    patient = patient_list[0]

    pid = patient["id"]
    name = get_name(patient)
    dob = patient.get("birthDate", "")
    gender = patient.get("gender", "")
    deceased = patient.get("deceasedDateTime")
    age = calc_age(dob, deceased)

    chunks = []

    # ── demographics chunk ────────────────────────────────────────────────
    demo_text = (
        f"Patient: {name}. "
        f"DOB: {dob}. Gender: {gender}. Age: {age}. "
        f"Status: {'deceased' if deceased else 'alive'}."
    )
    chunks.append({
        "id": f"{pid}__demographics",
        "text": demo_text,
        "metadata": {
            "patient_id": pid,
            "patient_name": name,
            "chunk_type": "demographics",
            "source": "fhir",
            "date": dob,
        },
    })

    # ── conditions chunk ──────────────────────────────────────────────────
    conditions = resources_by_type.get("Condition", [])
    if conditions:
        active, resolved = [], []
        for c in conditions:
            display = (c.get("code", {}).get("coding", [{}])[0].get("display", "") or
                       c.get("code", {}).get("text", ""))
            status = c.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "")
            onset = c.get("onsetDateTime", "")[:10]
            entry_str = f"{display} (onset: {onset})"
            (active if status == "active" else resolved).append(entry_str)
        cond_text = f"Patient {name} conditions. "
        if active:
            cond_text += "Active: " + "; ".join(active) + ". "
        if resolved:
            cond_text += "Resolved/inactive: " + "; ".join(resolved[:10]) + "."
        chunks.append({
            "id": f"{pid}__conditions",
            "text": clean(cond_text),
            "metadata": {
                "patient_id": pid,
                "patient_name": name,
                "chunk_type": "conditions",
                "source": "fhir",
                "date": dob,
            },
        })

    # ── medications chunk ─────────────────────────────────────────────────
    med_requests = resources_by_type.get("MedicationRequest", [])
    if med_requests:
        active_meds, past_meds = [], []
        for m in med_requests:
            display = (m.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("display", "") or
                       m.get("medicationCodeableConcept", {}).get("text", ""))
            status = m.get("status", "")
            (active_meds if status == "active" else past_meds).append(display)
        med_text = f"Patient {name} medications. "
        if active_meds:
            med_text += "Current: " + "; ".join(active_meds) + ". "
        if past_meds:
            med_text += "Past: " + "; ".join(past_meds[:10]) + "."
        chunks.append({
            "id": f"{pid}__medications",
            "text": clean(med_text),
            "metadata": {
                "patient_id": pid,
                "patient_name": name,
                "chunk_type": "medications",
                "source": "fhir",
                "date": dob,
            },
        })

    # ── encounters chunk ──────────────────────────────────────────────────
    encounters = resources_by_type.get("Encounter", [])
    if encounters:
        enc_lines = []
        for enc in encounters[-20:]:  # last 20 encounters
            etype = (enc.get("type", [{}])[0].get("coding", [{}])[0].get("display", "") or
                     enc.get("type", [{}])[0].get("text", ""))
            period = enc.get("period", {}).get("start", "")[:10]
            reason_list = enc.get("reasonCode", [])
            reason = reason_list[0].get("coding", [{}])[0].get("display", "") if reason_list else ""
            enc_lines.append(f"{period}: {etype}" + (f" for {reason}" if reason else ""))
        enc_text = f"Patient {name} encounter history. " + ". ".join(enc_lines) + "."
        chunks.append({
            "id": f"{pid}__encounters",
            "text": clean(enc_text),
            "metadata": {
                "patient_id": pid,
                "patient_name": name,
                "chunk_type": "encounters",
                "source": "fhir",
                "date": encounters[-1].get("period", {}).get("start", "")[:10] if encounters else "",
            },
        })

    # ── document references (clinical notes) ─────────────────────────────
    docrefs = resources_by_type.get("DocumentReference", [])
    for i, doc in enumerate(docrefs):
        text = decode_docref(doc)
        if not text.strip():
            continue
        date = doc.get("date", "")[:10]
        chunks.append({
            "id": f"{pid}__docref_{i}",
            "text": clean(text),
            "metadata": {
                "patient_id": pid,
                "patient_name": name,
                "chunk_type": "clinical_note",
                "source": "fhir_docref",
                "date": date,
            },
        })

    # ── summary for patient_index ─────────────────────────────────────────
    active_condition_names = []
    for c in conditions:
        display = (c.get("code", {}).get("coding", [{}])[0].get("display", "") or
                   c.get("code", {}).get("text", ""))
        status = c.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "")
        if status == "active" and display:
            active_condition_names.append(display)

    active_med_names = []
    for m in med_requests:
        display = (m.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("display", "") or
                   m.get("medicationCodeableConcept", {}).get("text", ""))
        if m.get("status") == "active" and display:
            active_med_names.append(display)

    last_enc_date = ""
    if encounters:
        last_enc_date = max(
            (e.get("period", {}).get("start", "") for e in encounters), default=""
        )[:10]

    summary_text = (
        f"{name}, {age} year old {gender}. "
        f"DOB {dob}. {'Deceased.' if deceased else ''} "
        f"Active conditions: {', '.join(active_condition_names) or 'none'}. "
        f"Current medications: {', '.join(active_med_names) or 'none'}. "
        f"Last encounter: {last_enc_date or 'unknown'}. "
        f"Total encounters: {len(encounters)}. "
        f"Total conditions: {len(conditions)}."
    )

    summary = {
        "id": pid,
        "text": clean(summary_text),
        "metadata": {
            "patient_id": pid,
            "patient_name": name,
            "dob": dob,
            "gender": gender,
            "age": age,
            "deceased": deceased or "",
            "condition_count": len(conditions),
            "active_condition_count": len(active_condition_names),
            "med_count": len(med_requests),
            "active_med_count": len(active_med_names),
            "encounter_count": len(encounters),
            "last_encounter": last_enc_date,
            "source": "fhir",
        },
    }

    return chunks, summary


def load_clinician_notes(notes_file: Path, patient_name: str) -> list[dict]:
    try:
        notes = json.loads(notes_file.read_text())
    except Exception:
        return []

    pid = notes_file.stem
    chunks = []
    for i, note in enumerate(notes):
        text = note.get("note_text", "").strip()
        if not text:
            continue
        chunks.append({
            "id": f"{pid}__clinician_note_{i}",
            "text": clean(text),
            "metadata": {
                "patient_id": pid,
                "patient_name": note.get("patient_name", patient_name),
                "chunk_type": "clinician_note",
                "note_type": note.get("note_type", ""),
                "source": "clinician_note",
                "date": note.get("timestamp", "")[:10],
            },
        })
    return chunks


# ── embedding ─────────────────────────────────────────────────────────────────

def embed_batch(model: str, texts: list[str]) -> list[list[float]]:
    response = ollama.embed(model=model, input=texts)
    return response["embeddings"]


def upsert_batch(collection, ids, texts, embeddings, metadatas):
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-embedding:8b")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--reset", action="store_true")
    return p.parse_args()


def run(args):
    # verify Ollama
    try:
        models = [m.model for m in ollama.list().models]
    except Exception as e:
        print(f"ERROR: Cannot reach Ollama — {e}")
        sys.exit(1)
    if args.model not in models:
        print(f"ERROR: Model '{args.model}' not found. Pull: ollama pull {args.model}")
        sys.exit(1)

    # setup ChromaDB
    CHROMA_PATH.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    if args.reset:
        for name in [COLLECTION_STRUCTURED, COLLECTION_INDEX]:
            try:
                client.delete_collection(name)
                print(f"Dropped '{name}'")
            except Exception:
                pass

    col_structured = client.get_or_create_collection(
        COLLECTION_STRUCTURED, metadata={"hnsw:space": "cosine"}
    )
    col_index = client.get_or_create_collection(
        COLLECTION_INDEX, metadata={"hnsw:space": "cosine"}
    )

    # existing IDs for resume
    existing_structured = set(col_structured.get(include=[])["ids"])
    existing_index = set(col_index.get(include=[])["ids"])
    print(f"synthea_structured: {len(existing_structured)} already embedded")
    print(f"patient_index:      {len(existing_index)} already embedded")

    # collect all FHIR files
    fhir_files = sorted(FHIR_PATH.glob("*.json"))
    print(f"FHIR files found: {len(fhir_files)}")

    all_structured_chunks = []
    index_chunks = []

    print("Extracting from FHIR bundles...")
    for fhir_file in tqdm(fhir_files, desc="Parsing FHIR", unit="patient"):
        chunks, summary = extract_patient_chunks(fhir_file)

        # structured chunks (skip already embedded)
        for c in chunks:
            if c["id"] not in existing_structured:
                all_structured_chunks.append(c)

        # patient index
        if summary and summary["id"] not in existing_index:
            # load clinician notes for this patient
            notes_file = NOTES_PATH / f"{summary['metadata']['patient_id']}.json"
            if notes_file.exists():
                note_chunks = load_clinician_notes(notes_file, summary["metadata"]["patient_name"])
                for nc in note_chunks:
                    if nc["id"] not in existing_structured:
                        all_structured_chunks.append(nc)
            index_chunks.append(summary)

    print(f"synthea_structured chunks to embed: {len(all_structured_chunks):,}")
    print(f"patient_index entries to embed:     {len(index_chunks):,}")

    bs = args.batch_size

    # ── embed synthea_structured ──────────────────────────────────────────
    if all_structured_chunks:
        with tqdm(total=len(all_structured_chunks), desc="synthea_structured", unit="chunk") as bar:
            for i in range(0, len(all_structured_chunks), bs):
                batch = all_structured_chunks[i: i + bs]
                texts = [c["text"] for c in batch]
                try:
                    embeddings = embed_batch(args.model, texts)
                except Exception as e:
                    print(f"\nBatch failed ({e}), 1-by-1 fallback...")
                    embeddings = []
                    for t in texts:
                        try:
                            embeddings.extend(embed_batch(args.model, [t]))
                        except Exception:
                            embeddings.append(None)
                    valid = [(c, e) for c, e in zip(batch, embeddings) if e is not None]
                    if not valid:
                        bar.update(len(batch))
                        continue
                    batch, embeddings = zip(*valid)
                    batch, embeddings = list(batch), list(embeddings)
                    texts = [c["text"] for c in batch]

                upsert_batch(
                    col_structured,
                    [c["id"] for c in batch],
                    texts,
                    embeddings,
                    [c["metadata"] for c in batch],
                )
                bar.update(len(batch))

    # ── embed patient_index ───────────────────────────────────────────────
    if index_chunks:
        with tqdm(total=len(index_chunks), desc="patient_index", unit="patient") as bar:
            for i in range(0, len(index_chunks), bs):
                batch = index_chunks[i: i + bs]
                texts = [c["text"] for c in batch]
                try:
                    embeddings = embed_batch(args.model, texts)
                except Exception as e:
                    print(f"\nBatch failed ({e}), 1-by-1 fallback...")
                    embeddings = []
                    for t in texts:
                        try:
                            embeddings.extend(embed_batch(args.model, [t]))
                        except Exception:
                            embeddings.append(None)
                    valid = [(c, e) for c, e in zip(batch, embeddings) if e is not None]
                    if not valid:
                        bar.update(len(batch))
                        continue
                    batch, embeddings = zip(*valid)
                    batch, embeddings = list(batch), list(embeddings)
                    texts = [c["text"] for c in batch]

                upsert_batch(
                    col_index,
                    [c["id"] for c in batch],
                    texts,
                    embeddings,
                    [c["metadata"] for c in batch],
                )
                bar.update(len(batch))

    print(f"\nDone.")
    print(f"synthea_structured: {col_structured.count():,} vectors")
    print(f"patient_index:      {col_index.count():,} vectors")


if __name__ == "__main__":
    run(parse_args())
