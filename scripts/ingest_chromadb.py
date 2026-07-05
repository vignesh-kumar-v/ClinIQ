#!/usr/bin/env python3
"""
ClinicalRecall — Master ChromaDB Ingestion Pipeline.

Runs everything in order:
  1. Normalize section names in SQLite (mtsamples)
  2. Embed Synthea FHIR bundles + clinician notes → synthea_structured, patient_index
  3. Embed MTSamples chunks → mtsamples_knowledge
  4. Setup logs DB (activity_log, audit_log)

Uses sentence-transformers (Qwen/Qwen3-Embedding-0.6B) for local embeddings.
Aligned with the ClinIQ backend.

Usage:
    python ingest_chromadb.py [--batch-size 100] [--reset]
    python ingest_chromadb.py --only normalize
    python ingest_chromadb.py --only patients
    python ingest_chromadb.py --only mtsamples
    python ingest_chromadb.py --only logs

Prerequisites:
    pip install chromadb sentence-transformers torch
"""

import argparse
import base64
import json
import os
import re
import sqlite3
import sys
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
import torch
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────────────────────

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 100
CHROMA_PATH = Path("chroma_db")
PROGRESS_DIR = Path("chroma_ingestion_progress")
FHIR_DIR = Path("data/fhir")
NOTES_DIR = Path("data/clinician_notes")
MTSAMPLES_DB = Path("data/mtsamples_staging.db")
LOGS_DB = Path("data/logs.db")
MAX_CHARS = 3000

_embed_model: Optional[SentenceTransformer] = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model {EMBED_MODEL_NAME} on {device}...")
        _embed_model = SentenceTransformer(
            EMBED_MODEL_NAME,
            device=device,
            model_kwargs={"dtype": torch.float16, "attn_implementation": "sdpa"}
            if device == "cuda" else {},
        )
        _embed_model.max_seq_length = 1024
        dim = _embed_model.get_sentence_embedding_dimension()
        print(f"  Loaded. Dimension: {dim}")
    return _embed_model

# ── Initialize ────────────────────────────────────────────────────────────

chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)


def get_collection(name: str) -> chromadb.Collection:
    return chroma_client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


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


def clean(text: str) -> str:
    text = text.lstrip(",").lstrip(" ")
    return text[:MAX_CHARS] if len(text) > MAX_CHARS else text


def embed_batch(texts: list[str]) -> list[list[float]]:
    model = _get_embed_model()
    vecs = model.encode(
        [clean(t) for t in texts],
        batch_size=min(BATCH_SIZE, len(texts)),
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
    return vecs


def safe_embed_and_upsert(collection, batch_ids, batch_docs, batch_metas):
    if not batch_ids:
        return
    try:
        embeddings = embed_batch(batch_docs)
    except Exception as e:
        print(f"\n  Batch embed failed ({e}), falling back to 1-by-1...")
        embeddings = []
        for t in batch_docs:
            try:
                embeddings.extend(embed_batch([t]))
            except Exception:
                embeddings.append(None)
        valid = [(i, d, m, e) for i, d, m, e in zip(batch_ids, batch_docs, batch_metas, embeddings) if e is not None]
        if not valid:
            return
        batch_ids, batch_docs, batch_metas, embeddings = zip(*valid)
        batch_ids, batch_docs, batch_metas = list(batch_ids), list(batch_docs), list(batch_metas)
        embeddings = list(embeddings)
    collection.upsert(ids=batch_ids, embeddings=embeddings, documents=batch_docs, metadatas=batch_metas)


# ── Progress Bar ───────────────────────────────────────────────────────────

class ProgressBar:
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
        bar = "\u2588" * filled + "\u2591" * (self.width - filled)
        elapsed = time.time() - self.start_time
        if self.current > 0 and pct < 1.0:
            eta_seconds = (elapsed / self.current) * (self.total - self.current)
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "0s"
        rate = self.current / elapsed if elapsed > 0 else 0
        elapsed_str = self._format_time(elapsed)
        line = (
            f"\r  {self.label} [{bar}] {pct * 100:5.1f}% "
            f"({self.current}/{self.total}) "
            f"[{elapsed_str}<{eta_str}, {rate:.1f}/s]"
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


# ── Section Normalization ─────────────────────────────────────────────────

SECTION_MAP = {
    "HPI": "HISTORY OF PRESENT ILLNESS",
    "HISTORY": "HISTORY OF PRESENT ILLNESS",
    "PRESENT ILLNESS": "HISTORY OF PRESENT ILLNESS",
    "HISTORY OF ILLNESS": "HISTORY OF PRESENT ILLNESS",
    "HISTORY OF PRESENT COMPLAINT": "HISTORY OF PRESENT ILLNESS",
    "HISTORY OF PRESENT PROBLEM": "HISTORY OF PRESENT ILLNESS",
    "HISTORY OF PRESENT INJURY": "HISTORY OF PRESENT ILLNESS",
    "HISTORY OF PRESENTING COMPLAINT": "HISTORY OF PRESENT ILLNESS",
    "HISTORY OF PRESENTING ILLNESS": "HISTORY OF PRESENT ILLNESS",
    "HISTORY OF PRESENTING PROBLEM": "HISTORY OF PRESENT ILLNESS",
    "HISTORY OF THE PRESENT ILLNESS": "HISTORY OF PRESENT ILLNESS",
    "BRIEF HISTORY OF PRESENT ILLNESS": "HISTORY OF PRESENT ILLNESS",
    "CURRENT HISTORY OF PRESENT ILLNESS": "HISTORY OF PRESENT ILLNESS",
    "CURRENT HISTORY": "HISTORY OF PRESENT ILLNESS",
    "INTERVAL HISTORY": "HISTORY OF PRESENT ILLNESS",
    "INTERIM HISTORY": "HISTORY OF PRESENT ILLNESS",
    "PRESENT COMPLAINTS": "HISTORY OF PRESENT ILLNESS",
    "PRESENT PROBLEMS": "HISTORY OF PRESENT ILLNESS",
    "PRESENTING PROBLEM": "HISTORY OF PRESENT ILLNESS",
    "PRESENTING PROBLEMS": "HISTORY OF PRESENT ILLNESS",
    "BRIEF HISTORY": "HISTORY OF PRESENT ILLNESS",
    "CLINICAL HISTORY": "HISTORY OF PRESENT ILLNESS",
    "PMH": "PAST MEDICAL HISTORY",
    "PAST MEDICAL HX": "PAST MEDICAL HISTORY",
    "PAST HISTORY": "PAST MEDICAL HISTORY",
    "PREVIOUS MEDICAL HISTORY": "PAST MEDICAL HISTORY",
    "PRIOR MEDICAL HISTORY": "PAST MEDICAL HISTORY",
    "PERTINENT MEDICAL HISTORY": "PAST MEDICAL HISTORY",
    "MEDICAL HISTORY": "PAST MEDICAL HISTORY",
    "ADULT MEDICAL PROBLEMS": "PAST MEDICAL HISTORY",
    "PAST MEDICAL CONDITIONS": "PAST MEDICAL HISTORY",
    "PRIMARY MEDICAL HISTORY": "PAST MEDICAL HISTORY",
    "PATIENT HISTORY": "PAST MEDICAL HISTORY",
    "PAST MEDICAL AND SURGICAL HISTORY": "PAST MEDICAL HISTORY",
    "PAST MEDICAL/SURGICAL HISTORY": "PAST MEDICAL HISTORY",
    "SIGNIFICANT PAST MEDICAL AND SURGICAL HISTORY": "PAST MEDICAL HISTORY",
    "OTHER SIGNIFICANT MEDICAL HISTORY/SURGERIES": "PAST MEDICAL HISTORY",
    "PAST MEDICAL HISTORY/SURGERIES/HOSPITALIZATIONS": "PAST MEDICAL HISTORY",
    "PAST MEDICAL HISTORY / SURGERY / HOSPITALIZATIONS": "PAST MEDICAL HISTORY",
    "PSH": "PAST SURGICAL HISTORY",
    "PAST SURGICAL HX": "PAST SURGICAL HISTORY",
    "SURGICAL HISTORY": "PAST SURGICAL HISTORY",
    "PREVIOUS SURGICAL HISTORY": "PAST SURGICAL HISTORY",
    "PREVIOUS SURGERIES": "PAST SURGICAL HISTORY",
    "PRIOR SURGERIES": "PAST SURGICAL HISTORY",
    "PRIOR SURGERIES AND INTERVENTIONS": "PAST SURGICAL HISTORY",
    "PAST SURGERIES": "PAST SURGICAL HISTORY",
    "SOCIAL HX": "SOCIAL HISTORY",
    "SOCIAL": "SOCIAL HISTORY",
    "SOCIAL FACTORS": "SOCIAL HISTORY",
    "PERSONAL AND SOCIAL HISTORY": "SOCIAL HISTORY",
    "PERSONAL/SOCIAL HISTORY": "SOCIAL HISTORY",
    "PERSONAL HISTORY": "SOCIAL HISTORY",
    "FAMILY HX": "FAMILY HISTORY",
    "FAMILY": "FAMILY HISTORY",
    "FAMILY MEDICAL HISTORY": "FAMILY HISTORY",
    "FH": "FAMILY HISTORY",
    "MEDS": "MEDICATIONS",
    "MEDICATION": "MEDICATIONS",
    "CURRENT MEDICATIONS": "MEDICATIONS",
    "HOME MEDICATIONS": "MEDICATIONS",
    "MEDICATIONS ON ADMISSION": "MEDICATIONS",
    "MEDICATIONS AT HOME": "MEDICATIONS",
    "MEDICATIONS PRIOR TO ADMISSION": "MEDICATIONS",
    "DISCHARGE MEDICATIONS": "MEDICATIONS",
    "ALLERGIES": "ALLERGIES",
    "ALLERGY": "ALLERGIES",
    "KNOWN ALLERGIES": "ALLERGIES",
    "DRUG ALLERGIES": "ALLERGIES",
    "PHYSICAL EXAM": "PHYSICAL EXAMINATION",
    "PHYSICAL EXAMINATION": "PHYSICAL EXAMINATION",
    "EXAM": "PHYSICAL EXAMINATION",
    "EXAMINATION": "PHYSICAL EXAMINATION",
    "PE": "PHYSICAL EXAMINATION",
    "PHYSICAL EXAM FINDINGS": "PHYSICAL EXAMINATION",
    "VITALS": "VITAL SIGNS",
    "VITAL SIGNS": "VITAL SIGNS",
    "VS": "VITAL SIGNS",
    "ROS": "REVIEW OF SYSTEMS",
    "REVIEW OF SYSTEMS": "REVIEW OF SYSTEMS",
    "SYSTEMS REVIEW": "REVIEW OF SYSTEMS",
    "CC": "CHIEF COMPLAINT",
    "CHIEF COMPLAINT": "CHIEF COMPLAINT",
    "COMPLAINT": "CHIEF COMPLAINT",
    "PRESENTING COMPLAINT": "CHIEF COMPLAINT",
    "REASON FOR VISIT": "CHIEF COMPLAINT",
    "REASON FOR CONSULT": "CHIEF COMPLAINT",
    "REASON FOR CONSULTATION": "CHIEF COMPLAINT",
    "ASSESSMENT": "ASSESSMENT AND PLAN",
    "PLAN": "ASSESSMENT AND PLAN",
    "ASSESSMENT AND PLAN": "ASSESSMENT AND PLAN",
    "A/P": "ASSESSMENT AND PLAN",
    "ASSESSMENT & PLAN": "ASSESSMENT AND PLAN",
    "ASSESSMENT/PLAN": "ASSESSMENT AND PLAN",
    "IMPRESSION": "IMPRESSION",
    "IMPRESSIONS": "IMPRESSION",
    "DIAGNOSIS": "DIAGNOSES",
    "DIAGNOSES": "DIAGNOSES",
    "PRIMARY DIAGNOSIS": "DIAGNOSES",
    "PRINCIPAL DIAGNOSIS": "DIAGNOSES",
    "FINAL DIAGNOSIS": "DIAGNOSES",
    "FINAL DIAGNOSES": "DIAGNOSES",
    "PREOPERATIVE DIAGNOSIS": "PREOPERATIVE DIAGNOSIS",
    "PREOP DIAGNOSIS": "PREOPERATIVE DIAGNOSIS",
    "POSTOPERATIVE DIAGNOSIS": "POSTOPERATIVE DIAGNOSIS",
    "POSTOP DIAGNOSIS": "POSTOPERATIVE DIAGNOSIS",
    "PROCEDURE": "PROCEDURE PERFORMED",
    "PROCEDURE PERFORMED": "PROCEDURE PERFORMED",
    "OPERATION": "PROCEDURE PERFORMED",
    "OPERATIVE PROCEDURE": "PROCEDURE PERFORMED",
    "SURGICAL PROCEDURE": "PROCEDURE PERFORMED",
    "DESCRIPTION OF PROCEDURE": "DESCRIPTION OF PROCEDURE",
    "PROCEDURE DETAILS": "DESCRIPTION OF PROCEDURE",
    "OPERATIVE NOTE": "DESCRIPTION OF PROCEDURE",
    "OPERATIVE REPORT": "DESCRIPTION OF PROCEDURE",
    "NEURO": "NEUROLOGICAL",
    "NEUROLOGICAL": "NEUROLOGICAL",
    "NEUROLOGIC": "NEUROLOGICAL",
    "LABS": "LABORATORY DATA",
    "LAB": "LABORATORY DATA",
    "LABORATORY": "LABORATORY DATA",
    "LABORATORY DATA": "LABORATORY DATA",
    "LABORATORY STUDIES": "LABORATORY DATA",
    "LABORATORY RESULTS": "LABORATORY DATA",
    "HOSPITAL COURSE": "HOSPITAL COURSE",
    "COURSE": "HOSPITAL COURSE",
    "CLINICAL COURSE": "HOSPITAL COURSE",
    "DISCHARGE DIAGNOSES": "DISCHARGE DIAGNOSES",
    "DISCHARGE DIAGNOSIS": "DISCHARGE DIAGNOSES",
    "INDICATIONS": "INDICATIONS",
    "INDICATION": "INDICATIONS",
    "FINDINGS": "FINDINGS",
    "HEENT": "HEENT",
    "CV": "CARDIOVASCULAR",
    "CARDIOVASCULAR": "CARDIOVASCULAR",
    "CARDIAC": "CARDIOVASCULAR",
    "RESP": "RESPIRATORY",
    "RESPIRATORY": "RESPIRATORY",
    "PULMONARY": "RESPIRATORY",
    "LUNGS": "RESPIRATORY",
    "CHEST": "RESPIRATORY",
    "MSK": "MUSCULOSKELETAL",
    "MUSCULOSKELETAL": "MUSCULOSKELETAL",
    "EXTREMITIES": "MUSCULOSKELETAL",
    "GU": "GENITOURINARY",
    "GENITOURINARY": "GENITOURINARY",
    "GI": "GASTROINTESTINAL",
    "GASTROINTESTINAL": "GASTROINTESTINAL",
    "ABDOMEN": "GASTROINTESTINAL",
    "ABDOMINAL": "GASTROINTESTINAL",
    "PSYCH": "PSYCHIATRIC",
    "PSYCHIATRIC": "PSYCHIATRIC",
    "PSYCHIATRY": "PSYCHIATRIC",
    "SKIN": "SKIN",
    "DERMATOLOGIC": "SKIN",
    "DERMATOLOGY": "SKIN",
    "HEME": "HEMATOLOGY",
    "HEMATOLOGY": "HEMATOLOGY",
    "RECOMMENDATIONS": "RECOMMENDATIONS",
    "RECOMMENDATION": "RECOMMENDATIONS",
    "SUMMARY": "SUMMARY",
    "ANESTHESIA": "ANESTHESIA",
    "ANESTHETIC": "ANESTHESIA",
    "EBL": "ESTIMATED BLOOD LOSS",
    "ESTIMATED BLOOD LOSS": "ESTIMATED BLOOD LOSS",
    "BLOOD LOSS": "ESTIMATED BLOOD LOSS",
    "FLUIDS": "FLUIDS",
    "IV FLUIDS": "FLUIDS",
    "DISPOSITION": "DISPOSITION",
    "DISCHARGE DISPOSITION": "DISPOSITION",
    "RADIOLOGY": "RADIOLOGY",
    "IMAGING": "RADIOLOGY",
    "TECHNIQUE": "TECHNIQUE",
    "SUBJECTIVE": "SUBJECTIVE",
    "FOLLOW-UP": "FOLLOW-UP",
    "FOLLOWUP": "FOLLOW-UP",
    "FOLLOW UP": "FOLLOW-UP",
    "SPECIMENS": "SPECIMENS",
    "SPECIMEN": "SPECIMENS",
    "PATHOLOGY": "SPECIMENS",
}


def normalize_section(section: str) -> str:
    return SECTION_MAP.get(section.upper().strip(), section)


# ── FHIR Helpers ──────────────────────────────────────────────────────────

def get_patient_name(patient: dict) -> str:
    names = patient.get("name", [])
    if not names:
        return "Unknown"
    n = names[0]
    given = " ".join(n.get("given", []))
    family = n.get("family", "")
    return f"{given} {family}".strip()


def calc_age(dob: str, deceased: Optional[str]) -> int:
    if not dob:
        return 0
    birth = date.fromisoformat(dob)
    end = date.fromisoformat(deceased[:10]) if deceased else date.today()
    return (end - birth).days // 365


def decode_docref(resource: dict) -> str:
    try:
        b64 = resource["content"][0]["attachment"].get("data", "")
        return base64.b64decode(b64).decode("utf-8") if b64 else ""
    except Exception:
        return ""


# ── 0. Section Normalization ───────────────────────────────────────────────

def step_normalize_sections():
    print("\n" + "=" * 60)
    print("STEP 0: NORMALIZING MTSAMPLES SECTION NAMES")
    print("=" * 60)

    if not MTSAMPLES_DB.exists():
        print(f"  SKIP: {MTSAMPLES_DB} not found.")
        return

    conn = sqlite3.connect(str(MTSAMPLES_DB))
    before = conn.execute("SELECT COUNT(DISTINCT section) FROM chunks").fetchone()[0]
    print(f"  Distinct sections before: {before}")

    rows = conn.execute("SELECT rowid, section FROM chunks").fetchall()
    updated = 0
    for rowid, section in rows:
        normalized = normalize_section(section or "")
        if normalized != section:
            conn.execute("UPDATE chunks SET section = ? WHERE rowid = ?", (normalized, rowid))
            updated += 1

    conn.commit()
    after = conn.execute("SELECT COUNT(DISTINCT section) FROM chunks").fetchone()[0]
    conn.close()

    print(f"  Updated {updated:,} rows")
    print(f"  Distinct sections after: {after}")


# ── 1. Synthea FHIR + Clinician Notes ─────────────────────────────────────

def parse_synthea_bundle(filepath: Path) -> tuple[Optional[str], list[dict], Optional[dict]]:
    with open(filepath) as f:
        bundle = json.load(f)

    entries = bundle.get("entry", [])
    patient_id = None
    patient_name = ""
    patient_dob = ""
    patient_gender = ""
    patient_deceased = None
    chunks = []

    for entry in entries:
        resource = entry.get("resource", {})
        rt = resource.get("resourceType")
        if rt == "Patient":
            patient_id = resource["id"]
            patient_name = get_patient_name(resource)
            patient_dob = resource.get("birthDate", "")
            patient_gender = resource.get("gender", "")
            patient_deceased = resource.get("deceasedDateTime")
            age = calc_age(patient_dob, patient_deceased)
            status = "deceased" if patient_deceased else "alive"
            chunks.append({
                "text": f"Patient {patient_name}, DOB {patient_dob}, Gender {patient_gender}, Age {age}, Status {status}",
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
        return None, [], None

    conditions_active = []
    conditions_inactive = []
    medications_active = []
    medications_past = []
    encounters_list = []

    for entry in entries:
        resource = entry.get("resource", {})
        rt = resource.get("resourceType")

        if rt == "Condition":
            code = resource.get("code", {}).get("coding", [{}])[0].get("display", "")
            onset = resource.get("onsetDateTime", "")
            clinical_status = resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "")
            if code:
                if clinical_status in ("active", "recurrence", "relapse"):
                    conditions_active.append(f"{code} (onset {onset})")
                else:
                    conditions_inactive.append(f"{code} (onset {onset}, status {clinical_status})")

        elif rt == "MedicationRequest":
            med = resource.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("display", "")
            authored = resource.get("authoredOn", "")
            dosage = ""
            if resource.get("dosageInstruction"):
                dosage = resource["dosageInstruction"][0].get("text", "")
            status = resource.get("status", "")
            if med:
                entry_text = f"{med}, dosage: {dosage}, prescribed {authored}, status {status}"
                if status == "active":
                    medications_active.append(entry_text)
                else:
                    medications_past.append(entry_text)

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
                encounters_list.append(f"{enc_type} ({reason}) on {start}")

        elif rt == "DocumentReference":
            doc_text = decode_docref(resource)
            if doc_text:
                doc_date = resource.get("date", "")
                chunks.append({
                    "text": clean(doc_text),
                    "metadata": {
                        "patient_id": patient_id,
                        "patient_name": patient_name,
                        "data_type": "document_reference",
                        "source": "synthea",
                        "date": doc_date,
                        "chunk_id": str(uuid.uuid4()),
                    },
                })

    if conditions_active:
        chunks.append({
            "text": f"Active conditions of patient {patient_name}:\n- " + "\n- ".join(conditions_active),
            "metadata": {
                "patient_id": patient_id,
                "patient_name": patient_name,
                "data_type": "condition",
                "source": "synthea",
                "date": "",
                "chunk_id": str(uuid.uuid4()),
            },
        })
    if conditions_inactive:
        chunks.append({
            "text": f"Resolved/inactive conditions of patient {patient_name}:\n- " + "\n- ".join(conditions_inactive),
            "metadata": {
                "patient_id": patient_id,
                "patient_name": patient_name,
                "data_type": "condition",
                "source": "synthea",
                "date": "",
                "chunk_id": str(uuid.uuid4()),
            },
        })
    if medications_active:
        chunks.append({
            "text": f"Current medications of patient {patient_name}:\n- " + "\n- ".join(medications_active),
            "metadata": {
                "patient_id": patient_id,
                "patient_name": patient_name,
                "data_type": "medication",
                "source": "synthea",
                "date": "",
                "chunk_id": str(uuid.uuid4()),
            },
        })
    if medications_past:
        chunks.append({
            "text": f"Past medications of patient {patient_name}:\n- " + "\n- ".join(medications_past),
            "metadata": {
                "patient_id": patient_id,
                "patient_name": patient_name,
                "data_type": "medication",
                "source": "synthea",
                "date": "",
                "chunk_id": str(uuid.uuid4()),
            },
        })
    if encounters_list:
        chunks.append({
            "text": f"Encounters for patient {patient_name} (last 20):\n- " + "\n- ".join(encounters_list[-20:]),
            "metadata": {
                "patient_id": patient_id,
                "patient_name": patient_name,
                "data_type": "encounter",
                "source": "synthea",
                "date": "",
                "chunk_id": str(uuid.uuid4()),
            },
        })

    last_visit = encounters_list[-1].split(" on ")[-1] if encounters_list else ""
    condition_summary = ", ".join(c.split(" (")[0] for c in conditions_active[:5])

    summary = {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "age": calc_age(patient_dob, patient_deceased),
        "gender": patient_gender,
        "active_conditions": ", ".join(c.split(" (")[0] for c in conditions_active),
        "current_medications": ", ".join(m.split(",")[0] for m in medications_active),
        "last_encounter_date": last_visit,
        "chunk_count": len(chunks),
    }

    return patient_id, chunks, summary


def step_embed_patients():
    print("\n" + "=" * 60)
    print("STEP 1: EMBEDDING SYNTHEA FHIR + CLINICIAN NOTES")
    print("=" * 60)

    collection = get_collection("synthea_structured")
    completed = load_progress("synthea")
    fhir_files = sorted(FHIR_DIR.glob("*.json"))
    fhir_files = [f for f in fhir_files if "hospitalInformation" not in f.name
                  and "practitionerInformation" not in f.name]

    total = len(fhir_files)
    remaining = total - len(completed)
    print(f"FHIR files: {total} total, {len(completed)} done, {remaining} remaining")

    batch_ids, batch_docs, batch_metas = [], [], []
    total_chunks = 0
    patient_index_entries = []
    errors = 0

    progress = ProgressBar(total, label="Parsing & embedding")

    for fpath in fhir_files:
        patient_id_key = fpath.stem
        if patient_id_key in completed:
            progress.update()
            continue

        try:
            patient_id, chunks, summary = parse_synthea_bundle(fpath)
            if not patient_id or not chunks:
                completed.add(patient_id_key)
                save_progress("synthea", patient_id_key)
                progress.update()
                continue

            if summary:
                patient_index_entries.append(summary)

            for chunk in chunks:
                batch_ids.append(chunk["metadata"]["chunk_id"])
                batch_docs.append(chunk["text"])
                batch_metas.append(chunk["metadata"])
                if len(batch_ids) >= BATCH_SIZE:
                    safe_embed_and_upsert(collection, batch_ids, batch_docs, batch_metas)
                    total_chunks += len(batch_ids)
                    batch_ids, batch_docs, batch_metas = [], [], []

            completed.add(patient_id_key)
            save_progress("synthea", patient_id_key)

        except Exception:
            errors += 1

        progress.update()

    progress.done()

    if batch_ids:
        safe_embed_and_upsert(collection, batch_ids, batch_docs, batch_metas)
        total_chunks += len(batch_ids)

    print(f"  FHIR chunks embedded: {total_chunks}")
    if errors:
        print(f"  Errors: {errors}")

    # ── Clinician Notes ──
    print("\n  Embedding clinician notes...")
    completed_notes = load_progress("clinician_notes")
    notes_files = sorted(NOTES_DIR.glob("*.json"))
    notes_total = len(notes_files)
    print(f"  Notes files: {notes_total} total, {len(completed_notes)} done")

    batch_ids, batch_docs, batch_metas = [], [], []
    notes_embedded = 0
    notes_errors = 0
    notes_progress = ProgressBar(notes_total, label="Embedding notes")

    for fpath in notes_files:
        patient_id = fpath.stem
        if patient_id in completed_notes:
            notes_progress.update()
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
                    safe_embed_and_upsert(collection, batch_ids, batch_docs, batch_metas)
                    notes_embedded += len(batch_ids)
                    batch_ids, batch_docs, batch_metas = [], [], []

            completed_notes.add(patient_id)
            save_progress("clinician_notes", patient_id)

        except Exception:
            notes_errors += 1

        notes_progress.update()

    notes_progress.done()

    if batch_ids:
        safe_embed_and_upsert(collection, batch_ids, batch_docs, batch_metas)
        notes_embedded += len(batch_ids)

    print(f"  Notes embedded: {notes_embedded}")
    if notes_errors:
        print(f"  Notes errors: {notes_errors}")

    # ── Patient Index ──
    if patient_index_entries:
        print("\n  Building patient_index...")
        idx_collection = get_collection("patient_index")
        idx_ids, idx_docs, idx_metas = [], [], []
        idx_progress = ProgressBar(len(patient_index_entries), label="Indexing patients")

        for entry in patient_index_entries:
            idx_ids.append(entry["patient_id"])
            idx_docs.append(
                f"Patient {entry['patient_name']}, Age {entry['age']}, Gender {entry['gender']}. "
                f"Active conditions: {entry['active_conditions']}. "
                f"Current medications: {entry['current_medications']}. "
                f"Last encounter: {entry['last_encounter_date']}. "
                f"Records: {entry['chunk_count']} chunks."
            )
            idx_metas.append({
                "patient_id": entry["patient_id"],
                "patient_name": entry["patient_name"],
                "age": entry["age"],
                "gender": entry["gender"],
                "active_conditions": entry["active_conditions"],
                "current_medications": entry["current_medications"],
                "last_encounter_date": entry["last_encounter_date"],
                "chunk_count": entry["chunk_count"],
            })
            if len(idx_ids) >= BATCH_SIZE:
                safe_embed_and_upsert(idx_collection, idx_ids, idx_docs, idx_metas)
                idx_progress.update(len(idx_ids))
                idx_ids, idx_docs, idx_metas = [], [], []

        if idx_ids:
            safe_embed_and_upsert(idx_collection, idx_ids, idx_docs, idx_metas)
            idx_progress.update(len(idx_ids))

        idx_progress.done()
        print(f"  {len(patient_index_entries)} patients indexed")


# ── 2. MTSamples Embedding ────────────────────────────────────────────────

def step_embed_mtsamples():
    print("\n" + "=" * 60)
    print("STEP 2: EMBEDDING MTSAMPLES CHUNKS")
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

            text = clean(text)
            if not text:
                completed.add(chunk_id)
                save_progress("mtsamples_embed", chunk_id)
                progress.update()
                continue

            keywords = json.loads(keywords_json) if keywords_json else []
            normalized_section = normalize_section(section or "")

            batch_ids.append(chunk_id)
            batch_docs.append(text)
            batch_metas.append({
                "specialty": specialty,
                "section": normalized_section,
                "keywords": ", ".join(keywords[:10]),
                "source": "mtsamples",
                "chunk_id": chunk_id,
            })

            if len(batch_ids) >= BATCH_SIZE:
                safe_embed_and_upsert(collection, batch_ids, batch_docs, batch_metas)
                for cid in batch_ids:
                    completed.add(cid)
                    save_progress("mtsamples_embed", cid)
                total_embedded += len(batch_ids)
                progress.update(len(batch_ids))
                batch_ids, batch_docs, batch_metas = [], [], []

        offset += limit

    if batch_ids:
        safe_embed_and_upsert(collection, batch_ids, batch_docs, batch_metas)
        for cid in batch_ids:
            completed.add(cid)
            save_progress("mtsamples_embed", cid)
        total_embedded += len(batch_ids)
        progress.update(len(batch_ids))

    progress.done()
    conn.close()
    print(f"  Chunks embedded: {total_embedded}")


# ── 3. Setup Logs ─────────────────────────────────────────────────────────

def step_setup_logs():
    print("\n" + "=" * 60)
    print("STEP 3: SETTING UP LOGS DATABASE")
    print("=" * 60)

    conn = sqlite3.connect(str(LOGS_DB))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS activity_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            query_text  TEXT NOT NULL,
            collection  TEXT NOT NULL,
            filters     TEXT,
            n_results   INTEGER,
            latency_ms  INTEGER
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            patient_id  TEXT,
            patient_name TEXT,
            action      TEXT NOT NULL,
            query_text  TEXT,
            collection  TEXT,
            source      TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_activity_timestamp ON activity_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_patient ON audit_log(patient_id);
        CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
    """)
    conn.commit()
    conn.close()
    print(f"  Logs DB ready: {LOGS_DB}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ClinicalRecall ChromaDB Ingestion Pipeline")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Embedding batch size")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate all collections")
    parser.add_argument("--only", choices=["normalize", "patients", "mtsamples", "logs"],
                        help="Run only one step")
    args = parser.parse_args()

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    _get_embed_model()

    print("\u2554" + "\u2550" * 58 + "\u2557")
    print("\u2551     ClinicalRecall \u2014 ChromaDB Ingestion Pipeline         \u2551")
    print("\u2560" + "\u2550" * 58 + "\u2563")
    print(f"\u2551  Embedding model : {EMBED_MODEL_NAME:<36}\u2551")
    print(f"\u2551  Batch size      : {BATCH_SIZE:<36}\u2551")
    print("\u255a" + "\u2550" * 58 + "\u255d")

    if args.reset:
        print("\nResetting all collections...")
        for name in ["synthea_structured", "mtsamples_knowledge", "patient_index"]:
            try:
                chroma_client.delete_collection(name)
                print(f"  Dropped: {name}")
            except Exception:
                pass
        import shutil
        if PROGRESS_DIR.exists():
            shutil.rmtree(PROGRESS_DIR)
            PROGRESS_DIR.mkdir()
            print("  Cleared progress files")

    overall_start = time.time()

    if args.only == "normalize":
        step_normalize_sections()
    elif args.only == "patients":
        step_embed_patients()
    elif args.only == "mtsamples":
        step_embed_mtsamples()
    elif args.only == "logs":
        step_setup_logs()
    else:
        step_normalize_sections()
        step_embed_patients()
        step_embed_mtsamples()
        step_setup_logs()

    overall_elapsed = time.time() - overall_start
    m, s = divmod(int(overall_elapsed), 60)
    h, m = divmod(m, 60)

    print(f"\n{'=' * 60}")
    print(f"INGESTION COMPLETE in {h}h {m}m {s}s")
    print(f"{'=' * 60}")

    for name in ["synthea_structured", "mtsamples_knowledge", "patient_index"]:
        try:
            col = chroma_client.get_collection(name)
            print(f"  {name}: {col.count():,} chunks")
        except Exception:
            print(f"  {name}: NOT FOUND")


if __name__ == "__main__":
    main()
