# ClinicalRecall — Product Requirements Document

*Version:* 1.0  
*Hackathon:* Global AI Hackathon Series with Qwen Cloud  
*Track:* Track 1 — MemoryAgent  
*Submission Deadline:* July 9, 2026  

---

## 1. Problem Statement

Clinical information is fragmented across visits, systems, and time. Doctors re-ask questions patients have already answered. Important trends in labs or symptoms go unnoticed because no one is looking across the full history. Handoffs between shifts or providers lose critical context. The average clinician spends 16 minutes per patient just orienting themselves before a visit.

ClinicalRecall solves this by giving clinicians a persistent AI memory layer over their patients. It ingests structured records, free-text notes, and lab data, and lets clinicians ask natural language questions that span everything it has ever seen about a patient.

---

## 2. Product Overview

ClinicalRecall is a clinical memory agent built on Qwen Cloud. It combines:

•⁠  ⁠*Synthea-generated patient records* (structured JSON: conditions, medications, labs, encounters)
•⁠  ⁠*MTSamples clinical transcriptions* (free-text: SOAP notes, radiology reports, discharge summaries)
•⁠  ⁠*Clinician-added notes* (freeform text entered directly in the app)

The agent stores all of this in a vector database (ChromaDB), namespaced per patient, and allows clinicians to query across all of it using natural language. It can also write back: adding notes, updating medications, and logging every change with a timestamp.

---

## 3. Goals

*Primary*
•⁠  ⁠Let a clinician ask any question about a patient and get a grounded, cross-session answer
•⁠  ⁠Support both read (retrieval) and write (note-taking, medication updates) workflows
•⁠  ⁠Demonstrate persistent memory that survives across sessions

*Secondary*
•⁠  ⁠Show judges a compelling live demo with real data sources
•⁠  ⁠Build a resume-worthy RAG architecture using Qwen embeddings + LLM

*Out of scope (for hackathon)*
•⁠  ⁠Real patient data / HIPAA compliance
•⁠  ⁠EHR integrations (Epic, Cerner)
•⁠  ⁠Multi-user authentication
•⁠  ⁠Production-grade security

---

## 4. Target Users

For the hackathon, the primary persona is a *demo user acting as a clinician*. In a real-world extension, the users would be:

| Persona | Use Case |
|---|---|
| Primary care physician | Track chronic disease progression across visits |
| Hospitalist | Get oriented on a patient during handoff |
| Medical resident | Review full history before rounds |
| Nurse practitioner | Check medication history and flag interactions |

---

## 5. Data Sources

### 5.1 Synthea (Structured)

Synthea is an open-source synthetic patient generator from MITRE. It outputs realistic patient records as JSON with no credentialing required.

*What it provides:*
•⁠  ⁠Patient demographics
•⁠  ⁠Conditions with onset dates and ICD codes
•⁠  ⁠Medications with start/stop dates and dosages
•⁠  ⁠Lab observations (HbA1c, creatinine, cholesterol, etc.)
•⁠  ⁠Encounter history with visit reasons

*How to generate:*
⁠ bash
./run_synthea -p 50   # generates 50 synthetic patients
 ⁠

*Output path:* ⁠ output/fhir/*.json ⁠

*Ingestion strategy:*
•⁠  ⁠Parse FHIR Bundle JSON — Synthea wraps all records in `Bundle.entry[]`, not flat objects
•⁠  ⁠Filter entries by `resourceType`: Patient, Condition, MedicationRequest, Observation, Encounter
•⁠  ⁠Extract `patient_id` from `subject.reference` (e.g., `"Patient/abc-123"` → `"abc-123"`)
•⁠  ⁠Chunk into semantic units: one chunk per condition, one per medication, one per lab result, one per encounter
•⁠  ⁠Tag each chunk with: ⁠ patient_id ⁠, ⁠ data_type ⁠, ⁠ date ⁠, ⁠ source: synthea ⁠, ⁠ chunk_id ⁠ (UUID for update/delete tracking)
•⁠  ⁠Embed with Qwen embedding model
•⁠  ⁠Store in ChromaDB under collection ⁠ synthea_structured ⁠

*Batch ingestion strategy (for full 5,772 patient dataset):*
•⁠  ⁠Phase 1 — Parse all FHIR files and extract chunks (no API calls, fast)
  - Stream files one at a time (do NOT load entire 23 GB into memory)
  - Extract chunks into a local SQLite staging database: `ingestion_staging.db`
  - Schema: `chunks(id, patient_id, text, data_type, date, source, chunk_id)`
  - This decouples parsing from embedding — if embedding fails, we don't re-parse
•⁠  ⁠Phase 2 — Batch embed via Qwen API
  - Read chunks from SQLite in batches of 100
  - Call Qwen embedding API with batch input (reduces API round-trips by 100x)
  - Store vectors + metadata in ChromaDB
  - Track progress with a checkpoint file — resume from last completed batch on failure
•⁠  ⁠Phase 3 — Build patient index
  - Create a lightweight `patient_index` collection in ChromaDB
  - One entry per patient: `{patient_id, name, age, conditions_summary, last_visit_date, chunk_count}`
  - Powers the sidebar patient list without scanning the full collection
•⁠  ⁠Estimated scale:
  - ~5,772 patients × ~500 chunks/patient = ~2.9M chunks
  - ~2.9M chunks / 100 per batch = ~29,000 API calls for embedding
  - At ~200ms per batch call: ~1.6 hours total embedding time
  - ChromaDB storage: ~2.9M × 1536-dim × 4 bytes = ~18 GB of vectors
  - SQLite staging DB: ~2.9M rows × ~500 bytes = ~1.5 GB

### 5.2 MTSamples (Unstructured)

MTSamples.com hosts 4,999 real (anonymized) medical transcription samples across 40+ specialties including cardiology, neurology, orthopedics, and radiology.

*What it provides:*
•⁠  ⁠Chief complaints
•⁠  ⁠History of present illness
•⁠  ⁠Physical examination findings
•⁠  ⁠Assessment and plan
•⁠  ⁠Specialty-specific clinical language

*How to use:*
•⁠  ⁠Download sample transcriptions as text files
•⁠  ⁠Use as a general clinical knowledge base (not patient-specific)
•⁠  ⁠Enables the agent to answer questions grounded in real clinical language and reasoning

*Ingestion strategy:*
•⁠  ⁠Chunk each transcription by section (HPI, Assessment, Plan)
•⁠  ⁠Tag each chunk with: ⁠ specialty ⁠, ⁠ condition_keywords ⁠, ⁠ source: mtsamples ⁠
•⁠  ⁠Embed and store in ChromaDB under collection ⁠ mtsamples_knowledge ⁠

### 5.3 Clinician-Added Notes (Runtime)

Clinicians can add freeform notes directly in the app during or after a visit. These are stored immediately to the patient's namespace.

*Examples:*
•⁠  ⁠"Patient came in today, HbA1c is 9.1, added insulin to the regimen"
•⁠  ⁠"Patient reported dizziness after starting Lisinopril, consider dose reduction"
•⁠  ⁠"Referred to nephrology given rising creatinine over last 3 visits"

*Storage:* ChromaDB, tagged ⁠ source: clinician_note ⁠, ⁠ patient_id ⁠, ⁠ timestamp ⁠

---

## 6. Architecture


┌─────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                        │
│                                                              │
│  Synthea JSON ──► Parser ──► Structured chunks               │
│  MTSamples TXT ──► Section splitter ──► Text chunks          │
│  Clinician notes ──► Direct input ──► Raw text chunks        │
│                                                              │
│  All chunks ──► Qwen Embedding Model ──► Vectors             │
│              ──► ChromaDB (namespaced by patient + source)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                        QUERY LAYER                           │
│                                                              │
│  User query ──► Intent classifier (read vs write)           │
│                                                              │
│  READ path:                                                  │
│    ──► Embed query                                           │
│    ──► Search synthea_structured (patient-scoped)            │
│    ──► Search mtsamples_knowledge (condition-scoped)         │
│    ──► Merge + rank top-k results                            │
│    ──► Feed context into Qwen LLM                            │
│    ──► Stream response to UI                                 │
│                                                              │
│  WRITE path:                                                 │
│    ──► Extract structured data from input                    │
│    ──► Insert new chunk OR update existing chunk             │
│    ──► Version old value (audit trail)                       │
│    ──► Confirm write to user                                 │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                           UI LAYER                           │
│                                                              │
│  Sidebar: patient list + upload button                       │
│  Header: patient details + Add Note + Activity Log           │
│  Tabs: Chat | Medications                                    │
│  Chat: message thread with suggestion chips                  │
│  Medications: inline editable fields                         │
│  Activity Log: timestamped write history                     │
└─────────────────────────────────────────────────────────────┘


---

## 7. Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| LLM | Qwen (via dashscope-intl API) | Required by hackathon |
| Embeddings | Qwen text-embedding model | Same API, no extra setup |
| Vector DB | ChromaDB (local) | Zero infra, fast to set up |
| Backend | FastAPI | Lightweight, async-ready |
| Frontend | React (Streamlit fallback) | Already prototyped |
| Data generation | Synthea CLI | One command, 50 patients |
| Clinical text | MTSamples (downloaded) | No auth required |

*API base URL:* ⁠ https://dashscope-intl.aliyuncs.com/compatible-mode/v1 ⁠  
*SDK:* OpenAI Python SDK (compatible with Qwen)

---

## 8. Features

### 8.1 Patient Selection
•⁠  ⁠Sidebar lists all loaded patients with name, condition, and last visit date
•⁠  ⁠Selecting a patient activates their memory namespace in ChromaDB
•⁠  ⁠"Memory Active" indicator confirms the context is loaded

### 8.2 Natural Language Query (Read)
•⁠  ⁠Clinician types any question about the selected patient
•⁠  ⁠Agent retrieves relevant chunks from both Synthea and MTSamples collections
•⁠  ⁠Qwen synthesizes a grounded answer with source attribution
•⁠  ⁠Suggestion chips provide starter queries for new users

*Example queries:*
•⁠  ⁠"Has this patient had elevated HbA1c before?"
•⁠  ⁠"What medications is this patient currently on?"
•⁠  ⁠"Summarize this patient's last 3 visits"
•⁠  ⁠"Are there any drug interactions I should be aware of?"
•⁠  ⁠"How do clinicians typically manage poorly controlled diabetes?"

### 8.2.1 Suggestion Chips
•⁠  ⁠Pre-generated per patient based on available data types in their Synthea record
•⁠  ⁠Generation logic: scan patient's conditions, medications, and lab types → produce targeted chips
•⁠  ⁠Examples by data profile:
  - Patient has diabetes + HbA1c labs → "Has HbA1c been trending up?", "What diabetes medications is this patient on?"
  - Patient has hypertension + BP readings → "What was the last blood pressure reading?", "Is blood pressure controlled?"
  - Patient has multiple medications → "Are there any drug interactions?", "Summarize medication history"
  - Patient has recent encounters → "Summarize the last 3 visits", "What was discussed in the last encounter?"
•⁠  ⁠Fallback chips (shown when no patient-specific data is available):
  - "What conditions does this patient have?"
  - "What medications is this patient on?"
  - "Show recent lab results"
•⁠  ⁠Chips disappear after the first message is sent (replaced by chat thread)

### 8.3 Add Note (Write)
•⁠  ⁠"Add Note" button opens a freeform text input
•⁠  ⁠Note is embedded and stored immediately to the patient's namespace
•⁠  ⁠Confirmation message appears in chat
•⁠  ⁠Entry appears in Activity Log with timestamp

### 8.4 Medication Management (Write)
•⁠  ⁠Medications tab shows all active medications per patient
•⁠  ⁠Any field (dose, frequency) is editable inline
•⁠  ⁠On save: ChromaDB chunk is updated, old value is versioned
•⁠  ⁠Change is logged to Activity Log

### 8.5 Activity Log
•⁠  ⁠Slide-in panel showing every write operation
•⁠  ⁠Fields: action description, timestamp
•⁠  ⁠Stored in ChromaDB `activity_log` collection (survives page refresh and server restart)
•⁠  ⁠Filtered by `patient_id` — only shows activity for the selected patient
•⁠  ⁠Empty state: "No activity yet"

### 8.6 Data Upload
•⁠  ⁠"Upload Patient Data" button in sidebar
•⁠  ⁠Accepts: Synthea JSON, plain text notes, MTSamples transcriptions
•⁠  ⁠Triggers ingestion pipeline: parse, chunk, embed, store
•⁠  ⁠Patient appears in sidebar after upload

---

## 9. Query Types and Expected Behavior

| Query type | Primary source | Secondary source | Example |
|---|---|---|---|
| Patient-specific factual | Synthea structured | Clinician notes | "What was the last HbA1c?" |
| Trend / longitudinal | Synthea observations | Clinician notes | "Is creatinine trending up?" |
| Medication history | Synthea medications | Clinician notes | "What has this patient been prescribed?" |
| Clinical reasoning | MTSamples | Synthea (for grounding) | "How do clinicians handle this?" |
| Cross-source synthesis | Both | Both | "Given this patient's labs, what should I watch for?" |
| Write intent | Clinician input | None | "Update Metformin to 1000mg" |

---

## 10. Data Flow: Ingestion Pipeline

⁠ python
# Pseudocode for ingestion

def ingest_synthea(fhir_bundle_json):
    """
    Synthea outputs FHIR Bundle JSON. Each Bundle.entry has a resourceType
    (Patient, Condition, MedicationRequest, Observation, Encounter).
    We extract patient_id from subject.reference and chunk by resource type.
    """
    entries = fhir_bundle_json.get("entry", [])
    chunks = []

    # First pass: extract patient_id from Patient resource
    patient_id = None
    for entry in entries:
        resource = entry["resource"]
        if resource.get("resourceType") == "Patient":
            patient_id = resource["id"]
            chunks.append({
                "text": f"Patient {resource.get('name', [{}])[0].get('given', [''])[0]} "
                        f"{resource.get('name', [{}])[0].get('family', '')}, "
                        f"DOB {resource.get('birthDate', '')}, "
                        f"Gender {resource.get('gender', '')}",
                "metadata": {"patient_id": patient_id, "type": "demographics", "source": "synthea", "chunk_id": str(uuid4())}
            })
            break

    if not patient_id:
        raise ValueError("No Patient resource found in FHIR bundle")

    for entry in entries:
        resource = entry["resource"]
        rt = resource.get("resourceType")

        if rt == "Condition":
            code = resource.get("code", {}).get("coding", [{}])[0].get("display", "")
            onset = resource.get("onsetDateTime", "")
            chunks.append({
                "text": f"Condition: {code}, onset {onset}",
                "metadata": {"patient_id": patient_id, "type": "condition", "source": "synthea", "chunk_id": str(uuid4())}
            })

        elif rt == "MedicationRequest":
            med_code = resource.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("display", "")
            dosage = resource.get("dosageInstruction", [{}])[0].get("text", "")
            authored = resource.get("authoredOn", "")
            chunks.append({
                "text": f"Medication: {med_code}, dosage: {dosage}, prescribed {authored}",
                "metadata": {"patient_id": patient_id, "type": "medication", "source": "synthea", "chunk_id": str(uuid4())}
            })

        elif rt == "Observation":
            obs_code = resource.get("code", {}).get("coding", [{}])[0].get("display", "")
            value = resource.get("valueQuantity", {}).get("value", "")
            unit = resource.get("valueQuantity", {}).get("unit", "")
            effective = resource.get("effectiveDateTime", "")
            chunks.append({
                "text": f"Lab: {obs_code} = {value} {unit} on {effective}",
                "metadata": {"patient_id": patient_id, "type": "observation", "source": "synthea", "date": effective, "chunk_id": str(uuid4())}
            })

        elif rt == "Encounter":
            reason = resource.get("reasonCode", [{}])[0].get("coding", [{}])[0].get("display", "")
            period_start = resource.get("period", {}).get("start", "")
            chunks.append({
                "text": f"Encounter: {reason} on {period_start}",
                "metadata": {"patient_id": patient_id, "type": "encounter", "source": "synthea", "date": period_start, "chunk_id": str(uuid4())}
            })

    embed_and_store(chunks, collection="synthea_structured")


def ingest_mtsamples(transcription_text, specialty):
    sections = split_by_section(transcription_text)
    for section_name, section_text in sections.items():
        chunk = {
            "text": section_text,
            "metadata": {"specialty": specialty, "section": section_name, "source": "mtsamples", "chunk_id": str(uuid4())}
        }
    embed_and_store([chunk], collection="mtsamples_knowledge")
 ⁠

---

## 11. Data Flow: Query Pipeline

⁠ python
# Pseudocode for query

# Write-intent trigger keywords (fast, no extra LLM call)
WRITE_KEYWORDS = ["update", "change", "add", "set", "modify", "prescribe", "discontinue", "start", "stop", "note:", "log:"]

def classify_intent(user_input: str) -> str:
    """Keyword-based intent classification. Fast, deterministic, no API cost."""
    lower = user_input.lower().strip()
    for kw in WRITE_KEYWORDS:
        if lower.startswith(kw) or f" {kw} " in f" {lower} ":
            return "write"
    return "read"

def query(user_input, patient_id):
    intent = classify_intent(user_input)

    if intent == "read":
        try:
            query_vector = embed(user_input)

            # Patient-scoped search
            synthea_results = chromadb.query(
                collection="synthea_structured",
                query_embeddings=[query_vector],
                where={"patient_id": patient_id},
                n_results=5
            )

            # General clinical knowledge
            mtsamples_results = chromadb.query(
                collection="mtsamples_knowledge",
                query_embeddings=[query_vector],
                n_results=3
            )

            context = merge_and_rank(synthea_results, mtsamples_results)
            response = qwen_llm.chat(system_prompt + context, user_input)
            return {"status": "ok", "response": response, "sources": extract_source_refs(context)}

        except APIError as e:
            return {"status": "error", "message": f"Qwen API error: {e}. Please retry."}
        except TimeoutError:
            return {"status": "error", "message": "Request timed out. Please try a shorter query."}

    elif intent == "write":
        try:
            extracted = extract_write_intent(user_input)
            update_chromadb(patient_id, extracted)
            log_activity(patient_id, extracted)
            return {"status": "ok", "message": "Memory updated.", "activity": extracted}
        except Exception as e:
            return {"status": "error", "message": f"Write failed: {e}"}


def update_chromadb(patient_id, extracted):
    """
    ChromaDB does not support in-place updates.
    Strategy: delete old chunk by chunk_id, then re-insert with same chunk_id.
    Old values are versioned in a separate audit_log collection.
    """
    collection = chromadb.get_collection("synthea_structured")

    if extracted.get("chunk_id"):
        # Fetch existing chunk for audit trail
        existing = collection.get(ids=[extracted["chunk_id"]])
        if existing and existing["documents"]:
            audit_collection = chromadb.get_or_create_collection("audit_log")
            audit_collection.add(
                documents=[existing["documents"][0]],
                metadatas=[{
                    "patient_id": patient_id,
                    "chunk_id": extracted["chunk_id"],
                    "action": "updated",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }],
                ids=[str(uuid4())]
            )
            collection.delete(ids=[extracted["chunk_id"]])

    # Insert new/updated chunk
    collection.add(
        documents=[extracted["text"]],
        metadatas=[extracted["metadata"]],
        ids=[extracted.get("chunk_id", str(uuid4()))]
    )


def log_activity(patient_id, extracted):
    """All write operations are persisted to ChromaDB for cross-session survival."""
    activity_collection = chromadb.get_or_create_collection("activity_log")
    activity_collection.add(
        documents=[f"{extracted['action']}: {extracted.get('summary', '')}"],
        metadatas=[{
            "patient_id": patient_id,
            "action": extracted["action"],
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }],
        ids=[str(uuid4())]
    )
 ⁠

---

## 12. UI Screens

### Screen 1: No Patient Selected
•⁠  ⁠Sidebar with patient list
•⁠  ⁠Main area shows empty state: "Select a patient to begin"
•⁠  ⁠Upload button visible

### Screen 2: Patient Selected, Chat Tab
•⁠  ⁠Header: patient name, age, condition, ID, MEMORY ACTIVE indicator
•⁠  ⁠Add Note and Activity Log buttons visible
•⁠  ⁠Suggestion chips shown until first message sent
•⁠  ⁠Chat thread below

### Screen 3: Add Note Open
•⁠  ⁠Yellow-green panel slides in below header
•⁠  ⁠Textarea for freeform note
•⁠  ⁠"Save to Memory" and "Cancel" buttons

### Screen 4: Medications Tab
•⁠  ⁠List of active medications
•⁠  ⁠Each medication shows drug name, dose, frequency
•⁠  ⁠Clicking a value opens inline edit with Save/Cancel
•⁠  ⁠Badge shows "Active" status

### Screen 5: Activity Log Open
•⁠  ⁠Right panel slides in (260px)
•⁠  ⁠Each entry: action text + timestamp
•⁠  ⁠Empty state: "No activity yet"

---

## 13. Demo Script (For Judges)

*Setup:* Synthea data for 3 patients pre-loaded. MTSamples cardiology and endocrinology transcriptions ingested.

*Step 1 — Show memory depth*
Select James Harlow (Type 2 Diabetes). Ask: "Has this patient had elevated HbA1c before?" Agent responds with trend across 3 sessions.

*Step 2 — Show cross-source synthesis*
Ask: "Given this patient's recent labs, what should I be watching for?" Agent pulls Synthea lab history AND MTSamples diabetic nephropathy notes to give a synthesized answer.

*Step 3 — Show write capability*
Click "Add Note." Type: "Patient came in today, HbA1c 9.1, adding insulin to regimen." Save. Open Activity Log to show it was logged.

*Step 4 — Show medication update*
Switch to Medications tab. Click Metformin dose. Change 500mg to 1000mg. Save. Show Activity Log entry.

*Step 5 — Show persistence*
Refresh the page. Reload the same patient. Ask the same HbA1c question. Agent still remembers everything including the note just added.

---

## 14. Build Plan

| Day | Task | Deliverable |
|---|---|---|
| 1 | Synthea data verified (5,772 patients, 23 GB). Build FHIR Bundle parser + SQLite staging. Generate clinician notes from Synthea data. | All 5,772 patients parsed into staging DB. Clinician notes generated. |
| 2 | MTSamples download, section splitter, ingestion. Batch embedding pipeline (Phase 2). | mtsamples_knowledge populated. Embedding pipeline running. |
| 3 | Complete batch embedding. Build patient_index. Qwen LLM integration, intent classifier, write pipeline. | Full 5,772 patients in ChromaDB. Read + write flow working. |
| 4 | FastAPI backend wiring, React frontend integration. Suggestion chips per patient. | End-to-end app running locally. |
| 5 | Demo script refinement, 3 killer demo queries, edge case handling. Activity Log persistence. | Polished demo. |
| 6 | README, Devpost writeup, video demo recording, submission | Submitted. |

---

## 15. Judging Criteria Alignment

| Criterion | How ClinicalRecall addresses it |
|---|---|
| Technical complexity | Dual-collection RAG, intent classification, read/write memory, Qwen embeddings + LLM |
| Innovation | Cross-source synthesis of structured + unstructured clinical data |
| Real-world impact | Directly addresses clinical context loss and information fragmentation |
| Demo quality | Live cross-session retrieval with a clear before/after moment |
| Use of Qwen Cloud | Qwen embedding model + Qwen LLM as the core of every query |

---

## 16. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Synthea data too sparse for compelling demo | Low | Generate 50+ patients, cherry-pick 3 with rich histories |
| MTSamples retrieval returning irrelevant chunks | Medium | Add specialty metadata filter to queries |
| Qwen API rate limits slowing demo | Low | Cache common query results for demo |
| ChromaDB namespace bleed between patients | Low | Enforce ⁠ patient_id ⁠ filter on every Synthea query |
| Write intent misclassification | Medium | Add a confirmation step before executing writes |

---

## 17. Future Extensions (Post-Hackathon)

•⁠  ⁠FHIR API integration for real EHR connectivity
•⁠  ⁠Multi-provider support (memories scoped to provider + patient)
•⁠  ⁠Anomaly detection: agent proactively flags concerning trends
•⁠  ⁠Voice input for hands-free note-taking in clinical settings
•⁠  ⁠Differential diagnosis assistant grounded in patient history
•⁠  ⁠HIPAA-compliant deployment on private cloud