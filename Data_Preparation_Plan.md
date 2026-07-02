# Project Plan: ClinicalRecall Data Preparation

## Target Output
- Destination: ChromaDB collections (`synthea_structured`, `mtsamples_knowledge`, `activity_log`, `audit_log`)
- Clinician notes: `data/clinician_notes/{patient_id}.json` (16,843 notes across 5,770 patients)

## Data Sources and Structures

### 1. Synthea (Core Demographic & Event Data)
- **Status**: Structured (FHIR Bundle JSON)
- **Fields**:
  - `patient_id` (extracted from `subject.reference` or `Patient.id`)
  - `data_type` (resourceType: Patient, Condition, MedicationRequest, Observation, Encounter)
  - `date` (from `onsetDateTime`, `effectiveDateTime`, `authoredOn`, `period.start`)
- **Source Tag**: `synthea_structured`
- **Preparation Requirements**:
  - **FHIR Bundle Parsing**: Synthea outputs `Bundle.entry[]`, not flat objects. Each entry has a `resource` with a `resourceType`. Filter and extract by type.
  - **Patient ID Extraction**: `patient_id` comes from `subject.reference` (e.g., `"Patient/abc-123"` → `"abc-123"`) or from the `Patient.id` field.
  - **Type Casting**: Ensure `patient_id` is a consistent String across all datasets.
  - **Time Normalization**: Convert all date fields to ISO 8601 format for standard timestamping.
  - **De-duplication**: Remove duplicate identity records to ensure profile integrity.
  - **Chunk ID Assignment**: Every chunk gets a UUID `chunk_id` for update/delete tracking in ChromaDB.

### 2. MTSamples (Medical Knowledge Reference)
- **Status**: Knowledge Extraction & Enrichment
- **Fields**:
  - `specialty`
  - `condition_keywords`
- **Source Tag**: `mtsamples_knowledge`
- **Preparation Requirements**:
  - **Case Normalization**: Convert all text to lowercase for consistent matching.
  - **Keyword Stemming**: Reduce keywords to their base forms (e.g., "infections" → "infection") to improve matching coverage.
  - **Search Optimization**: Structure as a mapping (Specialty → Keywords) for efficient lookup.

### 3. Clinician Note (Temporal/Clinical Events)
- **Fields**:
  - `patient_id`
  - `timestamp`
- **Preparation Requirements**:
  - **Text Sanitization**: Strip non-printable characters and normalize whitespace in note content.
  - **Time Normalization**: Convert `timestamp` to ISO 8601 to align with Synthea's timeline.
  - **Referential Integrity**: Validate that every `patient_id` exists in the Synthea primary dataset before processing.

## Integration & Transformation Workflow

### Step 1: Synthea FHIR Bundle Ingestion
- Parse each FHIR Bundle JSON file
- Extract `Patient` resource for demographics and `patient_id`
- Extract `Condition`, `MedicationRequest`, `Observation`, `Encounter` resources
- Chunk each resource into a semantic text representation
- Tag with metadata: `patient_id`, `data_type`, `date`, `source: synthea`, `chunk_id`
- Embed with Qwen embedding model
- Store in ChromaDB collection `synthea_structured`

### Step 2: MTSamples Knowledge Base Ingestion
- Split each transcription by clinical section (HPI, Assessment, Plan, etc.)
- Tag with metadata: `specialty`, `section`, `condition_keywords`, `source: mtsamples`, `chunk_id`
- Embed and store in ChromaDB collection `mtsamples_knowledge`

### Step 3: Clinician Note Ingestion (Runtime)
- Accept freeform text from the UI
- Sanitize and normalize
- Tag with metadata: `patient_id`, `timestamp`, `source: clinician_note`, `chunk_id`
- Embed and store in ChromaDB collection `synthea_structured` (same namespace as Synthea data for unified retrieval)

### Step 4: Knowledge Enrichment (Optional, for improved retrieval)
- Scan clinician notes for matches against MTSamples `condition_keywords`
- Tag clinical events with corresponding medical `specialty` based on keyword matches
- This enriches the metadata for better cross-collection retrieval relevance

## ChromaDB Collection Architecture

| Collection | Purpose | Key Metadata Fields |
|---|---|---|
| `synthea_structured` | All patient-scoped data (Synthea + clinician notes) | `patient_id`, `data_type`, `date`, `source`, `chunk_id` |
| `mtsamples_knowledge` | General clinical knowledge base | `specialty`, `section`, `condition_keywords`, `source`, `chunk_id` |
| `activity_log` | Persistent write operation history | `patient_id`, `action`, `timestamp` |
| `audit_log` | Versioned old values before updates | `patient_id`, `chunk_id`, `action`, `timestamp` |

## Integration Details
- **API Endpoint**: [https://dashscope-intl.aliyuncs.com/compatible-mode/v1](https://dashscope-intl.aliyuncs.com/compatible-mode/v1)
- **Embedding Model**: Qwen text-embedding (via dashscope-intl API)
- **LLM**: Qwen (via dashscope-intl API, OpenAI-compatible SDK)

## Data Samples

### 1. Synthea (FHIR Bundle — Structured Demographic/Event Data)
```json
{
  "resourceType": "Bundle",
  "entry": [
    {
      "resource": {
        "resourceType": "Patient",
        "id": "abc-123",
        "name": [{"given": ["James"], "family": "Harlow"}],
        "birthDate": "1965-03-15",
        "gender": "male"
      }
    },
    {
      "resource": {
        "resourceType": "Observation",
        "id": "obs-001",
        "subject": {"reference": "Patient/abc-123"},
        "code": {"coding": [{"display": "Hemoglobin A1c"}]},
        "valueQuantity": {"value": 9.1, "unit": "%"},
        "effectiveDateTime": "2026-06-30T09:00:00Z"
      }
    },
    {
      "resource": {
        "resourceType": "Condition",
        "id": "cond-001",
        "subject": {"reference": "Patient/abc-123"},
        "code": {"coding": [{"display": "Type 2 Diabetes Mellitus"}]},
        "onsetDateTime": "2020-01-15"
      }
    }
  ]
}
```

### 2. MTSamples (Medical Knowledge Lookup)
```json
[
  {
    "specialty": "Cardiology",
    "condition_keywords": ["heart", "tachycardia", "arrhythmia", "palpitations"]
  },
  {
    "specialty": "Pulmonology",
    "condition_keywords": ["breathing", "lung", "shortness of breath", "respiratory"]
  }
]
```

### 3. Clinician Notes (Raw Narrative Data)
| patient_id | timestamp | note_text |
| :--- | :--- | :--- |
| `abc-123` | `2026-06-30T09:15:00Z` | "Patient complains of acute shortness of breath and irregular heart rhythm." |
| `def-456` | `2026-07-01T14:30:00Z` | "Examination shows signs of pulmonary congestion and labored breathing." |

### Transformation Logic Example
**Process:**
1. Parse FHIR Bundle → extract `Patient/abc-123` as `patient_id`
2. For each `Observation` entry → chunk: `"Lab: Hemoglobin A1c = 9.1 % on 2026-06-30T09:00:00Z"`
3. Tag with metadata: `{"patient_id": "abc-123", "type": "observation", "source": "synthea", "date": "2026-06-30T09:00:00Z", "chunk_id": "<uuid>"}`
4. Embed with Qwen → store in `synthea_structured`
5. (Optional enrichment) Scan clinician note for MTSamples keywords → tag with `specialty: "Pulmonology"`

