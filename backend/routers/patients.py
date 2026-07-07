from datetime import date
from typing import Optional
from uuid import uuid4
from fastapi import APIRouter, Query
from models import NewPatientRequest, NewPatientResponse
from config import patient_index_col, synthea_col
from services import embedder, writer
from services.encounter_enricher import enrich_encounter
from logger import get_logger

log = get_logger("router.patients")
router = APIRouter()


@router.get("/patients")
async def list_patients(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("patient_name", pattern="^(patient_name|last_visit_date|chunk_count)$"),
    order: str = Query("asc", pattern="^(asc|desc)$"),
    q: Optional[str] = Query(None, description="Substring match on patient_name (case-insensitive)"),
):
    log.info(f"list_patients limit={limit} offset={offset} sort={sort_by} order={order} q={q!r}")
    results = patient_index_col.get(include=["metadatas"])
    patients = []
    for m in results["metadatas"]:
        if not m:
            continue
        patients.append({
            "patient_id": m.get("patient_id", ""),
            "patient_name": m.get("patient_name", ""),
            "last_visit_date": m.get("last_visit_date", ""),
            "chunk_count": int(m.get("chunk_count", 0)),
            "conditions_summary": m.get("conditions_summary", ""),
        })

    if q:
        q_lower = q.lower()
        before = len(patients)
        patients = [p for p in patients if q_lower in p["patient_name"].lower()]
        log.debug(f"Name filter {q!r}: {before} → {len(patients)}")

    reverse = order == "desc"
    if sort_by == "chunk_count":
        patients.sort(key=lambda p: p["chunk_count"], reverse=reverse)
    else:
        patients.sort(key=lambda p: p[sort_by] or "", reverse=reverse)

    total = len(patients)
    page = patients[offset : offset + limit]
    log.info(f"Returning {len(page)}/{total} patients")
    return {"total": total, "offset": offset, "limit": limit, "patients": page}


@router.get("/patients/search")
async def search_patients(
    q: str = Query(..., description="Natural language query over patient conditions/summaries"),
    limit: int = Query(10, ge=1, le=50),
):
    log.info(f"Semantic search q={q!r} limit={limit}")
    query_vector = embedder.embed(q, is_query=True)
    try:
        results = patient_index_col.query(
            query_embeddings=[query_vector],
            n_results=min(limit, patient_index_col.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        log.error(f"Semantic search failed: {e}")
        return {"query": q, "results": []}

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    patients = [
        {
            "patient_id": m.get("patient_id", ""),
            "patient_name": m.get("patient_name", ""),
            "last_visit_date": m.get("last_visit_date", ""),
            "chunk_count": int(m.get("chunk_count", 0)),
            "conditions_summary": m.get("conditions_summary", ""),
            "score": round(max(0.0, 1 - d / 2), 4),  # L2 on normalized vecs: d∈[0,2] → sim∈[0,1]
        }
        for m, d in zip(metas, distances)
        if m
    ]
    log.info(f"Semantic search returned {len(patients)} results for q={q!r}")
    return {"query": q, "results": patients}


@router.get("/patients/{patient_id}")
async def get_patient(
    patient_id: str,
    types: Optional[str] = Query(None, description="Comma-separated data_types: medication,condition,observation,note,encounter,demographics"),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    log.info(f"get_patient patient={patient_id} types={types!r} limit={limit} offset={offset}")
    # Normalise plural UI names → singular DB values (ingestion stores singular)
    _TO_SINGULAR = {
        "conditions": "condition", "medications": "medication",
        "observations": "observation", "notes": "note", "encounters": "encounter",
    }
    type_list = None
    if types:
        type_list = [_TO_SINGULAR.get(t.strip(), t.strip()) for t in types.split(",")]

    if type_list and len(type_list) == 1:
        where_filter = {"$and": [
            {"patient_id": {"$eq": patient_id}},
            {"data_type": {"$eq": type_list[0]}},
        ]}
    elif type_list:
        where_filter = {"$and": [
            {"patient_id": {"$eq": patient_id}},
            {"data_type": {"$in": type_list}},
        ]}
    else:
        where_filter = {"patient_id": {"$eq": patient_id}}

    try:
        results = synthea_col.get(
            where=where_filter,
            include=["documents", "metadatas"],
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        log.error(f"get_patient DB query failed for patient={patient_id}: {e}")
        empty = {"demographics": [], "conditions": [], "medications": [], "observations": [], "notes": [], "encounters": []}
        return {"patient_id": patient_id, "total": 0, "offset": offset, "limit": limit, **empty}

    # Ingestion uses singular data_type values ("condition", "medication", etc.)
    _BUCKET = {
        "demographics": "demographics",
        "condition": "conditions",   "conditions": "conditions",
        "medication": "medications", "medications": "medications",
        "observation": "observations", "observations": "observations",
        "note": "notes",             "notes": "notes",
        "clinician_note": "notes",
        "encounter": "encounters",   "encounters": "encounters",
    }
    grouped: dict[str, list] = {
        "demographics": [], "conditions": [], "medications": [],
        "observations": [], "notes": [], "encounters": [],
    }
    for doc, m in zip(results["documents"], results["metadatas"]):
        dt = (m or {}).get("data_type", "")
        bucket = _BUCKET.get(dt, "observations")
        grouped[bucket].append({"text": doc, "metadata": m})

    for enc in grouped["encounters"]:
        enc["text"] = enrich_encounter(patient_id, enc["text"], enc.get("metadata", {}))

    total = sum(len(v) for v in grouped.values())
    log.info(f"get_patient returned {total} chunks for patient={patient_id}")
    return {"patient_id": patient_id, "total": total, "offset": offset, "limit": limit, **grouped}


@router.post("/patients/new", response_model=NewPatientResponse)
async def create_patient(body: NewPatientRequest):
    patient_id = str(uuid4())
    patient_name = body.name
    today = date.today().isoformat()
    log.info(f"create_patient name={patient_name!r} id={patient_id}")

    chunks_text = []
    chunks_meta = []
    base = {"patient_id": patient_id, "patient_name": patient_name, "source": "manual", "timestamp": today}

    chunks_text.append(f"Patient: {patient_name}, {body.age}y, {body.gender}")
    chunks_meta.append({**base, "data_type": "demographics", "date": today})

    for condition in body.conditions:
        chunks_text.append(f"{condition} onset {today}")
        chunks_meta.append({**base, "data_type": "condition", "date": today})

    for med in body.medications:
        chunks_text.append(f"{med.drug} {med.dose} {med.frequency} started {today}")
        chunks_meta.append({**base, "data_type": "medication", "date": today, "drug": med.drug})

    if body.notes.strip():
        chunks_text.append(body.notes)
        chunks_meta.append({**base, "data_type": "note", "source": "clinician_note", "note_type": "follow_up"})

    log.debug(f"Embedding {len(chunks_text)} chunks for new patient {patient_id}")
    vectors = embedder.embed_batch(chunks_text)
    ids = [str(uuid4()) for _ in chunks_text]

    synthea_col.add(ids=ids, documents=chunks_text, embeddings=vectors, metadatas=chunks_meta)
    log.debug(f"Added {len(ids)} chunks to synthea_structured")

    index_doc = f"Patient {patient_name}. Conditions: {', '.join(body.conditions)}. Last visit: {today}."
    index_vec = embedder.embed(index_doc)
    patient_index_col.add(
        ids=[str(uuid4())],
        documents=[index_doc],
        embeddings=[index_vec],
        metadatas=[{
            "patient_id": patient_id,
            "patient_name": patient_name,
            "last_visit_date": today,
            "chunk_count": len(chunks_text),
            "conditions_summary": ", ".join(body.conditions),
        }],
    )
    log.info(f"Patient {patient_id} ({patient_name}) created with {len(chunks_text)} chunks")

    await writer.log_activity(patient_id, "patient_created", patient_name)
    return NewPatientResponse(patient_id=patient_id, status="created")
