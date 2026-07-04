from datetime import datetime, timezone
from uuid import uuid4
from config import synthea_col, chat_history_col, activity_log_col, MAX_HISTORY_TURNS
from logger import get_logger

log = get_logger("writer")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def add_note(
    patient_id: str,
    patient_name: str,
    text: str,
    note_type: str,
    vector: list[float],
) -> None:
    doc_id = str(uuid4())
    log.info(f"add_note patient={patient_id} note_type={note_type} id={doc_id}")
    synthea_col.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[vector],
        metadatas=[{
            "patient_id": patient_id,
            "patient_name": patient_name,
            "data_type": "note",
            "source": "clinician_note",
            "note_type": note_type,
            "timestamp": _now(),
        }],
    )


async def update_medication(
    patient_id: str,
    drug: str,
    field: str,
    value: str,
) -> None:
    log.info(f"update_medication patient={patient_id} drug={drug!r} {field}={value!r}")
    results = synthea_col.get(
        where={"$and": [
            {"patient_id": {"$eq": patient_id}},
            {"data_type": {"$eq": "medication"}},
        ]},
        include=["documents", "metadatas"],  # no embeddings — numpy array truth-value crashes
    )

    ids = results["ids"]
    docs = results["documents"]
    metas = results["metadatas"]
    log.debug(f"Found {len(ids)} medication chunks for patient {patient_id}")

    drug_lower = drug.lower()
    matched = False
    for i, doc in enumerate(docs):
        if drug_lower not in doc.lower():
            continue
        matched = True
        original_id = ids[i]
        log.debug(f"Matched medication chunk id={original_id}: {doc[:80]!r}")

        # Re-embed since we can't safely retrieve stored numpy embeddings
        from services.embedder import embed as _embed
        original_vec = _embed(doc)
        versioned_meta = {**metas[i], "versioned": "true", "versioned_at": _now()}
        synthea_col.add(
            ids=[str(uuid4())],
            documents=[doc],
            embeddings=[original_vec],
            metadatas=[versioned_meta],
        )

        if field == "dose":
            updated_doc = f"[Updated dose: {value}] {doc}"
        elif field == "frequency":
            updated_doc = f"[Updated frequency: {value}] {doc}"
        else:
            updated_doc = f"[Updated {field}: {value}] {doc}"

        updated_vec = _embed(updated_doc)
        synthea_col.update(
            ids=[original_id],
            documents=[updated_doc],
            embeddings=[updated_vec],
            metadatas=[{**metas[i], "updated_at": _now()}],
        )
        log.info(f"Medication updated id={original_id}")
        break

    if not matched:
        log.warning(f"No medication chunk found matching drug={drug!r} for patient={patient_id}")


async def add_observation(
    patient_id: str,
    patient_name: str,
    text: str,
    vector: list[float],
) -> None:
    doc_id = str(uuid4())
    log.info(f"add_observation patient={patient_id} id={doc_id}")
    synthea_col.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[vector],
        metadatas=[{
            "patient_id": patient_id,
            "patient_name": patient_name,
            "data_type": "observation",
            "source": "clinician_note",
            "timestamp": _now(),
        }],
    )


async def log_activity(patient_id: str, action: str, detail: str) -> None:
    doc_id = str(uuid4())
    log.info(f"log_activity patient={patient_id} action={action!r}")
    activity_log_col.add(
        ids=[doc_id],
        documents=[f"{action}: {detail}"],
        metadatas=[{
            "patient_id": patient_id,
            "action": action,
            "detail": detail,
            "timestamp": _now(),
        }],
    )


async def save_chat_turn(
    patient_id: str,
    session_id: str,
    role: str,
    content: str,
) -> None:
    doc_id = str(uuid4())
    log.debug(f"save_chat_turn patient={patient_id} session={session_id} role={role}")
    chat_history_col.add(
        ids=[doc_id],
        documents=[content],
        metadatas=[{
            "patient_id": patient_id,
            "session_id": session_id,
            "role": role,
            "timestamp": _now(),
        }],
    )


async def get_chat_history(patient_id: str) -> list[dict]:
    log.debug(f"get_chat_history patient={patient_id}")
    try:
        results = chat_history_col.get(
            where={"patient_id": {"$eq": patient_id}},
            include=["documents", "metadatas"],
        )
    except Exception as e:
        log.warning(f"chat_history fetch failed for patient={patient_id}: {e}")
        return []

    items = list(zip(results["metadatas"], results["documents"]))
    items.sort(key=lambda x: x[0].get("timestamp", ""))
    items = items[-MAX_HISTORY_TURNS:]
    log.debug(f"Returning {len(items)} history turns for patient={patient_id}")
    return [{"role": m.get("role", "user"), "content": doc} for m, doc in items]
