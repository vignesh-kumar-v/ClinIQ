from fastapi import APIRouter
from models import NoteRequest
from config import patient_index_col
from services import embedder, writer

router = APIRouter()


@router.post("/patients/{patient_id}/note")
async def add_note(patient_id: str, body: NoteRequest):
    patient_name = _get_patient_name(patient_id)
    vec = embedder.embed(body.text)
    await writer.add_note(patient_id, patient_name, body.text, body.note_type, vec)
    await writer.log_activity(patient_id, "note_added", body.text[:80])
    return {"status": "saved"}


def _get_patient_name(patient_id: str) -> str:
    try:
        results = patient_index_col.get(
            where={"patient_id": {"$eq": patient_id}},
            include=["metadatas"],
        )
        if results["metadatas"]:
            return results["metadatas"][0].get("patient_name", "Unknown")
    except Exception:
        pass
    return "Unknown"
