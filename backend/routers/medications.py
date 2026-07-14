from fastapi import APIRouter
from models import MedicationUpdateRequest
from services import writer, cache

router = APIRouter()


@router.patch("/patients/{patient_id}/medication")
async def update_medication(patient_id: str, body: MedicationUpdateRequest):
    await writer.update_medication(patient_id, body.drug, body.field, body.value)
    await writer.log_activity(
        patient_id,
        "medication_updated",
        f"{body.drug} {body.field} -> {body.value}",
    )
    await cache.invalidate(f"patient:{patient_id}")
    return {"status": "updated"}
