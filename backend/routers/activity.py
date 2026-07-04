from fastapi import APIRouter, Query
from models import ActivityEntry
from config import activity_log_col

router = APIRouter()


@router.get("/patients/{patient_id}/activity")
async def get_activity(
    patient_id: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    order: str = Query("desc", pattern="^(asc|desc)$"),
):
    try:
        results = activity_log_col.get(
            where={"patient_id": {"$eq": patient_id}},
            include=["metadatas"],
        )
    except Exception:
        return {"total": 0, "offset": offset, "limit": limit, "entries": []}

    entries = [
        {
            "action": m.get("action", ""),
            "detail": m.get("detail", ""),
            "timestamp": m.get("timestamp", ""),
        }
        for m in results["metadatas"]
        if m
    ]
    entries.sort(key=lambda x: x["timestamp"], reverse=(order == "desc"))

    total = len(entries)
    page = entries[offset : offset + limit]
    return {"total": total, "offset": offset, "limit": limit, "entries": page}
