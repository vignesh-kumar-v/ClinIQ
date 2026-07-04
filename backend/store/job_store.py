from typing import Optional

jobs: dict[str, dict] = {}


def init(job_id: str, total: int = 0) -> None:
    jobs[job_id] = {
        "status": "processing",
        "progress": 0,
        "total": total,
        "message": "",
    }


def update(job_id: str, progress: int, message: str = "") -> None:
    if job_id in jobs:
        jobs[job_id]["progress"] = progress
        jobs[job_id]["message"] = message


def complete(job_id: str) -> None:
    if job_id in jobs:
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["progress"] = jobs[job_id]["total"]


def fail(job_id: str, error: str) -> None:
    if job_id in jobs:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = error


def get(job_id: str) -> Optional[dict]:
    return jobs.get(job_id)
