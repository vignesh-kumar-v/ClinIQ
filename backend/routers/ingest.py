import json
import asyncio
from uuid import uuid4
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse
from models import IngestSyntheaRequest, IngestMTSamplesRequest
from config import synthea_col, mtsamples_col, EMBED_BATCH_SIZE
from services.embedder import embed_batch
from ingestion import synthea_parser, mtsamples_parser
from store import job_store

router = APIRouter()


@router.post("/ingest/synthea")
async def ingest_synthea(body: IngestSyntheaRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid4())
    job_store.init(job_id)
    background_tasks.add_task(run_synthea_ingestion, job_id, body.file_path)
    return {"job_id": job_id, "status": "queued"}


@router.post("/ingest/mtsamples")
async def ingest_mtsamples(body: IngestMTSamplesRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid4())
    job_store.init(job_id)
    background_tasks.add_task(run_mtsamples_ingestion, job_id, body.text, body.specialty)
    return {"job_id": job_id, "status": "queued"}


@router.get("/ingest/progress/{job_id}")
async def ingest_progress(job_id: str):
    async def event_stream():
        while True:
            status = job_store.get(job_id)
            if status is None:
                yield f"data: {json.dumps({'error': 'job not found'})}\n\n"
                break
            payload = {
                "job_id": job_id,
                "status": status["status"],
                "progress": status["progress"],
                "total": status["total"],
                "message": status["message"],
            }
            yield f"data: {json.dumps(payload)}\n\n"
            if status["status"] in ("complete", "failed"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def run_synthea_ingestion(job_id: str, file_path: str) -> None:
    try:
        with open(file_path, "r") as f:
            fhir_json = json.load(f)

        chunks = synthea_parser.parse_patient(fhir_json)
        job_store.init(job_id, total=len(chunks))

        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[i : i + EMBED_BATCH_SIZE]
            texts = [c["text"] for c in batch]
            metas = [c["metadata"] for c in batch]
            vectors = embed_batch(texts)
            ids = [str(uuid4()) for _ in batch]
            synthea_col.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)
            job_store.update(job_id, min(i + EMBED_BATCH_SIZE, len(chunks)), f"Ingested {min(i + EMBED_BATCH_SIZE, len(chunks))}/{len(chunks)}")

        job_store.complete(job_id)
    except Exception as e:
        job_store.fail(job_id, str(e))


def run_mtsamples_ingestion(job_id: str, text: str, specialty: str) -> None:
    try:
        chunks = mtsamples_parser.parse_transcription(text, specialty)
        job_store.init(job_id, total=len(chunks))

        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[i : i + EMBED_BATCH_SIZE]
            texts = [c["text"] for c in batch]
            metas = [c["metadata"] for c in batch]
            vectors = embed_batch(texts)
            ids = [str(uuid4()) for _ in batch]
            mtsamples_col.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)
            job_store.update(job_id, min(i + EMBED_BATCH_SIZE, len(chunks)), f"Ingested {min(i + EMBED_BATCH_SIZE, len(chunks))}/{len(chunks)}")

        job_store.complete(job_id)
    except Exception as e:
        job_store.fail(job_id, str(e))
