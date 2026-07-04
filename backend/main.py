import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import patients, query, notes, medications, activity, ingest

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload embedding model so first request doesn't time out
    log.info("Preloading embedding model...")
    from services.embedder import _get_model
    _get_model()
    log.info("Embedding model ready.")
    yield


app = FastAPI(title="ClinIQ", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(patients.router, prefix="/api")
app.include_router(query.router, prefix="/api")
app.include_router(notes.router, prefix="/api")
app.include_router(medications.router, prefix="/api")
app.include_router(activity.router, prefix="/api")
app.include_router(ingest.router, prefix="/api")


@app.get("/")
async def root():
    return {"status": "ok", "app": "ClinIQ", "version": "1.0"}
