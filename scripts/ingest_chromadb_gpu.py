#!/usr/bin/env python3
"""
ClinicalRecall — GPU-Optimized ChromaDB Ingestion Pipeline.

Same as ingest_chromadb.py but auto-tuned for NVIDIA GPUs:
  - Auto-detects CUDA and uses fp16
  - Sweeps batch sizes to find the optimal throughput before starting
  - Falls back to CPU if no GPU found

Usage:
    python ingest_chromadb_gpu.py [--batch-size N] [--reset]
    python ingest_chromadb_gpu.py --only synthea
    python ingest_chromadb_gpu.py --only patients
    python ingest_chromadb_gpu.py --only mtsamples
    python ingest_chromadb_gpu.py --only logs

Prerequisites:
    pip install chromadb sentence-transformers torch
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
CHROMA_PATH = Path("chroma_db")
PROGRESS_DIR = Path("chroma_ingestion_progress_gpu")
FHIR_DIR = Path("data/fhir")
NOTES_DIR = Path("data/clinician_notes")
MTSAMPLES_DB = Path("data/mtsamples_staging.db")
LOGS_DB = Path("data/logs.db")

AUTOTUNE_SAMPLES = 200
AUTOTUNE_CANDIDATES = [16, 32, 64, 128, 256, 512]


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _autotune_batch_size(device: str) -> int:
    if device == "cpu":
        print("  CPU detected — using batch_size=32")
        return 32

    from sentence_transformers import SentenceTransformer

    model_kwargs = {"dtype": torch.float16} if device in ("cuda", "mps") else {}
    print(f"  Loading model on {device} for auto-tune...")
    model = SentenceTransformer(EMBED_MODEL_NAME, device=device, model_kwargs=model_kwargs)
    model.max_seq_length = 1024

    texts = [
        "Patient: John Smith, 45y, male. Hypertension onset 2019-03-15 during General examination of patient (procedure)",
        "HbA1c: 7.2 on 2024-03-15 during Follow-up encounter (procedure)",
        "Lisinopril 10 MG Oral Tablet Take 1 tablet daily started 2019-03-22 during General examination of patient (procedure)",
        "Visit on 2024-06-10, reason: Diabetes mellitus type 2",
        "Hemoglobin A1c/Hemoglobin.total in Blood: 7.2 % on 2024-03-15 during Follow-up encounter (procedure)",
    ] * (AUTOTUNE_SAMPLES // 5)

    print(f"  Sweeping batch sizes with {len(texts)} sample texts...")
    best_bs = None
    best_rate = 0

    for bs in AUTOTUNE_CANDIDATES:
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
            t0 = time.time()
            model.encode(texts, batch_size=bs, normalize_embeddings=True, show_progress_bar=False)
            elapsed = time.time() - t0
            rate = len(texts) / elapsed
            print(f"    batch={bs:3d}: {elapsed:.2f}s ({rate:.0f} texts/s)")
            if rate > best_rate:
                best_rate = rate
                best_bs = bs
        except torch.cuda.OutOfMemoryError:
            print(f"    batch={bs:3d}: OOM — stopping sweep")
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    batch={bs:3d}: OOM — stopping sweep")
                break
            print(f"    batch={bs:3d}: error — {e}")
            break

    if best_bs is None:
        print("  Auto-tune failed — falling back to batch_size=16")
        return 16

    print(f"  Optimal batch size: {best_bs} ({best_rate:.0f} texts/s)")
    return best_bs


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ClinicalRecall GPU-Optimized ChromaDB Ingestion Pipeline")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Embedding batch size (auto-tuned if not set)")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate all collections")
    parser.add_argument("--only", choices=["normalize", "patients", "mtsamples", "logs", "synthea"],
                        help="Run only one step (synthea = reset + re-ingest FHIR + notes only)")
    parser.add_argument("--skip-autotune", action="store_true", help="Skip auto-tune, use default batch_size=32")
    args = parser.parse_args()

    device = _detect_device()
    print(f"Device: {device}")

    if args.skip_autotune:
        batch_size = args.batch_size or 32
        print(f"Batch size: {batch_size} (manual, auto-tune skipped)")
    elif args.batch_size:
        batch_size = args.batch_size
        print(f"Batch size: {batch_size} (manual override)")
    else:
        batch_size = _autotune_batch_size(device)

    # Import the shared pipeline and override globals
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ingest_chromadb as _ic

    _ic.BATCH_SIZE = batch_size
    _ic.CHROMA_PATH = CHROMA_PATH
    _ic.PROGRESS_DIR = PROGRESS_DIR
    _ic.FHIR_DIR = FHIR_DIR
    _ic.NOTES_DIR = NOTES_DIR
    _ic.MTSAMPLES_DB = MTSAMPLES_DB
    _ic.LOGS_DB = LOGS_DB

    import chromadb
    _ic.chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

    _ic._get_embed_model()

    print("\u2554" + "\u2550" * 58 + "\u2557")
    print("\u2551     ClinicalRecall — GPU Ingestion Pipeline            \u2551")
    print("\u2560" + "\u2550" * 58 + "\u2563")
    print(f"\u2551  Embedding model : {EMBED_MODEL_NAME:<36}\u2551")
    print(f"\u2551  Device          : {device:<36}\u2551")
    print(f"\u2551  Batch size      : {batch_size:<36}\u2551")
    print("\u255a" + "\u2550" * 58 + "\u255d")

    if args.reset:
        print("\nResetting all collections...")
        for name in ["synthea_structured", "mtsamples_knowledge", "patient_index"]:
            try:
                _ic.chroma_client.delete_collection(name)
                print(f"  Dropped: {name}")
            except Exception:
                pass
        import shutil
        if PROGRESS_DIR.exists():
            shutil.rmtree(PROGRESS_DIR)
            PROGRESS_DIR.mkdir()
            print("  Cleared progress files")

    overall_start = time.time()

    if args.only == "normalize":
        _ic.step_normalize_sections()
    elif args.only == "patients":
        _ic.step_embed_patients()
    elif args.only == "mtsamples":
        _ic.step_embed_mtsamples()
    elif args.only == "logs":
        _ic.step_setup_logs()
    elif args.only == "synthea":
        print("\nResetting synthea collections only (mtsamples_knowledge untouched)...")
        for name in ["synthea_structured", "patient_index"]:
            try:
                _ic.chroma_client.delete_collection(name)
                print(f"  Dropped: {name}")
            except Exception:
                pass
        import shutil
        for pfile in ["synthea.txt", "clinician_notes.txt"]:
            ppath = PROGRESS_DIR / pfile
            if ppath.exists():
                ppath.unlink()
                print(f"  Cleared progress: {pfile}")
        _ic.step_embed_patients()
    else:
        _ic.step_normalize_sections()
        _ic.step_embed_patients()
        _ic.step_embed_mtsamples()
        _ic.step_setup_logs()

    overall_elapsed = time.time() - overall_start
    m, s = divmod(int(overall_elapsed), 60)
    h, m = divmod(m, 60)

    print(f"\n{'=' * 60}")
    print(f"INGESTION COMPLETE in {h}h {m}m {s}s")
    print(f"{'=' * 60}")

    for name in ["synthea_structured", "mtsamples_knowledge", "patient_index"]:
        try:
            col = _ic.chroma_client.get_collection(name)
            print(f"  {name}: {col.count():,} chunks")
        except Exception:
            print(f"  {name}: NOT FOUND")


if __name__ == "__main__":
    main()
