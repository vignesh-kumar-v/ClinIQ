#!/usr/bin/env python3
"""
MTSamples ingestion pipeline.

Reads data/mtsamples.csv, splits each transcription into clinical sections,
cleans specialty tags, and stages chunks in SQLite for batch embedding.

Output: data/mtsamples_staging.db (SQLite)
"""

import csv
import json
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

CSV_PATH = Path("data/mtsamples.csv")
DB_PATH = Path("data/mtsamples_staging.db")
PROGRESS_FILE = Path("data/mtsamples_ingestion_progress.txt")

VALID_SPECIALTIES = {
    "Surgery", "Consult - History and Phy.", "Cardiovascular / Pulmonary",
    "Orthopedic", "Radiology", "General Medicine", "Gastroenterology",
    "Neurology", "SOAP / Chart / Progress Notes", "Obstetrics / Gynecology",
    "Urology", "Discharge Summary", "ENT - Otolaryngology", "Neurosurgery",
    "Hematology - Oncology", "Nephrology", "Emergency Room Reports",
    "Pediatrics - Neonatal", "Pain Management", "Psychiatry / Psychology",
    "Office Notes", "Podiatry", "Dermatology", "Dentistry",
    "Cosmetic / Plastic Surgery", "Letters", "Physical Medicine - Rehab",
    "Sleep Medicine", "Endocrinology", "Bariatrics", "IME-QME-Work Comp",
    "Chiropractic", "Rheumatology", "Diets and Nutritions",
    "Speech - Language", "Lab Medicine - Pathology", "Autopsy",
    "Allergy / Immunology", "Hospice - Palliative Care",
}

SECTION_PATTERN = re.compile(
    r"(?:^|,\s*)([A-Z][A-Z /&\-]{2,}?):\s*",
    re.MULTILINE,
)

MIN_SECTION_LENGTH = 20
MIN_CHUNK_LENGTH = 50


def clean_specialty(raw: str) -> Optional[str]:
    """Validate and clean specialty. Returns None if invalid."""
    cleaned = raw.strip()
    if cleaned in VALID_SPECIALTIES:
        return cleaned
    # Try case-insensitive match
    cleaned_lower = cleaned.lower()
    for valid in VALID_SPECIALTIES:
        if valid.lower() == cleaned_lower:
            return valid
    return None


def clean_keywords(raw: str) -> list[str]:
    """Parse comma-separated keywords, deduplicate, lowercase, strip."""
    if not raw or not raw.strip():
        return []
    keywords = []
    seen = set()
    for kw in raw.split(","):
        kw = kw.strip().lower()
        if kw and kw not in seen:
            seen.add(kw)
            keywords.append(kw)
    return keywords


def split_into_sections(text: str) -> list[dict]:
    """
    Split transcription text into clinical sections.

    Uses regex to find section headers like 'SUBJECTIVE:', 'PLAN:',
    'PREOPERATIVE DIAGNOSIS:', etc. Content between headers is assigned
    to the preceding section. Text before the first header is discarded
    (usually empty or preamble).
    """
    if not text or not text.strip():
        return []

    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        # No section headers found — treat entire text as one chunk
        cleaned = text.strip()
        if len(cleaned) >= MIN_CHUNK_LENGTH:
            return [{"section": "full_text", "content": cleaned}]
        return []

    sections = []
    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        # Remove trailing comma from previous section that bled into this header
        content = re.sub(r",\s*$", "", content)
        # Strip leading commas and whitespace
        content = re.sub(r"^,\s*", "", content)
        # Normalize whitespace
        content = re.sub(r"\s+", " ", content).strip()

        if len(content) >= MIN_SECTION_LENGTH:
            sections.append({"section": section_name, "content": content})

    return sections


def init_db():
    """Create SQLite staging database with chunks table."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT UNIQUE NOT NULL,
            text TEXT NOT NULL,
            specialty TEXT NOT NULL,
            section TEXT NOT NULL,
            keywords TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'mtsamples',
            row_index INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_specialty ON chunks(specialty)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_section ON chunks(section)")
    conn.commit()
    return conn


def ingest_row(conn: sqlite3.Connection, row: dict, row_index: int) -> int:
    """Process one CSV row: validate, split, insert chunks. Returns chunk count."""
    specialty = clean_specialty(row.get("medical_specialty", ""))
    if specialty is None:
        return 0

    transcription = row.get("transcription", "")
    if not transcription or not transcription.strip():
        return 0

    keywords = clean_keywords(row.get("keywords", ""))
    keywords_json = json.dumps(keywords)

    sections = split_into_sections(transcription)
    if not sections:
        return 0

    count = 0
    for sec in sections:
        chunk_id = str(uuid.uuid4())
        conn.execute(
            "INSERT OR IGNORE INTO chunks (chunk_id, text, specialty, section, keywords, row_index) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chunk_id, sec["content"], specialty, sec["section"], keywords_json, row_index),
        )
        if conn.total_changes > 0:
            count += 1

    return count


def main():
    conn = init_db()

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    print(f"Loaded {total} rows from {CSV_PATH}")

    completed = set()
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            completed = set(int(line.strip()) for line in f if line.strip().isdigit())
        print(f"Resuming: {len(completed)} already processed")

    total_chunks = 0
    skipped_empty = 0
    skipped_specialty = 0
    skipped_no_sections = 0

    for i, row in enumerate(rows):
        if i in completed:
            continue

        specialty = clean_specialty(row.get("medical_specialty", ""))
        if specialty is None:
            skipped_specialty += 1
            completed.add(i)
            with open(PROGRESS_FILE, "a") as f:
                f.write(f"{i}\n")
            continue

        transcription = row.get("transcription", "")
        if not transcription or not transcription.strip():
            skipped_empty += 1
            completed.add(i)
            with open(PROGRESS_FILE, "a") as f:
                f.write(f"{i}\n")
            continue

        chunks = ingest_row(conn, row, i)
        if chunks == 0:
            skipped_no_sections += 1
        else:
            total_chunks += chunks

        completed.add(i)
        with open(PROGRESS_FILE, "a") as f:
            f.write(f"{i}\n")

        if (i + 1) % 500 == 0:
            conn.commit()
            print(f"  Progress: {i + 1}/{total} ({total_chunks} chunks)")

    conn.commit()

    # Store metadata
    conn.execute(
        "INSERT OR REPLACE INTO ingestion_meta (key, value) VALUES (?, ?)",
        ("total_rows", str(total)),
    )
    conn.execute(
        "INSERT OR REPLACE INTO ingestion_meta (key, value) VALUES (?, ?)",
        ("total_chunks", str(total_chunks)),
    )
    conn.execute(
        "INSERT OR REPLACE INTO ingestion_meta (key, value) VALUES (?, ?)",
        ("skipped_empty", str(skipped_empty)),
    )
    conn.execute(
        "INSERT OR REPLACE INTO ingestion_meta (key, value) VALUES (?, ?)",
        ("skipped_specialty", str(skipped_specialty)),
    )
    conn.execute(
        "INSERT OR REPLACE INTO ingestion_meta (key, value) VALUES (?, ?)",
        ("skipped_no_sections", str(skipped_no_sections)),
    )
    conn.commit()

    # Print stats
    cursor = conn.execute("SELECT COUNT(*) FROM chunks")
    db_count = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(DISTINCT specialty) FROM chunks")
    unique_spec = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(DISTINCT section) FROM chunks")
    unique_sec = cursor.fetchone()[0]

    print(f"\nDone!")
    print(f"  Total rows processed: {total}")
    print(f"  Chunks created: {db_count}")
    print(f"  Unique specialties: {unique_spec}")
    print(f"  Unique section types: {unique_sec}")
    print(f"  Skipped (empty transcription): {skipped_empty}")
    print(f"  Skipped (invalid specialty): {skipped_specialty}")
    print(f"  Skipped (no valid sections): {skipped_no_sections}")
    print(f"  Database: {DB_PATH}")

    conn.close()


if __name__ == "__main__":
    main()
