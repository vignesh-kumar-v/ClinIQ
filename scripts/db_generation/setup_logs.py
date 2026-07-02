"""
Create activity_log and audit_log SQLite tables in logs.db.

Usage:
    python setup_logs.py

Provides helper functions for use by the app:
    from setup_logs import log_activity, log_audit
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

LOGS_DB = Path(__file__).parent / "data" / "logs.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(LOGS_DB)
    conn.row_factory = sqlite3.Row
    return conn


def setup():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS activity_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            query_text  TEXT NOT NULL,
            collection  TEXT NOT NULL,
            filters     TEXT,
            n_results   INTEGER,
            latency_ms  INTEGER
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            patient_id  TEXT,
            patient_name TEXT,
            action      TEXT NOT NULL,
            query_text  TEXT,
            collection  TEXT,
            source      TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_activity_timestamp ON activity_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_patient ON audit_log(patient_id);
        CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
    """)
    conn.commit()
    conn.close()
    print(f"Logs DB ready: {LOGS_DB}")


def log_activity(
    query_text: str,
    collection: str,
    filters: dict | None = None,
    n_results: int = 0,
    latency_ms: int = 0,
):
    conn = get_conn()
    conn.execute(
        """INSERT INTO activity_log
           (timestamp, query_text, collection, filters, n_results, latency_ms)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            query_text,
            collection,
            str(filters) if filters else None,
            n_results,
            latency_ms,
        ),
    )
    conn.commit()
    conn.close()


def log_audit(
    action: str,
    patient_id: str = "",
    patient_name: str = "",
    query_text: str = "",
    collection: str = "",
    source: str = "",
):
    conn = get_conn()
    conn.execute(
        """INSERT INTO audit_log
           (timestamp, patient_id, patient_name, action, query_text, collection, source)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            patient_id,
            patient_name,
            action,
            query_text,
            collection,
            source,
        ),
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    setup()
