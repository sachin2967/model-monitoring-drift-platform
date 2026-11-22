"""Durable report history for a single-host monitoring pilot."""

import json
import sqlite3
from pathlib import Path
from uuid import uuid4


def connect(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("""CREATE TABLE IF NOT EXISTS reports
        (report_id TEXT PRIMARY KEY, evaluated_at TEXT NOT NULL, report_json TEXT NOT NULL)""")
    return connection


def record_report(path: Path, report: dict):
    report_id = uuid4().hex
    connection = connect(path)
    try:
        with connection:
            connection.execute(
                "INSERT INTO reports VALUES (?,?,?)",
                (report_id, report["as_of"], json.dumps(report, allow_nan=False)),
            )
    finally:
        connection.close()
    return {"report_id": report_id, **report}


def read_report(path: Path, report_id: str):
    connection = connect(path)
    try:
        row = connection.execute(
            "SELECT report_json FROM reports WHERE report_id=?", (report_id,)
        ).fetchone()
        return {"report_id": report_id, **json.loads(row[0])} if row else None
    finally:
        connection.close()
