"""SQLite database for model metadata."""

import sqlite3

from .config import get_db_path

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS models (
    name TEXT PRIMARY KEY,
    repo_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    quantization TEXT,
    size_bytes INTEGER,
    path TEXT NOT NULL,
    downloaded_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


def _connect() -> sqlite3.Connection:
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


def add_model(
    name: str,
    repo_id: str,
    filename: str,
    path: str,
    quantization: str | None = None,
    size_bytes: int | None = None,
) -> None:
    """Register a downloaded model in the database."""
    conn = _connect()
    conn.execute(
        """INSERT OR REPLACE INTO models (name, repo_id, filename, quantization, size_bytes, path)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (name, repo_id, filename, quantization, size_bytes, path),
    )
    conn.commit()
    conn.close()


def remove_model(name: str) -> bool:
    """Remove a model from the database. Returns True if a row was deleted."""
    conn = _connect()
    cursor = conn.execute("DELETE FROM models WHERE name = ?", (name,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def get_model(name: str) -> dict | None:
    """Look up a model by name, filename, or path suffix. Returns dict or None."""
    conn = _connect()
    # Try exact name match first.
    row = conn.execute("SELECT * FROM models WHERE name = ?", (name,)).fetchone()
    if not row:
        # Fall back: match by filename or the tail of the stored path.
        row = conn.execute(
            "SELECT * FROM models WHERE filename = ? OR path LIKE ?",
            (name, f"%{name}"),
        ).fetchone()
    conn.close()
    return dict(row) if row else None


def list_models() -> list[dict]:
    """List all registered models."""
    conn = _connect()
    rows = conn.execute("SELECT * FROM models ORDER BY downloaded_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]
