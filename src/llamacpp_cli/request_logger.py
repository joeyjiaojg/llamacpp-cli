"""Request logging for replay/debugging."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

# Headers that contain credentials and must never be written to disk.
_REDACTED_HEADERS: frozenset[str] = frozenset(
    {
        "authorization",
        "x-api-key",
        "proxy-authorization",
        "cookie",
        "set-cookie",
    }
)


@dataclass
class RequestRecord:
    """Immutable snapshot of a single HTTP round-trip."""

    timestamp: float
    method: str
    path: str
    headers: dict[str, str]
    body: bytes
    response_status: int
    response_body: bytes | None = None
    error: str | None = None


@dataclass
class RequestLogger:
    """Logs requests to an in-memory ring buffer and optionally to a JSONL file.

    Usage::

        logger = RequestLogger(log_file=Path("requests.jsonl"), max_records=500)
        record = RequestRecord(
            timestamp=time.time(),
            method="POST",
            path="/v1/chat/completions",
            headers=dict(request.headers),
            body=body_bytes,
            response_status=200,
        )
        logger.record(record)
    """

    log_file: Path | None = None
    max_records: int = 1000
    log_failed_only: bool = False

    _records: list[RequestRecord] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, req: RequestRecord) -> None:
        """Add *req* to the in-memory buffer and (optionally) to the log file.

        When *log_failed_only* is True only requests whose response_status is
        >= 400 are retained.
        """
        if self.log_failed_only and req.response_status < 400:
            return

        self._records.append(req)

        # Trim the oldest entry when the ring buffer overflows.
        if len(self._records) > self.max_records:
            self._records.pop(0)

        if self.log_file is not None:
            self._append_to_file(req)

    def get_recent(self, limit: int = 100) -> list[RequestRecord]:
        """Return up to *limit* most-recently recorded requests."""
        return self._records[-limit:]

    def get_failed(self) -> list[RequestRecord]:
        """Return all recorded requests with response_status >= 400."""
        return [r for r in self._records if r.response_status >= 400]

    def clear(self) -> None:
        """Discard all in-memory records (log file is not affected)."""
        self._records.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_to_file(self, req: RequestRecord) -> None:
        """Append *req* as a single JSON line to *self.log_file*.

        Sensitive headers (Authorization, Cookie, …) are redacted so that the
        log file is safe to share for debugging.
        """
        assert self.log_file is not None  # guarded by caller
        record_dict: dict = {
            "timestamp": req.timestamp,
            "method": req.method,
            "path": req.path,
            "headers": {
                k: v
                for k, v in req.headers.items()
                if k.lower() not in _REDACTED_HEADERS
            },
            "body": req.body.decode("utf-8", errors="replace"),
            "response_status": req.response_status,
        }
        if req.error is not None:
            record_dict["error"] = req.error

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record_dict) + "\n")


# ---------------------------------------------------------------------------
# Replay utility
# ---------------------------------------------------------------------------


def load_records_from_file(log_file: Path) -> list[dict]:
    """Read all JSONL records from *log_file*.

    Returns a list of raw dicts (one per line).  Invalid / empty lines are
    silently skipped.
    """
    records: list[dict] = []
    if not log_file.exists():
        return records
    with log_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


async def replay_requests(
    log_file: Path,
    target: str,
    *,
    timeout: float = 30.0,
    verbose: bool = False,
) -> list[dict]:
    """Replay all requests stored in *log_file* against *target*.

    Args:
        log_file: Path to the JSONL request log.
        target:   Base URL of the target server, e.g. ``http://localhost:8080``.
        timeout:  Per-request HTTP timeout in seconds.
        verbose:  Print each request/response to stdout when True.

    Returns:
        List of result dicts with keys ``path``, ``status``, ``ok``, and
        optionally ``error``.
    """
    import httpx  # local import so the module is usable without httpx

    records = load_records_from_file(log_file)
    results: list[dict] = []
    target = target.rstrip("/")

    async with httpx.AsyncClient(timeout=timeout) as client:
        for rec in records:
            method: str = rec.get("method", "GET").upper()
            path: str = rec.get("path", "/")
            headers: dict[str, str] = rec.get("headers", {})
            body_str: str = rec.get("body", "")
            body_bytes: bytes = body_str.encode("utf-8") if body_str else b""

            url = target + path
            try:
                resp = await client.request(
                    method,
                    url,
                    headers=headers,
                    content=body_bytes,
                )
                result: dict = {"path": path, "status": resp.status_code, "ok": resp.is_success}
                if verbose:
                    print(f"  {method} {path} → {resp.status_code}")
            except Exception as exc:
                result = {"path": path, "status": None, "ok": False, "error": str(exc)}
                if verbose:
                    print(f"  {method} {path} → ERROR: {exc}")

            results.append(result)

    return results
