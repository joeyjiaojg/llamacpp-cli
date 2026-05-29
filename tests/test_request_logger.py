"""Tests for request_logger.py - request recording and replay."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from llamacpp_cli.request_logger import (
    RequestLogger,
    RequestRecord,
    load_records_from_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    *,
    method: str = "POST",
    path: str = "/v1/chat/completions",
    status: int = 200,
    headers: dict | None = None,
    body: bytes = b'{"model":"test"}',
    error: str | None = None,
) -> RequestRecord:
    return RequestRecord(
        timestamp=time.time(),
        method=method,
        path=path,
        headers=headers or {"content-type": "application/json"},
        body=body,
        response_status=status,
        error=error,
    )


# ---------------------------------------------------------------------------
# RequestRecord construction
# ---------------------------------------------------------------------------


class TestRequestRecord:
    def test_fields_accessible(self):
        rec = _make_record(status=201)
        assert rec.method == "POST"
        assert rec.response_status == 201
        assert rec.error is None

    def test_response_body_defaults_none(self):
        rec = _make_record()
        assert rec.response_body is None

    def test_error_field(self):
        rec = _make_record(error="timeout")
        assert rec.error == "timeout"


# ---------------------------------------------------------------------------
# RequestLogger.record  (basic)
# ---------------------------------------------------------------------------


class TestRequestLoggerRecord:
    def test_record_added_to_buffer(self):
        logger = RequestLogger()
        logger.record(_make_record())
        assert len(logger._records) == 1

    def test_multiple_records(self):
        logger = RequestLogger()
        for _ in range(5):
            logger.record(_make_record())
        assert len(logger._records) == 5

    def test_failed_only_skips_success(self):
        logger = RequestLogger(log_failed_only=True)
        logger.record(_make_record(status=200))
        logger.record(_make_record(status=201))
        assert len(logger._records) == 0

    def test_failed_only_keeps_4xx(self):
        logger = RequestLogger(log_failed_only=True)
        logger.record(_make_record(status=404))
        assert len(logger._records) == 1

    def test_failed_only_keeps_5xx(self):
        logger = RequestLogger(log_failed_only=True)
        logger.record(_make_record(status=503))
        assert len(logger._records) == 1

    def test_failed_only_boundary_399(self):
        logger = RequestLogger(log_failed_only=True)
        logger.record(_make_record(status=399))
        assert len(logger._records) == 0

    def test_failed_only_boundary_400(self):
        logger = RequestLogger(log_failed_only=True)
        logger.record(_make_record(status=400))
        assert len(logger._records) == 1


# ---------------------------------------------------------------------------
# Max records (ring buffer)
# ---------------------------------------------------------------------------


class TestMaxRecords:
    def test_max_records_enforced(self):
        logger = RequestLogger(max_records=3)
        for i in range(5):
            logger.record(_make_record(path=f"/path/{i}"))
        assert len(logger._records) == 3

    def test_oldest_evicted(self):
        logger = RequestLogger(max_records=2)
        logger.record(_make_record(path="/first"))
        logger.record(_make_record(path="/second"))
        logger.record(_make_record(path="/third"))
        paths = [r.path for r in logger._records]
        assert "/first" not in paths
        assert "/second" in paths
        assert "/third" in paths

    def test_max_records_one(self):
        logger = RequestLogger(max_records=1)
        logger.record(_make_record(path="/a"))
        logger.record(_make_record(path="/b"))
        assert len(logger._records) == 1
        assert logger._records[0].path == "/b"


# ---------------------------------------------------------------------------
# get_recent / get_failed
# ---------------------------------------------------------------------------


class TestGetRecentAndFailed:
    def test_get_recent_returns_last_n(self):
        logger = RequestLogger()
        for i in range(10):
            logger.record(_make_record(path=f"/p/{i}"))
        recent = logger.get_recent(limit=3)
        assert len(recent) == 3
        assert recent[-1].path == "/p/9"

    def test_get_recent_fewer_than_limit(self):
        logger = RequestLogger()
        logger.record(_make_record())
        recent = logger.get_recent(limit=100)
        assert len(recent) == 1

    def test_get_failed_empty_when_no_failures(self):
        logger = RequestLogger()
        logger.record(_make_record(status=200))
        assert logger.get_failed() == []

    def test_get_failed_returns_only_errors(self):
        logger = RequestLogger()
        logger.record(_make_record(status=200))
        logger.record(_make_record(status=500))
        logger.record(_make_record(status=404))
        failed = logger.get_failed()
        assert len(failed) == 2
        assert all(r.response_status >= 400 for r in failed)

    def test_clear_empties_buffer(self):
        logger = RequestLogger()
        logger.record(_make_record())
        logger.clear()
        assert logger._records == []


# ---------------------------------------------------------------------------
# File logging
# ---------------------------------------------------------------------------


class TestFileLogging:
    def test_file_created_on_first_record(self, tmp_path):
        log_file = tmp_path / "requests.jsonl"
        logger = RequestLogger(log_file=log_file)
        logger.record(_make_record())
        assert log_file.exists()

    def test_file_contains_one_line_per_record(self, tmp_path):
        log_file = tmp_path / "requests.jsonl"
        logger = RequestLogger(log_file=log_file)
        for _ in range(3):
            logger.record(_make_record())
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_file_line_is_valid_json(self, tmp_path):
        log_file = tmp_path / "requests.jsonl"
        logger = RequestLogger(log_file=log_file)
        logger.record(_make_record(path="/test", status=200))
        line = log_file.read_text().strip()
        obj = json.loads(line)
        assert obj["path"] == "/test"
        assert obj["response_status"] == 200

    def test_auth_header_redacted(self, tmp_path):
        log_file = tmp_path / "requests.jsonl"
        logger = RequestLogger(log_file=log_file)
        headers = {
            "authorization": "Bearer secret-token",
            "content-type": "application/json",
        }
        logger.record(_make_record(headers=headers))
        obj = json.loads(log_file.read_text().strip())
        assert "authorization" not in obj["headers"]
        assert obj["headers"].get("content-type") == "application/json"

    def test_x_api_key_redacted(self, tmp_path):
        log_file = tmp_path / "requests.jsonl"
        logger = RequestLogger(log_file=log_file)
        headers = {"x-api-key": "super-secret", "accept": "application/json"}
        logger.record(_make_record(headers=headers))
        obj = json.loads(log_file.read_text().strip())
        assert "x-api-key" not in obj["headers"]

    def test_cookie_redacted(self, tmp_path):
        log_file = tmp_path / "requests.jsonl"
        logger = RequestLogger(log_file=log_file)
        headers = {"cookie": "session=abc123", "host": "localhost"}
        logger.record(_make_record(headers=headers))
        obj = json.loads(log_file.read_text().strip())
        assert "cookie" not in obj["headers"]

    def test_non_sensitive_headers_preserved(self, tmp_path):
        log_file = tmp_path / "requests.jsonl"
        logger = RequestLogger(log_file=log_file)
        headers = {"content-type": "application/json", "x-request-id": "abc123"}
        logger.record(_make_record(headers=headers))
        obj = json.loads(log_file.read_text().strip())
        assert obj["headers"]["content-type"] == "application/json"
        assert obj["headers"]["x-request-id"] == "abc123"

    def test_file_parent_dirs_created(self, tmp_path):
        log_file = tmp_path / "deep" / "nested" / "requests.jsonl"
        logger = RequestLogger(log_file=log_file)
        logger.record(_make_record())
        assert log_file.exists()

    def test_no_file_logging_when_log_file_is_none(self):
        logger = RequestLogger(log_file=None)
        logger.record(_make_record())
        # Should not raise; just verify in-memory storage
        assert len(logger._records) == 1


# ---------------------------------------------------------------------------
# load_records_from_file
# ---------------------------------------------------------------------------


class TestLoadRecordsFromFile:
    def test_nonexistent_file_returns_empty(self, tmp_path):
        records = load_records_from_file(tmp_path / "no.jsonl")
        assert records == []

    def test_reads_all_lines(self, tmp_path):
        f = tmp_path / "r.jsonl"
        f.write_text(
            json.dumps({"method": "GET", "path": "/a"}) + "\n"
            + json.dumps({"method": "POST", "path": "/b"}) + "\n"
        )
        records = load_records_from_file(f)
        assert len(records) == 2
        assert records[0]["path"] == "/a"
        assert records[1]["path"] == "/b"

    def test_empty_lines_skipped(self, tmp_path):
        f = tmp_path / "r.jsonl"
        f.write_text("\n\n" + json.dumps({"x": 1}) + "\n\n")
        records = load_records_from_file(f)
        assert len(records) == 1

    def test_invalid_json_line_skipped(self, tmp_path):
        f = tmp_path / "r.jsonl"
        f.write_text(
            "not json\n"
            + json.dumps({"ok": True}) + "\n"
        )
        records = load_records_from_file(f)
        assert len(records) == 1
        assert records[0]["ok"] is True

    def test_error_field_preserved(self, tmp_path):
        log_file = tmp_path / "r.jsonl"
        logger = RequestLogger(log_file=log_file)
        logger.record(_make_record(status=500, error="upstream timeout"))
        records = load_records_from_file(log_file)
        assert records[0]["error"] == "upstream timeout"
