"""Tests for refresh_report.py — the failure traceback the orchestrators
leave for GET /api/refresh to report.

The contract these pin is cross-language: web/app/api/refresh/route.ts
reads this file, matches on `pid`, and returns `failed_at` + `traceback`.
"""

from __future__ import annotations

import json
import os

from refresh_report import report_path, write_failure_report


def _run_failing_task(mode: str = "daily") -> None:
    try:
        raise ValueError("anthropic exploded")
    except Exception:
        write_failure_report(mode)


def test_writes_traceback_keyed_by_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "anomaly_radar.duckdb"))
    _run_failing_task()

    report = json.loads((tmp_path / "refresh_error.json").read_text())
    assert report["pid"] == os.getpid(), "route.ts pairs the report to a run by pid"
    assert report["mode"] == "daily"
    assert "ValueError: anthropic exploded" in report["traceback"]
    assert report["failed_at"]


def test_report_sits_beside_the_status_file(tmp_path, monkeypatch):
    """resolveStatusPath() in route.ts puts refresh_status.json in the DB's
    directory; the report has to land in that same directory or the API
    will never find it."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "anomaly_radar.duckdb"))
    assert report_path() == tmp_path / "refresh_error.json"


def test_falls_back_to_tmpdir_without_db_path(tmp_path, monkeypatch):
    monkeypatch.delenv("DB_PATH", raising=False)
    assert report_path().name == "refresh_error.json"
    assert report_path().parent.is_dir()


def test_truncates_a_huge_traceback(tmp_path, monkeypatch):
    """The traceback rides in an HTTP response and a workflow log."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "anomaly_radar.duckdb"))
    try:
        raise ValueError("x" * 20_000)
    except Exception:
        write_failure_report("weekly")

    report = json.loads((tmp_path / "refresh_error.json").read_text())
    assert len(report["traceback"]) < 5_000
    assert report["traceback"].startswith("...(truncated)...")


def test_never_raises_when_the_report_cannot_be_written(tmp_path, monkeypatch):
    """It runs inside an `except` block on a process that is already
    failing; a reporting error must not mask the real traceback."""
    unwritable = tmp_path / "ro"
    unwritable.mkdir()
    monkeypatch.setenv("DB_PATH", str(unwritable / "anomaly_radar.duckdb"))
    os.chmod(unwritable, 0o500)
    try:
        _run_failing_task("weekly")  # must not raise
    finally:
        os.chmod(unwritable, 0o700)
