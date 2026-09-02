"""Failure reporting for the pipeline orchestrators (scripts/*.py).

The refresh API spawns these scripts detached and reads their outcome from
a status file, which records only an exit code -- so a failed cron says
"exit_code: 1" and nothing about why. Recovering the reason meant reading
the container's stderr in Railway, and on 2026-09-02 a failing daily run
had already aged out of log retention, leaving the failure unattributable.

So the orchestrator writes its own traceback next to the status file, on
the persistent volume. GET /api/refresh attaches it to a failed run, and
because the GitHub Actions poller echoes that body when it reports a
failure, the traceback lands in the workflow log on its own.

Keyed by pid, which is the same number the API records for the run: the
API spawns `python3 scripts/<mode>.py` directly, so this process IS the
child it tracked.
"""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Enough for a traceback plus the exception message, small enough to sit
# in an HTTP response and a workflow log without drowning them.
MAX_TRACEBACK_CHARS = 4000

REPORT_NAME = "refresh_error.json"


def report_path() -> Path:
    """Mirror of resolveStatusPath() in web/app/api/refresh/route.ts."""
    db_path = os.environ.get("DB_PATH")
    if db_path:
        directory = Path(db_path).parent
        if directory.is_dir():
            return directory / REPORT_NAME
    return Path(tempfile.gettempdir()) / REPORT_NAME


def write_failure_report(mode: str) -> None:
    """Persist the exception being handled. Call from an `except` block.

    Deliberately swallows its own errors: this runs while the process is
    already failing, and a broken report must not replace the real
    traceback on stderr with a confusing one from the reporter.
    """
    try:
        text = traceback.format_exc()
        if len(text) > MAX_TRACEBACK_CHARS:
            text = "...(truncated)...\n" + text[-MAX_TRACEBACK_CHARS:]
        payload = {
            "mode": mode,
            "pid": os.getpid(),
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "traceback": text,
        }
        path = report_path()
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, path)  # atomic: readers never see a partial file
    except Exception:
        pass
