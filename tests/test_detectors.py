"""Unit tests for Phase 2 anomaly detectors.

Each test seeds a fresh in-memory-ish DuckDB under tmp_path with synthetic
data shaped to exercise one detector behavior, then asserts on the
detector's polars output.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import duckdb
import polars as pl
import pytest

from ingestion.load_db import init_schema


# ── Shared synthetic-DB helpers ───────────────────────────────────────────


def _fresh_db(tmp_path: Path) -> str:
    """Return the path to an empty DuckDB with the production schema."""
    db_path = str(tmp_path / "test.duckdb")
    init_schema(db_path)
    return db_path


def _insert_awards(db_path: str, rows: list[dict]) -> None:
    """Insert award rows directly via DuckDB (bypassing parquet round-trip)."""
    if not rows:
        return
    cols = list(rows[0].keys())
    placeholders = ", ".join(["?"] * len(cols))
    sql = f"INSERT INTO awards ({', '.join(cols)}) VALUES ({placeholders})"
    con = duckdb.connect(db_path)
    try:
        for r in rows:
            con.execute(sql, [r[c] for c in cols])
    finally:
        con.close()


def _insert_entities(db_path: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    placeholders = ", ".join(["?"] * len(cols))
    sql = f"INSERT INTO entities ({', '.join(cols)}) VALUES ({placeholders})"
    con = duckdb.connect(db_path)
    try:
        for r in rows:
            con.execute(sql, [r[c] for c in cols])
    finally:
        con.close()


def _award(award_id: str, uei: str, amount: float, **overrides) -> dict:
    """Synthetic award row with sensible defaults; override any field."""
    base = {
        "award_id": award_id,
        "parent_award_id": None,
        "recipient_name": "Test Co",
        "recipient_uei": uei,
        "awarding_agency": "Department of Defense",
        "awarding_sub_agency": None,
        "award_type": "DEFINITIVE CONTRACT",
        "award_description": None,
        "naics_code": "541330",
        "naics_description": "Engineering Services",
        "total_obligation": amount,
        "base_and_all_options_value": amount,
        "period_of_performance_start": date(2024, 1, 1),
        "period_of_performance_end": date(2025, 1, 1),
        "award_date": date(2024, 6, 1),
        "competition_type": "FULL AND OPEN COMPETITION",
        "number_of_offers": 3,
        "modification_number": "0",
        "pulled_at": datetime(2026, 5, 1, 12, 0, 0),
    }
    base.update(overrides)
    return base


def _entity(uei: str, registration_date: date | None = date(2020, 1, 1), **overrides) -> dict:
    base = {
        "uei": uei,
        "legal_business_name": "Test Co",
        "dba_name": None,
        "physical_address_line1": "1 Test St",
        "physical_city": "Tampa",
        "physical_state": "FL",
        "physical_zip": "33601",
        "business_type": None,
        "entity_structure": None,
        "registration_date": registration_date,
        "expiration_date": date(2027, 1, 1),
        "cage_code": None,
        "exclusion_status": "N",
        "last_pulled_at": datetime(2026, 5, 1, 12, 0, 0),
    }
    base.update(overrides)
    return base


# ── Task 1 — Benford ──────────────────────────────────────────────────────


def test_benford_flags_uniform_high_digits(tmp_path):
    """An entity whose 30 awards all start with leading digit 9 wildly
    violates Benford (which expects ~4.6% nines, observed 100%) and must
    score near 1.0."""
    from detectors.benford import detect_benford

    db_path = _fresh_db(tmp_path)
    rows = [
        _award(f"FAKE_{i}", "SUSPECT00001", amount=9000.0 + i)
        for i in range(30)
    ]
    _insert_awards(db_path, rows)

    df = detect_benford(db_path)

    assert df.height == 1
    row = df.row(0, named=True)
    assert row["uei"] == "SUSPECT00001"
    assert row["detector"] == "benford"
    assert row["score"] > 0.9
    details = json.loads(row["details"])
    assert details["n_transactions"] == 30
    assert details["observed_distribution"]["9"] == pytest.approx(1.0)


def test_benford_skips_entity_below_min_transactions(tmp_path):
    """Entities with fewer than 30 transactions are excluded from results
    because the leading-digit distribution isn't statistically meaningful."""
    from detectors.benford import detect_benford

    db_path = _fresh_db(tmp_path)
    rows = [_award(f"X_{i}", "SMALL0000001", amount=9000.0 + i) for i in range(29)]
    _insert_awards(db_path, rows)

    df = detect_benford(db_path)

    assert df.height == 0


def test_benford_passes_natural_distribution(tmp_path):
    """An entity whose leading-digit distribution matches Benford should
    score low (high p-value, low 1-p)."""
    from detectors.benford import detect_benford

    # Construct 90 awards whose leading-digit counts approximate Benford:
    # 1:30, 2:18, 3:12, 4:10, 5:8, 6:7, 7:6, 8:5, 9:4 → ~Benford proportions.
    counts = {1: 30, 2: 18, 3: 12, 4: 10, 5: 8, 6: 7, 7: 6, 8: 5, 9: 4}
    db_path = _fresh_db(tmp_path)
    rows = []
    idx = 0
    for digit, n in counts.items():
        for _ in range(n):
            # Spread the magnitude so amounts aren't degenerate.
            amount = float(digit) * 1000.0 + idx
            rows.append(_award(f"NAT_{idx}", "NATURAL00001", amount=amount))
            idx += 1
    _insert_awards(db_path, rows)

    df = detect_benford(db_path)
    assert df.height == 1
    assert df.row(0, named=True)["score"] < 0.5


def test_benford_returns_correct_schema(tmp_path):
    """All detectors share the same 4-column output schema; lock it in."""
    from detectors.benford import detect_benford

    db_path = _fresh_db(tmp_path)
    rows = [_award(f"S_{i}", "SCHEMA000001", amount=9000.0 + i) for i in range(30)]
    _insert_awards(db_path, rows)

    df = detect_benford(db_path)
    assert df.columns == ["uei", "detector", "score", "details"]
    assert df.schema["uei"] == pl.Utf8
    assert df.schema["detector"] == pl.Utf8
    assert df.schema["score"] == pl.Float64
    assert df.schema["details"] == pl.Utf8
