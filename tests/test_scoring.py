"""Unit tests for Phase 3 composite scoring.

The composite scorer runs all detectors, normalizes their disparate raw
scores via per-detector percentile ranking, then averages those
percentiles into a single composite score per UEI. Output lands in the
`suspicion_scores` table keyed by `(uei, score_date)` to support daily
historical tracking and the dashboard's Trending view.

Tests use synthetic DuckDB databases seeded directly via `_insert_awards`
so detector inputs are controlled and assertions can be exact.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import duckdb
import polars as pl
import pytest

from ingestion.load_db import init_schema


# ── Shared synthetic-DB helpers (mirrors test_detectors.py) ──────────────


def _fresh_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "scoring.duckdb")
    init_schema(db_path)
    return db_path


def _insert_awards(db_path: str, rows: list[dict]) -> None:
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


def _award(award_id: str, uei: str, amount: float, **overrides) -> dict:
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


def _seed_multi_detector_signal(db_path: str) -> dict[str, str]:
    """Populate awards so several detectors flag entities with varying
    intensity. Returns a name -> uei map so tests can assert on specific
    entities by role rather than memorizing UEIs.
    """
    awards = []

    # `benford_only`: 30 awards starting with 9 → Benford flags strongly,
    # no other detector triggers.
    for i in range(30):
        awards.append(_award(
            f"BO_{i}", "BENFORDONLY1", amount=9000.0 + i,
        ))

    # `new_entity_only`: a single $5M sole-source first-ever award. Only
    # new_entity flags. Lots of awards for the same UEI to keep Benford
    # under the 30-transaction floor? Actually 1 award is enough — Benford
    # skips entities below 30, so new_entity is the only signal.
    awards.append(_award(
        "NE_0", "NEWENTITYONLY", amount=5_000_000.0,
        award_date=date(2024, 1, 1),
        competition_type="NOT COMPETED",
    ))

    # `multi_signal`: hit by Benford (30 9-amounts) AND mod_growth (a
    # parent contract that grows 50x relative to NAICS peers). NOT hit
    # by new_entity (lifetime award count > 5 cap). Should still score
    # higher composite than entities flagged by a single detector.
    for i in range(30):
        awards.append(_award(
            f"MS_BF_{i}", "MULTISIGNAL1", amount=9000.0 + i,
            award_date=date(2024, 6, 1),
        ))
    # Parent with high mod growth.
    awards.append(_award(
        "MS_P_0", "MULTISIGNAL1", amount=100_000.0,
        parent_award_id="MSP", modification_number="0",
        award_date=date(2024, 7, 1),
    ))
    awards.append(_award(
        "MS_P_1", "MULTISIGNAL1", amount=5_000_000.0,
        parent_award_id="MSP", modification_number="P00001",
        award_date=date(2024, 8, 1),
    ))
    # mod_growth NAICS peers — 20 entities each growing 1.2x so the
    # MULTISIGNAL1 ratio of 50x stands out as a z-score outlier.
    for i in range(20):
        peer = f"MSPEER{i:06d}"
        awards.append(_award(
            f"MSP_PEER_{i}_0", peer, amount=100_000.0,
            parent_award_id=f"MSPP_{i}", modification_number="0",
        ))
        awards.append(_award(
            f"MSP_PEER_{i}_1", peer, amount=20_000.0,
            parent_award_id=f"MSPP_{i}", modification_number="P00001",
        ))

    # 10 quiet peers — entities with one $50K competed award each. These
    # exist in the DB but no detector flags them; they should NOT appear
    # in suspicion_scores.
    for i in range(10):
        awards.append(_award(
            f"Q_{i}", f"QUIET{i:07d}", amount=50_000.0,
            award_date=date(2024, 6, 1),
        ))

    _insert_awards(db_path, awards)

    return {
        "benford_only": "BENFORDONLY1",
        "new_entity_only": "NEWENTITYONLY",
        "multi_signal": "MULTISIGNAL1",
        "quiet": "QUIET0000000",  # an example quiet UEI
    }


# ── Schema contract ──────────────────────────────────────────────────────


def test_composite_emits_expected_columns(tmp_path):
    """The composite scorer returns a DataFrame with all per-detector and
    composite columns plus the merged detector_details JSON."""
    from scoring.composite import compute_composite_scores

    db_path = _fresh_db(tmp_path)
    _seed_multi_detector_signal(db_path)

    df = compute_composite_scores(db_path, score_date=date(2026, 5, 27))

    expected_cols = {
        "uei",
        "score_date",
        "composite_score",
        "composite_percentile_rank",
        "benford_score",
        "benford_percentile",
        "new_entity_score",
        "new_entity_percentile",
        "mod_growth_score",
        "mod_growth_percentile",
        "isolation_score",
        "isolation_percentile",
        "sole_source_concentration_score",
        "sole_source_concentration_percentile",
        "award_velocity_score",
        "award_velocity_percentile",
        "detector_details",
        "generated_at",
    }
    assert set(df.columns) == expected_cols
    assert df.schema["composite_score"] == pl.Float64
    assert df.schema["composite_percentile_rank"] == pl.Float64
    assert df.schema["score_date"] == pl.Date


# ── Universe + missing-detector handling ─────────────────────────────────


def test_composite_skips_entities_no_detector_flagged(tmp_path):
    """Entities with awards but flagged by zero detectors don't appear in
    suspicion_scores. Reduces table size and keeps the leaderboard signal."""
    from scoring.composite import compute_composite_scores

    db_path = _fresh_db(tmp_path)
    roles = _seed_multi_detector_signal(db_path)

    df = compute_composite_scores(db_path, score_date=date(2026, 5, 27))

    flagged_ueis = set(df["uei"].to_list())
    # The three signal-bearing UEIs must appear.
    assert roles["benford_only"] in flagged_ueis
    assert roles["new_entity_only"] in flagged_ueis
    assert roles["multi_signal"] in flagged_ueis
    # All ten quiet peers must NOT appear.
    for i in range(10):
        assert f"QUIET{i:07d}" not in flagged_ueis


def test_composite_missing_detector_score_is_zero(tmp_path):
    """An entity flagged by only one detector has 0.0 for the others'
    raw scores and 0.0 for their percentiles."""
    from scoring.composite import compute_composite_scores

    db_path = _fresh_db(tmp_path)
    roles = _seed_multi_detector_signal(db_path)

    df = compute_composite_scores(db_path, score_date=date(2026, 5, 27))

    benford_only = df.filter(pl.col("uei") == roles["benford_only"]).row(0, named=True)
    assert benford_only["benford_score"] > 0.0
    assert benford_only["new_entity_score"] == 0.0
    assert benford_only["mod_growth_score"] == 0.0
    assert benford_only["isolation_score"] == 0.0
    assert benford_only["new_entity_percentile"] == 0.0
    assert benford_only["mod_growth_percentile"] == 0.0


# ── Composite math ──────────────────────────────────────────────────────


def test_composite_score_is_mean_of_detector_percentiles(tmp_path):
    """composite_score must equal the simple mean of the four
    per-detector percentile ranks for that entity."""
    from scoring.composite import compute_composite_scores

    db_path = _fresh_db(tmp_path)
    _seed_multi_detector_signal(db_path)

    df = compute_composite_scores(db_path, score_date=date(2026, 5, 27))

    for row in df.iter_rows(named=True):
        expected = (
            row["benford_percentile"]
            + row["new_entity_percentile"]
            + row["mod_growth_percentile"]
            + row["isolation_percentile"]
            + row["sole_source_concentration_percentile"]
            + row["award_velocity_percentile"]
        ) / 6.0
        assert row["composite_score"] == pytest.approx(expected, abs=1e-9), (
            f"Composite math broken for {row['uei']}: "
            f"got {row['composite_score']}, expected {expected}"
        )


def test_composite_more_detector_hits_raise_composite(tmp_path):
    """A controllable check that the composite math actually rewards multi-detector hits.

    Constructed against `_seed_multi_detector_signal` rather than asserting
    a specific ranking outcome (synthetic data is too fragile for that —
    isolation forest with a small universe can flag any sufficiently-
    extreme entity). Instead asserts the formula property: if entity A has
    >= each per-detector percentile of entity B, then A.composite >= B.composite.
    """
    from scoring.composite import compute_composite_scores

    db_path = _fresh_db(tmp_path)
    _seed_multi_detector_signal(db_path)

    df = compute_composite_scores(db_path, score_date=date(2026, 5, 27))

    pct_cols = [
        "benford_percentile",
        "new_entity_percentile",
        "mod_growth_percentile",
        "isolation_percentile",
        "sole_source_concentration_percentile",
        "award_velocity_percentile",
    ]
    rows = list(df.iter_rows(named=True))
    # Every pair where A dominates B on every detector percentile must
    # produce A.composite >= B.composite (strict if any strict).
    for a in rows:
        for b in rows:
            if a["uei"] == b["uei"]:
                continue
            a_dominates = all(a[c] >= b[c] for c in pct_cols)
            if a_dominates:
                assert a["composite_score"] >= b["composite_score"], (
                    f"{a['uei']} dominates {b['uei']} on all detectors but "
                    f"has lower composite ({a['composite_score']} vs {b['composite_score']})"
                )


def test_composite_percentile_rank_is_normalized(tmp_path):
    """composite_percentile_rank values land in [0, 1]; the maximum is exactly 1.0."""
    from scoring.composite import compute_composite_scores

    db_path = _fresh_db(tmp_path)
    _seed_multi_detector_signal(db_path)

    df = compute_composite_scores(db_path, score_date=date(2026, 5, 27))

    pcts = df["composite_percentile_rank"].to_list()
    assert all(0.0 <= p <= 1.0 for p in pcts)
    assert max(pcts) == pytest.approx(1.0, abs=1e-9)


# ── detector_details JSON merging ───────────────────────────────────────


def test_composite_merges_detector_details(tmp_path):
    """`detector_details` is a JSON object with one key per detector that
    flagged the entity; non-flagging detectors are absent (not null)."""
    from scoring.composite import compute_composite_scores

    db_path = _fresh_db(tmp_path)
    roles = _seed_multi_detector_signal(db_path)

    df = compute_composite_scores(db_path, score_date=date(2026, 5, 27))
    by_uei = {row["uei"]: row for row in df.iter_rows(named=True)}

    multi = json.loads(by_uei[roles["multi_signal"]]["detector_details"])
    assert "benford" in multi
    assert "mod_growth" in multi
    # new_entity is correctly suppressed by the lifetime-awards cap:
    # MULTISIGNAL1 has 32 awards, well above the 5-award default.
    assert "new_entity" not in multi

    benford_only = json.loads(by_uei[roles["benford_only"]]["detector_details"])
    assert set(benford_only.keys()) == {"benford"}
    # The merged details preserve per-detector internal structure.
    assert "n_transactions" in benford_only["benford"]


# ── Persistence ─────────────────────────────────────────────────────────


def test_composite_writes_to_suspicion_scores_table(tmp_path):
    """After running, the suspicion_scores table contains the same rows
    that the scorer returned."""
    from scoring.composite import compute_composite_scores

    db_path = _fresh_db(tmp_path)
    _seed_multi_detector_signal(db_path)

    df = compute_composite_scores(db_path, score_date=date(2026, 5, 27))

    con = duckdb.connect(db_path, read_only=True)
    try:
        n = con.execute("SELECT COUNT(*) FROM suspicion_scores").fetchone()[0]
        assert n == df.height
        # Spot-check one row round-trips through the DB.
        row = con.execute(
            "SELECT composite_score, composite_percentile_rank "
            "FROM suspicion_scores WHERE uei = ? AND score_date = ?",
            ["MULTISIGNAL1", date(2026, 5, 27)],
        ).fetchone()
        assert row is not None
        assert row[0] > 0.0
    finally:
        con.close()


def test_composite_same_day_rerun_upserts(tmp_path):
    """Running twice on the same score_date doesn't duplicate rows;
    INSERT OR REPLACE on (uei, score_date) overwrites."""
    from scoring.composite import compute_composite_scores

    db_path = _fresh_db(tmp_path)
    _seed_multi_detector_signal(db_path)

    compute_composite_scores(db_path, score_date=date(2026, 5, 27))
    n1 = _count_rows(db_path, "suspicion_scores")

    compute_composite_scores(db_path, score_date=date(2026, 5, 27))
    n2 = _count_rows(db_path, "suspicion_scores")

    assert n1 == n2, f"Same-day re-run duplicated rows: {n1} -> {n2}"


def test_composite_same_day_rerun_clears_shrunken_universe(tmp_path):
    """Regression: if a detector's universe shrinks between two same-day
    runs (e.g., a previously-flagged entity no longer triggers), the
    stale rows for that score_date must be cleared. Otherwise the
    leaderboard inherits ghost rows from a prior, looser detector run.

    Reproduced by running the scorer twice on the same date with
    different award seeds — first seed produces an entity in the
    universe, second seed (cleared awards table) produces none.
    """
    from scoring.composite import compute_composite_scores

    db_path = _fresh_db(tmp_path)
    _seed_multi_detector_signal(db_path)
    compute_composite_scores(db_path, score_date=date(2026, 5, 27))
    n_first = _count_rows(db_path, "suspicion_scores")
    assert n_first > 0, "First run should produce rows"

    # Wipe awards — second run has empty universe.
    con = duckdb.connect(db_path)
    try:
        con.execute("DELETE FROM awards")
    finally:
        con.close()
    compute_composite_scores(db_path, score_date=date(2026, 5, 27))
    n_second = _count_rows(db_path, "suspicion_scores")
    assert n_second == 0, (
        f"Same-day re-run with empty universe should wipe stale rows; "
        f"got {n_second} (was {n_first})"
    )


def test_composite_different_days_creates_new_rows(tmp_path):
    """Running on two different score_dates creates two rows per entity,
    enabling the daily historical tracking the Trending view depends on."""
    from scoring.composite import compute_composite_scores

    db_path = _fresh_db(tmp_path)
    _seed_multi_detector_signal(db_path)

    compute_composite_scores(db_path, score_date=date(2026, 5, 26))
    n1 = _count_rows(db_path, "suspicion_scores")

    compute_composite_scores(db_path, score_date=date(2026, 5, 27))
    n2 = _count_rows(db_path, "suspicion_scores")

    assert n2 == 2 * n1, (
        f"Two distinct score_dates should yield 2x rows: {n1} -> {n2}"
    )


# ── Degenerate cases ────────────────────────────────────────────────────


def test_composite_handles_empty_db(tmp_path):
    """An empty awards table yields an empty suspicion_scores table without
    crashing."""
    from scoring.composite import compute_composite_scores

    db_path = _fresh_db(tmp_path)

    df = compute_composite_scores(db_path, score_date=date(2026, 5, 27))

    assert df.height == 0
    assert _count_rows(db_path, "suspicion_scores") == 0


# ── helpers ─────────────────────────────────────────────────────────────


def _count_rows(db_path: str, table: str) -> int:
    con = duckdb.connect(db_path, read_only=True)
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()
