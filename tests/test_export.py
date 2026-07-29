"""Tests for export/publish.py — the DuckDB-to-JSON publisher that feeds
the Next.js frontend at build time.

Each test seeds a synthetic DuckDB and asserts the JSON files match the
contract documented in docs/superpowers/specs/2026-05-30-deployment-architecture.md
and the implementation plan.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import duckdb
import pytest

from ingestion.load_db import init_schema


# ── DB helpers (mirror tests/test_briefs.py) ─────────────────────────────


def _fresh_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "export.duckdb")
    init_schema(db_path)
    return db_path


def _insert_award(con, **kwargs) -> None:
    base = {
        "award_id": kwargs["award_id"],
        "parent_award_id": kwargs.get("parent_award_id"),
        "recipient_name": kwargs.get("recipient_name", "Test Co"),
        "recipient_uei": kwargs["recipient_uei"],
        "awarding_agency": kwargs.get("awarding_agency", "Department of Defense"),
        "awarding_sub_agency": None,
        "award_type": "DEFINITIVE CONTRACT",
        "award_description": None,
        "naics_code": kwargs.get("naics_code", "541330"),
        "naics_description": kwargs.get("naics_description", "Engineering Services"),
        "total_obligation": kwargs.get("total_obligation", 100_000.0),
        "base_and_all_options_value": kwargs.get("total_obligation", 100_000.0),
        "period_of_performance_start": date(2024, 1, 1),
        "period_of_performance_end": date(2025, 1, 1),
        "award_date": kwargs.get("award_date", date(2024, 6, 1)),
        "competition_type": kwargs.get("competition_type", "FULL AND OPEN COMPETITION"),
        "number_of_offers": 3,
        "modification_number": "0",
        "pulled_at": datetime(2026, 5, 1, 12, 0, 0),
    }
    cols = list(base.keys())
    placeholders = ", ".join(["?"] * len(cols))
    con.execute(
        f"INSERT INTO awards ({', '.join(cols)}) VALUES ({placeholders})",
        [base[c] for c in cols],
    )


def _insert_suspicion_score(
    con,
    *,
    uei: str,
    score_date: date,
    composite_score: float,
    composite_percentile_rank: float = 0.95,
    benford_score: float = 0.0,
    mod_growth_score: float = 0.0,
    new_entity_score: float = 0.0,
    isolation_score: float = 0.0,
    sole_source_concentration_score: float = 0.0,
    award_velocity_score: float = 0.0,
    detector_details: str = "{}",
) -> None:
    con.execute(
        """
        INSERT INTO suspicion_scores (
            uei, score_date, composite_score, composite_percentile_rank,
            benford_score, benford_percentile,
            new_entity_score, new_entity_percentile,
            mod_growth_score, mod_growth_percentile,
            isolation_score, isolation_percentile,
            sole_source_concentration_score, sole_source_concentration_percentile,
            award_velocity_score, award_velocity_percentile,
            detector_details, generated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            uei, score_date, composite_score, composite_percentile_rank,
            benford_score, 1.0 if benford_score > 0 else 0.0,
            new_entity_score, 1.0 if new_entity_score > 0 else 0.0,
            mod_growth_score, 1.0 if mod_growth_score > 0 else 0.0,
            isolation_score, 1.0 if isolation_score > 0 else 0.0,
            sole_source_concentration_score, 1.0 if sole_source_concentration_score > 0 else 0.0,
            award_velocity_score, 1.0 if award_velocity_score > 0 else 0.0,
            detector_details, datetime.now(),
        ],
    )


def _insert_brief(
    con,
    *,
    uei: str,
    score_date: date,
    brief_text: str,
    input_hash: str = "abc",
    model: str = "test-model",
    prompt_version: str = "v2",
) -> None:
    con.execute(
        """
        INSERT INTO entity_briefs (
            uei, score_date, input_hash, brief_text, model,
            prompt_version, generated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [uei, score_date, input_hash, brief_text, model, prompt_version, datetime.now()],
    )


def _seed_three_entities(db_path: str, score_date: date) -> None:
    """Three entities at differentiated composite scores so rank ordering
    is unambiguous in assertions."""
    con = duckdb.connect(db_path)
    try:
        # Top — Oshkosh-shaped: 4 detectors fired, mod_growth references a
        # specific worst_award_id we can find in awards.
        _insert_award(
            con, award_id="A1", recipient_uei="ENT0001",
            recipient_name="Acme Defense Corp",
            awarding_agency="Department of Defense",
            naics_description="Heavy Duty Truck Manufacturing",
            total_obligation=5_000_000.0,
            competition_type="NOT COMPETED",
        )
        _insert_award(
            con, award_id="A1_MOD", recipient_uei="ENT0001",
            recipient_name="Acme Defense Corp", parent_award_id="A1",
            total_obligation=3_500_000_000.0,
            award_date=date(2024, 8, 15),
            competition_type="NOT COMPETED",
        )
        _insert_suspicion_score(
            con,
            uei="ENT0001",
            score_date=score_date,
            composite_score=0.85,
            composite_percentile_rank=0.99,
            benford_score=0.92,
            mod_growth_score=0.95,
            isolation_score=0.80,
            sole_source_concentration_score=0.70,
            detector_details=json.dumps({
                "benford": {"n_transactions": 142, "ks_statistic": 0.128,
                            "p_value": 3.2e-22},
                "mod_growth": {"worst_award_id": "A1_MOD", "growth_ratio": 777.7,
                               "z_score": 7.24},
                "isolation": {"raw_anomaly_score": -0.78},
                "sole_source_concentration": {"ss_frac": 0.45, "z_score": 4.02},
            }),
        )

        # Middle.
        _insert_award(
            con, award_id="B1", recipient_uei="ENT0002",
            recipient_name="Middle Co.", awarding_agency="Department of Health and Human Services",
            naics_description="Pharma",
            total_obligation=200_000.0,
        )
        _insert_suspicion_score(
            con, uei="ENT0002", score_date=score_date,
            composite_score=0.50, composite_percentile_rank=0.66,
            benford_score=0.60,
            detector_details=json.dumps({"benford": {"max_z": 2.1}}),
        )

        # Low.
        _insert_award(
            con, award_id="C1", recipient_uei="ENT0003",
            recipient_name="Low Co.", awarding_agency="Department of Defense",
            naics_description="Other", total_obligation=50_000.0,
        )
        _insert_suspicion_score(
            con, uei="ENT0003", score_date=score_date,
            composite_score=0.20, composite_percentile_rank=0.33,
            isolation_score=0.30,
            detector_details=json.dumps({"isolation": {"raw_anomaly_score": -0.1}}),
        )

        # One brief for ENT0001 only — exercises the "missing brief" path
        # for the other two.
        _insert_brief(
            con, uei="ENT0001", score_date=score_date,
            brief_text=(
                "Four statistical detectors flagged Acme Defense Corp.\n\n"
                "The Benford test yielded a KS statistic of 0.128.\n\n"
                "These signals do not prove fraud."
            ),
        )
    finally:
        con.close()


# ── leaderboard.json ─────────────────────────────────────────────────────


def test_publish_writes_leaderboard_json(tmp_path):
    """leaderboard.json contains lead, featured (ranks 2..min(10, n)), and
    ranking (ranks 11..top_n). With only 3 entities seeded, lead=1,
    featured=[2,3], ranking=[]."""
    from export.publish import publish

    db_path = _fresh_db(tmp_path)
    out_dir = tmp_path / "out"
    _seed_three_entities(db_path, date(2026, 5, 29))

    publish(db_path, score_date=date(2026, 5, 29), top_n=50, out_dir=out_dir)

    payload = json.loads((out_dir / "leaderboard.json").read_text())
    assert payload["score_date"] == "2026-05-29"
    assert payload["total_scored"] == 3
    assert payload["total_flagged"] == 3

    assert payload["lead"]["uei"] == "ENT0001"
    assert payload["lead"]["rank"] == 1
    assert payload["lead"]["name"] == "Acme Defense Corp"
    # Brief text is the FULL brief on the lead (used by masthead spotlight).
    assert "Four statistical detectors" in payload["lead"]["brief_text"]

    featured_ueis = [e["uei"] for e in payload["featured"]]
    assert featured_ueis == ["ENT0002", "ENT0003"]
    assert payload["featured"][0]["rank"] == 2
    # Featured carry brief_excerpt (not full text).
    assert "brief_excerpt" in payload["featured"][0]

    # Ranking is empty (no entities past rank 10).
    assert payload["ranking"] == []


def test_publish_lead_includes_detectors_fired(tmp_path):
    from export.publish import publish

    db_path = _fresh_db(tmp_path)
    out_dir = tmp_path / "out"
    _seed_three_entities(db_path, date(2026, 5, 29))

    publish(db_path, score_date=date(2026, 5, 29), top_n=50, out_dir=out_dir)
    payload = json.loads((out_dir / "leaderboard.json").read_text())

    lead_detectors = set(payload["lead"]["detectors_fired"])
    assert lead_detectors == {"benford", "mod_growth", "isolation",
                              "sole_source_concentration"}


def test_publish_ranking_section_for_long_universe(tmp_path):
    """With > 10 entities, ranking[] contains ranks 11..top_n with the
    compact per-row schema (no brief excerpt, just detectors_fired_count)."""
    from export.publish import publish

    db_path = _fresh_db(tmp_path)
    out_dir = tmp_path / "out"

    score_date = date(2026, 5, 29)
    con = duckdb.connect(db_path)
    try:
        for i in range(15):
            uei = f"ENT{i:04d}"
            _insert_award(con, award_id=f"A{i}", recipient_uei=uei,
                          recipient_name=f"Entity {i}")
            _insert_suspicion_score(
                con, uei=uei, score_date=score_date,
                composite_score=1.0 - i * 0.05,
                composite_percentile_rank=0.5,
                benford_score=0.5,
                detector_details='{"benford": {"max_z": 2.0}}',
            )
    finally:
        con.close()

    publish(db_path, score_date=score_date, top_n=50, out_dir=out_dir)
    payload = json.loads((out_dir / "leaderboard.json").read_text())

    assert len(payload["featured"]) == 9
    assert len(payload["ranking"]) == 5  # ranks 11..15
    assert payload["ranking"][0]["rank"] == 11
    assert "detectors_fired_count" in payload["ranking"][0]
    assert "brief_excerpt" not in payload["ranking"][0]


# ── entity JSON files ────────────────────────────────────────────────────


def test_publish_writes_one_entity_json_per_top_n(tmp_path):
    """One JSON per top-N entity into entities/<uei>.json. With 3 entities
    seeded, exactly 3 entity files."""
    from export.publish import publish

    db_path = _fresh_db(tmp_path)
    out_dir = tmp_path / "out"
    _seed_three_entities(db_path, date(2026, 5, 29))

    publish(db_path, score_date=date(2026, 5, 29), top_n=50, out_dir=out_dir)

    entity_files = sorted((out_dir / "entities").glob("*.json"))
    assert {p.stem for p in entity_files} == {"ENT0001", "ENT0002", "ENT0003"}


def test_entity_json_includes_full_brief_when_available(tmp_path):
    from export.publish import publish

    db_path = _fresh_db(tmp_path)
    out_dir = tmp_path / "out"
    _seed_three_entities(db_path, date(2026, 5, 29))

    publish(db_path, score_date=date(2026, 5, 29), top_n=50, out_dir=out_dir)

    payload = json.loads((out_dir / "entities" / "ENT0001.json").read_text())
    assert payload["uei"] == "ENT0001"
    assert payload["name"] == "Acme Defense Corp"
    assert payload["agency"] == "Department of Defense"
    assert "Four statistical detectors" in payload["brief_text"]


def test_entity_json_brief_null_when_no_brief_row(tmp_path):
    """ENT0002 was seeded without a brief row; brief_text must be null."""
    from export.publish import publish

    db_path = _fresh_db(tmp_path)
    out_dir = tmp_path / "out"
    _seed_three_entities(db_path, date(2026, 5, 29))

    publish(db_path, score_date=date(2026, 5, 29), top_n=50, out_dir=out_dir)

    payload = json.loads((out_dir / "entities" / "ENT0002.json").read_text())
    assert payload["brief_text"] is None


def test_entity_json_detectors_list_emitted_only(tmp_path):
    """`detectors` list contains only the detectors with score > 0,
    with their score, percentile, and details JSON."""
    from export.publish import publish

    db_path = _fresh_db(tmp_path)
    out_dir = tmp_path / "out"
    _seed_three_entities(db_path, date(2026, 5, 29))

    publish(db_path, score_date=date(2026, 5, 29), top_n=50, out_dir=out_dir)

    payload = json.loads((out_dir / "entities" / "ENT0001.json").read_text())
    names = {d["name"] for d in payload["detectors"]}
    assert names == {"benford", "mod_growth", "isolation",
                     "sole_source_concentration"}
    by_name = {d["name"]: d for d in payload["detectors"]}
    assert by_name["mod_growth"]["details"]["worst_award_id"] == "A1_MOD"


def test_entity_json_includes_score_history_when_multi_day(tmp_path):
    """Two score_dates → history has two entries in chronological order
    with rank computed for each day."""
    from export.publish import publish

    db_path = _fresh_db(tmp_path)
    out_dir = tmp_path / "out"
    _seed_three_entities(db_path, date(2026, 5, 28))
    # Add second day for ENT0001 with a higher score.
    con = duckdb.connect(db_path)
    try:
        _insert_suspicion_score(
            con, uei="ENT0001", score_date=date(2026, 5, 29),
            composite_score=0.90, composite_percentile_rank=1.0,
            benford_score=0.95,
            detector_details='{"benford": {"max_z": 5.0}}',
        )
        _insert_brief(con, uei="ENT0001", score_date=date(2026, 5, 29),
                      brief_text="Day 2 brief.")
    finally:
        con.close()

    publish(db_path, score_date=date(2026, 5, 29), top_n=50, out_dir=out_dir)
    payload = json.loads((out_dir / "entities" / "ENT0001.json").read_text())

    assert len(payload["score_history"]) == 2
    assert payload["score_history"][0]["date"] == "2026-05-28"
    assert payload["score_history"][1]["date"] == "2026-05-29"
    assert payload["score_history"][1]["composite_score"] == pytest.approx(0.90)


def test_entity_json_score_delta_vs_previous_day(tmp_path):
    """composite_score_delta = today - yesterday; null if no prior day."""
    from export.publish import publish

    db_path = _fresh_db(tmp_path)
    out_dir = tmp_path / "out"
    _seed_three_entities(db_path, date(2026, 5, 28))
    con = duckdb.connect(db_path)
    try:
        _insert_suspicion_score(
            con, uei="ENT0001", score_date=date(2026, 5, 29),
            composite_score=0.90, composite_percentile_rank=1.0,
            benford_score=0.95,
            detector_details='{"benford": {"max_z": 5.0}}',
        )
    finally:
        con.close()

    publish(db_path, score_date=date(2026, 5, 29), top_n=50, out_dir=out_dir)
    payload = json.loads((out_dir / "entities" / "ENT0001.json").read_text())
    assert payload["composite_score_delta"] == pytest.approx(0.05, abs=1e-9)


def test_entity_json_score_delta_null_when_no_prior_day(tmp_path):
    """Single score_date → composite_score_delta is null."""
    from export.publish import publish

    db_path = _fresh_db(tmp_path)
    out_dir = tmp_path / "out"
    _seed_three_entities(db_path, date(2026, 5, 29))

    publish(db_path, score_date=date(2026, 5, 29), top_n=50, out_dir=out_dir)
    payload = json.loads((out_dir / "entities" / "ENT0001.json").read_text())
    assert payload["composite_score_delta"] is None


def test_entity_json_flagged_contracts_from_detector_details(tmp_path):
    """Contracts referenced in detector details (worst_award_id, etc.)
    appear in flagged_contracts with USAspending URL."""
    from export.publish import publish

    db_path = _fresh_db(tmp_path)
    out_dir = tmp_path / "out"
    _seed_three_entities(db_path, date(2026, 5, 29))

    publish(db_path, score_date=date(2026, 5, 29), top_n=50, out_dir=out_dir)
    payload = json.loads((out_dir / "entities" / "ENT0001.json").read_text())

    award_ids = {c["award_id"] for c in payload["flagged_contracts"]}
    assert "A1_MOD" in award_ids
    contract = next(c for c in payload["flagged_contracts"] if c["award_id"] == "A1_MOD")
    assert contract["triggered_detector"] == "mod_growth"
    assert "usaspending.gov" in contract["usaspending_url"]


# ── degenerate cases ─────────────────────────────────────────────────────


def test_publish_handles_empty_universe(tmp_path):
    """No suspicion_scores rows → leaderboard.json written with empty
    sections, no entity files."""
    from export.publish import publish

    db_path = _fresh_db(tmp_path)
    out_dir = tmp_path / "out"

    publish(db_path, score_date=date(2026, 5, 29), top_n=50, out_dir=out_dir)

    payload = json.loads((out_dir / "leaderboard.json").read_text())
    assert payload["total_flagged"] == 0
    assert payload["lead"] is None
    assert payload["featured"] == []
    assert payload["ranking"] == []
    # entities/ dir may not exist at all when there's nothing to write.
    assert not (out_dir / "entities").exists() or \
        list((out_dir / "entities").glob("*.json")) == []


def test_publish_creates_out_dir_if_missing(tmp_path):
    """out_dir is created (with parents) if not already present."""
    from export.publish import publish

    db_path = _fresh_db(tmp_path)
    out_dir = tmp_path / "deeply" / "nested" / "out"
    _seed_three_entities(db_path, date(2026, 5, 29))

    publish(db_path, score_date=date(2026, 5, 29), top_n=50, out_dir=out_dir)

    assert out_dir.exists()
    assert (out_dir / "leaderboard.json").exists()
