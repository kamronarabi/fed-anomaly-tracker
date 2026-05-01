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


# ── Task 2 — Mod growth ──────────────────────────────────────────────────


def test_mod_growth_flags_high_growth_outlier(tmp_path):
    """One entity has a parent contract that grew 50x; 20 peer NAICS contracts
    grew 1.2x. With the outlier z-score around 4.4 (using sample std), the
    sigmoid(z - 2) score saturates above 0.9."""
    from detectors.mod_growth import detect_mod_growth

    db_path = _fresh_db(tmp_path)
    rows = []

    # Suspect: parent PX, initial $100K, three mods totalling $4.9M -> ratio 50.
    rows.append(_award("PX_0", "SUSPECT00001", amount=100_000.0,
                       parent_award_id="PX", modification_number="0"))
    rows.append(_award("PX_1", "SUSPECT00001", amount=1_500_000.0,
                       parent_award_id="PX", modification_number="P00001"))
    rows.append(_award("PX_2", "SUSPECT00001", amount=1_700_000.0,
                       parent_award_id="PX", modification_number="P00002"))
    rows.append(_award("PX_3", "SUSPECT00001", amount=1_700_000.0,
                       parent_award_id="PX", modification_number="P00003"))

    # 20 peers in same NAICS, each with growth_ratio 1.2. With many peers,
    # the suspect's outlier ratio doesn't pull the mean toward itself.
    for i in range(20):
        peer_uei = f"PEER{i:08d}"
        parent = f"PEER_PARENT_{i}"
        rows.append(_award(f"{parent}_0", peer_uei, amount=100_000.0,
                           parent_award_id=parent, modification_number="0"))
        rows.append(_award(f"{parent}_1", peer_uei, amount=20_000.0,
                           parent_award_id=parent, modification_number="P00001"))

    _insert_awards(db_path, rows)

    df = detect_mod_growth(db_path)
    df = df.sort("score", descending=True)
    top = df.row(0, named=True)
    assert top["uei"] == "SUSPECT00001"
    assert top["score"] > 0.9
    details = json.loads(top["details"])
    assert details["worst_award_id"] == "PX"
    assert details["growth_ratio"] == pytest.approx(50.0, rel=0.01)


def test_mod_growth_does_not_flag_normal_growth(tmp_path):
    """An entity whose growth ratio matches the NAICS mean should score low."""
    from detectors.mod_growth import detect_mod_growth

    db_path = _fresh_db(tmp_path)
    rows = []
    # 10 peers all growing 1.2x.
    for i in range(10):
        uei = f"NORMAL{i:06d}"
        parent = f"NORM_P_{i}"
        rows.append(_award(f"{parent}_0", uei, amount=100_000.0,
                           parent_award_id=parent, modification_number="0"))
        rows.append(_award(f"{parent}_1", uei, amount=20_000.0,
                           parent_award_id=parent, modification_number="P00001"))
    _insert_awards(db_path, rows)

    df = detect_mod_growth(db_path)
    # All scores should be well below the sigmoid midpoint at z=2.
    if df.height > 0:
        assert df["score"].max() < 0.5


def test_mod_growth_returns_correct_schema(tmp_path):
    from detectors.mod_growth import detect_mod_growth

    db_path = _fresh_db(tmp_path)
    rows = [
        _award("S_0", "SCHEMA000001", amount=100_000.0,
               parent_award_id="SP", modification_number="0"),
        _award("S_1", "SCHEMA000001", amount=200_000.0,
               parent_award_id="SP", modification_number="P00001"),
        _award("S_2", "PEER00000001", amount=100_000.0,
               parent_award_id="PP", modification_number="0"),
        _award("S_3", "PEER00000001", amount=20_000.0,
               parent_award_id="PP", modification_number="P00001"),
    ]
    _insert_awards(db_path, rows)

    df = detect_mod_growth(db_path)
    assert df.columns == ["uei", "detector", "score", "details"]
    assert df.schema["score"] == pl.Float64


# ── Task 3 — New entity sole-source ─────────────────────────────────────


def test_new_entity_flags_recent_registration_with_sole_source(tmp_path):
    """Entity registered 30 days before a $1M sole-source award scores high."""
    from detectors.new_entity import detect_new_entity_sole_source

    db_path = _fresh_db(tmp_path)
    reg = date(2024, 1, 1)
    award_dt = date(2024, 1, 31)  # 30-day gap
    _insert_entities(db_path, [_entity("FRESH0000001", registration_date=reg)])
    _insert_awards(
        db_path,
        [
            _award(
                "A1", "FRESH0000001", amount=1_000_000.0,
                award_date=award_dt,
                competition_type="NOT COMPETED",
            )
        ],
    )

    df = detect_new_entity_sole_source(db_path)
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["uei"] == "FRESH0000001"
    assert row["detector"] == "new_entity"
    assert row["score"] > 0.5
    details = json.loads(row["details"])
    assert details["days_gap"] == 30
    assert details["competition_type"] == "NOT COMPETED"


def test_new_entity_does_not_flag_competed_award(tmp_path):
    """Full-and-open competition is exactly the opposite of the signal we want."""
    from detectors.new_entity import detect_new_entity_sole_source

    db_path = _fresh_db(tmp_path)
    _insert_entities(db_path, [_entity("COMPED000001", registration_date=date(2024, 1, 1))])
    _insert_awards(
        db_path,
        [
            _award(
                "A1", "COMPED000001", amount=1_000_000.0,
                award_date=date(2024, 1, 31),
                competition_type="FULL AND OPEN COMPETITION",
            )
        ],
    )

    df = detect_new_entity_sole_source(db_path)
    assert df.height == 0


def test_new_entity_does_not_flag_old_registration(tmp_path):
    """Sole-source to a 5-year-old entity is normal procurement, not a flag."""
    from detectors.new_entity import detect_new_entity_sole_source

    db_path = _fresh_db(tmp_path)
    _insert_entities(db_path, [_entity("OLD000000001", registration_date=date(2020, 1, 1))])
    _insert_awards(
        db_path,
        [
            _award(
                "A1", "OLD000000001", amount=1_000_000.0,
                award_date=date(2025, 6, 1),
                competition_type="NOT COMPETED",
            )
        ],
    )

    df = detect_new_entity_sole_source(db_path)
    assert df.height == 0


def test_new_entity_returns_correct_schema(tmp_path):
    from detectors.new_entity import detect_new_entity_sole_source

    db_path = _fresh_db(tmp_path)
    _insert_entities(db_path, [_entity("SCHEMA000002", registration_date=date(2024, 1, 1))])
    _insert_awards(
        db_path,
        [
            _award(
                "A1", "SCHEMA000002", amount=500_000.0,
                award_date=date(2024, 1, 15),
                competition_type="NOT COMPETED",
            )
        ],
    )

    df = detect_new_entity_sole_source(db_path)
    assert df.columns == ["uei", "detector", "score", "details"]
    assert df.schema["score"] == pl.Float64


# ── Task 4 — Isolation Forest ────────────────────────────────────────────


def _populate_iforest_dataset(db_path: str) -> str:
    """Insert 19 'normal' entities + 1 obvious outlier; return the
    outlier's UEI."""
    entities = []
    awards = []
    # 19 normal entities: one NAICS, single agency, one award each at
    # ~$50K, registration 5 years ago.
    for i in range(19):
        uei = f"NORM{i:08d}"
        entities.append(_entity(uei, registration_date=date(2019, 1, 1)))
        awards.append(_award(f"AN_{i}", uei, amount=50_000.0))
    # Outlier: extremely large dollars, many agencies, many NAICS,
    # mostly non-competed, brand-new registration, lots of mods.
    outlier_uei = "OUTLIER00001"
    entities.append(_entity(outlier_uei, registration_date=date(2026, 1, 1)))
    for i in range(20):
        awards.append(
            _award(
                f"AO_{i}",
                outlier_uei,
                amount=10_000_000.0,
                naics_code=f"5413{i % 5:02d}",
                awarding_agency=("DoD" if i % 2 == 0 else "HHS"),
                competition_type="NOT COMPETED",
                modification_number=("0" if i == 0 else f"P0000{i}"),
                parent_award_id="OUTLIER_P",
            )
        )
    _insert_entities(db_path, entities)
    _insert_awards(db_path, awards)
    return outlier_uei


def test_isolation_flags_obvious_outlier(tmp_path):
    """The synthetic outlier (giant dollars, many agencies, all sole-source,
    new registration) must appear in the flagged set with a high score."""
    from detectors.isolation import detect_isolation_outlier

    db_path = _fresh_db(tmp_path)
    outlier_uei = _populate_iforest_dataset(db_path)

    df = detect_isolation_outlier(db_path)
    df = df.sort("score", descending=True)
    top = df.row(0, named=True)
    assert top["uei"] == outlier_uei
    assert top["score"] > 0.5


def test_isolation_handles_missing_entity_age(tmp_path):
    """Entities with no SAM row get NaN entity_age_days, which the detector
    must impute (median) rather than crash."""
    from detectors.isolation import detect_isolation_outlier

    db_path = _fresh_db(tmp_path)
    awards = []
    entities = []
    for i in range(15):
        uei = f"ENRICHED{i:04d}"
        entities.append(_entity(uei, registration_date=date(2020, 1, 1)))
        awards.append(_award(f"E_{i}", uei, amount=100_000.0))
    for i in range(5):
        # No matching entity row.
        uei = f"BARE0000{i:04d}"
        awards.append(_award(f"B_{i}", uei, amount=100_000.0))
    _insert_entities(db_path, entities)
    _insert_awards(db_path, awards)

    df = detect_isolation_outlier(db_path)  # must not raise
    # All 20 entities are identical after imputation; IsolationForest correctly
    # returns 0 outliers — the point of the test is no crash, not a forced flag.
    assert df.columns == ["uei", "detector", "score", "details"]


def test_isolation_returns_correct_schema(tmp_path):
    from detectors.isolation import detect_isolation_outlier

    db_path = _fresh_db(tmp_path)
    _populate_iforest_dataset(db_path)
    df = detect_isolation_outlier(db_path)
    assert df.columns == ["uei", "detector", "score", "details"]
    assert df.schema["score"] == pl.Float64


def test_isolation_imputes_missing_entity_age_to_no_nans(tmp_path):
    """Regression: nulls/NaNs from missing SAM rows must be median-imputed
    to real numbers BEFORE the feature matrix is built. Previously the
    `.otherwise(None)` branch produced polars-null which `fill_nan` skipped,
    silently feeding NaN to sklearn."""
    import numpy as np
    from detectors.isolation import _build_features, detect_isolation_outlier

    db_path = _fresh_db(tmp_path)
    awards = []
    entities = []
    for i in range(15):
        uei = f"ENRICHED{i:04d}"
        entities.append(_entity(uei, registration_date=date(2020, 1, 1)))
        awards.append(_award(f"E_{i}", uei, amount=100_000.0))
    for i in range(5):
        uei = f"BARE0000{i:04d}"
        awards.append(_award(f"B_{i}", uei, amount=100_000.0))
    _insert_entities(db_path, entities)
    _insert_awards(db_path, awards)

    # detect_isolation_outlier internally builds + imputes — assert that
    # by reaching into _build_features and replicating the imputation
    # the post-imputation column has zero nulls AND zero NaNs.
    feats = _build_features(db_path)
    import polars as pl
    feats = feats.with_columns(
        pl.col("entity_age_days").fill_nan(
            pl.col("entity_age_days").median().fill_null(0.0)
        )
    )
    age = feats["entity_age_days"].to_numpy().astype(float)
    assert feats["entity_age_days"].null_count() == 0
    assert not np.isnan(age).any(), f"NaN survived imputation: {age}"

    # End-to-end completes without raising (all entities are homogeneous here,
    # so 0 flags is the correct answer — no fabricated output).
    df = detect_isolation_outlier(db_path)
    assert df.columns == ["uei", "detector", "score", "details"]


# ── Task 5 — Cross-detector contract ────────────────────────────────────


def _seed_full_synthetic_db(db_path: str) -> None:
    """Populate enough data that every detector returns at least one row."""
    entities = [
        _entity("FRESH0000001", registration_date=date(2024, 1, 1)),
        _entity("OLD000000001", registration_date=date(2018, 1, 1)),
    ]
    _insert_entities(db_path, entities)

    awards = []
    # Benford ammo: 30 awards starting with 9.
    for i in range(30):
        awards.append(_award(f"BF_{i}", "OLD000000001", amount=9000.0 + i))
    # Mod-growth ammo: parent with mods totalling 5x initial, plus peers.
    awards.append(_award("MG_0", "OLD000000001", amount=100_000.0,
                         parent_award_id="MGP", modification_number="0"))
    awards.append(_award("MG_1", "OLD000000001", amount=400_000.0,
                         parent_award_id="MGP", modification_number="P00001"))
    for i in range(8):
        peer = f"MGPEER{i:06d}"
        awards.append(_award(f"MG_P_{i}_0", peer, amount=100_000.0,
                             parent_award_id=f"MGPP_{i}", modification_number="0"))
        awards.append(_award(f"MG_P_{i}_1", peer, amount=20_000.0,
                             parent_award_id=f"MGPP_{i}", modification_number="P00001"))
    # New-entity ammo: fresh registration + non-competed award.
    awards.append(
        _award("NE_0", "FRESH0000001", amount=1_000_000.0,
               award_date=date(2024, 1, 31), competition_type="NOT COMPETED")
    )
    _insert_awards(db_path, awards)


def test_all_detectors_share_contract(tmp_path):
    """Every detector must return the same 4-column schema with scores in [0, 1]."""
    from detectors.benford import detect_benford
    from detectors.isolation import detect_isolation_outlier
    from detectors.mod_growth import detect_mod_growth
    from detectors.new_entity import detect_new_entity_sole_source

    db_path = _fresh_db(tmp_path)
    _seed_full_synthetic_db(db_path)

    detectors = {
        "benford": detect_benford,
        "mod_growth": detect_mod_growth,
        "new_entity": detect_new_entity_sole_source,
        "isolation": detect_isolation_outlier,
    }

    for name, fn in detectors.items():
        df = fn(db_path)
        assert df.columns == ["uei", "detector", "score", "details"], (
            f"{name} returned wrong columns: {df.columns}"
        )
        assert df.schema["uei"] == pl.Utf8
        assert df.schema["detector"] == pl.Utf8
        assert df.schema["score"] == pl.Float64
        assert df.schema["details"] == pl.Utf8
        if df.height > 0:
            assert df["score"].min() >= 0.0, f"{name} produced score < 0"
            assert df["score"].max() <= 1.0, f"{name} produced score > 1"
            # Every row's `detector` column matches the registered name.
            assert set(df["detector"].unique().to_list()) == {name}
