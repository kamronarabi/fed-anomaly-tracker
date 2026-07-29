"""Unit tests for Phase 2 anomaly detectors.

Each test seeds a fresh in-memory-ish DuckDB under tmp_path with synthetic
data shaped to exercise one detector behavior, then asserts on the
detector's polars output.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
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


# ── Task 3 — New-to-federal sole-source (USAspending-only, post-2026-05-27) ────


def test_new_entity_flags_first_award_sole_source_above_threshold(tmp_path):
    """An entity whose first-ever award is a $1M sole-source above the
    threshold scores high."""
    from detectors.new_entity import detect_new_entity_sole_source

    db_path = _fresh_db(tmp_path)
    _insert_awards(
        db_path,
        [
            _award(
                "A1", "FRESH0000001", amount=1_000_000.0,
                award_date=date(2024, 1, 31),
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
    assert details["first_award_id"] == "A1"
    assert details["competition_type"] == "NOT COMPETED"
    assert details["first_award_obligation"] == 1_000_000.0


def test_new_entity_does_not_flag_competed_first_award(tmp_path):
    """A competed first award does not trigger the signal even if later
    sole-source awards exist — the signal is specifically about the first."""
    from detectors.new_entity import detect_new_entity_sole_source

    db_path = _fresh_db(tmp_path)
    _insert_awards(
        db_path,
        [
            _award(
                "A1", "COMPED000001", amount=500_000.0,
                award_date=date(2024, 1, 15),
                competition_type="FULL AND OPEN COMPETITION",
            ),
            _award(
                "A2", "COMPED000001", amount=1_000_000.0,
                award_date=date(2024, 6, 1),
                competition_type="NOT COMPETED",
            ),
        ],
    )

    df = detect_new_entity_sole_source(db_path)
    assert df.height == 0


def test_new_entity_does_not_flag_first_award_below_threshold(tmp_path):
    """Sole-source first award below the $250K threshold is too small to flag."""
    from detectors.new_entity import detect_new_entity_sole_source

    db_path = _fresh_db(tmp_path)
    _insert_awards(
        db_path,
        [
            _award(
                "A1", "SMALL0000001", amount=50_000.0,
                award_date=date(2024, 1, 15),
                competition_type="NOT COMPETED",
            )
        ],
    )

    df = detect_new_entity_sole_source(db_path)
    assert df.height == 0


def test_new_entity_does_not_flag_first_award_above_amount_max(tmp_path):
    """Sole-source first award above the $5M ceiling is excluded — that
    range is dominated by IDV-megaprime contracts (Boeing, Lockheed, etc.)
    rather than fly-by-night entities."""
    from detectors.new_entity import detect_new_entity_sole_source

    db_path = _fresh_db(tmp_path)
    _insert_awards(
        db_path,
        [
            _award(
                "A1", "MEGAPRIME0001", amount=50_000_000.0,
                award_date=date(2024, 1, 15),
                competition_type="NOT COMPETED",
            )
        ],
    )

    df = detect_new_entity_sole_source(db_path)
    assert df.height == 0


def test_new_entity_does_not_flag_entity_with_many_lifetime_awards(tmp_path):
    """An entity with >5 lifetime federal awards is an established
    contractor, not a fly-by-night. Even if their first-in-window award
    is sole-source above $250K, the detector skips them."""
    from detectors.new_entity import detect_new_entity_sole_source

    db_path = _fresh_db(tmp_path)
    rows = [
        # First-in-window award: sole-source $500K — would qualify on its own.
        _award(
            "ESTAB_0", "ESTABLISHED1", amount=500_000.0,
            award_date=date(2024, 1, 1),
            competition_type="NOT COMPETED",
        ),
    ]
    # Six more awards (any competition) push lifetime count to 7 — above
    # the default cap of 5.
    for i in range(1, 7):
        rows.append(_award(
            f"ESTAB_{i}", "ESTABLISHED1", amount=200_000.0,
            award_date=date(2024, 6, i),
        ))
    _insert_awards(db_path, rows)

    df = detect_new_entity_sole_source(db_path)
    assert df.height == 0


def test_new_entity_does_not_flag_entity_with_huge_lifetime_total(tmp_path):
    """An entity whose lifetime federal contracting total exceeds $10M
    is excluded even with <= 5 awards — that's the IDV-megaprime
    profile (e.g., Bell Helicopter with 2 multi-billion-dollar IDVs)."""
    from detectors.new_entity import detect_new_entity_sole_source

    db_path = _fresh_db(tmp_path)
    _insert_awards(
        db_path,
        [
            # First award is within the dollar window but the second
            # blows past the $10M lifetime cap.
            _award(
                "IDV_0", "IDVMEGAPRIME1", amount=1_000_000.0,
                award_date=date(2024, 1, 1),
                competition_type="NOT COMPETED",
            ),
            _award(
                "IDV_1", "IDVMEGAPRIME1", amount=20_000_000.0,
                award_date=date(2024, 6, 1),
                competition_type="FULL AND OPEN COMPETITION",
            ),
        ],
    )

    df = detect_new_entity_sole_source(db_path)
    assert df.height == 0


def test_new_entity_details_include_lifetime_stats(tmp_path):
    """Post-refinement, details JSON exposes lifetime_awards and
    lifetime_total so the brief generator can explain why this entity
    looks fly-by-night-shaped."""
    from detectors.new_entity import detect_new_entity_sole_source

    db_path = _fresh_db(tmp_path)
    _insert_awards(
        db_path,
        [
            _award(
                "A1", "SHELL0000001", amount=2_500_000.0,
                award_date=date(2024, 1, 15),
                competition_type="NOT COMPETED",
            )
        ],
    )

    df = detect_new_entity_sole_source(db_path)
    assert df.height == 1
    details = json.loads(df.row(0, named=True)["details"])
    assert details["lifetime_awards"] == 1
    assert details["lifetime_total"] == pytest.approx(2_500_000.0)


def test_new_entity_picks_earliest_award_per_uei(tmp_path):
    """When an entity has multiple awards, the detector looks at the
    chronologically earliest one — not the largest or most recent."""
    from detectors.new_entity import detect_new_entity_sole_source

    db_path = _fresh_db(tmp_path)
    _insert_awards(
        db_path,
        [
            # Earliest award is sole-source above threshold — should flag.
            _award(
                "EARLY", "MULTI0000001", amount=500_000.0,
                award_date=date(2024, 1, 1),
                competition_type="NOT COMPETED",
            ),
            # Later award is competed and irrelevant to the signal.
            _award(
                "LATER", "MULTI0000001", amount=2_000_000.0,
                award_date=date(2024, 6, 1),
                competition_type="FULL AND OPEN COMPETITION",
            ),
        ],
    )

    df = detect_new_entity_sole_source(db_path)
    assert df.height == 1
    details = json.loads(df.row(0, named=True)["details"])
    assert details["first_award_id"] == "EARLY"
    assert details["lifetime_awards"] == 2


def test_new_entity_returns_correct_schema(tmp_path):
    from detectors.new_entity import detect_new_entity_sole_source

    db_path = _fresh_db(tmp_path)
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
    """Insert 19 'normal' entities + 1 obvious outlier; return the outlier's UEI.

    USAspending-only seeding after 2026-05-27 SAM removal — features come
    from contract patterns, not from SAM registration data.
    """
    awards = []
    # 19 normal entities: one NAICS, single agency, one award each at ~$50K.
    for i in range(19):
        uei = f"NORM{i:08d}"
        awards.append(_award(f"AN_{i}", uei, amount=50_000.0))
    # Outlier: extremely large dollars, many agencies, many NAICS,
    # mostly non-competed, lots of mods.
    outlier_uei = "OUTLIER00001"
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
    _insert_awards(db_path, awards)
    return outlier_uei


def test_isolation_flags_obvious_outlier(tmp_path):
    """The synthetic outlier (giant dollars, many agencies, all sole-source,
    lots of mods) must appear in the flagged set with a high score."""
    from detectors.isolation import detect_isolation_outlier

    db_path = _fresh_db(tmp_path)
    outlier_uei = _populate_iforest_dataset(db_path)

    df = detect_isolation_outlier(db_path)
    df = df.sort("score", descending=True)
    top = df.row(0, named=True)
    assert top["uei"] == outlier_uei
    assert top["score"] > 0.5


def test_isolation_features_are_usaspending_only(tmp_path):
    """Regression: post-2026-05-27, the isolation detector must not depend
    on any SAM-derived columns. Verifies the feature builder runs against
    a DB containing only the `awards` table."""
    from detectors.isolation import _build_features

    db_path = _fresh_db(tmp_path)
    _populate_iforest_dataset(db_path)

    feats = _build_features(db_path)
    # The current feature set is 6 USAspending-derived columns + uei.
    assert "entity_age_days" not in feats.columns
    assert "registration_date" not in feats.columns
    assert set(feats.columns) == {
        "uei",
        "log_total_dollars",
        "award_count",
        "unique_agencies_count",
        "naics_diversity",
        "competition_ratio",
        "modification_frequency",
    }


def test_isolation_returns_correct_schema(tmp_path):
    from detectors.isolation import detect_isolation_outlier

    db_path = _fresh_db(tmp_path)
    _populate_iforest_dataset(db_path)
    df = detect_isolation_outlier(db_path)
    assert df.columns == ["uei", "detector", "score", "details"]
    assert df.schema["score"] == pl.Float64


# ── Task 6 — Sole-source concentration vs NAICS peers (Wave 2) ──────────


def _seed_ss_population(
    db_path: str,
    naics: str,
    n_peers: int,
    peer_ss_fraction: float,
    peer_awards_per_entity: int = 10,
    peer_uei_prefix: str = "PEER",
) -> None:
    """Insert n_peers entities in `naics`, each with `peer_awards_per_entity`
    awards split so that `peer_ss_fraction` are sole-source. Establishes
    a stable peer-group baseline."""
    rows = []
    for i in range(n_peers):
        uei = f"{peer_uei_prefix}{i:08d}"
        n_ss = int(round(peer_awards_per_entity * peer_ss_fraction))
        n_open = peer_awards_per_entity - n_ss
        for j in range(n_ss):
            rows.append(_award(
                f"{uei}_SS_{j}", uei, amount=100_000.0,
                naics_code=naics,
                competition_type="NOT COMPETED",
            ))
        for j in range(n_open):
            rows.append(_award(
                f"{uei}_OPEN_{j}", uei, amount=100_000.0,
                naics_code=naics,
                competition_type="FULL AND OPEN COMPETITION",
            ))
    _insert_awards(db_path, rows)


def _seed_ss_focal_entity(
    db_path: str,
    uei: str,
    naics: str,
    n_awards: int,
    ss_fraction: float,
) -> None:
    """Insert a single entity with the given naics + sole-source fraction."""
    rows = []
    n_ss = int(round(n_awards * ss_fraction))
    n_open = n_awards - n_ss
    for j in range(n_ss):
        rows.append(_award(
            f"{uei}_SS_{j}", uei, amount=100_000.0,
            naics_code=naics,
            competition_type="NOT COMPETED",
        ))
    for j in range(n_open):
        rows.append(_award(
            f"{uei}_OPEN_{j}", uei, amount=100_000.0,
            naics_code=naics,
            competition_type="FULL AND OPEN COMPETITION",
        ))
    _insert_awards(db_path, rows)


def test_sole_source_flags_anomalously_high_share(tmp_path):
    """An entity at 95% sole-source in a NAICS where peers average 20%
    is well above the peer median; z should be high and the entity
    should be flagged."""
    from detectors.sole_source_concentration import (
        detect_sole_source_concentration,
    )

    db_path = _fresh_db(tmp_path)
    _seed_ss_population(db_path, naics="541330", n_peers=10, peer_ss_fraction=0.2)
    _seed_ss_focal_entity(
        db_path, uei="ANOMALOUS001", naics="541330",
        n_awards=20, ss_fraction=0.95,
    )

    df = detect_sole_source_concentration(
        db_path, min_awards=10, min_naics_entities=5, z_threshold=1.0,
    )

    rows = {r["uei"]: r for r in df.iter_rows(named=True)}
    assert "ANOMALOUS001" in rows, (
        f"Anomalous entity must be flagged; got UEIs: {sorted(rows.keys())}"
    )
    row = rows["ANOMALOUS001"]
    assert row["detector"] == "sole_source_concentration"
    assert row["score"] > 0.7, f"Score should be high for z >> 1; got {row['score']}"
    details = json.loads(row["details"])
    assert details["primary_naics"] == "541330"
    assert details["ss_frac"] == pytest.approx(0.95, abs=0.01)
    assert details["z_score"] > 1.0


def test_sole_source_does_not_flag_at_peer_median(tmp_path):
    """An entity right at the NAICS peer median produces z ≈ 0; the
    z_threshold filter must keep it out of the output."""
    from detectors.sole_source_concentration import (
        detect_sole_source_concentration,
    )

    db_path = _fresh_db(tmp_path)
    _seed_ss_population(db_path, naics="541330", n_peers=10, peer_ss_fraction=0.3)
    # AT the peer median, but with a different UEI so we can look it up.
    _seed_ss_focal_entity(
        db_path, uei="ATMEDIAN0001", naics="541330",
        n_awards=10, ss_fraction=0.3,
    )

    df = detect_sole_source_concentration(
        db_path, min_awards=10, min_naics_entities=5, z_threshold=1.0,
    )

    uei_set = set(df["uei"].to_list())
    assert "ATMEDIAN0001" not in uei_set, (
        f"At-median entity should not be flagged; got UEIs: {sorted(uei_set)}"
    )


def test_sole_source_does_not_flag_below_median(tmp_path):
    """An entity well below the peer median (lower SS share than peers)
    is not anomalous in the direction we care about. Negative z, no flag."""
    from detectors.sole_source_concentration import (
        detect_sole_source_concentration,
    )

    db_path = _fresh_db(tmp_path)
    _seed_ss_population(db_path, naics="541330", n_peers=10, peer_ss_fraction=0.7)
    _seed_ss_focal_entity(
        db_path, uei="LOWSHARE0001", naics="541330",
        n_awards=10, ss_fraction=0.1,
    )

    df = detect_sole_source_concentration(
        db_path, min_awards=10, min_naics_entities=5, z_threshold=1.0,
    )

    assert "LOWSHARE0001" not in set(df["uei"].to_list())


def test_sole_source_skips_entity_below_min_awards(tmp_path):
    """Entities with fewer than min_awards total awards have unstable
    SS fractions and must be excluded."""
    from detectors.sole_source_concentration import (
        detect_sole_source_concentration,
    )

    db_path = _fresh_db(tmp_path)
    _seed_ss_population(db_path, naics="541330", n_peers=10, peer_ss_fraction=0.2)
    # Only 3 awards (all sole-source) — too few to be stable.
    _seed_ss_focal_entity(
        db_path, uei="SMALL0000001", naics="541330",
        n_awards=3, ss_fraction=1.0,
    )

    df = detect_sole_source_concentration(
        db_path, min_awards=10, min_naics_entities=5, z_threshold=1.0,
    )

    assert "SMALL0000001" not in set(df["uei"].to_list())


def test_sole_source_skips_sparse_naics(tmp_path):
    """A NAICS code with fewer than min_naics_entities entities has too
    noisy a baseline; entities in it should be excluded."""
    from detectors.sole_source_concentration import (
        detect_sole_source_concentration,
    )

    db_path = _fresh_db(tmp_path)
    # Sparse NAICS: only 3 entities total in NAICS "999999".
    _seed_ss_population(
        db_path, naics="999999", n_peers=3, peer_ss_fraction=0.1,
        peer_uei_prefix="SPARSE",
    )
    # Focal entity in the sparse NAICS, would otherwise look anomalous.
    _seed_ss_focal_entity(
        db_path, uei="SPARSEANOM01", naics="999999",
        n_awards=10, ss_fraction=0.95,
    )

    df = detect_sole_source_concentration(
        db_path, min_awards=10, min_naics_entities=5, z_threshold=1.0,
    )

    assert "SPARSEANOM01" not in set(df["uei"].to_list()), (
        "Entity in sparse NAICS should be skipped due to unreliable baseline"
    )


def test_sole_source_returns_correct_schema(tmp_path):
    from detectors.sole_source_concentration import (
        detect_sole_source_concentration,
    )

    db_path = _fresh_db(tmp_path)
    _seed_ss_population(db_path, naics="541330", n_peers=10, peer_ss_fraction=0.2)
    _seed_ss_focal_entity(
        db_path, uei="SCHEMA000003", naics="541330",
        n_awards=20, ss_fraction=0.95,
    )

    df = detect_sole_source_concentration(
        db_path, min_awards=10, min_naics_entities=5, z_threshold=1.0,
    )

    assert df.columns == ["uei", "detector", "score", "details"]
    assert df.schema["uei"] == pl.Utf8
    assert df.schema["detector"] == pl.Utf8
    assert df.schema["score"] == pl.Float64
    assert df.schema["details"] == pl.Utf8


def test_sole_source_uses_plurality_primary_naics(tmp_path):
    """When an entity's awards span multiple NAICS codes, the detector
    uses the mode (plurality NAICS) as the comparison baseline."""
    from detectors.sole_source_concentration import (
        detect_sole_source_concentration,
    )

    db_path = _fresh_db(tmp_path)
    # Peer baseline in NAICS-A (low SS share).
    _seed_ss_population(
        db_path, naics="111111", n_peers=10, peer_ss_fraction=0.2,
        peer_uei_prefix="PEERA",
    )
    # Peer baseline in NAICS-B (high SS share).
    _seed_ss_population(
        db_path, naics="222222", n_peers=10, peer_ss_fraction=0.7,
        peer_uei_prefix="PEERB",
    )
    # Focal entity: 7 awards in NAICS-A (plurality) + 3 in NAICS-B,
    # all sole-source. Should be compared against NAICS-A peers (median ~0.2).
    rows = []
    for j in range(7):
        rows.append(_award(
            f"PLUR_A_{j}", "PLURALITY001", amount=100_000.0,
            naics_code="111111", competition_type="NOT COMPETED",
        ))
    for j in range(3):
        rows.append(_award(
            f"PLUR_B_{j}", "PLURALITY001", amount=100_000.0,
            naics_code="222222", competition_type="NOT COMPETED",
        ))
    _insert_awards(db_path, rows)

    df = detect_sole_source_concentration(
        db_path, min_awards=10, min_naics_entities=5, z_threshold=1.0,
    )

    rows = {r["uei"]: r for r in df.iter_rows(named=True)}
    assert "PLURALITY001" in rows
    details = json.loads(rows["PLURALITY001"]["details"])
    assert details["primary_naics"] == "111111", (
        f"Plurality should be NAICS-A; got {details['primary_naics']}"
    )


# ── Task 7 — Award velocity (Wave 2) ────────────────────────────────────


def _seed_velocity_entity(
    db_path: str,
    uei: str,
    baseline_dates: list[date],
    recent_dates: list[date],
) -> None:
    """Insert awards for `uei` at the explicit `baseline_dates` and `recent_dates`.

    Lets tests control the entity's award-arrival pattern exactly so
    we can reason about z-scores without surprises.
    """
    rows = []
    for i, d in enumerate(baseline_dates):
        rows.append(_award(
            f"{uei}_B_{i}", uei, amount=100_000.0, award_date=d,
        ))
    for i, d in enumerate(recent_dates):
        rows.append(_award(
            f"{uei}_R_{i}", uei, amount=100_000.0, award_date=d,
        ))
    _insert_awards(db_path, rows)


def test_velocity_flags_sudden_burst(tmp_path):
    """An entity with a steady monthly baseline (12 awards over 12 months)
    that suddenly wins 20 awards in the last 90 days should score high."""
    from detectors.award_velocity import detect_award_velocity

    db_path = _fresh_db(tmp_path)
    today = date(2026, 5, 29)
    # Baseline: one award per month from 2025-03 to 2026-02 (12 awards).
    baseline_dates = [date(2025, m, 1) for m in range(3, 13)] + [
        date(2026, 1, 1), date(2026, 2, 1),
    ]
    # Recent burst: 20 awards in last 90 days (all on 2026-04-15).
    recent_dates = [date(2026, 4, 15)] * 20
    _seed_velocity_entity(
        db_path, "BURST0000001", baseline_dates, recent_dates,
    )

    df = detect_award_velocity(db_path, today=today)

    rows = {r["uei"]: r for r in df.iter_rows(named=True)}
    assert "BURST0000001" in rows, (
        f"Burst entity must be flagged; got UEIs: {sorted(rows.keys())}"
    )
    row = rows["BURST0000001"]
    assert row["detector"] == "award_velocity"
    assert row["score"] > 0.7, f"Score should be high for big z; got {row['score']}"
    details = json.loads(row["details"])
    assert details["recent_count"] == 20
    assert details["baseline_count"] == 12
    assert details["z_score"] > 2.0


def test_velocity_does_not_flag_steady_pace(tmp_path):
    """An entity whose recent pace matches their baseline pace has z ≈ 0
    and must not be flagged."""
    from detectors.award_velocity import detect_award_velocity

    db_path = _fresh_db(tmp_path)
    today = date(2026, 5, 29)
    # 12 baseline + 2 recent ≈ baseline rate (12/640 days * 90 days ≈ 1.7)
    baseline_dates = [date(2025, m, 1) for m in range(3, 13)] + [
        date(2026, 1, 1), date(2026, 2, 1),
    ]
    recent_dates = [date(2026, 3, 1), date(2026, 5, 1)]
    _seed_velocity_entity(
        db_path, "STEADY000001", baseline_dates, recent_dates,
    )

    df = detect_award_velocity(db_path, today=today)
    assert "STEADY000001" not in set(df["uei"].to_list())


def test_velocity_skips_entity_below_min_baseline(tmp_path):
    """Entities with insufficient baseline data (< min_baseline_awards)
    have unstable rate estimates and must be excluded."""
    from detectors.award_velocity import detect_award_velocity

    db_path = _fresh_db(tmp_path)
    today = date(2026, 5, 29)
    # Only 3 baseline awards — below the default min of 5.
    baseline_dates = [date(2025, 4, 1), date(2025, 8, 1), date(2025, 12, 1)]
    # Big recent burst, but should not flag due to thin baseline.
    recent_dates = [date(2026, 4, 15)] * 15
    _seed_velocity_entity(
        db_path, "THIN00000001", baseline_dates, recent_dates,
    )

    df = detect_award_velocity(db_path, today=today)
    assert "THIN00000001" not in set(df["uei"].to_list())


def test_velocity_skips_entity_with_no_baseline(tmp_path):
    """An entity whose first award is recent (within window_days) has zero
    baseline and cannot be scored; must be silently excluded."""
    from detectors.award_velocity import detect_award_velocity

    db_path = _fresh_db(tmp_path)
    today = date(2026, 5, 29)
    # All awards in last 60 days; no baseline whatsoever.
    recent_dates = [date(2026, 4, 1) + timedelta(days=i) for i in range(0, 60, 2)]
    _seed_velocity_entity(
        db_path, "BRANDNEW0001", baseline_dates=[], recent_dates=recent_dates,
    )

    df = detect_award_velocity(db_path, today=today)
    assert "BRANDNEW0001" not in set(df["uei"].to_list())


def test_velocity_respects_today_parameter(tmp_path):
    """The `today` arg must override `MAX(award_date)`. Passing two
    different `today` values against identical data should produce
    different outputs because the recent window shifts."""
    from detectors.award_velocity import detect_award_velocity

    db_path = _fresh_db(tmp_path)
    baseline_dates = [date(2025, m, 1) for m in range(3, 13)] + [
        date(2026, 1, 1), date(2026, 2, 1),
    ]
    recent_dates = [date(2026, 4, 15)] * 20
    _seed_velocity_entity(
        db_path, "TIMETRAVEL01", baseline_dates, recent_dates,
    )

    # With today=2026-05-29, the 20 burst awards sit inside the 90-day window.
    df_now = detect_award_velocity(db_path, today=date(2026, 5, 29))
    # With today=2026-04-01, the burst awards are FUTURE relative to "today"
    # and don't count — neither does the entity look like a burst.
    df_past = detect_award_velocity(db_path, today=date(2026, 4, 1))

    flagged_now = set(df_now["uei"].to_list())
    flagged_past = set(df_past["uei"].to_list())
    assert "TIMETRAVEL01" in flagged_now, "Burst visible at today=2026-05-29"
    assert "TIMETRAVEL01" not in flagged_past, (
        "Burst should not be visible from earlier 'today'"
    )


def test_velocity_returns_correct_schema(tmp_path):
    from detectors.award_velocity import detect_award_velocity

    db_path = _fresh_db(tmp_path)
    today = date(2026, 5, 29)
    baseline_dates = [date(2025, m, 1) for m in range(3, 13)] + [
        date(2026, 1, 1), date(2026, 2, 1),
    ]
    recent_dates = [date(2026, 4, 15)] * 20
    _seed_velocity_entity(
        db_path, "SCHEMA000004", baseline_dates, recent_dates,
    )

    df = detect_award_velocity(db_path, today=today)
    assert df.columns == ["uei", "detector", "score", "details"]
    assert df.schema["uei"] == pl.Utf8
    assert df.schema["detector"] == pl.Utf8
    assert df.schema["score"] == pl.Float64
    assert df.schema["details"] == pl.Utf8


def test_velocity_details_include_as_of_date(tmp_path):
    """Details JSON must include the `as_of` date so consumers can tell
    when the velocity snapshot was taken (important for reproducibility
    and for the dashboard to label the trending window)."""
    from detectors.award_velocity import detect_award_velocity

    db_path = _fresh_db(tmp_path)
    today = date(2026, 5, 29)
    baseline_dates = [date(2025, m, 1) for m in range(3, 13)] + [
        date(2026, 1, 1), date(2026, 2, 1),
    ]
    recent_dates = [date(2026, 4, 15)] * 20
    _seed_velocity_entity(
        db_path, "ASOF00000001", baseline_dates, recent_dates,
    )

    df = detect_award_velocity(db_path, today=today)
    details = json.loads(df.row(0, named=True)["details"])
    assert details["as_of"] == "2026-05-29"


# ── Task 5 — Cross-detector contract ────────────────────────────────────


def _seed_full_synthetic_db(db_path: str) -> None:
    """Populate enough data that every detector returns at least one row.

    USAspending-only seeding after 2026-05-27 SAM removal.
    """
    awards = []
    # Benford ammo: 30 awards starting with 9 for OLD000000001.
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
    # New-entity ammo: a UEI whose first-ever award is a $1M sole-source.
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
            # Phase 3 pivot assumes one row per UEI per detector — lock it in.
            assert df["uei"].n_unique() == df.height, (
                f"{name} emitted duplicate UEIs"
            )
            # `details` is typed Utf8 but the implicit contract is JSON dict.
            for s in df["details"].to_list():
                assert isinstance(json.loads(s), dict), (
                    f"{name} produced non-dict details"
                )
