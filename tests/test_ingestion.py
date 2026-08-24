"""Unit tests for ingestion modules.

Phase 1 acceptance asks for an end-to-end mocked-API test, which Task 1.4
will round out (it owns Parquet → DuckDB load + dedup). For Task 1.2 we
cover the pure parsing surface: extract_row + fiscal_year_window.
"""

import asyncio
from datetime import date, datetime
from pathlib import Path

import polars as pl

import duckdb
import pytest

from ingestion.load_db import (
    effective_agency,
    get_watermark,
    init_schema,
    load_all_parquet,
    set_watermark,
)
from ingestion.pull_awards import (
    _apply_archive_filters,
    _archive_url,
    _build_bulk_body,
    _build_search_body,
    _map_bulk_to_schema,
    _midpoint,
    _parse_archive_listing,
    _select_prime_award_csvs,
    extract_row,
    fiscal_year_window,
    pull_awards,
)
# SAM puller (ingestion.pull_entities) and its tests removed in 2026-05-27 pivot.


def test_fiscal_year_window():
    assert fiscal_year_window(2024) == (date(2023, 10, 1), date(2024, 9, 30))
    assert fiscal_year_window(2025) == (date(2024, 10, 1), date(2025, 9, 30))


def test_midpoint_splits_for_bisection():
    # Even span: 30 days → mid is 15 days in
    assert _midpoint(date(2024, 1, 1), date(2024, 1, 31)) == date(2024, 1, 16)
    # Odd span: 3 days → floor-divide to 1
    assert _midpoint(date(2024, 1, 1), date(2024, 1, 4)) == date(2024, 1, 2)
    # Single day cannot be split further
    assert _midpoint(date(2024, 1, 1), date(2024, 1, 1)) is None
    # Zero/negative span
    assert _midpoint(date(2024, 1, 2), date(2024, 1, 1)) is None


def test_extract_row_full():
    pulled_at = datetime(2026, 4, 25, 12, 0, 0)
    record = {
        "Award ID": "FA8651-22-D-0010",
        "generated_internal_id": "CONT_AWD_FA8651_22_D_0010_9700",
        "Recipient Name": "Acme Corp",
        "Recipient UEI": "ABC123XYZ456",
        "Awarding Agency": "Department of Defense",
        "Awarding Sub Agency": "Department of the Air Force",
        "Contract Award Type": "DEFINITIVE CONTRACT",
        "Description": "Test description",
        "NAICS": "541330",
        "naics_description": "Engineering Services",
        "Total Obligations": "1234567.89",
        "Award Amount": "9999.0",  # ignored: Total Obligations wins
        "Base and All Options Value": "9999999.0",
        "Start Date": "2024-01-15",
        "End Date": "2026-12-31T00:00:00",
        "Action Date": "2024-01-10",
        "Extent Competed": "FULL AND OPEN COMPETITION",
        "Number of Offers Received": "3",
        "Mod Number": "P00002",
    }

    out = extract_row(record, "Department of Defense", pulled_at)

    # generated_internal_id wins over the human-readable PIID for stability
    assert out["award_id"] == "CONT_AWD_FA8651_22_D_0010_9700"
    assert out["recipient_uei"] == "ABC123XYZ456"
    assert out["awarding_agency"] == "Department of Defense"
    assert out["award_type"] == "DEFINITIVE CONTRACT"
    assert out["naics_code"] == "541330"
    assert out["total_obligation"] == 1234567.89
    assert out["base_and_all_options_value"] == 9999999.0
    assert out["period_of_performance_start"] == date(2024, 1, 15)
    assert out["period_of_performance_end"] == date(2026, 12, 31)
    assert out["award_date"] == date(2024, 1, 10)
    assert out["competition_type"] == "FULL AND OPEN COMPETITION"
    assert out["number_of_offers"] == 3
    assert out["modification_number"] == "P00002"
    assert out["pulled_at"] == pulled_at


def test_extract_row_unpacks_naics_object():
    """USAspending v2 returns NAICS as a {code, description} object on most
    contract responses; the schema needs the code as a bare string."""
    pulled_at = datetime(2026, 4, 25)
    record = {
        "Award ID": "X",
        "NAICS": {"code": "561422", "description": "TELEMARKETING BUREAUS"},
    }

    out = extract_row(record, "Department of Defense", pulled_at)

    assert out["naics_code"] == "561422"
    assert out["naics_description"] == "TELEMARKETING BUREAUS"


def test_extract_row_falls_back_to_award_amount():
    """When Total Obligations is missing, fall back to Award Amount."""
    pulled_at = datetime(2026, 4, 25)
    record = {"Award ID": "X", "Award Amount": "500.0"}

    out = extract_row(record, "Department of Defense", pulled_at)

    assert out["total_obligation"] == 500.0


def test_effective_agency_uses_overrides():
    """Per-agency settings override config-level defaults."""
    config = {
        "award_types": ["A", "B", "C", "D"],
        "award_amount_min": 0,
        "seed_strategy": "paginate",
    }
    dod = {
        "code": "097",
        "name": "Department of Defense",
        "short": "DoD",
        "award_types": ["C", "D"],
        "award_amount_min": 25000,
        "seed_strategy": "bulk",
    }
    eff = effective_agency(dod, config)
    assert eff["award_types"] == ["C", "D"]
    assert eff["award_amount_min"] == 25000
    assert eff["seed_strategy"] == "bulk"
    # Original fields are preserved
    assert eff["code"] == "097"
    assert eff["short"] == "DoD"


def test_effective_agency_falls_back_to_config_defaults():
    """When the agency omits a setting, the config default applies."""
    config = {
        "award_types": ["A", "B", "C", "D"],
        "award_amount_min": 0,
        "seed_strategy": "paginate",
    }
    hhs = {"code": "075", "name": "HHS", "short": "HHS"}
    eff = effective_agency(hhs, config)
    assert eff["award_types"] == ["A", "B", "C", "D"]
    assert eff["award_amount_min"] == 0
    assert eff["seed_strategy"] == "paginate"


def test_build_search_body_applies_overrides():
    """Per-agency award_types and award_amount_min reach the API request body."""
    agency = {
        "name": "Department of Defense",
        "short": "DoD",
        "award_types": ["C", "D"],
        "award_amount_min": 25000,
    }
    body = _build_search_body(agency, date(2024, 1, 1), date(2024, 1, 31))
    assert body["filters"]["award_type_codes"] == ["C", "D"]
    assert body["filters"]["agencies"][0]["name"] == "Department of Defense"
    assert body["filters"]["time_period"] == [
        {"start_date": "2024-01-01", "end_date": "2024-01-31"}
    ]
    assert body["filters"]["award_amounts"] == [{"lower_bound": 25000}]
    assert body["limit"] == 100


def test_build_search_body_omits_award_amounts_when_zero():
    """An amount min of 0 means no filter — don't add the array."""
    agency = {
        "name": "HHS",
        "short": "HHS",
        "award_types": ["A", "B", "C", "D"],
        "award_amount_min": 0,
    }
    body = _build_search_body(agency, date(2024, 1, 1), date(2024, 1, 2))
    assert "award_amounts" not in body["filters"]


def test_build_bulk_body_uses_prime_award_types_and_date_range():
    """Bulk download uses different filter keys than spending_by_award."""
    agency = {
        "name": "Department of Defense",
        "short": "DoD",
        "award_types": ["C", "D"],
        "award_amount_min": 25000,
    }
    body = _build_bulk_body(agency, date(2023, 10, 1), date(2024, 9, 30))
    # Bulk uses `prime_award_types`, not `award_type_codes`
    assert body["filters"]["prime_award_types"] == ["C", "D"]
    # Subawards must be explicitly excluded — without this, the endpoint
    # bleeds in subaward rows and crashes USAspending's backend on
    # large windows (HHS 1-week = ~764K rows w/ subs vs ~50K without).
    assert body["filters"]["sub_award_types"] == []
    # Bulk uses `date_range` (object), not `time_period` (array)
    assert body["filters"]["date_range"] == {
        "start_date": "2023-10-01",
        "end_date": "2024-09-30",
    }
    assert body["filters"]["date_type"] == "action_date"
    assert body["filters"]["award_amounts"] == [{"lower_bound": 25000}]
    assert body["file_format"] == "csv"


def test_map_bulk_to_schema_translates_snake_case_columns():
    """Bulk download snake_case column names → our awards schema."""
    pulled_at = datetime(2026, 4, 25, 12, 0, 0)
    df_raw = pl.DataFrame(
        {
            "contract_award_unique_key": ["CONT_AWD_X1"],
            "parent_award_id_piid": ["PARENT_X"],
            "recipient_name": ["Acme"],
            "recipient_uei": ["UEI123456789"],
            "awarding_agency_name": ["Department of Defense"],
            "awarding_sub_agency_name": ["Air Force"],
            "award_type": ["DEFINITIVE CONTRACT"],
            "naics_code": ["541330"],
            "naics_description": ["Engineering Services"],
            "total_dollars_obligated": [1_234_567.89],
            "base_and_all_options_value": [9_999_999.0],
            "period_of_performance_start_date": [date(2024, 1, 15)],
            "period_of_performance_current_end_date": [date(2026, 12, 31)],
            "action_date": [date(2024, 1, 10)],
            "extent_competed": ["FULL AND OPEN COMPETITION"],
            "number_of_offers_received": [3],
            "modification_number": ["P00002"],
            "prime_award_base_transaction_description": ["scope of work"],
        }
    )

    out = _map_bulk_to_schema(df_raw, "Department of Defense", pulled_at)
    row = out.to_dicts()[0]

    assert row["award_id"] == "CONT_AWD_X1"
    assert row["parent_award_id"] == "PARENT_X"
    assert row["recipient_uei"] == "UEI123456789"
    assert row["awarding_agency"] == "Department of Defense"
    assert row["awarding_sub_agency"] == "Air Force"
    assert row["award_type"] == "DEFINITIVE CONTRACT"
    assert row["total_obligation"] == 1_234_567.89
    assert row["base_and_all_options_value"] == 9_999_999.0
    assert row["period_of_performance_start"] == date(2024, 1, 15)
    assert row["period_of_performance_end"] == date(2026, 12, 31)
    assert row["award_date"] == date(2024, 1, 10)
    assert row["competition_type"] == "FULL AND OPEN COMPETITION"
    assert row["number_of_offers"] == 3
    assert row["modification_number"] == "P00002"
    assert row["award_description"] == "scope of work"
    assert row["pulled_at"] == pulled_at
    # Schema column order matches the awards table
    assert list(out.columns)[0] == "award_id"
    assert list(out.columns)[-1] == "pulled_at"


def test_map_bulk_to_schema_fills_missing_columns_with_nulls():
    """When the CSV omits some columns, the mapping shouldn't crash."""
    pulled_at = datetime(2026, 4, 25)
    df_raw = pl.DataFrame(
        {
            "award_id_piid": ["X"],
            "recipient_name": ["Sparse Co"],
            "total_dollars_obligated": [500.0],
        }
    )
    out = _map_bulk_to_schema(df_raw, "HHS", pulled_at)
    row = out.to_dicts()[0]
    assert row["award_id"] == "X"  # falls back to award_id_piid
    assert row["awarding_agency"] == "HHS"  # filled from agency_name arg
    assert row["recipient_uei"] is None
    assert row["competition_type"] is None
    assert row["modification_number"] is None
    assert row["total_obligation"] == 500.0


def test_extract_row_handles_nulls_and_missing_keys():
    pulled_at = datetime(2026, 4, 25)
    record = {
        "Award ID": "PIID-only",
        "Recipient Name": "Sparse Co",
        "Award Amount": None,
        "Start Date": "",
        "End Date": "null",
        "Number of Offers Received": "not-a-number",
    }

    out = extract_row(record, "Department of Health and Human Services", pulled_at)

    # Falls back to "Award ID" when generated_internal_id is missing
    assert out["award_id"] == "PIID-only"
    # Falls back to the agency_name argument when the API didn't return one
    assert out["awarding_agency"] == "Department of Health and Human Services"
    assert out["total_obligation"] is None
    assert out["period_of_performance_start"] is None
    assert out["period_of_performance_end"] is None
    assert out["number_of_offers"] is None
    assert out["recipient_uei"] is None
    assert out["modification_number"] is None


# Task 1.3 SAM puller tests removed in 2026-05-27 pivot (full SAM removal).


# ── Task 1.4 — Parquet → DuckDB loader ────────────────────────────────────


def _award_row(award_id: str, **overrides) -> dict:
    base = {
        "award_id": award_id,
        "parent_award_id": None,
        "recipient_name": "Acme Corp",
        "recipient_uei": "ABC123XYZ456",
        "awarding_agency": "Department of Defense",
        "awarding_sub_agency": None,
        "award_type": "DEFINITIVE CONTRACT",
        "award_description": "test",
        "naics_code": "541330",
        "naics_description": "Engineering Services",
        "total_obligation": 100.0,
        "base_and_all_options_value": 100.0,
        "period_of_performance_start": date(2024, 1, 1),
        "period_of_performance_end": date(2025, 1, 1),
        "award_date": date(2024, 1, 15),
        "competition_type": "FULL AND OPEN COMPETITION",
        "number_of_offers": 3,
        "modification_number": "0",
        "pulled_at": datetime(2026, 4, 26, 12, 0, 0),
    }
    base.update(overrides)
    return base


def _write_awards_parquet(path: Path, rows: list[dict]) -> Path:
    from ingestion.pull_awards import _empty_awards_schema
    pl.DataFrame(rows, schema=_empty_awards_schema()).write_parquet(path)
    return path


@pytest.fixture
def loader_dirs(tmp_path: Path) -> tuple[str, Path]:
    """Yield (db_path, parquet_dir) under a fresh tmp directory."""
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    db_path = str(tmp_path / "test.duckdb")
    init_schema(db_path)
    return db_path, parquet_dir


def test_init_schema_creates_awards_table(tmp_path: Path):
    """Post-2026-05-27 schema only creates the `awards` table.
    `entities` and `entity_snapshots` were removed when SAM was dropped."""
    db_path = str(tmp_path / "schema.duckdb")
    init_schema(db_path)
    con = duckdb.connect(db_path, read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    finally:
        con.close()
    assert "awards" in tables
    # SAM tables are no longer created on init.
    assert "entities" not in tables
    assert "entity_snapshots" not in tables


def test_load_all_parquet_loads_awards(loader_dirs):
    db_path, parquet_dir = loader_dirs

    _write_awards_parquet(
        parquet_dir / "awards_DoD_2024.parquet",
        [_award_row("A1"), _award_row("A2", recipient_uei="UEI2")],
    )

    deltas = load_all_parquet(db_path=db_path, parquet_dir=parquet_dir)
    assert deltas == {"awards": 2}

    con = duckdb.connect(db_path, read_only=True)
    try:
        assert con.execute("SELECT COUNT(*) FROM awards").fetchone()[0] == 2
        # Round-trip a typed column to make sure schemas align
        row = con.execute(
            "SELECT recipient_name, total_obligation, award_date "
            "FROM awards WHERE award_id = 'A1'"
        ).fetchone()
        assert row == ("Acme Corp", 100.0, date(2024, 1, 15))
    finally:
        con.close()


def test_load_all_parquet_dedups_awards_on_rerun(loader_dirs):
    """Re-running the loader with the same files must not duplicate rows;
    INSERT OR REPLACE on the PK keeps a single canonical row."""
    db_path, parquet_dir = loader_dirs

    _write_awards_parquet(
        parquet_dir / "awards_DoD_2024.parquet",
        [_award_row("A1", total_obligation=100.0)],
    )

    load_all_parquet(db_path=db_path, parquet_dir=parquet_dir)

    # Rewrite Parquet with updated value; loader should overwrite, not append.
    _write_awards_parquet(
        parquet_dir / "awards_DoD_2024.parquet",
        [_award_row("A1", total_obligation=999.0)],
    )

    load_all_parquet(db_path=db_path, parquet_dir=parquet_dir)

    con = duckdb.connect(db_path, read_only=True)
    try:
        assert con.execute("SELECT COUNT(*) FROM awards").fetchone()[0] == 1
        assert con.execute(
            "SELECT total_obligation FROM awards WHERE award_id='A1'"
        ).fetchone()[0] == 999.0
    finally:
        con.close()


def test_load_all_parquet_handles_missing_files(loader_dirs):
    """Empty parquet_dir is a no-op, not an error — important so the
    pipeline can run on a fresh machine before the first ingestion."""
    db_path, parquet_dir = loader_dirs
    deltas = load_all_parquet(db_path=db_path, parquet_dir=parquet_dir)
    assert deltas == {"awards": 0}


# ── Award Data Archive helpers ────────────────────────────────────────────


def test_archive_url_zero_pads_cgac_and_uses_int_fy():
    """The S3 keys use 3-digit zero-padded CGAC codes and bare integer FYs."""
    expected = (
        "https://files.usaspending.gov/award_data_archive/"
        "FY2024_075_Contracts_Full_20260406.zip"
    )
    assert _archive_url("075", 2024, "20260406") == expected
    # Numeric input is normalized the same way.
    assert _archive_url(75, 2024, "20260406") == expected


def test_parse_archive_listing_returns_latest_full_date():
    """Pick the most recent YYYYMMDD across `*_Contracts_Full_*.zip` keys
    while ignoring `*_Delta_*.zip` and assistance files."""
    xml = """
    <ListBucketResult>
      <Contents>
        <Key>FY2024_075_Contracts_Full_20260306.zip</Key>
      </Contents>
      <Contents>
        <Key>FY2024_097_Contracts_Full_20260406.zip</Key>
      </Contents>
      <Contents>
        <Key>FY(All)_075_Contracts_Delta_20260420.zip</Key>
      </Contents>
      <Contents>
        <Key>FY2024_075_Assistance_Full_20260406.zip</Key>
      </Contents>
    </ListBucketResult>
    """
    # Latest is 2026-04-06 (Delta is excluded; Assistance is excluded).
    assert _parse_archive_listing(xml) == "20260406"


def test_parse_archive_listing_returns_none_on_empty():
    assert _parse_archive_listing("<ListBucketResult/>") is None


def test_select_prime_award_csvs_drops_subawards_and_keeps_prime():
    names = [
        "Contracts_PrimeAwardSummaries_2025-09-21_H05.csv",
        "Contracts_Subawards_2025-09-21_H05.csv",
        "ReadMe.txt",
        "Assistance_PrimeAwardSummaries_2025-09-21_H05.csv",
    ]
    out = _select_prime_award_csvs(names)
    # Both PrimeAwardSummaries files match (Contracts AND Assistance prime
    # entries — but the agency-Contracts ZIPs we download don't actually
    # contain Assistance files; this just confirms the matcher's intent).
    assert "Contracts_PrimeAwardSummaries_2025-09-21_H05.csv" in out
    assert "Contracts_Subawards_2025-09-21_H05.csv" not in out
    # ReadMe.txt is non-CSV, excluded by extension.
    assert "ReadMe.txt" not in out


def test_select_prime_award_csvs_falls_back_when_no_primeaward_marker():
    """Older payloads use plain `Contracts_*.csv` without the
    PrimeAwardSummaries suffix — still drop subawards explicitly."""
    names = [
        "Contracts_Full_2025.csv",
        "Contracts_Subawards_Full_2025.csv",
    ]
    out = _select_prime_award_csvs(names)
    assert out == ["Contracts_Full_2025.csv"]


def test_apply_archive_filters_drops_wrong_type_and_below_min():
    """The archive CSV stores `award_type` as a description (e.g.
    "DEFINITIVE CONTRACT"); the filter must translate from API codes."""
    df = pl.DataFrame(
        {
            "award_id": ["A1", "B1", "C1", "D1"],
            "award_type": [
                "BPA CALL",          # code A — wrong type for DoD
                "PURCHASE ORDER",    # code B — below DoD's min anyway
                "DELIVERY ORDER",    # code C — kept
                "DEFINITIVE CONTRACT",  # code D — kept
            ],
            "total_obligation": [100_000.0, 5_000.0, 50_000.0, 30_000.0],
        }
    )
    agency = {"award_types": ["C", "D"], "award_amount_min": 25000}
    out = _apply_archive_filters(df.lazy(), agency).collect()
    assert sorted(out["award_id"].to_list()) == ["C1", "D1"]


def test_apply_archive_filters_no_op_when_filters_absent():
    """An agency with no overrides keeps all rows."""
    df = pl.DataFrame(
        {
            "award_id": ["A", "B"],
            "award_type": ["BPA CALL", "PURCHASE ORDER"],
            "total_obligation": [100.0, 200.0],
        }
    )
    out = _apply_archive_filters(df.lazy(), {}).collect()
    assert out.height == 2


def test_apply_archive_filters_skips_full_default_award_types():
    """`effective_agency()` populates `award_types` with the global default
    `["A","B","C","D"]` when an agency doesn't override. That means
    'no narrowing' — the filter must NOT translate this to a description
    set and drop everything."""
    df = pl.DataFrame(
        {
            "award_id": ["x", "y"],
            "award_type": ["DEFINITIVE CONTRACT", "BPA CALL"],
            "total_obligation": [100.0, 200.0],
        }
    )
    agency = {"award_types": ["A", "B", "C", "D"], "award_amount_min": 0}
    out = _apply_archive_filters(df.lazy(), agency).collect()
    assert out.height == 2


def test_apply_archive_filters_is_case_insensitive():
    """`award_type` in older snapshots may have mixed case; uppercase
    both sides before comparing."""
    df = pl.DataFrame(
        {
            "award_id": ["x"],
            "award_type": ["Delivery Order"],
            "total_obligation": [50_000.0],
        }
    )
    agency = {"award_types": ["C"], "award_amount_min": 0}
    out = _apply_archive_filters(df.lazy(), agency).collect()
    assert out.height == 1


# ── Ingest watermark (2026-08-24) ─────────────────────────────────────────
#
# The incremental window used to be derived from MAX(award_date), which
# silently stalls whenever an agency's recent awards are sparse: the
# watermark stops advancing, so every weekly run re-pulls a window that
# grows by 7 days each week (it had reached ~3.5 months and a 58-minute
# pull before this landed). The watermark is now recorded explicitly:
# "we have pulled this agency through date X", regardless of whether that
# window happened to contain any rows.


def test_watermark_roundtrip(tmp_path: Path):
    db_path = str(tmp_path / "wm.duckdb")
    init_schema(db_path)

    assert get_watermark(db_path, "Department of Defense") is None

    set_watermark(db_path, "Department of Defense", date(2026, 8, 17))
    assert get_watermark(db_path, "Department of Defense") == date(2026, 8, 17)

    # Upsert, not a second row.
    set_watermark(db_path, "Department of Defense", date(2026, 8, 24))
    assert get_watermark(db_path, "Department of Defense") == date(2026, 8, 24)
    assert get_watermark(db_path, "Department of Health and Human Services") is None


def test_get_watermark_returns_none_for_missing_db(tmp_path: Path):
    assert get_watermark(str(tmp_path / "nope.duckdb"), "DoD") is None


@pytest.fixture
def one_agency_config(monkeypatch) -> dict:
    """Point pull_awards at a single-agency config so the incremental path
    is exercised without touching the real config.yaml."""
    config = {
        "agencies": [
            {"code": "097", "name": "Department of Defense", "short": "DoD",
             "seed_strategy": "archive"},
        ],
        "fiscal_years": [2026],
        "parquet_dir": "unused",
        "db_path": "unused",
        "award_types": ["A", "B", "C", "D"],
        "award_amount_min": 0,
        "seed_strategy": "paginate",
    }
    monkeypatch.setattr("ingestion.pull_awards.load_config", lambda: config)
    return config


@pytest.fixture
def captured_windows(monkeypatch) -> list[tuple[date, date]]:
    """Record every (start, end) pull_window is asked for, pulling nothing."""
    windows: list[tuple[date, date]] = []

    def fake_pull_window(agency, start, end, label, config, out_dir):
        windows.append((start, end))
        return []

    monkeypatch.setattr("ingestion.pull_awards.pull_window", fake_pull_window)
    return windows


def test_incremental_window_starts_from_watermark(
    tmp_path: Path, one_agency_config, captured_windows, monkeypatch
):
    """A stale MAX(award_date) must not drag the window back: the
    watermark is the source of truth once it exists."""
    db_path = str(tmp_path / "incr.duckdb")
    init_schema(db_path)
    con = duckdb.connect(db_path)
    try:
        # Newest award in the DB is 3 months old -- the exact condition
        # that used to produce a 3-month re-pull every week.
        con.execute(
            "INSERT INTO awards (award_id, awarding_agency, award_date) VALUES (?, ?, ?)",
            ["A1", "Department of Defense", date(2026, 5, 19)],
        )
    finally:
        con.close()
    set_watermark(db_path, "Department of Defense", date(2026, 8, 17))
    monkeypatch.setattr("ingestion.pull_awards._today", lambda: date(2026, 8, 24))

    result = asyncio.run(pull_awards(incremental=True, db_path=db_path))

    assert captured_windows == [(date(2026, 8, 10), date(2026, 8, 24))]
    # Watermark is *reported*, not committed -- seed.py commits it only
    # after load_all_parquet has durably absorbed the Parquet.
    assert result.pulled_through == {"Department of Defense": date(2026, 8, 24)}
    assert get_watermark(db_path, "Department of Defense") == date(2026, 8, 17)


def test_incremental_bootstraps_from_max_award_date_when_no_watermark(
    tmp_path: Path, one_agency_config, captured_windows, monkeypatch
):
    """Existing deployments have awards but no watermark row yet; the
    first run after this change bootstraps from the data it already has
    rather than re-seeding from scratch."""
    db_path = str(tmp_path / "bootstrap.duckdb")
    init_schema(db_path)
    con = duckdb.connect(db_path)
    try:
        con.execute(
            "INSERT INTO awards (award_id, awarding_agency, award_date) VALUES (?, ?, ?)",
            ["A1", "Department of Defense", date(2026, 5, 19)],
        )
    finally:
        con.close()
    monkeypatch.setattr("ingestion.pull_awards._today", lambda: date(2026, 8, 24))

    asyncio.run(pull_awards(incremental=True, db_path=db_path))

    assert captured_windows == [(date(2026, 5, 12), date(2026, 8, 24))]


def test_incremental_falls_back_to_seed_when_db_is_empty(
    tmp_path: Path, one_agency_config, captured_windows, monkeypatch
):
    db_path = str(tmp_path / "empty.duckdb")
    init_schema(db_path)
    seeded: list[str] = []

    async def fake_seed(agency, config, out_dir):
        seeded.append(agency["short"])
        return []

    monkeypatch.setattr("ingestion.pull_awards._seed_agency", fake_seed)

    result = asyncio.run(pull_awards(incremental=True, db_path=db_path))

    assert seeded == ["DoD"]
    assert captured_windows == []
    # Nothing to commit: a seed's coverage end date isn't knowable from
    # here (archive snapshots lag), so the next run bootstraps instead.
    assert result.pulled_through == {}
