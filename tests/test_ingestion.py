"""Unit tests for ingestion modules.

Phase 1 acceptance asks for an end-to-end mocked-API test, which Task 1.4
will round out (it owns Parquet → DuckDB load + dedup). For Task 1.2 we
cover the pure parsing surface: extract_row + fiscal_year_window.
"""

from datetime import date, datetime
from pathlib import Path

import polars as pl

import duckdb
import pytest

from ingestion.load_db import (
    effective_agency,
    init_schema,
    load_all_parquet,
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
)
from ingestion.pull_entities import (
    _format_business_types,
    _parse_date as _parse_entity_date,
    _snapshot_rows_from_pulled,
    extract_entity,
)


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


# ── Task 1.3 — SAM.gov entity puller ──────────────────────────────────────


def test_extract_entity_full():
    """Map a fully-populated SAM v3 entityData record to the entities schema."""
    pulled_at = datetime(2026, 4, 26, 12, 0, 0)
    record = {
        "entityRegistration": {
            "ueiSAM": "ABC123XYZ456",
            "legalBusinessName": "Acme Corp",
            "dbaName": "Acme",
            "cageCode": "1ABC2",
            "registrationDate": "2024-01-15",
            "registrationExpirationDate": "2026-01-15",
            "exclusionStatusFlag": "N",
        },
        "coreData": {
            "physicalAddress": {
                "addressLine1": "123 Main St",
                "city": "Tampa",
                "stateOrProvinceCode": "FL",
                "zipCode": "33601",
            },
            "generalInformation": {
                "entityStructureCode": "2L",
                "entityStructureDesc": "Corporate Entity (Not Tax Exempt)",
            },
            "businessTypes": {
                "businessTypeList": [
                    {"businessTypeCode": "23", "businessTypeDesc": "Minority Owned Business"},
                    {"businessTypeCode": "OY", "businessTypeDesc": "Black American Owned"},
                ],
            },
        },
    }

    out = extract_entity(record, pulled_at)

    assert out["uei"] == "ABC123XYZ456"
    assert out["legal_business_name"] == "Acme Corp"
    assert out["dba_name"] == "Acme"
    assert out["physical_address_line1"] == "123 Main St"
    assert out["physical_city"] == "Tampa"
    assert out["physical_state"] == "FL"
    assert out["physical_zip"] == "33601"
    assert out["business_type"] == "Minority Owned Business, Black American Owned"
    # entityStructureDesc preferred over the code
    assert out["entity_structure"] == "Corporate Entity (Not Tax Exempt)"
    assert out["registration_date"] == date(2024, 1, 15)
    assert out["expiration_date"] == date(2026, 1, 15)
    assert out["cage_code"] == "1ABC2"
    assert out["exclusion_status"] == "N"
    assert out["last_pulled_at"] == pulled_at


def test_extract_entity_returns_none_without_uei():
    """Records missing a UEI are skipped silently rather than poisoning the
    Parquet write."""
    out = extract_entity({"entityRegistration": {}}, datetime(2026, 4, 26))
    assert out is None


def test_extract_entity_handles_missing_optional_fields():
    """coreData and most registration fields can be absent on inactive
    or partial registrations — the mapper should not crash."""
    pulled_at = datetime(2026, 4, 26)
    record = {
        "entityRegistration": {
            "ueiSAM": "PARTIAL00001",
            "legalBusinessName": "Sparse LLC",
        },
    }

    out = extract_entity(record, pulled_at)

    assert out["uei"] == "PARTIAL00001"
    assert out["legal_business_name"] == "Sparse LLC"
    assert out["physical_address_line1"] is None
    assert out["physical_state"] is None
    assert out["business_type"] is None
    assert out["entity_structure"] is None
    assert out["registration_date"] is None
    assert out["cage_code"] is None
    assert out["last_pulled_at"] == pulled_at


def test_extract_entity_falls_back_to_activation_date_and_state_field():
    """SAM responses occasionally use `activationDate` instead of
    `registrationDate`, and `state` instead of `stateOrProvinceCode`."""
    pulled_at = datetime(2026, 4, 26)
    record = {
        "entityRegistration": {
            "ueiSAM": "ALT0000000001",
            "activationDate": "03/15/2024",  # US-style date
            "expirationDate": "2027-03-15",
        },
        "coreData": {
            "physicalAddress": {"state": "CA", "zip": "90001"},
        },
    }
    out = extract_entity(record, pulled_at)
    assert out["registration_date"] == date(2024, 3, 15)
    assert out["expiration_date"] == date(2027, 3, 15)
    assert out["physical_state"] == "CA"
    assert out["physical_zip"] == "90001"


def test_format_business_types_falls_back_to_code():
    """When businessTypeDesc is missing, fall back to the code."""
    core = {
        "businessTypes": {
            "businessTypeList": [
                {"businessTypeCode": "OY"},
                {"businessTypeDesc": "Veteran Owned"},
            ],
        },
    }
    assert _format_business_types(core) == "OY, Veteran Owned"


def test_format_business_types_returns_none_on_empty():
    assert _format_business_types(None) is None
    assert _format_business_types({}) is None
    assert _format_business_types({"businessTypes": {}}) is None
    assert _format_business_types({"businessTypes": {"businessTypeList": []}}) is None


def test_parse_entity_date_handles_iso_us_and_garbage():
    assert _parse_entity_date("2024-01-15") == date(2024, 1, 15)
    assert _parse_entity_date("2024-01-15T00:00:00") == date(2024, 1, 15)
    assert _parse_entity_date("01/15/2024") == date(2024, 1, 15)
    assert _parse_entity_date("") is None
    assert _parse_entity_date(None) is None
    assert _parse_entity_date("not a date") is None
    # date object passes through
    assert _parse_entity_date(date(2024, 1, 15)) == date(2024, 1, 15)


def test_snapshot_rows_from_pulled_projects_to_snapshot_schema():
    """Snapshot rows pull only the address / name / cage_code fields out
    of the full entity row, plus a fixed snapshot_date."""
    pulled_at = datetime(2026, 4, 26)
    full_row = {
        "uei": "ABC123XYZ456",
        "legal_business_name": "Acme Corp",
        "dba_name": "Acme",
        "physical_address_line1": "123 Main St",
        "physical_city": "Tampa",
        "physical_state": "FL",
        "physical_zip": "33601",
        "business_type": "Minority Owned Business",
        "entity_structure": "Corp",
        "registration_date": date(2024, 1, 15),
        "expiration_date": date(2026, 1, 15),
        "cage_code": "1ABC2",
        "exclusion_status": "N",
        "last_pulled_at": pulled_at,
    }
    snap_date = date(2026, 4, 26)
    out = _snapshot_rows_from_pulled([full_row], snap_date)

    assert len(out) == 1
    assert out[0] == {
        "uei": "ABC123XYZ456",
        "snapshot_date": snap_date,
        "legal_business_name": "Acme Corp",
        "physical_address_line1": "123 Main St",
        "physical_city": "Tampa",
        "physical_state": "FL",
        "physical_zip": "33601",
        "cage_code": "1ABC2",
    }
    # No leakage of fields outside the snapshot schema
    assert "dba_name" not in out[0]
    assert "registration_date" not in out[0]
    assert "last_pulled_at" not in out[0]


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


def _entity_row(uei: str, **overrides) -> dict:
    base = {
        "uei": uei,
        "legal_business_name": "Acme Corp",
        "dba_name": None,
        "physical_address_line1": "123 Main St",
        "physical_city": "Tampa",
        "physical_state": "FL",
        "physical_zip": "33601",
        "business_type": "Minority Owned Business",
        "entity_structure": "Corporate Entity",
        "registration_date": date(2024, 1, 15),
        "expiration_date": date(2026, 1, 15),
        "cage_code": "1ABC2",
        "exclusion_status": "N",
        "last_pulled_at": datetime(2026, 4, 26, 12, 0, 0),
    }
    base.update(overrides)
    return base


def _snapshot_row(uei: str, snap_date: date, **overrides) -> dict:
    base = {
        "uei": uei,
        "snapshot_date": snap_date,
        "legal_business_name": "Acme Corp",
        "physical_address_line1": "123 Main St",
        "physical_city": "Tampa",
        "physical_state": "FL",
        "physical_zip": "33601",
        "cage_code": "1ABC2",
    }
    base.update(overrides)
    return base


def _write_awards_parquet(path: Path, rows: list[dict]) -> Path:
    from ingestion.pull_awards import _empty_awards_schema
    pl.DataFrame(rows, schema=_empty_awards_schema()).write_parquet(path)
    return path


def _write_entities_parquet(path: Path, rows: list[dict]) -> Path:
    from ingestion.pull_entities import _entities_schema
    pl.DataFrame(rows, schema=_entities_schema()).write_parquet(path)
    return path


def _write_snapshots_parquet(path: Path, rows: list[dict]) -> Path:
    from ingestion.pull_entities import _snapshot_schema
    pl.DataFrame(rows, schema=_snapshot_schema()).write_parquet(path)
    return path


@pytest.fixture
def loader_dirs(tmp_path: Path) -> tuple[str, Path]:
    """Yield (db_path, parquet_dir) under a fresh tmp directory."""
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    db_path = str(tmp_path / "test.duckdb")
    init_schema(db_path)
    return db_path, parquet_dir


def test_init_schema_creates_three_tables(tmp_path: Path):
    db_path = str(tmp_path / "schema.duckdb")
    init_schema(db_path)
    con = duckdb.connect(db_path, read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    finally:
        con.close()
    assert {"awards", "entities", "entity_snapshots"}.issubset(tables)


def test_load_all_parquet_loads_awards_entities_snapshots(loader_dirs):
    db_path, parquet_dir = loader_dirs

    _write_awards_parquet(
        parquet_dir / "awards_DoD_2024.parquet",
        [_award_row("A1"), _award_row("A2", recipient_uei="UEI2")],
    )
    _write_entities_parquet(
        parquet_dir / "entities.parquet",
        [_entity_row("ABC123XYZ456"), _entity_row("UEI2", legal_business_name="Beta LLC")],
    )
    _write_snapshots_parquet(
        parquet_dir / "entity_snapshots_2026-04-26.parquet",
        [
            _snapshot_row("ABC123XYZ456", date(2026, 4, 26)),
            _snapshot_row("UEI2", date(2026, 4, 26)),
        ],
    )

    deltas = load_all_parquet(db_path=db_path, parquet_dir=parquet_dir)
    assert deltas == {"awards": 2, "entities": 2, "entity_snapshots": 2}

    con = duckdb.connect(db_path, read_only=True)
    try:
        assert con.execute("SELECT COUNT(*) FROM awards").fetchone()[0] == 2
        assert con.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 2
        assert con.execute(
            "SELECT COUNT(*) FROM entity_snapshots"
        ).fetchone()[0] == 2
        # Round-trip a typed column to make sure schemas align
        row = con.execute(
            "SELECT recipient_name, total_obligation, award_date "
            "FROM awards WHERE award_id = 'A1'"
        ).fetchone()
        assert row == ("Acme Corp", 100.0, date(2024, 1, 15))
    finally:
        con.close()


def test_load_all_parquet_dedups_awards_and_entities_on_rerun(loader_dirs):
    """Re-running the loader with the same files must not duplicate rows;
    INSERT OR REPLACE on the PK keeps a single canonical row."""
    db_path, parquet_dir = loader_dirs

    _write_awards_parquet(
        parquet_dir / "awards_DoD_2024.parquet",
        [_award_row("A1", total_obligation=100.0)],
    )
    _write_entities_parquet(
        parquet_dir / "entities.parquet",
        [_entity_row("ABC123XYZ456", legal_business_name="Acme Corp")],
    )

    load_all_parquet(db_path=db_path, parquet_dir=parquet_dir)

    # Rewrite Parquets with updated values; loader should overwrite, not append.
    _write_awards_parquet(
        parquet_dir / "awards_DoD_2024.parquet",
        [_award_row("A1", total_obligation=999.0)],
    )
    _write_entities_parquet(
        parquet_dir / "entities.parquet",
        [_entity_row("ABC123XYZ456", legal_business_name="Acme Corp Renamed")],
    )

    load_all_parquet(db_path=db_path, parquet_dir=parquet_dir)

    con = duckdb.connect(db_path, read_only=True)
    try:
        assert con.execute("SELECT COUNT(*) FROM awards").fetchone()[0] == 1
        assert con.execute(
            "SELECT total_obligation FROM awards WHERE award_id='A1'"
        ).fetchone()[0] == 999.0
        assert con.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 1
        assert con.execute(
            "SELECT legal_business_name FROM entities WHERE uei='ABC123XYZ456'"
        ).fetchone()[0] == "Acme Corp Renamed"
    finally:
        con.close()


def test_load_all_parquet_appends_snapshots_across_days(loader_dirs):
    """Snapshots accumulate: a different snapshot_date adds a new row;
    the same snapshot_date is idempotent (INSERT OR IGNORE)."""
    db_path, parquet_dir = loader_dirs

    _write_snapshots_parquet(
        parquet_dir / "entity_snapshots_2026-04-26.parquet",
        [_snapshot_row("ABC123XYZ456", date(2026, 4, 26))],
    )
    load_all_parquet(db_path=db_path, parquet_dir=parquet_dir)

    # Same-day re-run: no new row, no clobber.
    load_all_parquet(db_path=db_path, parquet_dir=parquet_dir)

    # Next-day snapshot: one new row.
    _write_snapshots_parquet(
        parquet_dir / "entity_snapshots_2026-05-03.parquet",
        [_snapshot_row("ABC123XYZ456", date(2026, 5, 3), physical_city="Miami")],
    )
    load_all_parquet(db_path=db_path, parquet_dir=parquet_dir)

    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            "SELECT snapshot_date, physical_city FROM entity_snapshots "
            "WHERE uei='ABC123XYZ456' ORDER BY snapshot_date"
        ).fetchall()
        assert rows == [
            (date(2026, 4, 26), "Tampa"),
            (date(2026, 5, 3), "Miami"),
        ]
    finally:
        con.close()


def test_load_all_parquet_handles_missing_files(loader_dirs):
    """Empty parquet_dir is a no-op, not an error — important so the
    pipeline can run on a fresh machine before the first ingestion."""
    db_path, parquet_dir = loader_dirs
    deltas = load_all_parquet(db_path=db_path, parquet_dir=parquet_dir)
    assert deltas == {"awards": 0, "entities": 0, "entity_snapshots": 0}


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
