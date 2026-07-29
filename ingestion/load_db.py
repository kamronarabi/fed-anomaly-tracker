"""DuckDB schema and Parquet loader for the Anomaly Radar pipeline.

Task 1.1 defines the schema (`init_schema`).
Task 1.4 adds the Parquet -> DuckDB load with dedup/upsert semantics
(`load_all_parquet` and friends).

History: `entities` and `entity_snapshots` table init + loaders were
removed in the 2026-05-27 SAM-removal pivot. Empty SAM tables in any
already-populated DuckDB are left in place (harmless; can be dropped
later via `DROP TABLE`). The loader no longer creates them on init and
no longer reads any `entities*.parquet` / `entity_snapshots*.parquet`
files even if they exist on disk.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import duckdb
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS awards (
        award_id                       VARCHAR PRIMARY KEY,
        parent_award_id                VARCHAR,
        recipient_name                 VARCHAR,
        recipient_uei                  VARCHAR,
        awarding_agency                VARCHAR,
        awarding_sub_agency            VARCHAR,
        award_type                     VARCHAR,
        award_description              TEXT,
        naics_code                     VARCHAR,
        naics_description              VARCHAR,
        total_obligation               DOUBLE,
        base_and_all_options_value     DOUBLE,
        period_of_performance_start    DATE,
        period_of_performance_end      DATE,
        award_date                     DATE,
        competition_type               VARCHAR,
        number_of_offers               INTEGER,
        modification_number            VARCHAR,
        pulled_at                      TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS suspicion_scores (
        uei                                       VARCHAR,
        score_date                                DATE,
        composite_score                           DOUBLE,
        composite_percentile_rank                 DOUBLE,
        benford_score                             DOUBLE,
        benford_percentile                        DOUBLE,
        new_entity_score                          DOUBLE,
        new_entity_percentile                     DOUBLE,
        mod_growth_score                          DOUBLE,
        mod_growth_percentile                     DOUBLE,
        isolation_score                           DOUBLE,
        isolation_percentile                      DOUBLE,
        sole_source_concentration_score           DOUBLE,
        sole_source_concentration_percentile      DOUBLE,
        award_velocity_score                      DOUBLE,
        award_velocity_percentile                 DOUBLE,
        detector_details                          VARCHAR,
        generated_at                              TIMESTAMP,
        PRIMARY KEY (uei, score_date)
    )
    """,
    # Migration: add columns to suspicion_scores tables built before
    # the sole_source_concentration detector landed (2026-05-29).
    # Idempotent — IF NOT EXISTS prevents errors on fresh DBs.
    """
    ALTER TABLE suspicion_scores
        ADD COLUMN IF NOT EXISTS sole_source_concentration_score DOUBLE
    """,
    """
    ALTER TABLE suspicion_scores
        ADD COLUMN IF NOT EXISTS sole_source_concentration_percentile DOUBLE
    """,
    # Migration: add award_velocity columns (added 2026-05-29).
    """
    ALTER TABLE suspicion_scores
        ADD COLUMN IF NOT EXISTS award_velocity_score DOUBLE
    """,
    """
    ALTER TABLE suspicion_scores
        ADD COLUMN IF NOT EXISTS award_velocity_percentile DOUBLE
    """,
    # Phase 4a: AI-generated briefs. input_hash is a fingerprint of the
    # detector state we sent to the LLM; identical hash → reuse the prior
    # brief verbatim instead of calling the API. prompt_version is bumped
    # whenever the system prompt changes, invalidating all prior caches.
    """
    CREATE TABLE IF NOT EXISTS entity_briefs (
        uei                VARCHAR,
        score_date         DATE,
        input_hash         VARCHAR,
        brief_text         TEXT,
        model              VARCHAR,
        prompt_version     VARCHAR,
        generated_at       TIMESTAMP,
        PRIMARY KEY (uei, score_date)
    )
    """,
]


# Explicit column list — used to align the SELECT against the table even
# if a Parquet file's columns are reordered or have an extra column.
AWARD_COLUMNS = [
    "award_id",
    "parent_award_id",
    "recipient_name",
    "recipient_uei",
    "awarding_agency",
    "awarding_sub_agency",
    "award_type",
    "award_description",
    "naics_code",
    "naics_description",
    "total_obligation",
    "base_and_all_options_value",
    "period_of_performance_start",
    "period_of_performance_end",
    "award_date",
    "competition_type",
    "number_of_offers",
    "modification_number",
    "pulled_at",
]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def resolve_db_path(config: dict | None = None) -> str:
    """Return the DuckDB path, preferring the DB_PATH env var (used in prod on Railway)."""
    if config is None:
        config = load_config()
    return os.environ.get("DB_PATH", config["db_path"])


def effective_agency(agency: dict, config: dict) -> dict:
    """Resolve per-agency overrides against config-level defaults.

    Returns a new dict with `award_types`, `award_amount_min`, and
    `seed_strategy` always populated. Use this anywhere downstream code
    needs to know an agency's effective ingestion settings.
    """
    return {
        **agency,
        "award_types": (
            agency.get("award_types")
            or config.get("award_types")
            or ["A", "B", "C", "D"]
        ),
        "award_amount_min": int(
            agency.get("award_amount_min", config.get("award_amount_min", 0))
        ),
        "seed_strategy": (
            agency.get("seed_strategy")
            or config.get("seed_strategy")
            or "paginate"
        ),
    }


def init_schema(db_path: str | None = None) -> str:
    """Create the awards table if it doesn't exist.

    Returns the resolved db_path so callers can chain further work.
    """
    if db_path is None:
        db_path = resolve_db_path()

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db_path)
    try:
        for stmt in SCHEMA_STATEMENTS:
            con.execute(stmt)
    finally:
        con.close()

    return db_path


def _load_parquet_files(
    con: duckdb.DuckDBPyConnection,
    table: str,
    columns: list[str],
    paths: list[Path],
    mode: str,
) -> int:
    """Load `paths` into `table` via INSERT OR {mode} ... SELECT FROM read_parquet.

    `mode` is "REPLACE" (overwrite on PK conflict — newer pulls win) or
    "IGNORE" (keep existing rows on PK conflict).

    Returns the net delta in row count.
    """
    if not paths:
        logger.info("%s: no Parquet files to load", table)
        return 0

    col_list = ", ".join(columns)
    select_cols = ", ".join(columns)
    path_strs = [str(p) for p in paths]

    before = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    con.execute(
        f"INSERT OR {mode} INTO {table} ({col_list}) "
        f"SELECT {select_cols} "
        f"FROM read_parquet(?, union_by_name=true)",
        [path_strs],
    )
    after = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    delta = after - before
    logger.info(
        "%s: loaded %d Parquet file(s) (%+d rows, total=%d)",
        table, len(paths), delta, after,
    )
    return delta


def load_awards(con, parquet_dir: Path) -> int:
    """Upsert every `awards_*.parquet` under `parquet_dir` into `awards`."""
    paths = sorted(parquet_dir.glob("awards_*.parquet"))
    return _load_parquet_files(con, "awards", AWARD_COLUMNS, paths, mode="REPLACE")


def load_all_parquet(
    db_path: str | None = None, parquet_dir: Path | str | None = None
) -> dict[str, int]:
    """Initialize schema and load every Parquet partition into DuckDB.

    Returns a per-table delta dict for logging/tests:
        {"awards": +N}
    """
    config = load_config()
    db_path = db_path or resolve_db_path(config)
    parquet_dir = Path(parquet_dir or config["parquet_dir"])

    init_schema(db_path)

    if not parquet_dir.exists():
        logger.warning(
            "parquet_dir %s does not exist — nothing to load", parquet_dir
        )
        return {"awards": 0}

    con = duckdb.connect(db_path)
    try:
        deltas = {
            "awards": load_awards(con, parquet_dir),
        }
    finally:
        con.close()
    return deltas


def main():
    parser = argparse.ArgumentParser(
        description="Initialize schema and load Parquet files into DuckDB"
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Only create the schema; skip Parquet loading",
    )
    args = parser.parse_args()

    if args.init_only:
        path = init_schema()
        print(f"Initialized DuckDB schema at {path}")
        return

    deltas = load_all_parquet()
    print(f"Loaded: awards {deltas['awards']:+d}")


if __name__ == "__main__":
    main()
