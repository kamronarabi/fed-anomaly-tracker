"""Composite scoring: aggregate detector outputs into a single ranked table.

Runs all USAspending-only detectors, normalizes their disparate raw score
scales via **per-detector percentile ranking among emitted entities**, and
averages those four percentiles into one composite_score per entity. The
result is keyed by `(uei, score_date)` in the `suspicion_scores` table so
the daily orchestrator can append a new row each run and the dashboard's
Trending view can compute week-over-week percentile-rank deltas.

Why percentile rank instead of weighted average of raw scores:
- Benford emits absolute scores in [0, 1]
- mod_growth emits filter-floored scores in (0.5, 1.0)
- isolation emits batch-relative scores spanning exactly [0, 1] per run
- new_entity emits magnitude-floored scores in [0.7, 1.0]

A naive weighted average under-weights the absolute detectors and
over-weights the floored ones. Per-detector percentile rank within the
universe of *flagged* entities normalizes everything to the same
distribution before averaging. No hand-tuning of weights required.

Universe handling: an entity that's flagged by N of the 4 detectors gets
a percentile rank in [0, 1] for each of those N detectors and 0.0 for the
other 4-N. Composite = simple mean of all 4 numbers. This means:
- Entity flagged by 1 detector at the top of its emissions: composite = 0.25
- Entity flagged by 4 detectors at the top of each: composite = 1.0
- Entity flagged by 0 detectors: not emitted at all (no row).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path

import duckdb
import polars as pl
import yaml

from detectors.award_velocity import detect_award_velocity
from detectors.benford import detect_benford
from detectors.isolation import detect_isolation_outlier
from detectors.mod_growth import detect_mod_growth
from detectors.new_entity import detect_new_entity_sole_source
from detectors.sole_source_concentration import detect_sole_source_concentration
from ingestion.load_db import init_schema, resolve_db_path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Registered detectors. Order is fixed so column ordering in the
# suspicion_scores table is stable across runs.
DETECTORS: dict[str, callable] = {
    "benford": detect_benford,
    "new_entity": detect_new_entity_sole_source,
    "mod_growth": detect_mod_growth,
    "isolation": detect_isolation_outlier,
    "sole_source_concentration": detect_sole_source_concentration,
    "award_velocity": detect_award_velocity,
}

# Columns in the output DataFrame (and `suspicion_scores` table), in order.
OUTPUT_COLUMNS = [
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
]

OUTPUT_SCHEMA = {
    "uei": pl.Utf8,
    "score_date": pl.Date,
    "composite_score": pl.Float64,
    "composite_percentile_rank": pl.Float64,
    "benford_score": pl.Float64,
    "benford_percentile": pl.Float64,
    "new_entity_score": pl.Float64,
    "new_entity_percentile": pl.Float64,
    "mod_growth_score": pl.Float64,
    "mod_growth_percentile": pl.Float64,
    "isolation_score": pl.Float64,
    "isolation_percentile": pl.Float64,
    "sole_source_concentration_score": pl.Float64,
    "sole_source_concentration_percentile": pl.Float64,
    "award_velocity_score": pl.Float64,
    "award_velocity_percentile": pl.Float64,
    "detector_details": pl.Utf8,
    "generated_at": pl.Datetime,
}


def compute_composite_scores(
    db_path: str,
    score_date: date | None = None,
) -> pl.DataFrame:
    """Run all detectors, compute composite percentile-rank scores, persist to DB.

    Args:
        db_path: Path to the DuckDB file. Must already contain the `awards`
            table populated with USAspending data; `suspicion_scores` table
            will be created if missing.
        score_date: Date to attach to the run. Defaults to today. Multiple
            calls with the same score_date upsert (no duplicates); calls
            with different score_dates create separate rows.

    Returns:
        Polars DataFrame with one row per flagged UEI and the full set of
        per-detector + composite columns. Empty DataFrame if no detectors
        emit any rows.
    """
    if score_date is None:
        score_date = date.today()

    init_schema(db_path)
    generated_at = datetime.now()

    # 1. Run every detector. Each returns a DataFrame of (uei, detector,
    #    score, details) with one row per flagged entity.
    detector_outputs: dict[str, pl.DataFrame] = {}
    for name, fn in DETECTORS.items():
        logger.info("running detector: %s", name)
        df = fn(db_path)
        logger.info("  %s emitted %d rows", name, df.height)
        detector_outputs[name] = df

    # 2. Universe = union of UEIs flagged by any detector. Entities flagged
    #    by zero detectors don't make the leaderboard at all (saves space
    #    and avoids polluting the composite-percentile-rank distribution
    #    with non-signal entries).
    universe: set[str] = set()
    for df in detector_outputs.values():
        universe.update(df["uei"].to_list())

    if not universe:
        logger.info("no entities flagged by any detector — wiping prior rows for this date")
        # Still call the upsert path so any stale rows from a prior (looser)
        # run on the same score_date are cleared. Otherwise an empty re-run
        # silently inherits ghost rows.
        _upsert_to_suspicion_scores(db_path, _empty(), score_date)
        return _empty()

    # 3. Pivot to wide: one row per UEI with per-detector raw score
    #    + details payload. Missing detector → score=0, details=None.
    wide_rows = {uei: {"uei": uei} for uei in universe}
    for name, df in detector_outputs.items():
        score_col = f"{name}_score"
        details_col = f"{name}_details_raw"
        # Per-detector lookup of (score, details) by uei.
        emitted = {
            row["uei"]: (row["score"], row["details"])
            for row in df.iter_rows(named=True)
        }
        for uei in universe:
            if uei in emitted:
                wide_rows[uei][score_col] = float(emitted[uei][0])
                wide_rows[uei][details_col] = emitted[uei][1]
            else:
                wide_rows[uei][score_col] = 0.0
                wide_rows[uei][details_col] = None

    # infer_schema_length=None: scan every row, not just a sample. `universe`
    # is a set, so its iteration order (and thus row order here) varies by
    # process (string hash randomization) — a sampled inference can land on
    # an all-None window for a sparsely-emitted detector's `*_details_raw`
    # column (e.g. award_velocity, ~4% emission rate) and infer it as a
    # null-only type, then crash on the first real value outside the sample.
    wide = pl.DataFrame(list(wide_rows.values()), infer_schema_length=None)

    # 4. Per-detector percentile rank computed over **emitted entities only**.
    #    Normalization: `rank / emitted_count`. Highest emitted entity gets
    #    1.0; lowest emitted gets 1/emitted_count (positive, distinguishing
    #    "flagged but low" from "not flagged at all"). Non-emitted entities
    #    stay at 0.0. Ties get average rank.
    #
    #    Why not min-max [0, 1]: min-max would map the lowest-emitted
    #    entity to 0.0, collapsing it with non-emitted entities. We'd
    #    lose the "this detector flagged you" signal in the composite.
    n_universe = wide.height
    for name in DETECTORS.keys():
        score_col = f"{name}_score"
        pct_col = f"{name}_percentile"
        emitted_count = int((wide[score_col] > 0.0).sum())
        if emitted_count == 0:
            wide = wide.with_columns(pl.lit(0.0).alias(pct_col))
            continue

        emitted_mask = pl.col(score_col) > 0.0
        # Rank among emitted only. Need to rank within the filtered subset,
        # not the full DataFrame (otherwise the 0-score rows compete in the
        # rank tie-break and the math goes wonky).
        emitted_df = wide.filter(emitted_mask).with_columns(
            pl.col(score_col).rank(method="average").alias("_rank_tmp")
        )
        # Map uei -> percentile, attach back to wide.
        uei_to_pct = {
            r["uei"]: r["_rank_tmp"] / emitted_count
            for r in emitted_df.iter_rows(named=True)
        }
        wide = wide.with_columns(
            pl.col("uei")
            .map_elements(
                lambda u, lookup=uei_to_pct: lookup.get(u, 0.0),
                return_dtype=pl.Float64,
            )
            .alias(pct_col)
        )

    # 5. Composite = simple mean of the four detector percentiles.
    pct_cols = [f"{name}_percentile" for name in DETECTORS.keys()]
    wide = wide.with_columns(
        pl.mean_horizontal([pl.col(c) for c in pct_cols]).alias("composite_score")
    )

    # 6. composite_percentile_rank: rank composite_scores in [0, 1] across the universe.
    if n_universe == 1:
        wide = wide.with_columns(pl.lit(1.0).alias("composite_percentile_rank"))
    else:
        wide = wide.with_columns(
            pl.col("composite_score").rank(method="average").alias("_c_rank")
        )
        min_c = wide.select(pl.col("_c_rank").min()).item()
        max_c = wide.select(pl.col("_c_rank").max()).item()
        if max_c == min_c:
            wide = wide.with_columns(pl.lit(1.0).alias("composite_percentile_rank"))
        else:
            wide = wide.with_columns(
                (
                    (pl.col("_c_rank") - min_c) / (max_c - min_c)
                ).alias("composite_percentile_rank")
            )
        wide = wide.drop("_c_rank")

    # 7. Merge per-detector details JSON blobs into a single object keyed
    #    by detector name. Detectors that didn't flag the entity are absent
    #    (no null entry) so consumers can iterate over keys without
    #    filtering.
    detail_cols = [f"{name}_details_raw" for name in DETECTORS.keys()]
    merged_details = []
    for row in wide.iter_rows(named=True):
        merged: dict = {}
        for name in DETECTORS.keys():
            raw = row[f"{name}_details_raw"]
            if raw is not None:
                try:
                    merged[name] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    merged[name] = raw
        merged_details.append(json.dumps(merged) if merged else None)

    wide = wide.with_columns(
        pl.Series("detector_details", merged_details, dtype=pl.Utf8)
    ).drop(detail_cols)

    # 8. Add score_date and generated_at columns, project to the public schema.
    wide = wide.with_columns(
        pl.lit(score_date).alias("score_date"),
        pl.lit(generated_at).alias("generated_at"),
    )
    out = wide.select(OUTPUT_COLUMNS)
    out = out.cast({col: dtype for col, dtype in OUTPUT_SCHEMA.items()})

    # 9. Persist: wipe any prior rows for this score_date, then insert the
    #    current emissions. Different score_dates accumulate; same-date
    #    re-runs are fully idempotent (including universe shrinkage).
    _upsert_to_suspicion_scores(db_path, out, score_date)

    logger.info("composite scored %d entities for %s", out.height, score_date)
    return out


def _upsert_to_suspicion_scores(
    db_path: str, df: pl.DataFrame, score_date: date
) -> None:
    """Replace all rows for `score_date`, then insert the new emissions.

    Why DELETE-then-INSERT rather than per-row INSERT OR REPLACE: if the
    detector universe shrinks between runs (e.g., a detector's threshold
    is tightened, or a previously-flagged entity no longer triggers any
    detector), per-row upserts leave orphaned rows for that score_date
    from the previous run. Wiping the score_date first guarantees the
    table state matches the current run's emissions exactly.

    Past score_dates are untouched (historical scores are preserved).
    """
    con = duckdb.connect(db_path)
    try:
        # Wipe any prior emissions for this score_date.
        con.execute(
            "DELETE FROM suspicion_scores WHERE score_date = ?", [score_date]
        )

        if df.height == 0:
            return

        col_list = ", ".join(OUTPUT_COLUMNS)
        placeholders = ", ".join(["?"] * len(OUTPUT_COLUMNS))
        rows = list(df.select(OUTPUT_COLUMNS).iter_rows())
        con.executemany(
            f"INSERT INTO suspicion_scores ({col_list}) "
            f"VALUES ({placeholders})",
            rows,
        )
    finally:
        con.close()


def _empty() -> pl.DataFrame:
    return pl.DataFrame([], schema=OUTPUT_SCHEMA)


def main():
    parser = argparse.ArgumentParser(
        description="Run all detectors, compute composite scores, write to DuckDB."
    )
    parser.add_argument(
        "--score-date",
        type=date.fromisoformat,
        default=None,
        help="ISO date (YYYY-MM-DD) to label this run. Defaults to today.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Print the top N entities after scoring (default 10).",
    )
    args = parser.parse_args()

    db_path = resolve_db_path()
    df = compute_composite_scores(db_path, score_date=args.score_date)

    if df.height == 0:
        print("No entities flagged by any detector.")
        return

    print(
        df.sort("composite_score", descending=True)
        .head(args.top_n)
        .select(
            [
                "uei",
                "composite_score",
                "composite_percentile_rank",
                "benford_percentile",
                "new_entity_percentile",
                "mod_growth_percentile",
                "isolation_percentile",
            ]
        )
    )


if __name__ == "__main__":
    main()
