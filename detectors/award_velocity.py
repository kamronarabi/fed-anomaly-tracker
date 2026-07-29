"""Award velocity (sudden-burst) detector.

Compares each entity's recent award arrival rate to their own historical
baseline rate. The signal: most federal contractors have a steady
contracting pace. A sudden burst — 10x normal velocity in the last 90
days — can indicate preferential awarding, fraudulent rapid extraction
before disappearance, or improper insider relationships. Distinguishes
"established vendor at normal pace" from "established vendor in a
sudden burst."

Method:
  recent_count    = awards in (today - window_days, today]
  baseline_count  = awards in (today - history_days, today - window_days]
  baseline_rate   = baseline_count / (history_days - window_days)
  expected_recent = baseline_rate * window_days
  std (Poisson)   = sqrt(expected_recent)
  z               = (recent_count - expected_recent) / std
  score           = sigmoid(z - z_threshold)

We use the Poisson approximation (variance = mean for count data) rather
than computing the empirical std of monthly buckets. Federal contract
arrivals are approximately Poisson, so this is statistically defensible
and avoids both bucketing complexity and the small-sample noise of
empirical std on irregular monthly counts.

Reproducibility / `today` parameter:
  Same I2-style issue we fixed in `isolation` — if "today" is implicitly
  `date.today()`, the detector emits different output for the same DB on
  different days. To make output reproducible (important for tests, for
  composite scoring's percentile rank being stable, and for the daily
  cron's idempotence), `today` is an explicit parameter. Defaults to
  `MAX(award_date)` from the data, which is the right anchor because:
    (1) the archive snapshot is what bounds "now" anyway
    (2) it lets the same DB produce the same detector output regardless
        of when the detector is invoked

Known limitations (documented, not fixed in v1):
  - **Fiscal-year seasonality.** September is "use-it-or-lose-it" month;
    many legitimate vendors have a Q4 spike that's a seasonal artifact
    rather than fraud. v1 accepts this noise. Future refinement: compute
    baseline from same-month-of-prior-year rather than rolling 640-day
    window.
  - **Archive snapshot lag.** Our data is current through the archive
    snapshot date (typically 2-4 weeks behind real time). "Last 90 days"
    means 90 days before the archive snapshot, not 90 days before real
    today. Using `today = MAX(award_date)` keeps this consistent.

Score interpretation: absolute, in (sigmoid(0), 1.0) = (0.5, 1.0) at
the default `z_threshold=2.0`. Cross-detector comparisons use percentile
rank downstream so the floor doesn't matter for ranking.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import duckdb
import polars as pl

DEFAULT_WINDOW_DAYS = 90
DEFAULT_HISTORY_DAYS = 730
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_MIN_BASELINE_AWARDS = 5


def detect_award_velocity(
    db_path: str,
    window_days: int = DEFAULT_WINDOW_DAYS,
    history_days: int = DEFAULT_HISTORY_DAYS,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    min_baseline_awards: int = DEFAULT_MIN_BASELINE_AWARDS,
    today: date | None = None,
) -> pl.DataFrame:
    """Score entities whose recent contracting velocity is anomalously
    high vs. their own historical baseline.

    Returns the four-column detector schema. Score = sigmoid(z - z_threshold).
    """
    if window_days <= 0 or history_days <= window_days:
        raise ValueError(
            "history_days must be greater than window_days; both positive"
        )

    if today is None:
        # Anchor on the data's most recent award_date rather than wall-clock
        # today so the detector is reproducible across runs.
        con = duckdb.connect(db_path, read_only=True)
        try:
            result = con.execute("SELECT MAX(award_date) FROM awards").fetchone()
        finally:
            con.close()
        if not result or result[0] is None:
            return _empty()
        today = result[0]

    recent_start = today - timedelta(days=window_days)
    baseline_start = today - timedelta(days=history_days)
    baseline_end = recent_start  # = today - window_days

    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            """
            WITH entity_counts AS (
              SELECT
                recipient_uei AS uei,
                COUNT(*) FILTER (
                  WHERE award_date > ? AND award_date <= ?
                ) AS recent_count,
                COUNT(*) FILTER (
                  WHERE award_date > ? AND award_date <= ?
                ) AS baseline_count,
                MIN(award_date) AS earliest_award,
                MAX(award_date) AS latest_award
              FROM awards
              WHERE recipient_uei IS NOT NULL
                AND recipient_uei != ''
                AND award_date IS NOT NULL
                AND total_obligation IS NOT NULL
                AND total_obligation > 0
              GROUP BY recipient_uei
            )
            SELECT
              uei,
              recent_count,
              baseline_count,
              earliest_award,
              latest_award
            FROM entity_counts
            WHERE baseline_count >= ?
            """,
            [
                recent_start, today,
                baseline_start, baseline_end,
                min_baseline_awards,
            ],
        ).fetchall()
    finally:
        con.close()

    if not rows:
        return _empty()

    baseline_window_days = history_days - window_days
    out_rows = []
    for r in rows:
        uei, recent_count, baseline_count, earliest, latest = r
        baseline_rate = baseline_count / baseline_window_days  # awards/day
        expected_recent_count = baseline_rate * window_days
        if expected_recent_count <= 0:
            # No baseline rate → cannot score. Filter already excludes
            # baseline_count < min_baseline_awards, but defensive guard.
            continue
        recent_std = math.sqrt(expected_recent_count)  # Poisson std
        z = (recent_count - expected_recent_count) / recent_std
        if z < z_threshold:
            continue

        score = 1.0 / (1.0 + math.exp(-(z - z_threshold)))
        score = max(0.0, min(1.0, score))

        details = {
            "recent_count": recent_count,
            "baseline_count": baseline_count,
            "recent_window_days": window_days,
            "baseline_window_days": baseline_window_days,
            "recent_rate_per_day": recent_count / window_days,
            "baseline_rate_per_day": baseline_rate,
            "expected_recent_count": expected_recent_count,
            "z_score": z,
            "earliest_award": earliest.isoformat() if earliest else None,
            "latest_award": latest.isoformat() if latest else None,
            "as_of": today.isoformat(),
        }
        out_rows.append(
            {
                "uei": uei,
                "detector": "award_velocity",
                "score": score,
                "details": json.dumps(details),
            }
        )

    return pl.DataFrame(out_rows, schema=_schema())


def _schema() -> dict:
    return {
        "uei": pl.Utf8,
        "detector": pl.Utf8,
        "score": pl.Float64,
        "details": pl.Utf8,
    }


def _empty() -> pl.DataFrame:
    return pl.DataFrame([], schema=_schema())


if __name__ == "__main__":
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load(Path("config.yaml").read_text())
    df = (
        detect_award_velocity(cfg["db_path"])
        .sort("score", descending=True)
        .head(10)
    )
    print(df)
