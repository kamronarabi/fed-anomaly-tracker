"""Benford's Law leading-digit anomaly detector.

For each UEI with at least MIN_TRANSACTIONS award rows, compare the observed
distribution of leading digits in `total_obligation` against the theoretical
Benford distribution using a Kolmogorov-Smirnov test. A low p-value means
the observed distribution is unlikely under Benford -- score = 1 - p_value.

Score interpretation: absolute, in [0, 1]. score ≈ 1 means the observed
leading-digit distribution is overwhelmingly unlikely under Benford
(strong fabrication signal). score ≈ 0 means the distribution is
consistent with Benford. Comparable across runs and across entities.

Implementation note on the KS statistic: rather than passing raw integer
digit values (1-9) to scipy.stats.kstest (which is designed for continuous
distributions and produces a spuriously large D-statistic on discrete data),
we compute the KS statistic directly as the maximum absolute difference
between the observed 9-bin cumulative distribution and the Benford CDF, then
derive the p-value from the Kolmogorov asymptotic distribution via
scipy.stats.kstwobign.sf. This gives the correct, interpretable p-value for
a KS-style test of a 9-bin discrete distribution against Benford.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np
import duckdb
import polars as pl
from scipy import stats

MIN_TRANSACTIONS = 30

# P(d) = log10(1 + 1/d) for d in 1..9.
_BENFORD_PMF = [math.log10(1 + 1 / d) for d in range(1, 10)]
_BENFORD_CDF = []
_acc = 0.0
for _p in _BENFORD_PMF:
    _acc += _p
    _BENFORD_CDF.append(_acc)
_BENFORD_CDF_ARR = np.array(_BENFORD_CDF)


def _leading_digit(amount: float) -> int | None:
    """Return the first non-zero decimal digit of `amount`, or None if
    `amount` is None, zero, or negative."""
    if amount is None or amount <= 0:
        return None
    s = f"{amount:.10g}"
    for ch in s:
        if ch.isdigit() and ch != "0":
            return int(ch)
    return None


def _ks_score(digits: list[int]) -> tuple[float, float, float]:
    """Compute KS statistic, p-value, and score for a list of leading digits.

    Returns (ks_statistic, p_value, score) where score = 1 - p_value, clamped
    to [0, 1].

    The KS statistic is the max absolute difference between the observed
    cumulative digit-proportion vector and the Benford CDF vector (both
    evaluated at the 9 integer digit points). The asymptotic p-value is
    derived via scipy.stats.kstwobign.sf(ks_stat * sqrt(n)).
    """
    n = len(digits)
    obs_counts = [digits.count(d) for d in range(1, 10)]
    obs_cdf = np.cumsum([c / n for c in obs_counts])
    ks_stat = float(np.max(np.abs(obs_cdf - _BENFORD_CDF_ARR)))
    p_value = float(stats.kstwobign.sf(ks_stat * math.sqrt(n)))
    score = max(0.0, min(1.0, 1.0 - p_value))
    return ks_stat, p_value, score


def detect_benford(db_path: str) -> pl.DataFrame:
    """Score each UEI's leading-digit distribution against Benford."""
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            """
            SELECT recipient_uei AS uei, total_obligation
            FROM awards
            WHERE recipient_uei IS NOT NULL
              AND total_obligation IS NOT NULL
              AND total_obligation > 0
            """
        ).fetchall()
    finally:
        con.close()

    by_uei: dict[str, list[int]] = defaultdict(list)
    for uei, amount in rows:
        d = _leading_digit(amount)
        if d is not None:
            by_uei[uei].append(d)

    results: list[dict] = []
    for uei, digits in by_uei.items():
        n = len(digits)
        if n < MIN_TRANSACTIONS:
            continue

        observed_counts = [digits.count(d) for d in range(1, 10)]
        observed_dist = [c / n for c in observed_counts]

        ks_stat, p_value, score = _ks_score(digits)

        details = {
            "observed_distribution": {
                str(d): observed_dist[d - 1] for d in range(1, 10)
            },
            "expected_distribution": {
                str(d): _BENFORD_PMF[d - 1] for d in range(1, 10)
            },
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "n_transactions": n,
        }
        results.append(
            {
                "uei": uei,
                "detector": "benford",
                "score": score,
                "details": json.dumps(details),
            }
        )

    return pl.DataFrame(
        results,
        schema={
            "uei": pl.Utf8,
            "detector": pl.Utf8,
            "score": pl.Float64,
            "details": pl.Utf8,
        },
    )


if __name__ == "__main__":
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load(Path("config.yaml").read_text())
    df = detect_benford(cfg["db_path"]).sort("score", descending=True).head(10)
    print(df)
