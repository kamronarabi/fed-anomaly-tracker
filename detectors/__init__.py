"""Phase 2 anomaly detectors.

Each module exposes a single `detect_<name>(db_path) -> polars.DataFrame`
function returning the shared 4-column contract `(uei, detector, score, details)`.

Implemented (Wave 1):
  - benford      — leading-digit Kolmogorov-Smirnov goodness-of-fit
  - mod_growth   — z-scored modification-growth ratio within NAICS group
  - new_entity   — sole-source within N days of SAM registration
  - isolation    — sklearn IsolationForest on a 7-feature entity vector

Deferred (Wave 3):
  - address_churn — needs accumulated SAM snapshots, scheduled once the
    SAM API quota bump enables daily re-snapshots.
"""
