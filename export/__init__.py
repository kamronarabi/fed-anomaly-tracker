"""Data publisher: writes JSON contracts from DuckDB for the web frontend.

The nightly cron calls `python -m export.publish` after the scoring + brief
generation steps. It produces `leaderboard.json` plus one
`entities/<uei>.json` per top-N entity into `web/public/data/`. The Next.js
frontend reads these at build time; no runtime DuckDB access from the
frontend.
"""
