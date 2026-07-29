"""Phase 4a: AI-generated briefs over the composite-scored leaderboard.

Top-N entities (default 50) for a given score_date are summarized by
Claude Haiku into a short plain-text brief explaining which detectors
fired and why. Briefs are cached by `(uei, input_hash, prompt_version)`
so unchanged entities forward-carry yesterday's brief verbatim instead
of paying for a fresh API call.

Run via `python -m briefs.main --score-date YYYY-MM-DD --top-n 50`.
"""
