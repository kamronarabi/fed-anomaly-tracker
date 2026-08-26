"""Brief generation core: input hash, prompt assembly, Anthropic call, orchestration.

Public entry point is `generate_briefs(db_path, score_date, top_n, ...)`.
It picks the top-N entities by composite_score for the given score_date,
fingerprints each via `compute_input_hash`, and either forward-carries a
prior brief (cache hit) or calls Anthropic Haiku (cache miss).

The system prompt + detector reference is identical across all briefs in
a single run, so we mark it `cache_control=ephemeral`. Within Anthropic's
5-minute prompt-cache TTL this means only the first call in the batch
pays full input-token cost; the rest get a ~90% discount on that prefix.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import duckdb

from briefs.cache import find_cached_brief, write_brief

logger = logging.getLogger(__name__)


# Bumped whenever SYSTEM_PROMPT below changes. Old briefs with a stale
# prompt_version are NOT reused — they get regenerated. Bumping is the
# safest way to invalidate the entire cache.
PROMPT_VERSION = "v2"

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_TOP_N = 50
DEFAULT_MAX_TOKENS = 750

SYSTEM_PROMPT = """You write short fraud-detection briefs for federal contract awardees. Your reader is a journalist or compliance officer triaging which entities deserve a closer look.

You receive structured data per entity: identifying info, a composite suspicion score, and a list of statistical detectors that fired with their findings.

OUTPUT FORMAT — read carefully, this is enforced:
- Exactly three paragraphs separated by ONE blank line.
- 120-180 words TOTAL across all three paragraphs.
- Plain prose only. NO markdown. NO headers. NO bold. NO bullets. NO numbered lists. NO leading labels like "Why flagged:" or "Caveats:".
- Do not restate the entity name in every paragraph.

Paragraph 1: which detectors fired and what each one detected for THIS entity, in plain language.
Paragraph 2: the specific numbers from the detector details (z-scores, dollar amounts, counts) so a reader can verify.
Paragraph 3: caveats — these detectors surface statistical anomalies, not proof of fraud, and any limitation specific to this entity (industry context that may explain a signal, small peer group, etc.).

Detector reference:
- benford: digit-frequency anomalies in award amounts. max_z is the worst leading-digit deviation, in sigmas (>3 unusual, >5 very unusual).
- new_entity: entity is brand-new and won a large sole-source award as one of its first awards. lifetime_awards and lifetime_total bound fly-by-night profiles.
- mod_growth: a parent contract's modifications grew much faster in dollar value than same-NAICS peers. z_score measures growth-ratio deviation in sigmas.
- isolation: multivariate outlier via Isolation Forest. Score in [0,1]; higher = more outlier.
- sole_source_concentration: entity does much more sole-source work than peers in the same NAICS. z_score measures pct_sole_source deviation in sigmas.
- award_velocity: recent award count is far above the entity's own historical rate. z_score computed over a Poisson baseline.

Use only the data provided. Do not invent facts."""


@dataclass(frozen=True)
class BriefInput:
    """Everything the LLM needs to write a brief for one entity.

    Fields chosen so two entities producing the same `BriefInput` would
    receive the same brief — this is the input to `compute_input_hash`.
    """
    uei: str
    entity_name: str
    awarding_agency: str | None
    primary_naics: str | None
    total_obligated_lifetime: float | None
    award_count_lifetime: int | None
    composite_score: float
    composite_percentile_rank: float
    detectors_fired: list[dict] = field(default_factory=list)


# ── Input hash ───────────────────────────────────────────────────────────


def _canonicalize(obj: Any) -> Any:
    """Recursively normalize floats and sort dict keys so equivalent
    payloads hash identically regardless of insertion order or float noise.
    """
    if isinstance(obj, float):
        # 4dp is below the noise floor of every detector's score
        # (sigmoids saturate, z-scores rarely need more precision).
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj.keys(), key=str)}
    if isinstance(obj, list):
        return [_canonicalize(x) for x in obj]
    return obj


def compute_input_hash(
    brief_input: BriefInput,
    prompt_version: str = PROMPT_VERSION,
) -> str:
    """Deterministic sha256 of everything that would shape the LLM output.

    Stable against: detector ordering in detectors_fired, dict key order
    in details JSON, sub-precision float noise.

    Sensitive to: any score change >= 1e-4, any new/removed detector,
    any change to entity identifying info, prompt_version bumps.
    """
    detectors_sorted = sorted(
        brief_input.detectors_fired, key=lambda d: d.get("name", "")
    )
    payload = {
        "entity_name": brief_input.entity_name,
        "awarding_agency": brief_input.awarding_agency,
        "primary_naics": brief_input.primary_naics,
        "total_obligated_lifetime": brief_input.total_obligated_lifetime,
        "award_count_lifetime": brief_input.award_count_lifetime,
        "composite_score": brief_input.composite_score,
        "composite_percentile_rank": brief_input.composite_percentile_rank,
        "detectors_fired": detectors_sorted,
        "prompt_version": prompt_version,
    }
    canon = _canonicalize(payload)
    blob = json.dumps(canon, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ── Top-N selection ──────────────────────────────────────────────────────


# Detector name → (score_col, percentile_col). Same order as scoring/composite.py
# so a brief's `detectors_fired` lists detectors in the canonical order before
# we sort it for hashing/prompt.
_DETECTOR_COLS = [
    ("benford", "benford_score"),
    ("new_entity", "new_entity_score"),
    ("mod_growth", "mod_growth_score"),
    ("isolation", "isolation_score"),
    ("sole_source_concentration", "sole_source_concentration_score"),
    ("award_velocity", "award_velocity_score"),
]


def select_top_n(
    db_path: str, score_date: date, top_n: int
) -> list[BriefInput]:
    """Top-N entities by composite_score for `score_date`, enriched with
    awards-derived context (recipient name, primary agency/NAICS, totals).

    Returns an empty list if `suspicion_scores` has no rows for that date.
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        # Per-UEI awards aggregates. ARG_MAX picks the value associated with
        # the latest award_date — most-recent recipient_name handles
        # entities whose name slightly varies across awards.
        score_cols = ", ".join(c for _, c in _DETECTOR_COLS)
        rows = con.execute(
            f"""
            WITH agg AS (
                SELECT
                    recipient_uei,
                    ARG_MAX(recipient_name, award_date) AS recipient_name,
                    ARG_MAX(awarding_agency, award_date) AS awarding_agency,
                    ARG_MAX(naics_description, award_date) AS primary_naics,
                    SUM(total_obligation) AS total_obligated_lifetime,
                    COUNT(*) AS award_count_lifetime
                FROM awards
                WHERE recipient_uei IS NOT NULL
                GROUP BY recipient_uei
            )
            SELECT
                s.uei,
                COALESCE(agg.recipient_name, s.uei) AS entity_name,
                agg.awarding_agency,
                agg.primary_naics,
                agg.total_obligated_lifetime,
                agg.award_count_lifetime,
                s.composite_score,
                s.composite_percentile_rank,
                s.detector_details,
                {score_cols}
            FROM suspicion_scores s
            LEFT JOIN agg ON s.uei = agg.recipient_uei
            WHERE s.score_date = ?
            ORDER BY s.composite_score DESC
            LIMIT ?
            """,
            [score_date, top_n],
        ).fetchall()
    finally:
        con.close()

    out: list[BriefInput] = []
    for row in rows:
        (
            uei, entity_name, awarding_agency, primary_naics,
            total_obligated_lifetime, award_count_lifetime,
            composite_score, composite_percentile_rank, detector_details_json,
            *score_values,
        ) = row

        try:
            details_map = json.loads(detector_details_json) if detector_details_json else {}
        except (json.JSONDecodeError, TypeError):
            details_map = {}

        detectors_fired: list[dict] = []
        for (name, _col), score in zip(_DETECTOR_COLS, score_values):
            if score is None or score <= 0.0:
                continue
            detectors_fired.append({
                "name": name,
                "score": float(score),
                "details": details_map.get(name, {}),
            })

        out.append(BriefInput(
            uei=uei,
            entity_name=entity_name,
            awarding_agency=awarding_agency,
            primary_naics=primary_naics,
            total_obligated_lifetime=(
                float(total_obligated_lifetime)
                if total_obligated_lifetime is not None else None
            ),
            award_count_lifetime=(
                int(award_count_lifetime)
                if award_count_lifetime is not None else None
            ),
            composite_score=float(composite_score),
            composite_percentile_rank=float(composite_percentile_rank),
            detectors_fired=detectors_fired,
        ))
    return out


# ── Prompt assembly ──────────────────────────────────────────────────────


def build_messages(brief_input: BriefInput) -> tuple[list[dict], list[dict]]:
    """Return `(system, messages)` ready for `client.messages.create`.

    `system` is a list of content blocks rather than a bare string so we
    can attach `cache_control` to the prompt+detector reference — the part
    that's identical across every brief in a run.
    """
    payload = {
        "uei": brief_input.uei,
        "entity_name": brief_input.entity_name,
        "awarding_agency": brief_input.awarding_agency,
        "primary_naics": brief_input.primary_naics,
        "total_obligated_lifetime": brief_input.total_obligated_lifetime,
        "award_count_lifetime": brief_input.award_count_lifetime,
        "composite_score": round(brief_input.composite_score, 4),
        "composite_percentile_rank": round(brief_input.composite_percentile_rank, 4),
        "detectors_fired": brief_input.detectors_fired,
    }
    user_text = (
        "Write the brief for the following entity. "
        "Use only the data provided.\n\n"
        + json.dumps(payload, indent=2, default=str)
    )
    system = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    messages = [{"role": "user", "content": user_text}]
    return system, messages


# ── Anthropic call ───────────────────────────────────────────────────────


def _extract_text(response: Any) -> str:
    """Pull plain text out of an Anthropic Messages response."""
    parts: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts).strip()


def call_anthropic(
    brief_input: BriefInput,
    *,
    client: Any,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """One sync Anthropic call. Returns the brief text."""
    system, messages = build_messages(brief_input)
    # No sampling params: temperature/top_p/top_k were removed from the
    # Messages API on current models and dropped from the SDK signature
    # in anthropic 1.0.0, where passing one is a TypeError. Brief
    # reproducibility comes from the input_hash cache (find_cached_brief),
    # not from pinning temperature to 0.
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    return _extract_text(response)


def _default_client():
    """Lazy-import + construct the Anthropic SDK client.

    Loaded on demand so tests that pass their own client never need the
    `anthropic` package installed, and the CLI fails loudly with a clear
    error when ANTHROPIC_API_KEY is unset.
    """
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise RuntimeError(
            "anthropic SDK not installed. `pip install anthropic`."
        ) from e

    # python-dotenv is already a project dep; load .env if present.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to .env or export it."
        )
    return Anthropic(api_key=api_key)


# ── Orchestration ────────────────────────────────────────────────────────


# Safety ceiling on fresh (uncached) Anthropic calls per run, independent
# of top_n -- e.g. if the brief cache ever regresses to near-0 hit rate
# again (see detectors/isolation.py's ORDER BY fix), this bounds the
# damage instead of silently paying for top_n fresh calls every run.
# Overridable per-environment without a code change.
DEFAULT_MAX_FRESH_CALLS = int(os.environ.get("MAX_FRESH_BRIEFS_PER_RUN", "60"))


def generate_briefs(
    db_path: str,
    score_date: date,
    top_n: int = DEFAULT_TOP_N,
    *,
    client: Any = None,
    model: str = DEFAULT_MODEL,
    prompt_version: str = PROMPT_VERSION,
    max_fresh_calls: int = DEFAULT_MAX_FRESH_CALLS,
) -> int:
    """For each of the top-N entities on `score_date`, write or forward-carry
    a brief into `entity_briefs`.

    Stops making fresh API calls once `max_fresh_calls` is reached; any
    remaining entities are simply left without a brief row for this
    score_date (degraded, not fatal -- the dashboard just shows no brief
    text for them today).

    Returns the number of fresh Anthropic API calls made (cache misses).
    The total brief count written is `len(select_top_n(...))` regardless,
    minus whatever was skipped by the cap.
    """
    picks = select_top_n(db_path, score_date=score_date, top_n=top_n)
    if not picks:
        logger.info("no entities to brief for %s", score_date)
        return 0

    api_calls = 0
    for i, bi in enumerate(picks):
        if api_calls >= max_fresh_calls:
            logger.warning(
                "max_fresh_calls cap (%d) reached; skipping remaining %d entities for %s",
                max_fresh_calls, len(picks) - i, score_date,
            )
            break
        input_hash = compute_input_hash(bi, prompt_version=prompt_version)
        cached = find_cached_brief(
            db_path, uei=bi.uei, input_hash=input_hash, prompt_version=prompt_version,
        )
        if cached is not None:
            # Forward-carry: reuse the prior text under today's score_date.
            # If cached.score_date == score_date already (same-day re-run),
            # this is a no-op rewrite — INSERT OR REPLACE keeps the row.
            write_brief(
                db_path,
                uei=bi.uei,
                score_date=score_date,
                input_hash=input_hash,
                brief_text=cached["brief_text"],
                model=cached["model"],
                prompt_version=prompt_version,
                generated_at=cached["generated_at"],
            )
            continue

        # Cache miss — need a fresh API call. Construct client lazily if
        # the caller didn't pass one (CLI path).
        if client is None:
            client = _default_client()

        logger.info("generating brief for %s (%s)", bi.uei, bi.entity_name)
        brief_text = call_anthropic(bi, client=client, model=model)
        api_calls += 1
        write_brief(
            db_path,
            uei=bi.uei,
            score_date=score_date,
            input_hash=input_hash,
            brief_text=brief_text,
            model=model,
            prompt_version=prompt_version,
            generated_at=datetime.now(),
        )

    logger.info(
        "briefs for %s: %d total (%d new API calls, %d cached)",
        score_date, len(picks), api_calls, len(picks) - api_calls,
    )
    return api_calls
