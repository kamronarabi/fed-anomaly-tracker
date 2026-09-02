"""Unit tests for Phase 4a brief generation.

The brief generator picks the top-N entities by composite_score for a given
score_date, fingerprints each entity's detector state via `compute_input_hash`,
and either reuses a prior brief (if the same fingerprint exists for a prior
score_date) or calls the Anthropic API to generate a fresh one.

Tests use a fake Anthropic client so no network or API key is required.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest

from ingestion.load_db import init_schema


# ── Fake Anthropic client ────────────────────────────────────────────────


class FakeAnthropic:
    """Captures calls to `messages.create` and returns a canned response."""

    def __init__(self, response_text: str = "Mocked brief paragraph one.\n\nParagraph two.\n\nParagraph three."):
        self.response_text = response_text
        self.calls: list[dict] = []
        self.messages = self  # so client.messages.create works

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.response_text)],
            stop_reason="end_turn",
        )


# ── DB helpers ───────────────────────────────────────────────────────────


def _fresh_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "briefs.duckdb")
    init_schema(db_path)
    return db_path


def _insert_award(con, **kwargs) -> None:
    base = {
        "award_id": kwargs["award_id"],
        "parent_award_id": None,
        "recipient_name": kwargs.get("recipient_name", "Test Co"),
        "recipient_uei": kwargs["recipient_uei"],
        "awarding_agency": kwargs.get("awarding_agency", "Department of Defense"),
        "awarding_sub_agency": None,
        "award_type": "DEFINITIVE CONTRACT",
        "award_description": None,
        "naics_code": kwargs.get("naics_code", "541330"),
        "naics_description": kwargs.get("naics_description", "Engineering Services"),
        "total_obligation": kwargs.get("total_obligation", 100_000.0),
        "base_and_all_options_value": kwargs.get("total_obligation", 100_000.0),
        "period_of_performance_start": date(2024, 1, 1),
        "period_of_performance_end": date(2025, 1, 1),
        "award_date": kwargs.get("award_date", date(2024, 6, 1)),
        "competition_type": "FULL AND OPEN COMPETITION",
        "number_of_offers": 3,
        "modification_number": "0",
        "pulled_at": datetime(2026, 5, 1, 12, 0, 0),
    }
    cols = list(base.keys())
    placeholders = ", ".join(["?"] * len(cols))
    con.execute(
        f"INSERT INTO awards ({', '.join(cols)}) VALUES ({placeholders})",
        [base[c] for c in cols],
    )


def _insert_suspicion_score(
    con,
    *,
    uei: str,
    score_date: date,
    composite_score: float,
    composite_percentile_rank: float = 0.95,
    benford_score: float = 0.0,
    new_entity_score: float = 0.0,
    mod_growth_score: float = 0.0,
    isolation_score: float = 0.0,
    sole_source_concentration_score: float = 0.0,
    award_velocity_score: float = 0.0,
    detector_details: str = "{}",
) -> None:
    con.execute(
        """
        INSERT INTO suspicion_scores (
            uei, score_date, composite_score, composite_percentile_rank,
            benford_score, benford_percentile,
            new_entity_score, new_entity_percentile,
            mod_growth_score, mod_growth_percentile,
            isolation_score, isolation_percentile,
            sole_source_concentration_score, sole_source_concentration_percentile,
            award_velocity_score, award_velocity_percentile,
            detector_details, generated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            uei, score_date, composite_score, composite_percentile_rank,
            benford_score, 1.0 if benford_score > 0 else 0.0,
            new_entity_score, 1.0 if new_entity_score > 0 else 0.0,
            mod_growth_score, 1.0 if mod_growth_score > 0 else 0.0,
            isolation_score, 1.0 if isolation_score > 0 else 0.0,
            sole_source_concentration_score, 1.0 if sole_source_concentration_score > 0 else 0.0,
            award_velocity_score, 1.0 if award_velocity_score > 0 else 0.0,
            detector_details, datetime.now(),
        ],
    )


def _seed_minimal(db_path: str, score_date: date, with_award: bool = True) -> None:
    """One entity with benford + new_entity firing, plus (optionally) one
    award row to provide recipient_name + awarding_agency for the brief
    input. Pass `with_award=False` on subsequent calls so the awards-table
    PK isn't violated when seeding multiple score_dates for the same UEI.
    """
    import json

    con = duckdb.connect(db_path)
    try:
        if with_award:
            _insert_award(
                con,
                award_id="A1",
                recipient_uei="ENT0001",
                recipient_name="Acme Defense Corp",
                awarding_agency="Department of Defense",
                naics_description="Engineering Services",
                total_obligation=5_000_000.0,
            )
        _insert_suspicion_score(
            con,
            uei="ENT0001",
            score_date=score_date,
            composite_score=0.85,
            composite_percentile_rank=0.99,
            benford_score=0.92,
            new_entity_score=0.80,
            detector_details=json.dumps({
                "benford": {"n_transactions": 142, "max_z": 4.3},
                "new_entity": {
                    "first_award_id": "A1",
                    "first_award_obligation": 5_000_000.0,
                    "competition_type": "NOT COMPETED",
                    "lifetime_awards": 1,
                },
            }),
        )
    finally:
        con.close()


# ── Input hash ───────────────────────────────────────────────────────────


def test_input_hash_deterministic():
    """The same BriefInput produces the same hash every time."""
    from briefs.generator import BriefInput, compute_input_hash

    bi = BriefInput(
        uei="ENT0001",
        entity_name="Acme",
        awarding_agency="DoD",
        primary_naics="Engineering Services",
        total_obligated_lifetime=5_000_000.0,
        award_count_lifetime=1,
        composite_score=0.85,
        composite_percentile_rank=0.99,
        detectors_fired=[
            {"name": "benford", "score": 0.92, "details": {"max_z": 4.3}},
            {"name": "new_entity", "score": 0.80, "details": {"lifetime_awards": 1}},
        ],
    )
    h1 = compute_input_hash(bi)
    h2 = compute_input_hash(bi)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_input_hash_changes_on_composite_score():
    from briefs.generator import BriefInput, compute_input_hash

    bi = BriefInput(
        uei="ENT0001", entity_name="Acme",
        awarding_agency="DoD", primary_naics="Eng",
        total_obligated_lifetime=5e6, award_count_lifetime=1,
        composite_score=0.85, composite_percentile_rank=0.99,
        detectors_fired=[{"name": "benford", "score": 0.92, "details": {}}],
    )
    other = replace(bi, composite_score=0.86)
    assert compute_input_hash(bi) != compute_input_hash(other)


def test_input_hash_changes_on_detector_score():
    from briefs.generator import BriefInput, compute_input_hash

    bi = BriefInput(
        uei="ENT0001", entity_name="Acme",
        awarding_agency="DoD", primary_naics="Eng",
        total_obligated_lifetime=5e6, award_count_lifetime=1,
        composite_score=0.85, composite_percentile_rank=0.99,
        detectors_fired=[{"name": "benford", "score": 0.92, "details": {}}],
    )
    other = replace(
        bi,
        detectors_fired=[{"name": "benford", "score": 0.93, "details": {}}],
    )
    assert compute_input_hash(bi) != compute_input_hash(other)


def test_input_hash_changes_on_prompt_version():
    from briefs.generator import BriefInput, compute_input_hash

    bi = BriefInput(
        uei="ENT0001", entity_name="Acme",
        awarding_agency="DoD", primary_naics="Eng",
        total_obligated_lifetime=5e6, award_count_lifetime=1,
        composite_score=0.85, composite_percentile_rank=0.99,
        detectors_fired=[{"name": "benford", "score": 0.92, "details": {}}],
    )
    assert compute_input_hash(bi, prompt_version="v1") != compute_input_hash(bi, prompt_version="v2")


def test_input_hash_stable_against_detector_ordering():
    """Detectors_fired in different order must produce the same hash so that
    upstream call sites don't have to sort before hashing."""
    from briefs.generator import BriefInput, compute_input_hash

    bi_a = BriefInput(
        uei="ENT0001", entity_name="Acme",
        awarding_agency="DoD", primary_naics="Eng",
        total_obligated_lifetime=5e6, award_count_lifetime=1,
        composite_score=0.85, composite_percentile_rank=0.99,
        detectors_fired=[
            {"name": "benford", "score": 0.92, "details": {"max_z": 4.3}},
            {"name": "new_entity", "score": 0.80, "details": {"lifetime_awards": 1}},
        ],
    )
    bi_b = replace(
        bi_a,
        detectors_fired=[
            {"name": "new_entity", "score": 0.80, "details": {"lifetime_awards": 1}},
            {"name": "benford", "score": 0.92, "details": {"max_z": 4.3}},
        ],
    )
    assert compute_input_hash(bi_a) == compute_input_hash(bi_b)


# ── Top-N selection ──────────────────────────────────────────────────────


def test_select_top_n_returns_highest_composite(tmp_path):
    from briefs.generator import select_top_n

    db_path = _fresh_db(tmp_path)
    score_date = date(2026, 5, 29)
    con = duckdb.connect(db_path)
    try:
        for i, score in enumerate([0.10, 0.50, 0.90, 0.30, 0.70]):
            _insert_award(
                con,
                award_id=f"A{i}",
                recipient_uei=f"ENT{i:04d}",
                recipient_name=f"Entity {i}",
            )
            _insert_suspicion_score(
                con,
                uei=f"ENT{i:04d}",
                score_date=score_date,
                composite_score=score,
                benford_score=score,
                detector_details='{"benford": {"max_z": 2.0}}',
            )
    finally:
        con.close()

    picks = select_top_n(db_path, score_date=score_date, top_n=3)
    ueis = [p.uei for p in picks]
    # 0.90, 0.70, 0.50 — by composite_score desc
    assert ueis == ["ENT0002", "ENT0004", "ENT0001"]


def test_select_top_n_enriches_with_awards_metadata(tmp_path):
    """recipient_name + awarding_agency + naics come from the awards table."""
    from briefs.generator import select_top_n

    db_path = _fresh_db(tmp_path)
    score_date = date(2026, 5, 29)
    _seed_minimal(db_path, score_date)

    picks = select_top_n(db_path, score_date=score_date, top_n=10)
    assert len(picks) == 1
    p = picks[0]
    assert p.uei == "ENT0001"
    assert p.entity_name == "Acme Defense Corp"
    assert p.awarding_agency == "Department of Defense"
    assert p.primary_naics == "Engineering Services"
    assert p.total_obligated_lifetime == pytest.approx(5_000_000.0)
    assert p.award_count_lifetime == 1


def test_select_top_n_extracts_detectors_fired(tmp_path):
    """detectors_fired contains only detectors with score > 0; raw score and
    details JSON survive."""
    from briefs.generator import select_top_n

    db_path = _fresh_db(tmp_path)
    score_date = date(2026, 5, 29)
    _seed_minimal(db_path, score_date)

    picks = select_top_n(db_path, score_date=score_date, top_n=10)
    fired = picks[0].detectors_fired
    fired_names = {d["name"] for d in fired}
    assert fired_names == {"benford", "new_entity"}
    by_name = {d["name"]: d for d in fired}
    assert by_name["benford"]["score"] == pytest.approx(0.92)
    assert by_name["new_entity"]["details"]["lifetime_awards"] == 1


# ── Prompt assembly ──────────────────────────────────────────────────────


def test_build_messages_uses_cached_system_prompt():
    """The system block must include cache_control so prompt caching activates."""
    from briefs.generator import BriefInput, build_messages

    bi = BriefInput(
        uei="ENT0001", entity_name="Acme",
        awarding_agency="DoD", primary_naics="Eng",
        total_obligated_lifetime=5e6, award_count_lifetime=1,
        composite_score=0.85, composite_percentile_rank=0.99,
        detectors_fired=[{"name": "benford", "score": 0.92, "details": {}}],
    )
    system, messages = build_messages(bi)
    # system is a list of blocks (per Anthropic SDK format), with cache_control
    assert isinstance(system, list)
    assert any(
        block.get("cache_control", {}).get("type") == "ephemeral"
        for block in system
    ), "system prompt must mark a block with cache_control=ephemeral"


def test_build_messages_includes_entity_data():
    """The user message must include the entity name, composite score, and
    each fired detector's score and details."""
    from briefs.generator import BriefInput, build_messages

    bi = BriefInput(
        uei="ENT0001", entity_name="Acme Defense Corp",
        awarding_agency="DoD", primary_naics="Eng",
        total_obligated_lifetime=5e6, award_count_lifetime=1,
        composite_score=0.85, composite_percentile_rank=0.99,
        detectors_fired=[
            {"name": "benford", "score": 0.92, "details": {"max_z": 4.3}},
        ],
    )
    _, messages = build_messages(bi)
    user_text = messages[0]["content"]
    assert "Acme Defense Corp" in user_text
    assert "benford" in user_text
    assert "4.3" in user_text


# ── Caching / forward-carry ──────────────────────────────────────────────


def test_generate_briefs_calls_api_first_run(tmp_path):
    """On first run with no prior briefs, the API is called for each top-N
    entity and a row is written."""
    from briefs.generator import generate_briefs

    db_path = _fresh_db(tmp_path)
    score_date = date(2026, 5, 29)
    _seed_minimal(db_path, score_date)

    fake = FakeAnthropic("First brief text.")
    n_calls = generate_briefs(
        db_path, score_date=score_date, top_n=10, client=fake, model="test-model"
    )
    assert n_calls == 1
    assert len(fake.calls) == 1

    con = duckdb.connect(db_path, read_only=True)
    try:
        row = con.execute(
            "SELECT brief_text, model FROM entity_briefs WHERE uei = ? AND score_date = ?",
            ["ENT0001", score_date],
        ).fetchone()
    finally:
        con.close()
    assert row is not None
    assert row[0] == "First brief text."
    assert row[1] == "test-model"


def test_generate_briefs_forward_carries_on_matching_hash(tmp_path):
    """If a prior day has a brief with the same input_hash, today's brief
    reuses the text and makes ZERO API calls."""
    from briefs.generator import generate_briefs

    db_path = _fresh_db(tmp_path)
    day1 = date(2026, 5, 28)
    day2 = date(2026, 5, 29)
    # Seed day 1 with the entity and generate.
    _seed_minimal(db_path, day1)
    fake_day1 = FakeAnthropic("Original brief from day 1.")
    generate_briefs(db_path, score_date=day1, top_n=10, client=fake_day1, model="test-model")

    # Day 2: same entity, same scores → same hash.
    _seed_minimal(db_path, day2, with_award=False)
    fake_day2 = FakeAnthropic("This should NOT be used.")
    n_calls = generate_briefs(
        db_path, score_date=day2, top_n=10, client=fake_day2, model="test-model"
    )
    assert n_calls == 0, "Forward-carry should skip the API call"
    assert len(fake_day2.calls) == 0

    con = duckdb.connect(db_path, read_only=True)
    try:
        row = con.execute(
            "SELECT brief_text FROM entity_briefs WHERE uei = ? AND score_date = ?",
            ["ENT0001", day2],
        ).fetchone()
    finally:
        con.close()
    assert row[0] == "Original brief from day 1."


def test_generate_briefs_regenerates_on_score_change(tmp_path):
    """If composite_score moves between days, input_hash changes and a fresh
    API call is made."""
    from briefs.generator import generate_briefs
    import json

    db_path = _fresh_db(tmp_path)
    day1 = date(2026, 5, 28)
    day2 = date(2026, 5, 29)
    _seed_minimal(db_path, day1)
    fake_day1 = FakeAnthropic("Day 1 brief.")
    generate_briefs(db_path, score_date=day1, top_n=10, client=fake_day1, model="test-model")

    # Day 2: same entity, but composite_score changed.
    con = duckdb.connect(db_path)
    try:
        _insert_award(con, award_id="A2", recipient_uei="ENT0001",
                      recipient_name="Acme Defense Corp")
        _insert_suspicion_score(
            con,
            uei="ENT0001",
            score_date=day2,
            composite_score=0.50,  # changed from 0.85
            composite_percentile_rank=0.80,
            benford_score=0.50,
            new_entity_score=0.80,
            detector_details=json.dumps({
                "benford": {"n_transactions": 142, "max_z": 4.3},
                "new_entity": {"lifetime_awards": 1},
            }),
        )
    finally:
        con.close()

    fake_day2 = FakeAnthropic("Day 2 brief.")
    n_calls = generate_briefs(
        db_path, score_date=day2, top_n=10, client=fake_day2, model="test-model"
    )
    assert n_calls == 1


def test_generate_briefs_idempotent_same_day(tmp_path):
    """Running twice on the same score_date doesn't duplicate rows and makes
    zero additional API calls (the second run finds today's own brief)."""
    from briefs.generator import generate_briefs

    db_path = _fresh_db(tmp_path)
    score_date = date(2026, 5, 29)
    _seed_minimal(db_path, score_date)
    fake = FakeAnthropic("Brief.")
    generate_briefs(db_path, score_date=score_date, top_n=10, client=fake, model="test-model")
    n_after_first = _count_briefs(db_path, score_date)

    fake2 = FakeAnthropic("Should not be used.")
    n_calls = generate_briefs(
        db_path, score_date=score_date, top_n=10, client=fake2, model="test-model"
    )
    assert n_calls == 0
    assert _count_briefs(db_path, score_date) == n_after_first


def test_generate_briefs_no_suspicion_scores_no_calls(tmp_path):
    """Empty suspicion_scores table → no API calls, no rows written."""
    from briefs.generator import generate_briefs

    db_path = _fresh_db(tmp_path)
    fake = FakeAnthropic("Unused.")
    n_calls = generate_briefs(
        db_path, score_date=date(2026, 5, 29), top_n=50, client=fake, model="test-model"
    )
    assert n_calls == 0
    assert _count_briefs(db_path, date(2026, 5, 29)) == 0


# ── Helpers ──────────────────────────────────────────────────────────────


def _count_briefs(db_path: str, score_date: date) -> int:
    con = duckdb.connect(db_path, read_only=True)
    try:
        return con.execute(
            "SELECT COUNT(*) FROM entity_briefs WHERE score_date = ?", [score_date]
        ).fetchone()[0]
    finally:
        con.close()


# ── SDK parameter compatibility (2026-08-26) ──────────────────────────────


def test_call_anthropic_sends_no_removed_sampling_params():
    """Sampling params are gone from the current Messages API.

    `temperature` (and `top_p`/`top_k`) were removed on current Claude
    models and dropped from the SDK's `Messages.create` signature in
    anthropic 1.0.0, so passing one is a TypeError, not a soft warning.
    That's what took the daily pipeline down on 2026-08-26: requirements
    .txt pinned nothing, a rebuild pulled 1.0.0, and every brief call
    raised `Messages.create() got an unexpected keyword argument
    'temperature'`.

    FakeAnthropic accepts **kwargs, so it can't reproduce that on its own
    -- assert on what we actually send instead.
    """
    from briefs.generator import BriefInput, call_anthropic

    bi = BriefInput(
        uei="ENT0001",
        entity_name="Acme",
        awarding_agency="DoD",
        primary_naics="Eng",
        total_obligated_lifetime=5e6,
        award_count_lifetime=1,
        composite_score=0.85,
        composite_percentile_rank=0.99,
        detectors_fired=[{"name": "benford", "score": 0.92, "details": {}}],
    )
    client = FakeAnthropic()
    call_anthropic(bi, client=client)

    assert len(client.calls) == 1
    sent = client.calls[0]
    for removed in ("temperature", "top_p", "top_k"):
        assert removed not in sent, (
            f"{removed} was removed from the Messages API -- sending it "
            f"raises TypeError on anthropic>=1.0.0"
        )
    # The params we do rely on must still be the documented ones.
    assert {"model", "max_tokens", "system", "messages"} <= set(sent)


class FlakyAnthropic(FakeAnthropic):
    """Raises on the first call, succeeds afterwards."""

    def __init__(self, response_text: str = "Recovered brief."):
        super().__init__(response_text)
        self.failures = 0

    def create(self, **kwargs):
        if not self.failures:
            self.failures += 1
            raise RuntimeError("Simulated Anthropic API error")
        return super().create(**kwargs)


def test_generate_briefs_survives_a_failed_api_call(tmp_path):
    """One bad Anthropic call must not abandon the whole run.

    call_anthropic was unguarded, so a single API error propagated out of
    generate_briefs and killed the orchestrator before it reached
    export.publish -- the site then kept whatever JSON was already on the
    volume. That is how 2026-08-31's blanked briefs survived two daily
    runs: each one died in step 2 and never republished. Skip the entity
    that failed (publish carries its prior brief forward) and keep going.
    """
    from briefs.generator import generate_briefs

    db_path = _fresh_db(tmp_path)
    score_date = date(2026, 9, 2)
    _seed_minimal(db_path, score_date)
    con = duckdb.connect(db_path)
    try:
        _insert_award(
            con, award_id="B1", recipient_uei="ENT0002",
            recipient_name="Beta Systems", awarding_agency="Department of Defense",
            naics_description="Engineering Services", total_obligation=2_000_000.0,
        )
        _insert_suspicion_score(
            con, uei="ENT0002", score_date=score_date,
            composite_score=0.70, composite_percentile_rank=0.90,
            benford_score=0.75,
            detector_details=json.dumps({"benford": {"n_transactions": 90, "max_z": 3.1}}),
        )
    finally:
        con.close()

    flaky = FlakyAnthropic()
    n_calls = generate_briefs(
        db_path, score_date=score_date, top_n=10, client=flaky, model="test-model"
    )

    # The run completed and the surviving entity got its brief.
    assert n_calls == 1, "the successful call should still be counted"
    con = duckdb.connect(db_path, read_only=True)
    try:
        written = con.execute(
            "SELECT count(*) FROM entity_briefs WHERE score_date = ?", [score_date]
        ).fetchone()[0]
    finally:
        con.close()
    assert written == 1, "the entity whose call succeeded must be persisted"
