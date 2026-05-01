"""USAspending.gov contract award puller.

Task 1.2: pull federal contract awards per agency × fiscal year (initial
seed) or per agency over the recent window (incremental), writing one
Parquet file per partition under `data/parquet/`.

Run:
    python -m ingestion.pull_awards               # full seed
    python -m ingestion.pull_awards --incremental # since last load
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import re
import shutil
import tempfile
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import polars as pl
from dotenv import load_dotenv
from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ingestion.load_db import effective_agency, load_config, resolve_db_path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Loaded once at import so SAM_API_KEY (used as USAspending's X-Api-Key
# via api.data.gov) is available regardless of which entrypoint imports
# this module — smoke scripts, orchestrators, or `python -m ingestion.…`.
load_dotenv()


# Fields requested from spending_by_award. The API requires this list and
# the keys here must match its documented response keys for contract
# award_type_codes A/B/C/D. extract_row() defensively falls back across
# variant names (e.g. "Award ID" vs "generated_internal_id").
USASPENDING_FIELDS = [
    "Award ID",
    "Recipient Name",
    "Recipient UEI",
    "Awarding Agency",
    "Awarding Sub Agency",
    "Award Type",
    "Contract Award Type",
    "Award Amount",
    "Total Obligations",
    "Description",
    "NAICS",
    "naics_description",
    "Start Date",
    "End Date",
    "Base and All Options Value",
    "Last Modified Date",
    "Mod Number",
    "Number of Offers Received",
    "Type of Contract Pricing",
    "Solicitation Procedures",
    "Extent Competed",
    "Action Date",
]

PAGE_LIMIT = 100
HTTP_TIMEOUT = 60.0

# USAspending is fronted by api.data.gov; an X-Api-Key (the same SAM key
# stored in .env) lifts the unauthenticated rate limit. Without the key,
# the WAF disconnects after ~300 cumulative POSTs to spending_by_award.
# With the key, we've sustained 150+ consecutive pages cleanly.
PAGE_SLEEP_SECONDS = 0.0

# Sleep between leaf chunks so the WAF doesn't see one continuous burst.
# 5s lets a hot connection's blacklist age out before we open the next.
INTER_CHUNK_SECONDS = 5

# Chunk-level recovery: when page-level retries can't get past a WAF
# disconnect, sleep this long before tearing down the client and
# starting the chunk over with a fresh TCP+TLS handshake. The WAF
# rate-limit window appears to be 60-90s; 90s gives us margin.
CHUNK_RECOVERY_SLEEP = 90
CHUNK_MAX_ATTEMPTS = 4

# Award Data Archive — USAspending publishes monthly Full+Delta ZIPs
# of every contracts/assistance partition at this public bucket. Used
# by `seed_strategy: "archive"`, which is the default for v1 because it
# bypasses the API's WAF + queue + psql_process backend entirely.
ARCHIVE_BASE_URL = "https://files.usaspending.gov/award_data_archive/"
ARCHIVE_LIST_TIMEOUT = 30.0
ARCHIVE_DOWNLOAD_TIMEOUT = 1800.0  # 30 min — DoD FY ZIP is ~1 GB

# Headers attached to every USAspending client. `_default_headers()`
# adds X-Api-Key from .env at runtime — USAspending is fronted by
# api.data.gov, so the SAM_API_KEY works here too and lifts the
# WAF rate limit that drops unauth clients after ~300 POSTs.
DEFAULT_HEADERS_BASE = {
    "User-Agent": "anomaly-radar/0.1 (+https://github.com/kamronarabi/fed-anomaly-tracker)",
}


def _default_headers() -> dict:
    """Return the headers to attach to every USAspending HTTP client.

    Reads SAM_API_KEY from the environment lazily so .env values picked
    up via `load_dotenv()` are honored. Falls back to the base headers
    (no key) if the env var is unset, which is fine for tests but will
    blow up the WAF rate limit during a real seed.
    """
    headers = dict(DEFAULT_HEADERS_BASE)
    api_key = os.environ.get("SAM_API_KEY")
    if api_key:
        headers["X-Api-Key"] = api_key
    return headers

# USAspending caps `spending_by_award` at page * limit <= 10,000 per query.
# When a window exceeds that, we bisect the date range and recurse.
API_RECORD_CAP = 10_000
MAX_SPLIT_DEPTH = 12  # 2^12 = 4096 leaves; sufficient even for DoD

# Bulk download (used by seed_strategy="bulk") submits a job, polls until
# the server has assembled the CSV, then we fetch the resulting ZIP.
BULK_POLL_SECONDS = 15
BULK_TIMEOUT_SECONDS = 60 * 60  # 1 hour
BULK_DOWNLOAD_TIMEOUT = 600.0   # large ZIP downloads can take minutes

# Map our awards-table fields to the candidate column names we look for in
# bulk-download CSVs (snake_case, sometimes varying across endpoint
# versions). First match wins.
BULK_COLUMN_MAP: dict[str, list[str]] = {
    "award_id": ["contract_award_unique_key", "award_id_piid", "generated_internal_id"],
    "parent_award_id": ["parent_award_id_piid", "parent_award_unique_key"],
    "recipient_name": ["recipient_name"],
    "recipient_uei": ["recipient_uei"],
    "awarding_agency": ["awarding_agency_name"],
    "awarding_sub_agency": ["awarding_sub_agency_name"],
    "award_type": ["award_type", "type_of_contract_pricing"],
    "award_description": [
        "prime_award_base_transaction_description",
        "transaction_description",
        "award_description",
    ],
    "naics_code": ["naics_code"],
    "naics_description": ["naics_description"],
    "total_obligation": [
        "total_dollars_obligated",
        "current_total_value_of_award",
    ],
    "base_and_all_options_value": [
        "base_and_all_options_value",
        "potential_total_value_of_award",
    ],
    "period_of_performance_start": ["period_of_performance_start_date"],
    "period_of_performance_end": [
        "period_of_performance_current_end_date",
        "period_of_performance_potential_end_date",
    ],
    "award_date": ["action_date", "last_modified_date"],
    "competition_type": ["extent_competed"],
    "number_of_offers": ["number_of_offers_received"],
    "modification_number": ["modification_number"],
}


def fiscal_year_window(fy: int) -> tuple[date, date]:
    """Federal fiscal year FY{N} runs Oct 1 of N-1 through Sep 30 of N."""
    return date(fy - 1, 10, 1), date(fy, 9, 30)


def _parse_date(value) -> date | None:
    if value in (None, "", "null"):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return datetime.fromisoformat(str(value).split("T")[0]).date()
    except (ValueError, TypeError):
        return None


def _parse_float(value) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _parse_int(value) -> int | None:
    if value in (None, "", "null"):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _parse_naics(value) -> tuple[str | None, str | None]:
    """USAspending returns NAICS as either a bare code string or
    {'code': '...', 'description': '...'}. Normalize to (code, description)."""
    if value in (None, "", "null"):
        return None, None
    if isinstance(value, dict):
        return value.get("code"), value.get("description")
    return str(value), None


def extract_row(record: dict, agency_name: str, pulled_at: datetime) -> dict:
    """Map a single USAspending result row into the awards schema."""
    award_id = (
        record.get("generated_internal_id")
        or record.get("Award ID")
        or record.get("award_id")
    )
    naics_code, naics_desc_from_obj = _parse_naics(record.get("NAICS"))
    return {
        "award_id": str(award_id) if award_id is not None else None,
        "parent_award_id": (
            record.get("parent_award_id")
            or record.get("Parent Award ID")
            or record.get("Parent IDV Agency Name")
        ),
        "recipient_name": record.get("Recipient Name"),
        "recipient_uei": record.get("Recipient UEI") or record.get("recipient_uei"),
        "awarding_agency": record.get("Awarding Agency") or agency_name,
        "awarding_sub_agency": record.get("Awarding Sub Agency"),
        "award_type": record.get("Contract Award Type") or record.get("Award Type"),
        "award_description": record.get("Description"),
        "naics_code": naics_code,
        "naics_description": (
            record.get("naics_description")
            or record.get("NAICS Description")
            or naics_desc_from_obj
        ),
        "total_obligation": _parse_float(
            record.get("Total Obligations") or record.get("Award Amount")
        ),
        "base_and_all_options_value": _parse_float(
            record.get("Base and All Options Value")
        ),
        "period_of_performance_start": _parse_date(record.get("Start Date")),
        "period_of_performance_end": _parse_date(record.get("End Date")),
        "award_date": _parse_date(
            record.get("Action Date") or record.get("Base Obligation Date")
        ),
        "competition_type": record.get("Extent Competed"),
        "number_of_offers": _parse_int(record.get("Number of Offers Received")),
        "modification_number": (
            record.get("Mod Number") or record.get("modification_number")
        ),
        "pulled_at": pulled_at,
    }


def _empty_awards_schema() -> dict:
    """Polars schema for empty-page Parquet writes — matches the awards table."""
    return {
        "award_id": pl.Utf8,
        "parent_award_id": pl.Utf8,
        "recipient_name": pl.Utf8,
        "recipient_uei": pl.Utf8,
        "awarding_agency": pl.Utf8,
        "awarding_sub_agency": pl.Utf8,
        "award_type": pl.Utf8,
        "award_description": pl.Utf8,
        "naics_code": pl.Utf8,
        "naics_description": pl.Utf8,
        "total_obligation": pl.Float64,
        "base_and_all_options_value": pl.Float64,
        "period_of_performance_start": pl.Date,
        "period_of_performance_end": pl.Date,
        "award_date": pl.Date,
        "competition_type": pl.Utf8,
        "number_of_offers": pl.Int64,
        "modification_number": pl.Utf8,
        "pulled_at": pl.Datetime,
    }


def _warmup(client: httpx.Client, base: str) -> None:
    """Fire a tiny `spending_by_award` POST so the WAF establishes a session
    on this client before we send the real (heavier) request bodies.

    Sync `httpx.Client` is the deliberate choice here: empirically, the
    async client gets disconnected by USAspending's WAF on the first
    heavy POST while the sync client goes through cleanly. We still keep
    a brief warmup loop because even the sync client occasionally drops
    on first contact.
    """
    url = f"{base.rstrip('/')}/search/spending_by_award/"
    payload = {
        "filters": {
            "time_period": [{"start_date": "2024-01-01", "end_date": "2024-01-31"}],
            "award_type_codes": ["A", "B", "C", "D"],
        },
        "fields": ["Award ID", "Recipient Name", "Award Amount"],
        "limit": 1,
        "page": 1,
    }
    for i in range(6):
        try:
            client.post(url, json=payload, timeout=20)
            logger.info("WAF warmup ok after %d attempt(s)", i + 1)
            return
        except (httpx.HTTPError, httpx.TimeoutException):
            time.sleep(2 + i)
    logger.warning("WAF warmup failed after 6 attempts — proceeding anyway")


def _post_page(client: httpx.Client, url: str, payload: dict) -> dict:
    """POST one page with tenacity retries (6 attempts, exponential backoff).

    Sync — runs in a worker thread when called from async code via
    `asyncio.to_thread`. We use sync httpx because USAspending's WAF
    handles `httpx.Client` reliably and `httpx.AsyncClient` flakily;
    that asymmetry was the root cause of the seed-blocking flakes.
    """
    for attempt in Retrying(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    ):
        with attempt:
            resp = client.post(url, json=payload, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
    return {}


def _build_search_body(agency: dict, start: date, end: date) -> dict:
    """Build a `spending_by_award` request body for one window.

    `agency` is expected to be the resolved dict from `effective_agency()`,
    so `award_types` and `award_amount_min` are always populated.
    """
    body = {
        "filters": {
            "agencies": [
                {"type": "awarding", "tier": "toptier", "name": agency["name"]}
            ],
            "time_period": [
                {"start_date": start.isoformat(), "end_date": end.isoformat()}
            ],
            "award_type_codes": agency["award_types"],
        },
        "fields": USASPENDING_FIELDS,
        "limit": PAGE_LIMIT,
        "sort": "Award Amount",
        "order": "desc",
    }
    if agency.get("award_amount_min"):
        body["filters"]["award_amounts"] = [
            {"lower_bound": agency["award_amount_min"]}
        ]
    return body


def _pull_chunk(
    agency: dict,
    start: date,
    end: date,
    config: dict,
    pulled_at: datetime,
    label: str,
) -> tuple[list[dict], int, bool]:
    """Pull every page for one (agency, window). Returns (rows, page_count, capped).

    Each chunk opens its OWN `httpx.Client` and runs a fresh WAF warmup,
    rather than sharing a long-lived client across the whole window. The
    USAspending WAF eventually poisons a long-lived connection after
    sustained use (observed: ~200 successful pages then a hard
    RemoteProtocolError that retries don't clear). Per-chunk clients
    cap that blast radius — a poisoned connection only loses one chunk,
    and the next chunk starts with a fresh TCP+TLS handshake.

    The ~1s warmup-per-chunk overhead is negligible against ~50-100s of
    actual page-pulling per chunk.
    """
    base = config["usaspending_api_base"].rstrip("/")
    url = f"{base}/search/spending_by_award/"
    body = _build_search_body(agency, start, end)

    last_error: Exception | None = None
    for outer in range(CHUNK_MAX_ATTEMPTS):
        try:
            rows, page, capped = _pull_chunk_once(
                agency, url, body, pulled_at, label, start, end,
                config["usaspending_api_base"],
            )
            return rows, page, capped
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            last_error = e
            if outer + 1 >= CHUNK_MAX_ATTEMPTS:
                break
            logger.warning(
                "%s [%s] chunk failed on attempt %d/%d (%s); sleeping %ds "
                "before retrying with a fresh connection",
                agency["short"], label, outer + 1, CHUNK_MAX_ATTEMPTS,
                type(e).__name__, CHUNK_RECOVERY_SLEEP,
            )
            time.sleep(CHUNK_RECOVERY_SLEEP)
    raise RuntimeError(
        f"chunk {agency['short']} [{label}] failed after "
        f"{CHUNK_MAX_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def _pull_chunk_once(
    agency: dict,
    url: str,
    body: dict,
    pulled_at: datetime,
    label: str,
    start: date,
    end: date,
    base_url_for_warmup: str,
) -> tuple[list[dict], int, bool]:
    """One attempt at pulling all pages for a chunk with a fresh client."""
    rows: list[dict] = []
    page = 1
    with httpx.Client(headers=_default_headers()) as client:
        _warmup(client, base_url_for_warmup)
        while True:
            data = _post_page(client, url, {**body, "page": page})
            results = data.get("results") or []
            for record in results:
                rows.append(extract_row(record, agency["name"], pulled_at))

            has_next = bool((data.get("page_metadata") or {}).get("hasNext", False))
            if not results or not has_next:
                break
            if len(rows) >= API_RECORD_CAP:
                # Hit USAspending's hard cap; caller must subdivide the window.
                break
            page += 1
            if PAGE_SLEEP_SECONDS:
                time.sleep(PAGE_SLEEP_SECONDS)

    capped = len(rows) >= API_RECORD_CAP
    logger.info(
        "%s [%s] %s→%s: %d rows in %d pages%s",
        agency["short"], label, start, end, len(rows), page,
        " (CAPPED, will split)" if capped else "",
    )
    return rows, page, capped


def _midpoint(start: date, end: date) -> date | None:
    """Return the date halfway through [start, end]. None if window is one day."""
    span = (end - start).days
    if span <= 0:
        return None
    return start + timedelta(days=span // 2)


def _write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = _empty_awards_schema()
    if rows:
        pl.DataFrame(rows, schema=schema).write_parquet(path)
    else:
        pl.DataFrame(schema=schema).write_parquet(path)


def pull_window(
    agency: dict,
    start: date,
    end: date,
    label: str,
    config: dict,
    out_dir: Path,
) -> list[Path]:
    """Pull awards over [start, end], adaptively bisecting on the API's 10K cap.

    Sync function — uses `httpx.Client` because USAspending's WAF is
    consistent with the sync client and flaky with the async one. Async
    callers should wrap this with `asyncio.to_thread(pull_window, ...)`.

    Each leaf chunk writes its own Parquet partition file; one window may
    produce many files when the cap forces subdivision. Returns the list of
    written Parquet paths. The DB loader (Task 1.4) globs and deduplicates.
    """
    pulled_at = datetime.now(timezone.utc).replace(tzinfo=None)
    t0 = time.monotonic()
    written: list[Path] = []
    total_rows = 0

    logger.info("Pulling %s [%s]: %s → %s", agency["short"], label, start, end)

    # Iterative bisection via a stack of (start, end, label, depth).
    stack: list[tuple[date, date, str, int]] = [(start, end, label, 0)]

    while stack:
        s, e, lbl, depth = stack.pop()
        rows, _pages, capped = _pull_chunk(
            agency, s, e, config, pulled_at, lbl
        )

        if capped and depth < MAX_SPLIT_DEPTH:
            mid = _midpoint(s, e)
            if mid is not None:
                # Discard partial rows; the two sub-windows will refetch
                # them in full. Push right child first so left runs next.
                stack.append((mid + timedelta(days=1), e, lbl + "b", depth + 1))
                stack.append((s, mid, lbl + "a", depth + 1))
                continue
            logger.warning(
                "%s [%s] %s: 1-day window still capped — keeping partial 10K rows",
                agency["short"], lbl, s,
            )
        elif capped:
            logger.warning(
                "%s [%s] hit MAX_SPLIT_DEPTH; keeping partial 10K rows for %s→%s",
                agency["short"], lbl, s, e,
            )

        out_path = out_dir / f"awards_{agency['short']}_{lbl}.parquet"
        _write_parquet(rows, out_path)
        written.append(out_path)
        total_rows += len(rows)

        # Brief pause between leaf chunks so the WAF doesn't see one
        # continuous burst across all bisected windows.
        if stack and INTER_CHUNK_SECONDS:
            time.sleep(INTER_CHUNK_SECONDS)

    elapsed = time.monotonic() - t0
    logger.info(
        "DONE %s [%s]: %d rows across %d partition(s), %.1fs",
        agency["short"], label, total_rows, len(written), elapsed,
    )
    return written


def _build_bulk_body(agency: dict, start: date, end: date) -> dict:
    """Build a `download/awards/` request body for one (agency, window).

    Includes `sub_award_types: []` explicitly. Without that, USAspending's
    bulk endpoint sweeps in subawards alongside primes — observed empirically
    as a ~10× row inflation that crashed their backend `psql_process` on a
    764K-row HHS week. Setting the list to empty means "prime contracts only".
    """
    body = {
        "filters": {
            "prime_award_types": agency["award_types"],
            "sub_award_types": [],
            "agencies": [
                {"type": "awarding", "tier": "toptier", "name": agency["name"]}
            ],
            "date_type": "action_date",
            "date_range": {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            },
        },
        "columns": [],
        "file_format": "csv",
    }
    if agency.get("award_amount_min"):
        body["filters"]["award_amounts"] = [
            {"lower_bound": agency["award_amount_min"]}
        ]
    return body


def _map_bulk_to_schema(
    df_raw: pl.DataFrame | pl.LazyFrame,
    agency_name: str,
    pulled_at: datetime,
) -> pl.DataFrame | pl.LazyFrame:
    """Map a bulk-download CSV frame to our awards schema.

    Accepts either a `pl.DataFrame` (eager — bulk path, tests) or
    `pl.LazyFrame` (streaming — archive path for DoD's 1 GB ZIPs which
    won't fit in 8 GB RAM eagerly). Returns the same kind it was given,
    so existing callers keep their current shape.
    """
    return_lazy = isinstance(df_raw, pl.LazyFrame)
    # `LazyFrame.columns` resolves the schema with a PerformanceWarning
    # on each call; use the documented helper instead. Both branches
    # produce the same set so downstream code is unchanged.
    if return_lazy:
        available = set(df_raw.collect_schema().names())
    else:
        available = set(df_raw.columns)
    target_schema = _empty_awards_schema()

    selects: list[pl.Expr] = []
    for target_col, dtype in target_schema.items():
        if target_col == "pulled_at":
            continue
        candidates = BULK_COLUMN_MAP.get(target_col, [])
        src = next((c for c in candidates if c in available), None)
        if src is None:
            selects.append(pl.lit(None).cast(dtype).alias(target_col))
        else:
            selects.append(pl.col(src).cast(dtype, strict=False).alias(target_col))

    mapped = df_raw.select(selects)
    mapped = mapped.with_columns(
        pl.col("awarding_agency").fill_null(agency_name),
        pl.col("award_id").cast(pl.Utf8),
    )
    mapped = mapped.with_columns(
        pl.lit(pulled_at).cast(pl.Datetime).alias("pulled_at")
    )
    # award_id is the awards-table PK and must be NOT NULL. Bulk CSVs
    # occasionally have rows with no contract_award_unique_key (malformed
    # source records); drop them rather than letting the DB load fail.
    if return_lazy:
        # Skip the dropped-count log on the lazy path — computing it
        # would require materializing the frame, defeating streaming.
        mapped = mapped.filter(pl.col("award_id").is_not_null())
    else:
        before = mapped.height
        mapped = mapped.filter(pl.col("award_id").is_not_null())
        dropped = before - mapped.height
        if dropped:
            logger.warning(
                "dropped %d row(s) with null award_id (out of %d)", dropped, before
            )
    return mapped.select(list(target_schema.keys()))


# ── Award Data Archive (seed_strategy: "archive") ─────────────────────────


def _select_prime_award_csvs(zip_names: list[str]) -> list[str]:
    """Return only the CSV members of a Contracts ZIP that hold prime award
    rows. Used by both the bulk download path and the archive path —
    USAspending bundles a `Contracts_Subawards_*.csv` (different schema)
    in the same ZIP, and we want only `*PrimeAwardSummaries*.csv`.
    """
    all_csvs = [n for n in zip_names if n.lower().endswith(".csv")]
    primary = [n for n in all_csvs if "primeaward" in n.lower()]
    if primary:
        return primary
    # Fallback: older payloads may use plain `Contracts_*.csv` without
    # the PrimeAwardSummaries suffix. Still drop subawards explicitly.
    return [
        n for n in all_csvs
        if "contract" in n.lower() and "subaward" not in n.lower()
    ]


def _archive_url(cgac: str, fy: int, archive_date: str) -> str:
    """Build the URL for one (FY, agency) archive ZIP.

    `cgac` is zero-padded to 3 digits ("075" not "75"); `archive_date` is
    a YYYYMMDD string matching one of the published snapshot suffixes.
    """
    cgac_str = str(cgac).zfill(3)
    return (
        f"{ARCHIVE_BASE_URL}"
        f"FY{int(fy)}_{cgac_str}_Contracts_Full_{archive_date}.zip"
    )


# Pulled from a Key like "FY2024_075_Contracts_Full_20260406.zip".
# Not anchored to end of string — these keys appear inside an XML
# document where `<` follows the `.zip`.
_ARCHIVE_DATE_RE = re.compile(r"_Contracts_Full_(\d{8})\.zip")


def _parse_archive_listing(xml_text: str) -> str | None:
    """Find the most recent YYYYMMDD suffix among `*Contracts_Full_*.zip`
    keys in an S3 ListBucketResult XML body. Returns None if no match.

    Pure-functional and HTTP-free so the bucket-listing path is testable
    against a fixture without hitting S3.
    """
    dates = _ARCHIVE_DATE_RE.findall(xml_text)
    if not dates:
        return None
    return max(dates)  # YYYYMMDD strings sort lexicographically


_archive_date_cache: str | None = None


def _discover_archive_date(client: httpx.Client) -> str:
    """Return the most recent archive snapshot date (YYYYMMDD string).

    Performs one GET on the bucket index (~10 KB XML response, 1000-key
    page is enough — `Full` files are alphabetically near the top) and
    caches the result for the rest of the process. Subsequent calls
    across the 4 agency-FY pulls share one round trip.
    """
    global _archive_date_cache
    if _archive_date_cache:
        return _archive_date_cache
    resp = client.get(ARCHIVE_BASE_URL, timeout=ARCHIVE_LIST_TIMEOUT)
    resp.raise_for_status()
    # The index is paginated; one page (1000 keys) easily covers all the
    # `*_Contracts_Full_*.zip` files since they share one snapshot date.
    found = _parse_archive_listing(resp.text)
    if not found:
        # Fall through to a follow-up page if the first didn't include
        # any Full files (extremely unlikely given alphabetical ordering).
        m = re.search(r"<NextMarker>([^<]+)</NextMarker>", resp.text)
        if m:
            resp2 = client.get(
                ARCHIVE_BASE_URL,
                params={"marker": m.group(1)},
                timeout=ARCHIVE_LIST_TIMEOUT,
            )
            resp2.raise_for_status()
            found = _parse_archive_listing(resp2.text)
    if not found:
        raise RuntimeError(
            "could not find any *_Contracts_Full_*.zip in archive listing"
        )
    _archive_date_cache = found
    return found


# USAspending stores award_type in the archive CSV as a human-readable
# description, not the single-letter API code. Map our config's codes
# to the description strings the column actually contains. Strings are
# matched case-insensitively at filter time.
AWARD_TYPE_CODE_TO_DESC = {
    "A": "BPA CALL",
    "B": "PURCHASE ORDER",
    "C": "DELIVERY ORDER",
    "D": "DEFINITIVE CONTRACT",
}

# The "all four codes" set means no narrowing — treat it as no filter.
# This is what `effective_agency()` resolves to when a config entry
# doesn't override award_types.
_ALL_CONTRACT_TYPES = {"A", "B", "C", "D"}


def _apply_archive_filters(
    df: pl.LazyFrame, agency: dict
) -> pl.LazyFrame:
    """Apply per-agency post-download filters to a mapped LazyFrame.

    The archive ZIPs are unfiltered by award_type or amount, and the CSV
    `award_type` column holds descriptions like "DEFINITIVE CONTRACT"
    rather than the API's letter codes — we translate codes → descriptions
    before filtering. When the agency's `award_types` is the full A/B/C/D
    default (i.e. no override), the filter is skipped entirely.
    """
    types = agency.get("award_types") or []
    amount_min = int(agency.get("award_amount_min") or 0)

    if types and set(types) != _ALL_CONTRACT_TYPES:
        descs = [
            AWARD_TYPE_CODE_TO_DESC[c]
            for c in types
            if c in AWARD_TYPE_CODE_TO_DESC
        ]
        if descs:
            df = df.filter(
                pl.col("award_type").str.to_uppercase().is_in(descs)
            )
    if amount_min > 0:
        df = df.filter(
            pl.col("total_obligation").fill_null(0) >= amount_min
        )
    return df


ARCHIVE_DOWNLOAD_RETRIES = 20
# Backoff per attempt; values past the end of the tuple reuse the last
# entry. With 20 attempts and a 120s ceiling, worst-case cumulative
# backoff is ~35 min — acceptable for a monthly seed.
ARCHIVE_DOWNLOAD_RESUME_BACKOFF = (5, 15, 30, 60, 120)


def _resume_download(
    client: httpx.Client,
    url: str,
    dest: Path,
    log_label: str,
    t0: float,
) -> int:
    """Stream `url` into `dest`, resuming from where we left off via HTTP
    Range requests if the connection drops mid-transfer.

    files.usaspending.gov sets `Accept-Ranges: bytes` and serves 1 GB+ ZIPs;
    in practice connections sometimes drop part-way through and a fresh
    GET would re-download everything. Resume keeps the previously-received
    bytes on disk and asks for `bytes={offset}-` on the retry.

    Returns the total bytes written. Raises if every retry exhausted.
    """
    expected_size: int | None = None
    last_error: Exception | None = None
    for attempt in range(ARCHIVE_DOWNLOAD_RETRIES):
        already = dest.stat().st_size if dest.exists() else 0
        if expected_size is not None and already >= expected_size:
            return already

        headers = {}
        mode = "wb"
        if already > 0:
            headers["Range"] = f"bytes={already}-"
            mode = "ab"
            logger.info(
                "ARCHIVE %s: resuming download at %.1f MB (attempt %d/%d)",
                log_label, already / (1 << 20),
                attempt + 1, ARCHIVE_DOWNLOAD_RETRIES,
            )

        try:
            with client.stream(
                "GET", url,
                headers=headers,
                timeout=ARCHIVE_DOWNLOAD_TIMEOUT,
                follow_redirects=True,
            ) as r:
                r.raise_for_status()
                # Capture expected size on first response (or via
                # Content-Range if we sent a Range request).
                if expected_size is None:
                    if "content-range" in r.headers:
                        # Format: "bytes <start>-<end>/<total>"
                        try:
                            expected_size = int(
                                r.headers["content-range"].split("/")[-1]
                            )
                        except (ValueError, IndexError):
                            pass
                    elif "content-length" in r.headers and already == 0:
                        try:
                            expected_size = int(r.headers["content-length"])
                        except ValueError:
                            pass

                with open(dest, mode) as out:
                    for chunk in r.iter_bytes(chunk_size=1 << 20):
                        out.write(chunk)
            final_size = dest.stat().st_size
            logger.info(
                "ARCHIVE %s: downloaded %.1f MB in %.1fs (attempt %d)",
                log_label, final_size / (1 << 20),
                time.monotonic() - t0, attempt + 1,
            )
            return final_size
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            last_error = e
            wait = ARCHIVE_DOWNLOAD_RESUME_BACKOFF[
                min(attempt, len(ARCHIVE_DOWNLOAD_RESUME_BACKOFF) - 1)
            ]
            already_after = dest.stat().st_size if dest.exists() else 0
            logger.warning(
                "ARCHIVE %s: download error after %.1f MB on attempt %d (%s); "
                "sleeping %ds before resume",
                log_label, already_after / (1 << 20),
                attempt + 1, type(e).__name__, wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"archive download {url} failed after "
        f"{ARCHIVE_DOWNLOAD_RETRIES} resume attempts: {last_error}"
    )


def pull_awards_archive(
    agency: dict,
    fy: int,
    config: dict,
    out_dir: Path,
    *,
    archive_date: str | None = None,
) -> list[Path]:
    """Seed one (agency, fiscal year) by downloading the matching ZIP from
    USAspending's public Award Data Archive at files.usaspending.gov.

    Sync function — caller wraps with `asyncio.to_thread` if running in
    an async orchestrator. Returns the list with the single Parquet path
    written, matching the shape of `pull_window` / `pull_awards_bulk`.

    Architecture rationale: the API's `download/awards` endpoint is
    queue-backed and unreliable (psql_process crashes, multi-hour stalls);
    the archive is a static S3 GET that completes in seconds-to-minutes
    even for the 1 GB DoD partitions. See the plan doc for the full
    justification.
    """
    pulled_at = datetime.now(timezone.utc).replace(tzinfo=None)
    t0 = time.monotonic()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"awards_{agency['short']}_{fy}_archive.parquet"

    with httpx.Client(headers=_default_headers()) as client:
        if archive_date is None:
            archive_date = _discover_archive_date(client)
        url = _archive_url(agency["code"], fy, archive_date)
        log_label = f"{agency['short']} FY{fy}"
        logger.info("ARCHIVE %s: GET %s", log_label, url)

        # Stream the ZIP to disk so memory stays bounded for DoD's ~1 GB
        # downloads. The download itself is wrapped in a resume loop:
        # files.usaspending.gov sometimes drops mid-stream (observed at
        # 217 MB of 1.1 GB) and supports `Accept-Ranges: bytes`, so we
        # retry from where we left off rather than starting over.
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f"archive_{agency['short']}_{fy}_", suffix=".zip"
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        try:
            _resume_download(client, url, tmp_path, log_label, t0)
            rows_total = _archive_zip_to_parquet(
                tmp_path, agency, pulled_at, out_path, log_label,
            )
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

    elapsed = time.monotonic() - t0
    logger.info(
        "DONE ARCHIVE %s: %d rows written → %s (%.1fs total)",
        log_label, rows_total, out_path, elapsed,
    )
    return [out_path]


def _archive_zip_to_parquet(
    zip_path: Path,
    agency: dict,
    pulled_at: datetime,
    out_path: Path,
    log_label: str,
) -> int:
    """Open the downloaded archive ZIP, stream every prime-contracts CSV
    through polars's lazy engine, apply schema mapping + per-agency
    filters, and `sink_parquet` to disk.

    Memory matters here: DoD's compressed 1 GB ZIP unpacks to ~5-10 GB
    CSV (504 columns) per fiscal year. Eager `pl.read_csv` would OOM on
    a 8 GB Mac, so we extract each CSV member to a temp file and use
    `pl.scan_csv` + projection (via `_map_bulk_to_schema` on a LazyFrame)
    + `sink_parquet`, which streams chunks rather than materializing the
    full DataFrame.
    """
    rows_total = 0
    with zipfile.ZipFile(zip_path) as zf:
        contract_csvs = _select_prime_award_csvs(zf.namelist())
        if not contract_csvs:
            logger.warning(
                "ARCHIVE %s: ZIP contained no prime-contract CSV", log_label
            )
            pl.DataFrame(schema=_empty_awards_schema()).write_parquet(out_path)
            return 0

        # Extract each CSV to disk so polars can scan it lazily.
        # tempfile dir is auto-cleaned at the end of the with-block.
        with tempfile.TemporaryDirectory(
            prefix=f"archive_{agency['short']}_csvs_"
        ) as csv_dir:
            csv_dir_path = Path(csv_dir)
            extracted: list[Path] = []
            for csv_name in contract_csvs:
                target = csv_dir_path / Path(csv_name).name
                with zf.open(csv_name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1 << 20)  # 1 MB chunks
                if target.stat().st_size == 0:
                    target.unlink()
                    continue
                extracted.append(target)
                logger.info(
                    "ARCHIVE %s: extracted %s (%.1f MB on disk)",
                    log_label, csv_name, target.stat().st_size / (1 << 20),
                )

            if not extracted:
                pl.DataFrame(schema=_empty_awards_schema()).write_parquet(out_path)
                return 0

            lazy_frames: list[pl.LazyFrame] = []
            for csv_path in extracted:
                lf = pl.scan_csv(
                    csv_path,
                    infer_schema_length=10_000,
                    ignore_errors=True,
                    try_parse_dates=True,
                )
                mapped = _map_bulk_to_schema(lf, agency["name"], pulled_at)
                # Filter post-mapping so we use our schema column names.
                filtered = _apply_archive_filters(mapped, agency)
                lazy_frames.append(filtered)

            combined = (
                lazy_frames[0]
                if len(lazy_frames) == 1
                else pl.concat(lazy_frames, how="vertical_relaxed")
            )
            # sink_parquet streams the result to disk in batches, keeping
            # peak memory bounded regardless of source CSV size.
            combined.sink_parquet(out_path)

    # The row count is recorded by re-scanning the just-written Parquet
    # (cheap — only the metadata/footer is read, not the row data).
    rows_total = pl.scan_parquet(out_path).select(pl.len()).collect().item()
    logger.info(
        "ARCHIVE %s: streamed %d rows after filters → %s",
        log_label, rows_total, out_path,
    )
    return rows_total


async def _retrying_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    json=None,
    params=None,
    timeout: float = HTTP_TIMEOUT,
    follow_redirects: bool = False,
) -> httpx.Response:
    """Issue one HTTP request with the same retry policy as `_post_page`.

    USAspending's WAF disconnects without response on a non-trivial
    fraction of requests — both POST and GET, and across both
    `spending_by_award/` and `download/*` endpoints. Wrap every USAspending
    call in this helper so a single dropped connection doesn't blow up an
    in-flight bulk job.
    """
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    ):
        with attempt:
            resp = await client.request(
                method,
                url,
                json=json,
                params=params,
                timeout=timeout,
                follow_redirects=follow_redirects,
            )
            resp.raise_for_status()
            return resp
    raise RuntimeError("unreachable: tenacity should have raised or returned")


async def _bulk_submit_and_wait(
    client: httpx.AsyncClient,
    base: str,
    body: dict,
    label_for_log: str,
) -> tuple[bytes, str]:
    """Submit a bulk download job, poll until ready, return (zip_bytes, file_name)."""
    submit = await _retrying_request(
        client, "POST", f"{base}/download/awards/", json=body
    )
    job = submit.json()
    file_name = job.get("file_name") or job.get("file") or ""
    file_url = job.get("file_url") or job.get("url")
    if not file_name or not file_url:
        raise RuntimeError(
            f"bulk download submit returned unexpected payload: {job!r}"
        )

    logger.info("BULK [%s] submitted: %s", label_for_log, file_name)

    deadline = time.monotonic() + BULK_TIMEOUT_SECONDS
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"bulk download {file_name} did not finish within "
                f"{BULK_TIMEOUT_SECONDS}s"
            )
        await asyncio.sleep(BULK_POLL_SECONDS)
        s = await _retrying_request(
            client,
            "GET",
            f"{base}/download/status",
            params={"file_name": file_name},
        )
        status = s.json()
        phase = status.get("status")
        if phase in ("finished", "ready"):
            break
        if phase == "failed":
            raise RuntimeError(f"bulk download failed: {status}")
        logger.info(
            "BULK [%s] %s rows=%s",
            label_for_log, phase, status.get("total_rows"),
        )

    download = await _retrying_request(
        client,
        "GET",
        file_url,
        follow_redirects=True,
        timeout=BULK_DOWNLOAD_TIMEOUT,
    )
    return download.content, file_name


async def pull_awards_bulk(
    agency: dict,
    start: date,
    end: date,
    label: str,
    config: dict,
    out_dir: Path,
) -> list[Path]:
    """Use USAspending's bulk download endpoint for the seed of large agencies.

    Submits one job for the full window, polls until the server has built
    the CSV, downloads the ZIP, maps each Contracts CSV inside it to our
    schema, and writes one combined Parquet file. Returns a list with the
    single output path so callers can `.extend()` like with `pull_window`.
    """
    base = config["usaspending_api_base"].rstrip("/")
    body = _build_bulk_body(agency, start, end)
    pulled_at = datetime.now(timezone.utc).replace(tzinfo=None)
    t0 = time.monotonic()
    log_label = f"{agency['short']} {label}"

    logger.info(
        "BULK %s: requesting %s → %s, types=%s, min=%s",
        log_label, start, end, agency["award_types"], agency.get("award_amount_min"),
    )

    # No warmup here: warmup was a paginate-specific workaround for
    # `spending_by_award`. Bulk's single submit + polling pattern doesn't
    # trigger the same WAF behavior, and `_retrying_request` already
    # covers transient drops.
    async with httpx.AsyncClient(timeout=300.0, headers=_default_headers()) as client:
        zip_bytes, file_name = await _bulk_submit_and_wait(
            client, base, body, log_label
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"awards_{agency['short']}_{label}_bulk.parquet"
    rows_total = 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # USAspending bundles `Contracts_PrimeAwardSummaries_*.csv` and
        # `Contracts_Subawards_*.csv` together regardless of our
        # `sub_award_types: []` filter; keep only the prime CSVs.
        contract_csvs = _select_prime_award_csvs(zf.namelist())

        dfs: list[pl.DataFrame] = []
        for csv_name in contract_csvs:
            with zf.open(csv_name) as f:
                raw_bytes = f.read()
            if not raw_bytes.strip():
                continue
            df_raw = pl.read_csv(
                raw_bytes,
                infer_schema_length=10_000,
                ignore_errors=True,
                try_parse_dates=True,
            )
            mapped = _map_bulk_to_schema(df_raw, agency["name"], pulled_at)
            dfs.append(mapped)
            rows_total += mapped.height
            logger.info(
                "BULK %s: parsed %s (%d rows)", log_label, csv_name, mapped.height
            )

        if dfs:
            combined = pl.concat(dfs, how="vertical_relaxed")
            combined.write_parquet(out_path)
        else:
            logger.warning(
                "BULK %s: ZIP %s contained no usable CSVs", log_label, file_name
            )
            pl.DataFrame(schema=_empty_awards_schema()).write_parquet(out_path)

    elapsed = time.monotonic() - t0
    logger.info(
        "DONE BULK %s: %d rows in %.1fs → %s",
        log_label, rows_total, elapsed, out_path,
    )
    return [out_path]


def _max_award_date_for(db_path: str, agency_name: str) -> date | None:
    """Return the most recent award_date for an agency, or None if no data."""
    if not Path(db_path).exists():
        return None
    import duckdb

    con = duckdb.connect(db_path, read_only=True)
    try:
        try:
            row = con.execute(
                "SELECT MAX(award_date) FROM awards WHERE awarding_agency = ?",
                [agency_name],
            ).fetchone()
        except duckdb.CatalogException:
            return None
        return row[0] if row and row[0] else None
    finally:
        con.close()


async def pull_awards(
    incremental: bool = False, db_path: str | None = None
) -> list[Path]:
    config = load_config()
    db_path = db_path or resolve_db_path(config)
    out_dir = Path(config["parquet_dir"])

    written: list[Path] = []
    for raw_agency in config["agencies"]:
        agency = effective_agency(raw_agency, config)
        if incremental:
            # Incremental refreshes always paginate — the weekly window is
            # tiny relative to the API cap, so bulk download's job overhead
            # isn't worth it.
            max_date = _max_award_date_for(db_path, agency["name"])
            if max_date is None:
                logger.info(
                    "%s: no prior data, falling back to seed for FYs %s",
                    agency["short"], config["fiscal_years"],
                )
                written.extend(await _seed_agency(agency, config, out_dir))
                continue
            start = max_date - timedelta(days=7)
            end = date.today()
            label = f"incr_{end.isoformat()}"
            written.extend(
                await asyncio.to_thread(
                    pull_window, agency, start, end, label, config, out_dir
                )
            )
        else:
            written.extend(await _seed_agency(agency, config, out_dir))
    return written


async def _seed_agency(
    agency: dict, config: dict, out_dir: Path
) -> list[Path]:
    """Run the initial seed for one agency, routing on seed_strategy.

    Three strategies are supported:
      - "archive" (default for v1): one HTTPS GET per FY against
        files.usaspending.gov's public award_data_archive bucket. Fast,
        deterministic, no API/WAF/queue.
      - "bulk": legacy path through `POST /api/v2/download/awards/`.
        Kept as a fallback if the archive ever changes shape.
      - "paginate": `POST /api/v2/search/spending_by_award/`. Used for
        weekly incremental refreshes (small windows).
    """
    written: list[Path] = []
    strategy = agency["seed_strategy"]
    for fy in config["fiscal_years"]:
        if strategy == "archive":
            written.extend(
                await asyncio.to_thread(
                    pull_awards_archive, agency, fy, config, out_dir
                )
            )
            continue
        s, e = fiscal_year_window(fy)
        if strategy == "bulk":
            written.extend(
                await pull_awards_bulk(agency, s, e, str(fy), config, out_dir)
            )
        else:
            written.extend(
                await asyncio.to_thread(
                    pull_window, agency, s, e, str(fy), config, out_dir
                )
            )
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Pull federal contract awards from USAspending.gov"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Pull only since the most recent award_date already loaded",
    )
    args = parser.parse_args()
    asyncio.run(pull_awards(incremental=args.incremental))


if __name__ == "__main__":
    main()
