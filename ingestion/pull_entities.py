"""SAM.gov entity puller.

Task 1.3: enrich each unique UEI from the awards Parquet files with its
SAM.gov registration record, write a consolidated `entities.parquet`, and
append a daily `entity_snapshots_{date}.parquet` capturing point-in-time
address / name / cage-code state for the address-churn detector
(Task 2.4).

Run:
    python -m ingestion.pull_entities               # full enrich
    python -m ingestion.pull_entities --incremental # only new + stale UEIs
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import httpx
import polars as pl
from dotenv import load_dotenv
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ingestion.load_db import load_config, resolve_db_path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# SAM.gov public Entity API allows ~10 req/sec. Daily quotas vary by API
# key tier; with ~100K UEIs the seed will likely brush against the cap, so
# we log and continue rather than fail the whole run on quota errors.
SAM_RATE_LIMIT = 10
SAM_TIMEOUT = 30.0
STALE_AFTER_DAYS = 30
PROGRESS_LOG_EVERY = 200

DEFAULT_HEADERS = {
    "User-Agent": "anomaly-radar/0.1 (+https://github.com/kamronarabi/fed-anomaly-tracker)",
}


def _entities_schema() -> dict:
    return {
        "uei": pl.Utf8,
        "legal_business_name": pl.Utf8,
        "dba_name": pl.Utf8,
        "physical_address_line1": pl.Utf8,
        "physical_city": pl.Utf8,
        "physical_state": pl.Utf8,
        "physical_zip": pl.Utf8,
        "business_type": pl.Utf8,
        "entity_structure": pl.Utf8,
        "registration_date": pl.Date,
        "expiration_date": pl.Date,
        "cage_code": pl.Utf8,
        "exclusion_status": pl.Utf8,
        "last_pulled_at": pl.Datetime,
    }


def _snapshot_schema() -> dict:
    return {
        "uei": pl.Utf8,
        "snapshot_date": pl.Date,
        "legal_business_name": pl.Utf8,
        "physical_address_line1": pl.Utf8,
        "physical_city": pl.Utf8,
        "physical_state": pl.Utf8,
        "physical_zip": pl.Utf8,
        "cage_code": pl.Utf8,
    }


def _parse_date(value) -> date | None:
    if value in (None, "", "null"):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).split("T")[0]
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _format_business_types(core: dict | None) -> str | None:
    """Join SAM businessTypeList descriptions/codes into one comma-separated
    string. SAM nests this as coreData.businessTypes.businessTypeList[]."""
    if not core:
        return None
    bt = core.get("businessTypes") or {}
    items = bt.get("businessTypeList") or []
    descs: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        d = item.get("businessTypeDesc") or item.get("businessTypeCode")
        if d:
            descs.append(str(d))
    return ", ".join(descs) if descs else None


def extract_entity(record: dict, pulled_at: datetime) -> dict | None:
    """Map one SAM v3 `entityData` record to the entities-table schema.

    Returns None if the record has no UEI — these are skipped silently
    rather than poisoning the Parquet write.
    """
    reg = record.get("entityRegistration") or {}
    core = record.get("coreData") or {}
    addr = core.get("physicalAddress") or {}
    general = core.get("generalInformation") or {}

    uei = reg.get("ueiSAM") or reg.get("uei") or record.get("ueiSAM")
    if not uei:
        return None

    return {
        "uei": uei,
        "legal_business_name": reg.get("legalBusinessName"),
        "dba_name": reg.get("dbaName"),
        "physical_address_line1": addr.get("addressLine1"),
        "physical_city": addr.get("city"),
        "physical_state": (
            addr.get("stateOrProvinceCode") or addr.get("state")
        ),
        "physical_zip": addr.get("zipCode") or addr.get("zip"),
        "business_type": _format_business_types(core),
        "entity_structure": (
            general.get("entityStructureDesc")
            or general.get("entityStructureCode")
        ),
        "registration_date": _parse_date(
            reg.get("registrationDate") or reg.get("activationDate")
        ),
        "expiration_date": _parse_date(
            reg.get("registrationExpirationDate")
            or reg.get("expirationDate")
        ),
        "cage_code": reg.get("cageCode"),
        "exclusion_status": reg.get("exclusionStatusFlag"),
        "last_pulled_at": pulled_at,
    }


async def _fetch_entity(
    client: httpx.AsyncClient,
    uei: str,
    api_key: str,
    base_url: str,
) -> dict | None:
    """GET one entity from SAM with tenacity retries. Returns the raw
    `entityData[0]` record, or None if the entity is not registered."""
    params = {"ueiSAM": uei, "api_key": api_key}
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    ):
        with attempt:
            resp = await client.get(base_url, params=params, timeout=SAM_TIMEOUT)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            records = data.get("entityData") or []
            return records[0] if records else None
    return None


async def _pull_one_throttled(
    client: httpx.AsyncClient,
    uei: str,
    api_key: str,
    base_url: str,
    semaphore: asyncio.Semaphore,
    min_seconds_per_slot: float,
) -> tuple[str, dict | None]:
    """Hold one of `SAM_RATE_LIMIT` semaphore slots for at least
    `min_seconds_per_slot` seconds. With a 10-slot semaphore and a 1.0s
    floor, steady-state throughput is bounded at ~10 req/sec."""
    async with semaphore:
        t0 = time.monotonic()
        try:
            return uei, await _fetch_entity(client, uei, api_key, base_url)
        finally:
            held = time.monotonic() - t0
            if held < min_seconds_per_slot:
                await asyncio.sleep(min_seconds_per_slot - held)


async def _pull_many(
    ueis: list[str], api_key: str, base_url: str
) -> list[dict]:
    """Pull every UEI in `ueis`, bounded to ~10 req/sec.

    Returns successfully extracted entity rows (skipping not-found and
    errored UEIs). Stops early on 401/403/429 since those usually mean
    the API key is bad or the daily quota is exhausted — continuing would
    just burn through the retry budget.
    """
    if not ueis:
        return []

    semaphore = asyncio.Semaphore(SAM_RATE_LIMIT)
    pulled_at = datetime.now(timezone.utc).replace(tzinfo=None)
    rows: list[dict] = []
    not_found = 0
    errors = 0
    quota_hit = False
    completed = 0
    t_start = time.monotonic()

    async with httpx.AsyncClient(headers=DEFAULT_HEADERS) as client:
        tasks = [
            asyncio.create_task(
                _pull_one_throttled(
                    client, uei, api_key, base_url, semaphore,
                    min_seconds_per_slot=1.0,
                )
            )
            for uei in ueis
        ]
        try:
            for coro in asyncio.as_completed(tasks):
                try:
                    _uei, record = await coro
                except httpx.HTTPStatusError as e:
                    errors += 1
                    if e.response.status_code in (401, 403, 429):
                        quota_hit = True
                        logger.warning(
                            "SAM auth/quota error %s — stopping further requests",
                            e.response.status_code,
                        )
                        break
                    continue
                except Exception as e:
                    errors += 1
                    logger.warning("SAM fetch error: %s", e)
                    continue

                if record is None:
                    not_found += 1
                else:
                    row = extract_entity(record, pulled_at)
                    if row is not None:
                        rows.append(row)

                completed += 1
                if completed % PROGRESS_LOG_EVERY == 0:
                    logger.info(
                        "SAM progress: %d/%d (found=%d not_found=%d errors=%d)",
                        completed, len(ueis), len(rows), not_found, errors,
                    )
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.monotonic() - t_start
    logger.info(
        "SAM pulled %d/%d UEIs in %.1fs (not_found=%d errors=%d quota_hit=%s)",
        len(rows), len(ueis), elapsed, not_found, errors, quota_hit,
    )
    return rows


def _read_ueis_from_parquet(parquet_dir: Path) -> set[str]:
    """Set of unique recipient_uei values across every awards_*.parquet
    file under `parquet_dir`. Empty set if the directory or files are
    missing — the caller is expected to handle the empty case."""
    if not parquet_dir.exists():
        return set()
    paths = sorted(parquet_dir.glob("awards_*.parquet"))
    if not paths:
        return set()

    ueis: set[str] = set()
    for path in paths:
        try:
            col = (
                pl.scan_parquet(path)
                .select("recipient_uei")
                .collect()
                .get_column("recipient_uei")
            )
        except Exception as e:
            logger.warning("could not read %s: %s", path, e)
            continue
        ueis.update(v for v in col.to_list() if v)
    return ueis


def _select_incremental_ueis(db_path: str, parquet_dir: Path) -> list[str]:
    """For incremental runs, return the UEIs we must hit SAM for:
      - new UEIs that appeared in this week's awards Parquet but aren't in
        the entities table yet, OR
      - stale entities with an active contract (pop_end > today) whose
        last_pulled_at is older than STALE_AFTER_DAYS days.
    """
    awards_ueis = _read_ueis_from_parquet(parquet_dir)
    if not Path(db_path).exists():
        return sorted(awards_ueis)

    con = duckdb.connect(db_path, read_only=True)
    try:
        try:
            existing = {
                row[0] for row in con.execute("SELECT uei FROM entities").fetchall()
            }
        except duckdb.CatalogException:
            return sorted(awards_ueis)

        new_ueis = [u for u in awards_ueis if u not in existing]

        try:
            stale_rows = con.execute(
                f"""
                SELECT e.uei
                FROM entities e
                JOIN (
                    SELECT recipient_uei, MAX(period_of_performance_end) AS pop_end
                    FROM awards
                    GROUP BY recipient_uei
                ) a ON a.recipient_uei = e.uei
                WHERE a.pop_end > CURRENT_DATE
                  AND e.last_pulled_at < CURRENT_TIMESTAMP - INTERVAL {STALE_AFTER_DAYS} DAY
                """
            ).fetchall()
            stale = [r[0] for r in stale_rows]
        except duckdb.CatalogException:
            stale = []

        return sorted(set(new_ueis) | set(stale))
    finally:
        con.close()


def _snapshot_rows_from_pulled(
    rows: list[dict], snapshot_date: date
) -> list[dict]:
    """Project the freshly-pulled entity rows down to the snapshot schema."""
    return [
        {
            "uei": r["uei"],
            "snapshot_date": snapshot_date,
            "legal_business_name": r["legal_business_name"],
            "physical_address_line1": r["physical_address_line1"],
            "physical_city": r["physical_city"],
            "physical_state": r["physical_state"],
            "physical_zip": r["physical_zip"],
            "cage_code": r["cage_code"],
        }
        for r in rows
    ]


def _snapshot_rows_from_db(
    db_path: str, snapshot_date: date, exclude: set[str]
) -> list[dict]:
    """Snapshot rows for entities already in the DB that we did NOT re-pull
    this run. The plan calls for one snapshot per entity per snapshot_date,
    so we fall back to the cached values for the rest."""
    if not Path(db_path).exists():
        return []
    con = duckdb.connect(db_path, read_only=True)
    try:
        try:
            results = con.execute(
                """
                SELECT uei, legal_business_name, physical_address_line1,
                       physical_city, physical_state, physical_zip, cage_code
                FROM entities
                """
            ).fetchall()
        except duckdb.CatalogException:
            return []
    finally:
        con.close()

    out: list[dict] = []
    for uei, name, line1, city, state, zip_, cage in results:
        if uei in exclude:
            continue
        out.append({
            "uei": uei,
            "snapshot_date": snapshot_date,
            "legal_business_name": name,
            "physical_address_line1": line1,
            "physical_city": city,
            "physical_state": state,
            "physical_zip": zip_,
            "cage_code": cage,
        })
    return out


async def pull_entities(
    incremental: bool = False, db_path: str | None = None
) -> tuple[Path | None, Path | None]:
    """Run the SAM enrichment + snapshot writes. Returns
    (entities_path, snapshot_path); either may be None if there's nothing
    to write (no UEIs in awards parquet, no successful SAM pulls)."""
    load_dotenv()
    api_key = os.environ.get("SAM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SAM_API_KEY not set — add it to .env "
            "(register at https://sam.gov/content/entity-information)"
        )

    config = load_config()
    db_path = db_path or resolve_db_path(config)
    out_dir = Path(config["parquet_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = config["sam_api_base"]

    if incremental:
        ueis = _select_incremental_ueis(db_path, out_dir)
        logger.info("Incremental: %d UEIs to enrich (new + stale)", len(ueis))
    else:
        ueis = sorted(_read_ueis_from_parquet(out_dir))
        logger.info("Seed: %d unique UEIs in awards parquet", len(ueis))

    pulled = await _pull_many(ueis, api_key, base_url)

    entities_path: Path | None = None
    if pulled:
        entities_path = out_dir / "entities.parquet"
        pl.DataFrame(pulled, schema=_entities_schema()).write_parquet(entities_path)
        logger.info("Wrote %d entities → %s", len(pulled), entities_path)

    snapshot_date = date.today()
    snapshot_rows = _snapshot_rows_from_pulled(pulled, snapshot_date)
    snapshot_rows.extend(
        _snapshot_rows_from_db(
            db_path, snapshot_date, exclude={r["uei"] for r in pulled}
        )
    )

    snapshot_path: Path | None = None
    if snapshot_rows:
        snapshot_path = out_dir / f"entity_snapshots_{snapshot_date.isoformat()}.parquet"
        pl.DataFrame(snapshot_rows, schema=_snapshot_schema()).write_parquet(snapshot_path)
        logger.info("Wrote %d snapshots → %s", len(snapshot_rows), snapshot_path)

    return entities_path, snapshot_path


def main():
    parser = argparse.ArgumentParser(
        description="Enrich UEIs from awards parquet with SAM.gov registration data"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only pull new UEIs and stale entities with active contracts",
    )
    args = parser.parse_args()
    asyncio.run(pull_entities(incremental=args.incremental))


if __name__ == "__main__":
    main()
