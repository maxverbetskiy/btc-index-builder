"""
BTC address index builder — final version.

Free stack:
  AWS Public Blockchain Data (s3://aws-public-blockchain, no keys, no egress cost)
  -> DuckDB (on-disk, per-window shards, bounded memory)
  -> Hugging Face dataset (free storage, used by app.py as BTCINDEXREPO)

Design decisions (each one fixes a real crash seen in production):

1. No persistent, ever-growing "addr" table. Each checkpoint window
   (CHECKPOINT_DAYS days) is processed into its OWN small table that is
   dropped after export. Memory use is bounded by ONE window, not by the
   total backfilled history.

2. No PRIMARY KEY / index on the per-window table, no global re-sort.
   Each window's data is aggregated and sorted only once, only for that
   window (a few million rows at most).

3. Each window is exported as its OWN small shard file
   (pre_<start>_<end>.parquet / suf_<start>_<end>.parquet) instead of
   partitioning by 2-char address bucket (hundreds/thousands of tiny
   files per checkpoint).

4. Duplicate addresses across different shards are expected and fine:
   app.py's search already does GROUP BY address / max(last_seen) when
   reading multiple parquet files, so cross-shard duplicates collapse
   at query time.

5. Publish step retries on transient errors (including HF 429) with
   backoff.

6. Explicit throttle: enforces a minimum interval between HF commits
   (MIN_PUBLISH_INTERVAL_S, default 30s) regardless of how fast the
   day-processing loop runs, so the commit rate can never exceed roughly
   120/hour even if AWS/DuckDB happen to be faster than expected on a
   given day -> keeps a safety margin under HF's 128/hour commit limit.

Required env vars:
  HF_TOKEN               - Hugging Face token with WRITE access to the dataset repo
  HF_REPO                - dataset repo id, e.g. "yourname/btc-address-index"
  INDEX_FROM              - first day to index if the dataset does not exist yet (YYYY-MM-DD)
  CHECKPOINT_DAYS         - optional, days per shard/checkpoint (default 7)
  MIN_PUBLISH_INTERVAL_S  - optional, min seconds between HF commits (default 30)
"""
import json
import os
import sys
import time
from datetime import date, timedelta

import duckdb
from huggingface_hub import HfApi, hf_hub_download, upload_folder
from huggingface_hub.utils import HfHubHTTPError

HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO = os.environ["HF_REPO"]
INDEX_FROM = os.environ.get("INDEX_FROM", "").strip() or "2025-01-01"
CHECKPOINT_DAYS = int(os.environ.get("CHECKPOINT_DAYS", "7"))
MIN_PUBLISH_INTERVAL_S = float(os.environ.get("MIN_PUBLISH_INTERVAL_S", "30"))

WORK = "work"
_last_publish_ts = 0.0


def log(msg: str) -> None:
    print(f"[btc-index] {msg}", flush=True)


def new_con():
    con = duckdb.connect(":memory:")
    con.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='us-east-2';")
    con.execute("SET enable_progress_bar=false;")
    con.execute("SET preserve_insertion_order=false;")
    con.execute("SET threads=2;")
    con.execute("SET memory_limit='13GB';")
    con.execute(f"SET temp_directory='{os.path.join(WORK, 'tmp')}';")
    con.execute("CREATE TABLE window_addr(address VARCHAR, last_seen TIMESTAMP)")
    return con


def export_shard(con, win_start: str, win_end: str) -> tuple[str, str]:
    pre_name = f"pre_{win_start}_{win_end}.parquet"
    suf_name = f"suf_{win_start}_{win_end}.parquet"
    pre_path = os.path.join(WORK, pre_name)
    suf_path = os.path.join(WORK, suf_name)

    con.execute(f"""
        COPY (SELECT address, max(last_seen) AS last_seen FROM window_addr
              GROUP BY 1 ORDER BY 1)
        TO '{pre_path}' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (SELECT reverse(address) AS reverse, max(last_seen) AS last_seen
              FROM window_addr GROUP BY 1 ORDER BY 1)
        TO '{suf_path}' (FORMAT PARQUET)
    """)
    return pre_name, suf_name


def throttle_before_publish() -> None:
    global _last_publish_ts
    elapsed = time.time() - _last_publish_ts
    if elapsed < MIN_PUBLISH_INTERVAL_S:
        wait = MIN_PUBLISH_INTERVAL_S - elapsed
        log(f"throttling: waiting {wait:.1f}s before next HF commit "
            f"(keeps commit rate safely under HF's 128/hour limit)")
        time.sleep(wait)


def upload_with_retry(pre_name: str, suf_name: str, manifest: dict, attempts: int = 4) -> None:
    global _last_publish_ts
    manifest_path = os.path.join(WORK, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    throttle_before_publish()

    delay = 30
    for attempt in range(1, attempts + 1):
        try:
            upload_folder(
                repo_id=HF_REPO,
                repo_type="dataset",
                folder_path=WORK,
                path_in_repo=".",
                token=HF_TOKEN,
                allow_patterns=[pre_name, suf_name, "manifest.json"],
                commit_message=f"index shard {pre_name}",
            )
            _last_publish_ts = time.time()
            log(f"published shard {pre_name} / {suf_name}, last_day={manifest['last_day']}")
            return
        except HfHubHTTPError as e:
            is_rate_limit = "429" in str(e) or "rate limit" in str(e).lower()
            if attempt == attempts:
                raise
            wait = 300 if is_rate_limit else delay  # a real 429 needs minutes, not seconds
            log(f"publish attempt {attempt} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
        except Exception as e:  # noqa: BLE001
            if attempt == attempts:
                raise
            log(f"publish attempt {attempt} failed ({e}); retrying in {delay}s")
            time.sleep(delay)


def main() -> None:
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(os.path.join(WORK, "tmp"), exist_ok=True)

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO, repo_type="dataset", exist_ok=True)

    manifest = None
    try:
        mpath = hf_hub_download(repo_id=HF_REPO, filename="manifest.json",
                                 repo_type="dataset", token=HF_TOKEN)
        manifest = json.load(open(mpath, encoding="utf-8"))
        log(f"found existing manifest, last_day={manifest.get('last_day')}, "
            f"shards={len(manifest.get('files', {}).get('pre', []))}")
    except Exception as e:  # noqa: BLE001
        log(f"no existing manifest ({e}); building from scratch from {INDEX_FROM}")

    first_day = manifest["first_day"] if manifest else INDEX_FROM
    pre_files = list(manifest["files"]["pre"]) if manifest else []
    suf_files = list(manifest["files"]["suf"]) if manifest else []
    start = (date.fromisoformat(manifest["last_day"]) + timedelta(days=1)
             if manifest else date.fromisoformat(INDEX_FROM))
    end = date.today() - timedelta(days=1)

    if start > end:
        log("nothing new to index (already up to date)")
        sys.exit(0)

    d = start
    while d <= end:
        win_start = d.isoformat()
        con = new_con()
        days_in_window = 0
        last_ok = None
        stop_all = False

        while d <= end and days_in_window < CHECKPOINT_DAYS:
            path = f"s3://aws-public-blockchain/v1.0/btc/transactions/date={d.isoformat()}/*.parquet"
            sql = f"""
                INSERT INTO window_addr
                SELECT a AS address, max(t) AS last_seen FROM (
                    SELECT unnest(list_filter(list_transform(outputs, o -> o.address),
                                               a -> a IS NOT NULL)) AS a,
                           block_timestamp AS t
                    FROM read_parquet('{path}')
                ) GROUP BY 1
            """
            try:
                con.execute(sql)
                last_ok = d.isoformat()
                days_in_window += 1
                log(f"indexed {d.isoformat()}")
                d += timedelta(days=1)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "No files found" in msg or "404" in msg:
                    log(f"{d.isoformat()}: no data yet, stopping for this run")
                else:
                    log(f"{d.isoformat()}: error, stopping for this run: {msg[:300]}")
                stop_all = True
                break

        if last_ok:
            pre_name, suf_name = export_shard(con, win_start, last_ok)
            pre_files.append(pre_name)
            suf_files.append(suf_name)
            manifest_out = {
                "first_day": first_day,
                "last_day": last_ok,
                "files": {"pre": pre_files, "suf": suf_files},
            }
            upload_with_retry(pre_name, suf_name, manifest_out)
            manifest = manifest_out

        con.close()
        if stop_all:
            break

    log(f"run complete, last_day={manifest['last_day'] if manifest else None}")


if __name__ == "__main__":
    main()
