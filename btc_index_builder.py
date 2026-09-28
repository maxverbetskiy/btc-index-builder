"""
BTC address index builder — free stack:
  AWS Public Blockchain Data (s3://aws-public-blockchain, no keys, no egress cost)
  -> DuckDB (on-disk, upsert-based, bounded size)
  -> Hugging Face dataset (free storage, used by app.py as BTCINDEXREPO)

v4: the `addr` table now has a PRIMARY KEY on address and every day is
merged via INSERT ... ON CONFLICT DO UPDATE. This means the table can
never hold more than one row per unique address — it is bounded by
construction, not by a periodic compaction step. This is the fix for the
OOM seen in v1/v2 (temp dir filling to 47.2GiB from unbounded duplicate
rows: one row per address PER DAY it appeared).

Required env vars:
  HF_TOKEN        - Hugging Face token with WRITE access to the dataset repo
  HF_REPO         - dataset repo id, e.g. "yourname/btc-address-index"
  INDEX_FROM      - first day to index if the dataset does not exist yet (YYYY-MM-DD)
  CHECKPOINT_DAYS - optional, days per checkpoint publish (default 10)
"""
import glob
import json
import os
import shutil
import sys
from datetime import date, timedelta

import duckdb
from huggingface_hub import HfApi, hf_hub_download, upload_folder

HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO = os.environ["HF_REPO"]
INDEX_FROM = os.environ.get("INDEX_FROM", "").strip() or "2025-01-01"
CHECKPOINT_DAYS = int(os.environ.get("CHECKPOINT_DAYS", "10"))

WORK = "work"
PRE_DIR = os.path.join(WORK, "pre")
SUF_DIR = os.path.join(WORK, "suf")
SNAP = os.path.join(WORK, "snapshot.parquet")
DB_PATH = os.path.join(WORK, "idx.duckdb")


def log(msg: str) -> None:
    print(f"[btc-index] {msg}", flush=True)


def publish(con, manifest_first_day: str, last_day: str) -> None:
    for p in glob.glob(os.path.join(PRE_DIR, "**"), recursive=True):
        if os.path.isfile(p):
            os.remove(p)
    for p in glob.glob(os.path.join(SUF_DIR, "**"), recursive=True):
        if os.path.isfile(p):
            os.remove(p)

    con.execute(f"COPY addr TO '{SNAP}' (FORMAT PARQUET)")

    con.execute(
        f"COPY (SELECT address, last_seen, substr(address, 1, 2) AS bucket FROM addr) "
        f"TO '{PRE_DIR}' (FORMAT PARQUET, PARTITION_BY (bucket), OVERWRITE_OR_IGNORE 1)"
    )
    con.execute(
        f"COPY (SELECT reverse(address) AS reverse, last_seen, "
        f"substr(reverse(address), 1, 2) AS bucket FROM addr) "
        f"TO '{SUF_DIR}' (FORMAT PARQUET, PARTITION_BY (bucket), OVERWRITE_OR_IGNORE 1)"
    )

    pre_files = sorted(glob.glob(os.path.join(PRE_DIR, "**", "*.parquet"), recursive=True))
    suf_files = sorted(glob.glob(os.path.join(SUF_DIR, "**", "*.parquet"), recursive=True))
    pre_rel = [os.path.relpath(f, WORK).replace("\\", "/") for f in pre_files]
    suf_rel = [os.path.relpath(f, WORK).replace("\\", "/") for f in suf_files]

    manifest_out = {
        "first_day": manifest_first_day,
        "last_day": last_day,
        "files": {"pre": pre_rel, "suf": suf_rel},
    }
    with open(os.path.join(WORK, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_out, f)

    upload_folder(
        repo_id=HF_REPO,
        repo_type="dataset",
        folder_path=WORK,
        path_in_repo=".",
        token=HF_TOKEN,
        allow_patterns=["pre/**", "suf/**", "manifest.json", "snapshot.parquet"],
        commit_message=f"index checkpoint through {last_day}",
    )
    log(f"checkpoint published: last_day={last_day}, pre_files={len(pre_rel)}, suf_files={len(suf_rel)}")


def main() -> None:
    os.makedirs(PRE_DIR, exist_ok=True)
    os.makedirs(SUF_DIR, exist_ok=True)
    os.makedirs(os.path.join(WORK, "tmp"), exist_ok=True)

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO, repo_type="dataset", exist_ok=True)

    manifest = None
    try:
        mpath = hf_hub_download(repo_id=HF_REPO, filename="manifest.json",
                                 repo_type="dataset", token=HF_TOKEN)
        manifest = json.load(open(mpath, encoding="utf-8"))
        log(f"found existing manifest, last_day={manifest.get('last_day')}")
    except Exception as e:  # noqa: BLE001
        log(f"no existing manifest ({e}); building from scratch from {INDEX_FROM}")

    con = duckdb.connect(DB_PATH)
    con.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='us-east-2';")
    con.execute("SET enable_progress_bar=false;")
    con.execute("SET preserve_insertion_order=false;")
    con.execute("SET threads=2;")
    con.execute("SET memory_limit='10GB';")
    con.execute(f"SET temp_directory='{os.path.join(WORK, 'tmp')}';")
    # PRIMARY KEY -> table can never exceed one row per unique address
    con.execute("CREATE TABLE IF NOT EXISTS addr(address VARCHAR PRIMARY KEY, last_seen TIMESTAMP)")

    if manifest:
        try:
            p = hf_hub_download(repo_id=HF_REPO, filename="snapshot.parquet",
                                 repo_type="dataset", token=HF_TOKEN)
            shutil.copy(p, SNAP)
            if con.execute("SELECT count(*) FROM addr").fetchone()[0] == 0:
                con.execute(f"INSERT INTO addr SELECT * FROM read_parquet('{SNAP}')")
            log("loaded previous snapshot into DuckDB")
        except Exception as e:  # noqa: BLE001
            log(f"could not load previous snapshot: {e}")

    first_day = manifest["first_day"] if manifest else INDEX_FROM
    start = (date.fromisoformat(manifest["last_day"]) + timedelta(days=1)
             if manifest else date.fromisoformat(INDEX_FROM))
    end = date.today() - timedelta(days=1)

    if start > end:
        log("nothing new to index (already up to date)")
        sys.exit(0)

    last_ok = manifest["last_day"] if manifest else None
    days_since_checkpoint = 0
    d = start
    while d <= end:
        path = f"s3://aws-public-blockchain/v1.0/btc/transactions/date={d.isoformat()}/*.parquet"
        sql = f"""
            INSERT INTO addr
            SELECT a AS address, max(t) AS last_seen FROM (
                SELECT unnest(list_filter(list_transform(outputs, o -> o.address),
                                           a -> a IS NOT NULL)) AS a,
                       block_timestamp AS t
                FROM read_parquet('{path}')
            ) GROUP BY 1
            ON CONFLICT (address) DO UPDATE SET
                last_seen = GREATEST(excluded.last_seen, addr.last_seen)
        """
        try:
            con.execute(sql)
            last_ok = d.isoformat()
            days_since_checkpoint += 1
            log(f"indexed {d.isoformat()}")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "No files found" in msg or "404" in msg:
                log(f"{d.isoformat()}: no data yet, stopping for this run")
            else:
                log(f"{d.isoformat()}: error, stopping for this run: {msg[:300]}")
            break

        if days_since_checkpoint >= CHECKPOINT_DAYS:
            publish(con, first_day, last_ok)
            days_since_checkpoint = 0

        d += timedelta(days=1)

    if days_since_checkpoint > 0 or last_ok != (manifest["last_day"] if manifest else None):
        publish(con, first_day, last_ok)
    else:
        log("no new days processed since last checkpoint")

    log(f"run complete, last_day={last_ok}")


if __name__ == "__main__":
    main()
