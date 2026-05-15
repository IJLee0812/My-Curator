#!/usr/bin/env python3
"""
cleanup_curator_db.py — Wipe PG, Milvus, and MinIO for a fresh My-Curator run.

Usage (from repo root):
    .venv/bin/python3.10 infra/cleanup_curator_db.py [OPTIONS]

Options:
    --pg      Truncate Postgres tables only
    --milvus  Drop Milvus collections only
    --minio   Empty MinIO buckets only
    --all     Wipe everything (default when no flag is given)

Credentials are read from .env at the repo root.
"""

import argparse
import asyncio
import sys
from pathlib import Path

ENV_PATH = Path(__file__).parent.parent / ".env"

PG_TABLES = ["sessions", "clips", "scenario_dna", "review_queue"]
MILVUS_COLLECTIONS = ["clip_video_embed"]
MINIO_BUCKETS = ["frames", "clips", "raw", "artifacts"]


# ── .env loader ───────────────────────────────────────────────────────────────

def load_env(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"[ERROR] .env not found at {path}")
    env: dict = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip()
    return env


# ── PG cleanup (asyncpg) ──────────────────────────────────────────────────────

async def _cleanup_pg_async(env: dict) -> None:
    import asyncpg  # type: ignore

    dsn = (
        f"postgresql://{env.get('PG_USER', 'curation')}"
        f":{env['PG_PASSWORD']}"
        f"@{env.get('PG_HOST', 'localhost')}"
        f":{env.get('PG_PORT', '5432')}"
        f"/{env.get('PG_DB', 'curation')}"
    )
    print("[PG] Connecting...")
    conn = await asyncpg.connect(dsn)
    try:
        tables = ", ".join(PG_TABLES)
        await conn.execute(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE;")
    finally:
        await conn.close()
    print(f"[PG] TRUNCATED {', '.join(PG_TABLES)} — OK")


def cleanup_pg(env: dict) -> None:
    asyncio.run(_cleanup_pg_async(env))


# ── Milvus cleanup ────────────────────────────────────────────────────────────

def cleanup_milvus(env: dict) -> None:
    from pymilvus import MilvusClient  # type: ignore

    uri = env.get("MILVUS_URI", "http://localhost:19530")
    print(f"[Milvus] Connecting to {uri} ...")
    client = MilvusClient(uri=uri)
    for col in MILVUS_COLLECTIONS:
        if client.has_collection(col):
            client.drop_collection(col)
            print(f"[Milvus] Dropped '{col}' — OK")
        else:
            print(f"[Milvus] '{col}' not found — skip")


# ── MinIO cleanup ─────────────────────────────────────────────────────────────

def cleanup_minio(env: dict) -> None:
    import boto3  # type: ignore
    from botocore.exceptions import ClientError  # type: ignore

    endpoint = env.get("MINIO_ENDPOINT", "http://localhost:9000")
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=env.get("MINIO_USER", "minio-admin"),
        aws_secret_access_key=env["MINIO_PASSWORD"],
    )

    for bucket in MINIO_BUCKETS:
        try:
            paginator = s3.get_paginator("list_objects_v2")
            deleted = 0
            for page in paginator.paginate(Bucket=bucket):
                objects = page.get("Contents", [])
                if not objects:
                    continue
                s3.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
                )
                deleted += len(objects)
            print(f"[MinIO] '{bucket}': deleted {deleted} objects — OK")
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "NoSuchBucket":
                print(f"[MinIO] '{bucket}' does not exist — skip")
            else:
                raise


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wipe My-Curator data stores (PG + Milvus + MinIO)"
    )
    parser.add_argument("--pg", action="store_true", help="Truncate Postgres tables")
    parser.add_argument("--milvus", action="store_true", help="Drop Milvus collections")
    parser.add_argument("--minio", action="store_true", help="Empty MinIO buckets")
    parser.add_argument("--all", dest="all_", action="store_true", help="Wipe everything (default)")
    args = parser.parse_args()

    if not any([args.pg, args.milvus, args.minio, args.all_]):
        args.all_ = True

    env = load_env(ENV_PATH)

    if args.pg or args.all_:
        cleanup_pg(env)
    if args.milvus or args.all_:
        cleanup_milvus(env)
    if args.minio or args.all_:
        cleanup_minio(env)

    print("\n[DONE] Cleanup complete.")


if __name__ == "__main__":
    main()
