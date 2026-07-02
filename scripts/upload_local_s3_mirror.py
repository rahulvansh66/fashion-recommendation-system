#!/usr/bin/env python3
"""Upload the repo-local ``s3/`` mirror to an AWS S3 bucket.

Creates the target bucket when it does not exist yet. Reads ``.env.local`` via
``fashion_recommendation_system.config`` (see ``configs/data/s3_paths.yaml`` for paths).

Usage (from repo root)::

python scripts/upload_local_s3_mirror.py              # upload + verify
python scripts/upload_local_s3_mirror.py --dry-run    # preview only
python scripts/upload_local_s3_mirror.py --verify-only  # check without uploading
python scripts/upload_local_s3_mirror.py --bucket my-bucket --region us-east-1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from fashion_recommendation_system import config

DEFAULT_LOCAL_ROOT = REPO_ROOT / "s3"
DEFAULT_BUCKET = config.S3_BUCKET
DEFAULT_REGION = config.AWS_REGION


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync the local s3/ directory to an AWS S3 bucket."
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=DEFAULT_LOCAL_ROOT,
        help=f"Local mirror root (default: {DEFAULT_LOCAL_ROOT})",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=f"S3 bucket name (default: {DEFAULT_BUCKET})",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be uploaded without writing to S3",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Compare local mirror with S3; do not upload",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip post-upload verification",
    )
    return parser.parse_args()


def _iter_local_files(local_root: Path) -> list[Path]:
    """Return all files under ``local_root``, sorted for stable output."""
    if not local_root.is_dir():
        raise FileNotFoundError(f"Local mirror directory not found: {local_root}")
    files = [path for path in local_root.rglob("*") if path.is_file()]
    if not files:
        raise ValueError(f"No files found under {local_root}")
    return sorted(files)


def _object_key(local_root: Path, file_path: Path) -> str:
    """Map a local file path to its S3 object key (POSIX-style)."""
    return file_path.relative_to(local_root).as_posix()


def ensure_bucket_exists(s3_client, bucket: str, region: str) -> None:
    """Create the bucket when ``head_bucket`` reports that it is missing."""
    try:
        s3_client.head_bucket(Bucket=bucket)
        print(f"Bucket s3://{bucket}/ already exists.")
        return
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchBucket", "NotFound"}:
            print(f"Bucket s3://{bucket}/ not found — creating in {region}...")
            create_kwargs: dict = {"Bucket": bucket}
            if region != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": region
                }
            s3_client.create_bucket(**create_kwargs)
            s3_client.get_waiter("bucket_exists").wait(Bucket=bucket)
            print(f"Created bucket s3://{bucket}/")
            return
        if code == "403":
            raise PermissionError(
                f"Bucket {bucket!r} is not accessible (exists in another account or "
                "missing s3:ListBucket permission)."
            ) from exc
        raise


def upload_mirror(
    s3_client,
    local_root: Path,
    bucket: str,
    *,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Upload every file under ``local_root`` preserving relative keys.

    Returns:
        Tuple of (files_uploaded, total_bytes_uploaded).
    """
    local_files = _iter_local_files(local_root)
    uploaded = 0
    total_bytes = 0

    for file_path in local_files:
        key = _object_key(local_root, file_path)
        size = file_path.stat().st_size
        if dry_run:
            print(f"[dry-run] s3://{bucket}/{key} ({size} bytes)")
            uploaded += 1
            total_bytes += size
            continue

        s3_client.upload_file(str(file_path), bucket, key)
        uploaded += 1
        total_bytes += size
        if uploaded % 25 == 0 or uploaded == len(local_files):
            print(f"Uploaded {uploaded}/{len(local_files)} objects...")

    return uploaded, total_bytes


def _local_inventory(local_root: Path) -> dict[str, int]:
    """Build a mapping of S3 key -> file size for the local mirror."""
    return {
        _object_key(local_root, path): path.stat().st_size
        for path in _iter_local_files(local_root)
    }


def _remote_inventory(s3_client, bucket: str) -> dict[str, int]:
    """List all objects in ``bucket`` and return key -> size."""
    inventory: dict[str, int] = {}
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            inventory[obj["Key"]] = obj["Size"]
    return inventory


def verify_mirror(s3_client, local_root: Path, bucket: str) -> bool:
    """Verify S3 contents match the local mirror (keys and sizes)."""
    local = _local_inventory(local_root)
    remote = _remote_inventory(s3_client, bucket)

    missing = sorted(set(local) - set(remote))
    extra = sorted(set(remote) - set(local))
    size_mismatch = sorted(
        key
        for key in set(local) & set(remote)
        if local[key] != remote[key]
    )

    if not missing and not extra and not size_mismatch:
        total_bytes = sum(local.values())
        print(
            f"Verification OK: {len(local)} objects, "
            f"{total_bytes:,} bytes match s3://{bucket}/"
        )
        return True

    print("Verification FAILED:")
    if missing:
        print(f"  Missing in S3 ({len(missing)}): {missing[:5]}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")
    if extra:
        print(f"  Extra in S3 ({len(extra)}): {extra[:5]}")
        if len(extra) > 5:
            print(f"    ... and {len(extra) - 5} more")
    if size_mismatch:
        print(f"  Size mismatch ({len(size_mismatch)}): {size_mismatch[:5]}")
        if len(size_mismatch) > 5:
            print(f"    ... and {len(size_mismatch) - 5} more")
    return False


def main() -> int:
    args = _parse_args()
    local_root = args.local_root.resolve()

    profile = config.AWS_PROFILE or None
    bucket = args.bucket or config.S3_BUCKET
    region = args.region or config.AWS_REGION

    print(f"Local mirror : {local_root}")
    print(f"Target bucket: s3://{bucket}/")
    print(f"Region       : {region}")
    if profile:
        print(f"AWS profile  : {profile}")

    session = boto3.Session(profile_name=profile, region_name=region)
    s3_client = session.client("s3")

    if not args.verify_only:
        ensure_bucket_exists(s3_client, bucket, region)
        uploaded, total_bytes = upload_mirror(
            s3_client,
            local_root,
            bucket,
            dry_run=args.dry_run,
        )
        action = "Would upload" if args.dry_run else "Uploaded"
        print(f"{action} {uploaded} objects ({total_bytes:,} bytes).")

    if args.dry_run:
        return 0

    if args.skip_verify:
        return 0

    ok = verify_mirror(s3_client, local_root, bucket)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, PermissionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
