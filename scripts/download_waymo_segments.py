#!/usr/bin/env python3
"""
scripts/download_waymo_segments.py
====================================
Download a random sample of Waymo Open Dataset TFRecord segments from
Google Cloud Storage and write a segments_to_run.txt file for the pipeline.

Prerequisites
-------------
1. Accept the Waymo Open Dataset license: https://waymo.com/open/terms/
2. Authenticate with GCS (choose one):
     gcloud auth application-default login
     export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
3. Install dependencies:
     pip install google-cloud-storage       # for the default SDK mode
     # OR have 'gsutil' on your PATH        # for --use-gsutil mode

Usage examples
--------------
  # Download 5 random training segments (SDK mode, default)
  python scripts/download_waymo_segments.py --num-segments 5

  # Reproducible sample of 10 segments using gsutil (resumable, faster)
  python scripts/download_waymo_segments.py -n 10 --use-gsutil --seed 42

  # Custom bucket version and output directory
  python scripts/download_waymo_segments.py -n 3 \\
      --bucket waymo_open_dataset_v_1_4_3 \\
      --output-dir /data/waymo_segments

After running, execute the pipeline with:
  python pipeline_input.py
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download random Waymo Open Dataset segments from GCS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--num-segments", "-n",
        type=int, default=5,
        help="Number of segments to download.",
    )
    p.add_argument(
        "--output-dir", "-o",
        default=str(Path(__file__).resolve().parent.parent / "src" / "data"),
        help="Local directory where .tfrecord files are saved.",
    )
    p.add_argument(
        "--bucket",
        default="waymo_open_dataset_v_1_4_2",
        help="GCS bucket name (without gs:// prefix).",
    )
    p.add_argument(
        "--prefix",
        default="individual_files/training/",
        help="Object prefix (sub-directory) inside the bucket.",
    )
    p.add_argument(
        "--filter-suffix",
        default="_with_camera_labels.tfrecord",
        help="Only consider blobs whose name ends with this suffix.",
    )
    p.add_argument(
        "--segment-list",
        default=str(Path(__file__).resolve().parent.parent / "segments_to_run.txt"),
        help="Output path for the generated segments_to_run.txt file.",
    )
    p.add_argument(
        "--seed",
        type=int, default=None,
        help="Random seed for reproducible sampling (omit for true random).",
    )
    p.add_argument(
        "--use-gsutil",
        action="store_true",
        help=(
            "Use the 'gsutil' CLI tool instead of the google-cloud-storage "
            "Python SDK.  Gsutil supports resumable downloads and is often "
            "faster for large files."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List the selected segments without downloading them.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# SDK mode  (google-cloud-storage Python library)
# ---------------------------------------------------------------------------

def _sdk_list_segments(
    bucket_name: str,
    prefix:      str,
    suffix:      str,
) -> list[str]:
    """Return all blob names in the bucket that match the given prefix and suffix."""
    try:
        from google.cloud import storage  # type: ignore[import]
    except ImportError:
        sys.exit(
            "[error] google-cloud-storage is not installed.\n"
            "  Run:  pip install google-cloud-storage\n"
            "  Or use the --use-gsutil flag to rely on the gsutil CLI instead."
        )

    print(f"[sdk] Listing blobs under gs://{bucket_name}/{prefix} …")
    try:
        client = storage.Client()
        blobs  = client.list_blobs(bucket_name, prefix=prefix)
        names  = [b.name for b in blobs if b.name.endswith(suffix)]
    except Exception as exc:
        sys.exit(
            f"[error] GCS listing failed: {exc}\n"
            "  Ensure you are authenticated:\n"
            "    gcloud auth application-default login\n"
            "  And that your account has been granted read access to the bucket."
        )

    print(f"[sdk] Found {len(names)} matching blobs.")
    return names


def _sdk_download(bucket_name: str, blob_name: str, dest_path: str) -> None:
    """Download a single blob to dest_path using the Python SDK."""
    from google.cloud import storage  # already checked in _sdk_list_segments

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
    print(f"  [sdk] Downloading {os.path.basename(blob_name)} …")
    blob.download_to_filename(dest_path)


# ---------------------------------------------------------------------------
# gsutil mode  (subprocess)
# ---------------------------------------------------------------------------

def _gsutil_list_segments(
    bucket_name: str,
    prefix:      str,
    suffix:      str,
) -> list[str]:
    """Return blob names via `gsutil ls` subprocess."""
    _require_gsutil()
    gs_uri = f"gs://{bucket_name}/{prefix}**{suffix}"
    print(f"[gsutil] Listing: {gs_uri} …")
    try:
        result = subprocess.run(
            ["gsutil", "ls", gs_uri],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        sys.exit(
            f"[error] gsutil ls failed:\n{exc.stderr}\n"
            "  Ensure you are authenticated:  gcloud auth login"
        )

    # Each line is a full gs:// URI; strip the bucket prefix to get the blob name.
    gs_prefix = f"gs://{bucket_name}/"
    names = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith(gs_prefix) and line.endswith(suffix):
            names.append(line[len(gs_prefix):])

    print(f"[gsutil] Found {len(names)} matching blobs.")
    return names


def _gsutil_download(bucket_name: str, blob_name: str, dest_path: str) -> None:
    """Download a single blob via `gsutil cp` subprocess."""
    gs_uri = f"gs://{bucket_name}/{blob_name}"
    print(f"  [gsutil] Downloading {os.path.basename(blob_name)} …")
    try:
        subprocess.run(
            ["gsutil", "-q", "cp", gs_uri, dest_path],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        sys.exit(f"[error] gsutil cp failed for {gs_uri}:\n{exc}")


def _require_gsutil() -> None:
    """Abort with a clear message if gsutil is not on PATH."""
    result = subprocess.run(
        ["gsutil", "version"],
        capture_output=True,
    )
    if result.returncode != 0:
        sys.exit(
            "[error] 'gsutil' is not available on your PATH.\n"
            "  Install the Google Cloud SDK:  https://cloud.google.com/sdk/docs/install\n"
            "  Or omit --use-gsutil to use the Python SDK instead."
        )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _sample(names: list[str], n: int, seed: int | None) -> list[str]:
    """Return a random sample of at most n names from the list."""
    if seed is not None:
        random.seed(seed)
    if n >= len(names):
        print(f"[info] Requested {n} segments but only {len(names)} available — using all.")
        return list(names)
    return random.sample(names, n)


def _write_segment_list(segment_list_path: str, local_paths: list[str]) -> None:
    """Overwrite (or create) the segments_to_run.txt file with the downloaded paths."""
    with open(segment_list_path, "w") as f:
        f.write(
            "# segments_to_run.txt — auto-generated by scripts/download_waymo_segments.py\n"
            "# One absolute path per line.  Lines starting with # are ignored.\n\n"
        )
        for p in local_paths:
            f.write(p + "\n")
    print(f"[info] Segment list written → {segment_list_path}  ({len(local_paths)} entries)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Discover available segments ──────────────────────────────────────
    if args.use_gsutil:
        all_blobs = _gsutil_list_segments(args.bucket, args.prefix, args.filter_suffix)
    else:
        all_blobs = _sdk_list_segments(args.bucket, args.prefix, args.filter_suffix)

    if not all_blobs:
        sys.exit(
            "[error] No matching blobs found.\n"
            "  Check --bucket, --prefix, and --filter-suffix values.\n"
            f"  Searched: gs://{args.bucket}/{args.prefix}*{args.filter_suffix}"
        )

    # ── 2. Random sample ────────────────────────────────────────────────────
    selected = _sample(all_blobs, args.num_segments, args.seed)
    print(f"\n[info] Selected {len(selected)} segment(s):")
    for blob in selected:
        print(f"  {os.path.basename(blob)}")

    if args.dry_run:
        print("\n[dry-run] No files downloaded.  Remove --dry-run to proceed.")
        # Still write the list so the user can inspect what would be downloaded.
        local_paths = [
            os.path.join(args.output_dir, os.path.basename(b)) for b in selected
        ]
        _write_segment_list(args.segment_list, local_paths)
        return

    # ── 3. Download ──────────────────────────────────────────────────────────
    print()
    local_paths: list[str] = []
    for i, blob_name in enumerate(selected, start=1):
        fname     = os.path.basename(blob_name)
        dest_path = os.path.join(args.output_dir, fname)

        if os.path.exists(dest_path):
            size_mb = os.path.getsize(dest_path) / 1024 / 1024
            print(f"  [{i}/{len(selected)}] SKIP (already exists, {size_mb:.0f} MB): {fname}")
        else:
            print(f"  [{i}/{len(selected)}] ", end="", flush=True)
            if args.use_gsutil:
                _gsutil_download(args.bucket, blob_name, dest_path)
            else:
                _sdk_download(args.bucket, blob_name, dest_path)
            size_mb = os.path.getsize(dest_path) / 1024 / 1024
            print(f"    → saved ({size_mb:.0f} MB)")

        local_paths.append(dest_path)

    # ── 4. Write segment list ─────────────────────────────────────────────────
    print()
    _write_segment_list(args.segment_list, local_paths)

    print(
        f"\n[done] Run the pipeline with:\n"
        f"  python pipeline_input.py\n"
        f"  (reads {args.segment_list} by default via dataset.segment_list)"
    )


if __name__ == "__main__":
    main()
