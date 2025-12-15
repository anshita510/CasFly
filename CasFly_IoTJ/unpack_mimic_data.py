#!/usr/bin/env python3
"""
unpack_mimic_data.py

Prepares raw MIMIC data for use by the CasFly pipeline.

Two modes:
  1. If --src points to a .zip file (e.g. mimic-iv-clinical-database-demo-2.2.zip),
     the archive is extracted safely (path-traversal check included) and the
     resulting directory is used as the source root.
  2. If --src points to a directory, it is used directly.

In both modes, every *.csv.gz file found recursively under the source root is
decompressed to *.csv under --outdir, preserving the original directory
structure relative to the source root.

Inputs:
    --src     MIMIC root folder OR a .zip archive
    --outdir  destination for decompressed CSV files

Outputs:
    <outdir>/.../*.csv  -- decompressed MIMIC CSV files (structure preserved)

Optional flags:
    --overwrite   overwrite existing .csv files (default: skip)
    --delete-gz   delete the original .csv.gz after successful decompression
"""

import argparse
import gzip
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, Tuple, Optional


def secure_extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    """
    Extract a zip archive to dest_dir with path-traversal protection.
    Returns the single top-level subdirectory if the zip contains exactly one.
    """
    import zipfile
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Block any member whose resolved path escapes dest_dir
        for member in zf.infolist():
            member_path = dest_dir / member.filename
            if not str(member_path.resolve()).startswith(str(dest_dir.resolve())):
                raise RuntimeError(f"Blocked unsafe path in zip: {member.filename}")
        zf.extractall(dest_dir)
        # If zip contains a single top-level folder, return that as root
        names            = [n.split("/")[0] for n in zf.namelist() if "/" in n]
        root_candidates  = list(set(names))
        if len(root_candidates) == 1:
            return dest_dir / root_candidates[0]
    return dest_dir

def find_gz_csvs(root: Path) -> Iterable[Path]:
    """Recursively find all *.csv.gz files under root."""
    return root.rglob("*.csv.gz")

def decompress_gz(src_gz: Path, dst_csv: Path, overwrite: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Decompress src_gz to dst_csv.
    Returns (True, None) on success, (False, reason) on skip or error.
    """
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    if dst_csv.exists() and not overwrite:
        return False, "exists"
    try:
        with gzip.open(src_gz, "rb") as f_in, open(dst_csv, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    ap = argparse.ArgumentParser(
        description="Unpack MIMIC: extract zip (optional) and decompress all *.csv.gz to *.csv."
    )
    ap.add_argument("--src", required=True,
                    help="Path to MIMIC root folder OR the ZIP file.")
    ap.add_argument("--outdir", required=True,
                    help="Where to write the decompressed tree.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing .csv files if present.")
    ap.add_argument("--delete-gz", action="store_true",
                    help="Delete the original .csv.gz after successful decompression.")
    args = ap.parse_args()

    src_path = Path(args.src).expanduser().resolve()
    outdir   = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Step 1: extract zip if needed
    if src_path.suffix.lower() == ".zip":
        print(f"[1/3] Extracting zip: {src_path} -> {outdir}")
        extracted_root = secure_extract_zip(src_path, outdir)
        root = extracted_root
        print(f"      Extracted root set to: {root}")
    else:
        root = src_path
        print(f"[1/3] Using source directory: {root}")

    # Step 2: locate all compressed CSVs
    print("[2/3] Scanning for *.csv.gz files...")
    gz_files = sorted(find_gz_csvs(root))
    if not gz_files:
        print("      No *.csv.gz files found. Nothing to do.")
        return
    print(f"      Found {len(gz_files)} compressed CSVs.")

    # Step 3: decompress each, preserving directory structure
    print("[3/3] Decompressing...")
    t0 = time.time()
    created, skipped, failed = 0, 0, 0
    for gz in gz_files:
        rel = gz.relative_to(root)
        dst = outdir / rel.with_suffix("")  # strip .gz to yield .csv
        ok, err = decompress_gz(gz, dst, overwrite=args.overwrite)
        if ok:
            created += 1
            if args.delete_gz:
                try:
                    gz.unlink()
                except Exception as e:
                    print(f"      Warning: could not delete {gz}: {e}")
        else:
            if err == "exists":
                skipped += 1
            else:
                failed += 1
                print(f"      ERROR decompressing {gz} -> {dst}: {err}")

    dt = time.time() - t0
    print(f"Done in {dt:.1f}s -- created: {created}, skipped: {skipped}, failed: {failed}")

if __name__ == "__main__":
    main()
