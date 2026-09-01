#!/usr/bin/env python3
"""
Build reproducible, deterministic dataset manifests for BigEarthNet.txt.

Generates split manifests (train, val, test, bench) with zero spatial patch leakage,
preserves complete remote sensing metadata, and produces validation statistics.

Usage:
    python scripts/build_bigearthnet_manifest.py \
        --data-root data/bigearthnet_txt \
        --output-dir data/manifests \
        --seed 42
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.bigearthnet_txt.parser import BigEarthNetParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_manifest")


def deterministic_patch_split(
    patch_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Deterministically assign unique patch IDs to splits based on a fixed random seed.
    
    All samples sharing the same patch_id will belong to the exact same split,
    guaranteeing zero spatial or multi-sensor data leakage.
    
    Returns:
        Dict mapping patch_id -> split_name ('train', 'validation', 'test').
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    assert np.isclose(total_ratio, 1.0), f"Ratios must sum to 1.0, got {total_ratio}"

    unique_patches = sorted(list(set(patch_ids)))
    rng = np.random.default_rng(seed)
    shuffled_patches = list(unique_patches)
    rng.shuffle(shuffled_patches)

    n_total = len(shuffled_patches)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    patch_to_split: Dict[str, str] = {}
    for i, p in enumerate(shuffled_patches):
        if i < n_train:
            patch_to_split[p] = "train"
        elif i < n_train + n_val:
            patch_to_split[p] = "validation"
        else:
            patch_to_split[p] = "test"

    return patch_to_split


def build_manifest(
    data_root: str = "data/bigearthnet_txt",
    output_dir: str = "data/manifests",
    metadata_file: Optional[str] = None,
    s1_dir: Optional[str] = None,
    s2_dir: Optional[str] = None,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    use_official_splits: bool = True,
    validate_files: bool = False,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute manifest generation and export split files.
    """
    start_time = time.time()
    data_root_path = Path(data_root)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Parsing BigEarthNet dataset from: {data_root_path}")
    parser = BigEarthNetParser(
        data_root=data_root_path,
        metadata_file=metadata_file,
        s1_dir=s1_dir,
        s2_dir=s2_dir,
    )

    logger.info("Extracting records...")
    records = list(parser.parse_records(validate_files=validate_files, max_samples=max_samples))
    total_records = len(records)
    logger.info(f"Loaded {total_records} records from dataset.")

    if total_records == 0:
        raise ValueError(f"No records found in {data_root_path}. Please check data path and metadata.")

    # Check if dataset has official splits
    has_official_splits = any(r.get("split") in ("train", "validation", "test", "bench") for r in records)

    if use_official_splits and has_official_splits:
        logger.info("Preserving official BigEarthNet.txt splits (train/val/test/bench).")
    else:
        logger.info(f"Generating deterministic patch-level splits (seed={seed}, ratios: {train_ratio}/{val_ratio}/{test_ratio})...")
        patch_ids = [r["image_id"] for r in records]
        patch_split_map = deterministic_patch_split(
            patch_ids=patch_ids,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        for r in records:
            r["split"] = patch_split_map.get(r["image_id"], "train")

    # Group into splits
    split_records: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
        "bench": [],
    }

    valid_count = 0
    invalid_count = 0
    task_types_dist: Dict[str, int] = {}
    categories_dist: Dict[str, int] = {}
    countries_dist: Dict[str, int] = {}

    for r in records:
        sp = r.get("split", "train")
        if sp in ("val", "valid"):
            sp = "validation"
            r["split"] = "validation"
        if sp not in split_records:
            split_records[sp] = []
        split_records[sp].append(r)

        if r.get("is_valid", True):
            valid_count += 1
        else:
            invalid_count += 1

        t_type = r.get("task_type", "unknown")
        task_types_dist[t_type] = task_types_dist.get(t_type, 0) + 1

        t_cat = r.get("task_category", "unknown")
        categories_dist[t_cat] = categories_dist.get(t_cat, 0) + 1

        country = r.get("metadata", {}).get("country", "unknown") or "unknown"
        countries_dist[country] = countries_dist.get(country, 0) + 1

    # Verify split isolation (zero patch overlap)
    train_patches = {r["image_id"] for r in split_records["train"]}
    val_patches = {r["image_id"] for r in split_records["validation"]}
    test_patches = {r["image_id"] for r in split_records["test"]}
    bench_patches = {r["image_id"] for r in split_records["bench"]}

    overlap_train_val = train_patches.intersection(val_patches)
    overlap_train_test = train_patches.intersection(test_patches)
    overlap_val_test = val_patches.intersection(test_patches)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise RuntimeError(
            f"Split leakage detected! Overlap sizes: train-val={len(overlap_train_val)}, "
            f"train-test={len(overlap_train_test)}, val-test={len(overlap_val_test)}"
        )
    logger.info("Split isolation verified: 0% patch overlap between train, validation, and test.")

    # Write output manifest files
    # 1. Full manifest
    full_manifest_path = output_dir_path / "manifest_full.jsonl"
    with open(full_manifest_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Saved full manifest -> {full_manifest_path} ({total_records} samples)")

    # 2. Split manifests
    for sp_name, sp_recs in split_records.items():
        if not sp_recs and sp_name == "bench":
            continue
        out_path = output_dir_path / f"manifest_{sp_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in sp_recs:
                f.write(json.dumps(r) + "\n")
        logger.info(f"Saved split manifest -> {out_path} ({len(sp_recs)} samples, {len({r['image_id'] for r in sp_recs})} patches)")

    elapsed = time.time() - start_time

    # 3. Summary metadata
    summary = {
        "dataset_name": "BigEarthNet.txt",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "random_seed": seed,
        "total_samples": total_records,
        "total_unique_patches": len({r["image_id"] for r in records}),
        "valid_samples": valid_count,
        "invalid_samples": invalid_count,
        "split_counts": {k: len(v) for k, v in split_records.items() if len(v) > 0},
        "split_patches": {
            "train": len(train_patches),
            "validation": len(val_patches),
            "test": len(test_patches),
            "bench": len(bench_patches),
        },
        "task_types": task_types_dist,
        "categories": categories_dist,
        "countries": countries_dist,
    }

    summary_path = output_dir_path / "manifest_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary statistics -> {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Build reproducible deterministic manifests for BigEarthNet.txt"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/bigearthnet_txt",
        help="Root directory containing BigEarthNet files (default: data/bigearthnet_txt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/manifests",
        help="Output directory for manifests (default: data/manifests)",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default=None,
        help="Path to BigEarthNet.txt.parquet/csv/jsonl (default: auto-discover in data-root)",
    )
    parser.add_argument(
        "--s1-dir",
        type=str,
        default=None,
        help="Directory containing Sentinel-1 patch folders (default: auto-discover)",
    )
    parser.add_argument(
        "--s2-dir",
        type=str,
        default=None,
        help="Directory containing Sentinel-2 patch folders (default: auto-discover)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting (default: 42)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training split ratio (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--force-resplit",
        action="store_true",
        help="Force deterministic patch re-splitting even if official split column exists",
    )
    parser.add_argument(
        "--validate-files",
        action="store_true",
        help="Validate existence and integrity of GeoTIFF band files on disk during build",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process (useful for development/testing)",
    )

    args = parser.parse_args()

    summary = build_manifest(
        data_root=args.data_root,
        output_dir=args.output_dir,
        metadata_file=args.metadata_file,
        s1_dir=args.s1_dir,
        s2_dir=args.s2_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        use_official_splits=not args.force_resplit,
        validate_files=args.validate_files,
        max_samples=args.max_samples,
    )

    print("\n" + "=" * 60)
    print("MANIFEST GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total Samples:       {summary['total_samples']}")
    print(f"Unique Patches:      {summary['total_unique_patches']}")
    print(f"Valid Samples:       {summary['valid_samples']}")
    print(f"Invalid Samples:     {summary['invalid_samples']}")
    print(f"Splits:              {summary['split_counts']}")
    print(f"Output Directory:    {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
