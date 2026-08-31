#!/usr/bin/env python3
"""
Comprehensive validation pipeline and integrity audit for BigEarthNet.txt.

Scans manifests and physical GeoTIFF files on disk to verify:
- S1 band completeness (VV, VH)
- S2 band completeness (12 multispectral bands)
- S1/S2 co-registration pairing validity
- Raster file readability, no-data corruptions, and dimension correctness
- Spatial coordinates and environmental metadata integrity
- Annotation completeness

Usage:
    python scripts/validate_bigearthnet.py \
        --manifest data/manifests/manifest_full.jsonl \
        --data-root data/bigearthnet_txt \
        --report-output reports/validation_report.json
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.bigearthnet_txt.constants import S1_BAND_NAMES, S2_BAND_NAMES
from data.bigearthnet_txt.parser import BigEarthNetParser
from data.bigearthnet_txt.utils import (
    read_geotiff,
    validate_coordinates,
    validate_patch_bands,
    validate_s1_s2_pairing,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validate_bigearthnet")


def validate_sample(
    sample: Dict[str, Any],
    data_root: Path,
    check_raster_content: bool = True,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Perform rigorous multi-level validation on a single dataset sample.
    
    Returns:
        (is_valid, list_of_error_codes, detail_metrics_dict)
    """
    errors: List[str] = []
    details: Dict[str, Any] = {
        "s1_present": False,
        "s2_present": False,
        "annotation_present": False,
        "pairing_valid": False,
        "metadata_valid": False,
        "raster_valid": False,
    }

    patch_id = sample.get("image_id") or sample.get("patch_id", "")
    s1_name = sample.get("s1_name", "")
    text_in = sample.get("text_input") or sample.get("input", "")
    text_out = sample.get("text_output") or sample.get("output", "")
    metadata = sample.get("metadata", {})

    # 1. Annotation & Text Validation
    if text_in and str(text_in).strip():
        details["annotation_present"] = True
    else:
        errors.append("MISSING_INPUT_ANNOTATION")

    if not text_out or not str(text_out).strip():
        errors.append("MISSING_OUTPUT_ANNOTATION")

    # 2. Metadata Validation
    lat = metadata.get("latitude") if isinstance(metadata, dict) else sample.get("latitude")
    lon = metadata.get("longitude") if isinstance(metadata, dict) else sample.get("longitude")
    if lat is not None and lon is not None:
        coord_valid, coord_err = validate_coordinates(lat, lon)
        if coord_valid:
            details["metadata_valid"] = True
        else:
            errors.append(f"INVALID_COORDINATES: {coord_err}")
    else:
        errors.append("MISSING_COORDINATES")

    # 3. Pairing Validation
    pair_valid, pair_err = validate_s1_s2_pairing(s1_name, patch_id)
    if pair_valid:
        details["pairing_valid"] = True
    else:
        errors.append(f"INVALID_PAIRING: {pair_err}")

    # 4. Physical File & Directory Checks
    s1_path_rel = sample.get("s1_path")
    s2_path_rel = sample.get("s2_path")

    # Resolve S1 folder
    s1_dir: Optional[Path] = None
    if s1_path_rel:
        p = data_root / s1_path_rel
        if p.exists():
            s1_dir = p
        elif Path(s1_path_rel).exists():
            s1_dir = Path(s1_path_rel)
    if s1_dir is None and s1_name:
        candidates = [
            data_root / "images_s1" / s1_name,
            data_root / "s1" / s1_name,
            data_root / s1_name,
        ]
        for c in candidates:
            if c.exists():
                s1_dir = c
                break

    # Resolve S2 folder
    s2_dir: Optional[Path] = None
    if s2_path_rel:
        p = data_root / s2_path_rel
        if p.exists():
            s2_dir = p
        elif Path(s2_path_rel).exists():
            s2_dir = Path(s2_path_rel)
    if s2_dir is None and patch_id:
        candidates = [
            data_root / "images_s2" / patch_id,
            data_root / "s2" / patch_id,
            data_root / patch_id,
        ]
        for c in candidates:
            if c.exists():
                s2_dir = c
                break

    # Validate S1 bands
    if s1_dir and s1_dir.exists():
        s1_res = validate_patch_bands(s1_dir, S1_BAND_NAMES, prefix=s1_name)
        if s1_res["valid"]:
            details["s1_present"] = True
        else:
            if s1_res["missing_bands"]:
                errors.append(f"MISSING_S1_BANDS: {s1_res['missing_bands']}")
            if s1_res["corrupted_bands"]:
                errors.append(f"CORRUPTED_S1_BANDS: {list(s1_res['corrupted_bands'].keys())}")
    else:
        errors.append(f"MISSING_S1_DIR: {s1_name}")

    # Validate S2 bands
    if s2_dir and s2_dir.exists():
        s2_res = validate_patch_bands(s2_dir, S2_BAND_NAMES, prefix=patch_id)
        if s2_res["valid"]:
            details["s2_present"] = True
        else:
            if s2_res["missing_bands"]:
                errors.append(f"MISSING_S2_BANDS: {s2_res['missing_bands']}")
            if s2_res["corrupted_bands"]:
                errors.append(f"CORRUPTED_S2_BANDS: {list(s2_res['corrupted_bands'].keys())}")
    else:
        errors.append(f"MISSING_S2_DIR: {patch_id}")

    if details["s1_present"] and details["s2_present"]:
        details["raster_valid"] = True

    is_valid = len(errors) == 0
    return is_valid, errors, details


def run_validation(
    manifest_path: Optional[str] = None,
    data_root: str = "data/bigearthnet_txt",
    report_output: str = "reports/validation_report.json",
    invalid_log: str = "reports/invalid_samples.jsonl",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute full dataset validation.
    """
    start_time = time.time()
    data_root_path = Path(data_root)
    report_out_path = Path(report_output)
    invalid_log_path = Path(invalid_log)

    report_out_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_log_path.parent.mkdir(parents=True, exist_ok=True)

    samples: List[Dict[str, Any]] = []
    if manifest_path and Path(manifest_path).exists():
        logger.info(f"Loading samples from manifest: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line.strip()))
    else:
        logger.info(f"No manifest provided or found. Parsing data root directly: {data_root_path}")
        parser = BigEarthNetParser(data_root=data_root_path)
        samples = list(parser.parse_records(validate_files=False, max_samples=max_samples))

    if max_samples is not None and len(samples) > max_samples:
        samples = samples[:max_samples]

    total_samples = len(samples)
    logger.info(f"Validating {total_samples} samples...")

    valid_count = 0
    invalid_count = 0
    s1_available_count = 0
    s2_available_count = 0
    annotation_available_count = 0
    pairing_valid_count = 0
    metadata_valid_count = 0
    split_counts: Dict[str, int] = {}
    error_summary: Dict[str, int] = {}

    invalid_records: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        is_valid, errors, details = validate_sample(sample, data_root_path)

        sp = sample.get("split", "unknown")
        split_counts[sp] = split_counts.get(sp, 0) + 1

        if details["s1_present"]:
            s1_available_count += 1
        if details["s2_present"]:
            s2_available_count += 1
        if details["annotation_present"]:
            annotation_available_count += 1
        if details["pairing_valid"]:
            pairing_valid_count += 1
        if details["metadata_valid"]:
            metadata_valid_count += 1

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            for err in errors:
                error_prefix = err.split(":")[0]
                error_summary[error_prefix] = error_summary.get(error_prefix, 0) + 1

            invalid_entry = {
                "sample_id": sample.get("sample_id") or sample.get("ID", f"idx_{idx}"),
                "image_id": sample.get("image_id") or sample.get("patch_id", ""),
                "s1_name": sample.get("s1_name", ""),
                "split": sp,
                "errors": errors,
            }
            invalid_records.append(invalid_entry)

    # Write invalid samples log
    with open(invalid_log_path, "w", encoding="utf-8") as f:
        for r in invalid_records:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Saved invalid samples log -> {invalid_log_path} ({len(invalid_records)} entries)")

    elapsed = time.time() - start_time

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "data_root": str(data_root_path),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "metrics": {
            "total_samples": total_samples,
            "valid_samples": valid_count,
            "invalid_samples": invalid_count,
            "valid_percentage": round((valid_count / total_samples * 100.0) if total_samples > 0 else 0.0, 2),
            "s1_availability": s1_available_count,
            "s1_availability_pct": round((s1_available_count / total_samples * 100.0) if total_samples > 0 else 0.0, 2),
            "s2_availability": s2_available_count,
            "s2_availability_pct": round((s2_available_count / total_samples * 100.0) if total_samples > 0 else 0.0, 2),
            "annotation_availability": annotation_available_count,
            "annotation_availability_pct": round((annotation_available_count / total_samples * 100.0) if total_samples > 0 else 0.0, 2),
            "pairing_validity": pairing_valid_count,
            "pairing_validity_pct": round((pairing_valid_count / total_samples * 100.0) if total_samples > 0 else 0.0, 2),
            "metadata_validity": metadata_valid_count,
            "metadata_validity_pct": round((metadata_valid_count / total_samples * 100.0) if total_samples > 0 else 0.0, 2),
        },
        "split_counts": split_counts,
        "error_distribution": error_summary,
    }

    with open(report_out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved validation audit report -> {report_out_path}")

    return report


def print_report_table(report: Dict[str, Any]) -> None:
    m = report["metrics"]
    splits = report["split_counts"]
    errors = report["error_distribution"]

    print("\n" + "=" * 70)
    print("           BIGEARTHNET.TXT PRODUCTION VALIDATION REPORT           ")
    print("=" * 70)
    print(f"Timestamp:             {report['timestamp']}")
    print(f"Elapsed Time:          {report['elapsed_seconds']}s")
    print("-" * 70)
    print(f"Total Samples:         {m['total_samples']:>10}")
    print(f"Valid Samples:         {m['valid_samples']:>10} ({m['valid_percentage']}%)")
    print(f"Invalid Samples:       {m['invalid_samples']:>10}")
    print(f"S1 SAR Availability:   {m['s1_availability']:>10} ({m['s1_availability_pct']}%)")
    print(f"S2 MS Availability:    {m['s2_availability']:>10} ({m['s2_availability_pct']}%)")
    print(f"Annotation Avail.:     {m['annotation_availability']:>10} ({m['annotation_availability_pct']}%)")
    print(f"Pairing Validity:      {m['pairing_validity']:>10} ({m['pairing_validity_pct']}%)")
    print(f"Metadata Validity:     {m['metadata_validity']:>10} ({m['metadata_validity_pct']}%)")
    print("-" * 70)
    print("Dataset Split Distribution:")
    for sp_name, count in splits.items():
        print(f"  - {sp_name:<16}: {count:>8}")
    if errors:
        print("-" * 70)
        print("Detected Error Breakdown:")
        for err_type, count in errors.items():
            print(f"  - {err_type:<24}: {count:>6}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run production validation and integrity audit for BigEarthNet.txt"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/manifest_full.jsonl",
        help="Path to manifest file to validate (default: data/manifests/manifest_full.jsonl)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/bigearthnet_txt",
        help="Root directory containing imagery data (default: data/bigearthnet_txt)",
    )
    parser.add_argument(
        "--report-output",
        type=str,
        default="reports/validation_report.json",
        help="Output path for JSON validation report (default: reports/validation_report.json)",
    )
    parser.add_argument(
        "--invalid-log",
        type=str,
        default="reports/invalid_samples.jsonl",
        help="Output path for invalid samples log (default: reports/invalid_samples.jsonl)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to validate",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status code if invalid samples are encountered",
    )

    args = parser.parse_args()

    report = run_validation(
        manifest_path=args.manifest if Path(args.manifest).exists() else None,
        data_root=args.data_root,
        report_output=args.report_output,
        invalid_log=args.invalid_log,
        max_samples=args.max_samples,
    )

    print_report_table(report)

    if args.strict and report["metrics"]["invalid_samples"] > 0:
        logger.error(f"Strict validation failed with {report['metrics']['invalid_samples']} invalid samples.")
        sys.exit(1)


if __name__ == "__main__":
    main()
