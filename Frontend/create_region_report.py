import json
import rasterio
import numpy as np
from geospatial.alignment import check_alignment
from evidence_status import determine_region_status, SUPPORTED, UNCERTAIN, CONFLICTING

def create_region_report(
    cluster_file_path,
    ndvi_file_path,
    ndbi_file_path,
    vv_file_path,
    vh_file_path,
    output_file_path,
    *,
    ndvi_threshold=-0.20,
    ndbi_threshold=0.20,
    vv_threshold=-0.07333232,
    vh_threshold=-0.02110010
):
    """
    Dynamically extract statistics for each cluster region and generate a JSON report.
    Now includes scientific status determination based on pixel-level evidence composition.
    """
    # Enforce alignment
    align1 = check_alignment(cluster_file_path, ndvi_file_path)
    align2 = check_alignment(cluster_file_path, ndbi_file_path)
    align3 = check_alignment(cluster_file_path, vv_file_path)
    align4 = check_alignment(cluster_file_path, vh_file_path)

    if not (align1["aligned"] and align2["aligned"] and align3["aligned"] and align4["aligned"]):
        raise ValueError("Inputs are not perfectly spatially aligned. Run Step 6 alignment first.")

    with rasterio.open(cluster_file_path) as src:
        clusters = src.read(1)
        transform = src.transform
        crs = str(src.crs)

    with rasterio.open(ndvi_file_path) as src:
        ndvi = src.read(1).astype(np.float32)
    with rasterio.open(ndbi_file_path) as src:
        ndbi = src.read(1).astype(np.float32)
    with rasterio.open(vv_file_path) as src:
        vv = src.read(1).astype(np.float32)
    with rasterio.open(vh_file_path) as src:
        vh = src.read(1).astype(np.float32)

    region_ids = np.unique(clusters)
    region_ids = region_ids[region_ids != 0]

    regions = []

    for region_id in region_ids:
        mask = clusters == region_id
        pixels = int(mask.sum())

        if pixels == 0:
            continue

        rows, cols = np.where(mask)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        left, top = rasterio.transform.xy(
            transform, min_row, min_col, offset="ul"
        )
        right, bottom = rasterio.transform.xy(
            transform, max_row, max_col, offset="lr"
        )

        area_m2 = pixels * transform.a * (-transform.e)  # Assumes square pixels correctly
        area_ha = area_m2 / 10000.0

        ndvi_change = float(np.mean(ndvi[mask]))
        ndbi_change = float(np.mean(ndbi[mask]))
        vv_change = float(np.mean(vv[mask]))
        vh_change = float(np.mean(vh[mask]))

        # =====================================================================
        # OPTICAL/SAR EVIDENCE ANALYSIS FOR PIXEL-LEVEL COMPOSITION
        # =====================================================================
        
        # Evidence per indicator within this region
        ndvi_evidence = ndvi[mask] < ndvi_threshold
        ndbi_evidence = ndbi[mask] > ndbi_threshold
        vv_evidence = vv[mask] < vv_threshold
        vh_evidence = vh[mask] < vh_threshold
        
        # Count supporting pixels per indicator
        ndvi_supporting = np.sum(ndvi_evidence)
        ndbi_supporting = np.sum(ndbi_evidence)
        vv_supporting = np.sum(vv_evidence)
        vh_supporting = np.sum(vh_evidence)
        
        # Optical evidence: NDVI decrease OR NDBI increase (either indicates change)
        optical_supporting = ndvi_supporting + ndbi_supporting - np.sum(ndvi_evidence & ndbi_evidence)
        sar_supporting = vv_supporting + vh_supporting - np.sum(vv_evidence & vh_evidence)
        
        # Classification: pixels with strong optical, strong SAR, both, or conflict
        strong_optical = ndvi_evidence | ndbi_evidence
        strong_sar = vv_evidence | vh_evidence
        
        strong_both = strong_optical & strong_sar
        strong_optical_only = strong_optical & ~strong_sar
        strong_sar_only = strong_sar & ~strong_optical
        
        # CONFLICT: optical and SAR disagree
        # We define conflict as pixels where one modality is strong and the other is weak
        optical_strong_sar_weak = (ndvi_evidence | ndbi_evidence) & ~(vv_evidence | vh_evidence)
        sar_strong_optical_weak = (vv_evidence | vh_evidence) & ~(ndvi_evidence | ndbi_evidence)
        
        conflicting_pixels = optical_strong_sar_weak | sar_strong_optical_weak
        
        strong_count = int(np.sum(strong_both))
        weak_optical_only = int(np.sum(strong_optical_only))
        weak_sar_only = int(np.sum(strong_sar_only))
        conflict_count = int(np.sum(conflicting_pixels))
        
        # Fractions
        strong_fraction = strong_count / pixels if pixels > 0 else 0.0
        conflict_fraction = conflict_count / pixels if pixels > 0 else 0.0
        uncertain_fraction = (weak_optical_only + weak_sar_only) / pixels if pixels > 0 else 0.0
        
        # Evidence score (number of indicators supporting change)
        score = sum([
            1 if ndvi_evidence.any() else 0,
            1 if ndbi_evidence.any() else 0,
            1 if vv_evidence.any() else 0,
            1 if vh_evidence.any() else 0
        ])
        
        # Determine region status using scientific rules
        optical_count = (1 if ndvi_evidence.any() else 0) + (1 if ndbi_evidence.any() else 0)
        sar_count = (1 if vv_evidence.any() else 0) + (1 if vh_evidence.any() else 0)
        
        region_status = determine_region_status(
            evidence_score=score,
            strong_pixel_fraction=strong_fraction,
            uncertain_pixel_fraction=uncertain_fraction,
            conflicting_pixel_fraction=conflict_fraction,
            valid_pixel_count=pixels,
            optical_count=optical_count,
            sar_count=sar_count
        )

        regions.append({
            "region_id": int(region_id),
            "pixels": pixels,
            "area_m2": area_m2,
            "area_ha": round(area_ha, 4),
            "bbox_utm": {
                "left": float(left),
                "bottom": float(bottom),
                "right": float(right),
                "top": float(top)
            },
            "crs": crs,
            "changes": {
                "ndvi": round(ndvi_change, 6),
                "ndbi": round(ndbi_change, 6),
                "vv": round(vv_change, 6),
                "vh": round(vh_change, 6)
            },
            "evidence": {
                "ndvi_decrease": bool(ndvi_evidence.any()),
                "ndbi_increase": bool(ndbi_evidence.any()),
                "vv_decrease": bool(vv_evidence.any()),
                "vh_decrease": bool(vh_evidence.any())
            },
            "evidence_score": int(score),
            "pixel_composition": {
                "total_pixels": pixels,
                "strong_consensus_pixels": strong_count,
                "conflicting_pixels": conflict_count,
                "weak_optical_only_pixels": weak_optical_only,
                "weak_sar_only_pixels": weak_sar_only,
                "strong_fraction": round(strong_fraction, 4),
                "conflict_fraction": round(conflict_fraction, 4),
                "uncertain_fraction": round(uncertain_fraction, 4)
            },
            "optical_evidence": {
                "ndvi_supporting_pixels": int(ndvi_supporting),
                "ndbi_supporting_pixels": int(ndbi_supporting),
                "total_supporting": int(optical_supporting)
            },
            "sar_evidence": {
                "vv_supporting_pixels": int(vv_supporting),
                "vh_supporting_pixels": int(vh_supporting),
                "total_supporting": int(sar_supporting)
            },
            "status": region_status
        })

    # Sort largest regions first
    regions.sort(key=lambda x: x["area_ha"], reverse=True)

    report = {
        "project": "SatQuery",
        "region_count": len(regions),
        "regions": regions
    }

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report

if __name__ == "__main__":
    import sys
    print("create_region_report.py should be invoked via orchestration.")
    sys.exit(1)