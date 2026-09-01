import json
from pathlib import Path
from evidence_schema import EvidenceItem
from evidence_status import SUPPORTED, UNCERTAIN, CONFLICTING, INSUFFICIENT

def create_evidence_record(
    summary_file,
    regions_file,
    output_file,
    before_year,
    after_year,
    ndvi_source="Sentinel-2",
    ndbi_source="Sentinel-2",
    vv_source="Sentinel-1",
    vh_source="Sentinel-1"
):
    """
    Create a comprehensive evidence record from summary and regions data.
    Includes status, interpretation, and full provenance for each evidence item.
    """
    with open(summary_file, "r") as f:
        summary = json.load(f)

    with open(regions_file, "r") as f:
        regions_data = json.load(f)

    # Support both GeoJSON features and raw regions list
    if "features" in regions_data:
        regions_list = [f.get("properties", {}) for f in regions_data.get("features", [])]
    else:
        regions_list = regions_data.get("regions", [])

    strong_regions = []
    
    # Build evidence items from actual region data with status
    evidence_items = []
    for region in regions_list:
        region_id = region.get("region_id")
        score = region.get("evidence_score", 0)
        status = region.get("status", UNCERTAIN)
        area_ha = region.get("area_ha", 0)
        pixels = region.get("pixels", 0)
        
        # Only include high-confidence regions in summary
        if score >= 3:
            strong_regions.append({
                "region_id": region_id,
                "area_ha": area_ha,
                "evidence_score": score,
                "status": status
            })

        # Extract actual values from region
        ndvi_change = region.get("changes", {}).get("ndvi")
        ndbi_change = region.get("changes", {}).get("ndbi")
        vv_change = region.get("changes", {}).get("vv")
        vh_change = region.get("changes", {}).get("vh")
        
        # Get pixel composition for quality metrics
        pixel_composition = region.get("pixel_composition", {})
        strong_pixels = pixel_composition.get("strong_consensus_pixels", 0)
        total_pixels = pixel_composition.get("total_pixels", pixels)
        usable_fraction = (strong_pixels / total_pixels) if total_pixels > 0 else 0.0
        
        # Create evidence items with status and interpretation
        
        # NDVI Evidence
        if ndvi_change is not None:
            ndvi_evidence = region.get("evidence", {}).get("ndvi_decrease", False)
            ndvi_interpretation = (
                f"NDVI decreased by {ndvi_change:.4f}, indicating vegetation loss"
                if ndvi_evidence
                else f"NDVI increased slightly ({ndvi_change:.4f}), indicating vegetation recovery or no significant change"
            )
            
            evidence_items.append(EvidenceItem(
                evidence_id=f"reg{region_id}_ndvi",
                type="NDVI",
                source=ndvi_source,
                region=region_id,
                value=ndvi_change,
                quality="PRESENT" if ndvi_evidence else "WEAK",
                usable_pixel_fraction=usable_fraction,
                registration_ok=True,
                provenance={
                    "sensor": ndvi_source,
                    "source_raster": "ndvi_change.tif",
                    "before_year": before_year,
                    "after_year": after_year,
                    "processing_method": "NDVI change",
                    "threshold": -0.20,
                    "interpretation": ndvi_interpretation,
                    "supporting_pixels": int(region.get("optical_evidence", {}).get("ndvi_supporting_pixels", 0))
                }
            ))
        
        # NDBI Evidence
        if ndbi_change is not None:
            ndbi_evidence = region.get("evidence", {}).get("ndbi_increase", False)
            ndbi_interpretation = (
                f"NDBI increased by {ndbi_change:.4f}, indicating built-up/urban area increase"
                if ndbi_evidence
                else f"NDBI remained stable ({ndbi_change:.4f}), indicating no significant built-up area change"
            )
            
            evidence_items.append(EvidenceItem(
                evidence_id=f"reg{region_id}_ndbi",
                type="NDBI",
                source=ndbi_source,
                region=region_id,
                value=ndbi_change,
                quality="PRESENT" if ndbi_evidence else "WEAK",
                usable_pixel_fraction=usable_fraction,
                registration_ok=True,
                provenance={
                    "sensor": ndbi_source,
                    "source_raster": "ndbi_change.tif",
                    "before_year": before_year,
                    "after_year": after_year,
                    "processing_method": "NDBI change",
                    "threshold": 0.20,
                    "interpretation": ndbi_interpretation,
                    "supporting_pixels": int(region.get("optical_evidence", {}).get("ndbi_supporting_pixels", 0))
                }
            ))
        
        # SAR VV Evidence
        if vv_change is not None:
            vv_evidence = region.get("evidence", {}).get("vv_decrease", False)
            vv_interpretation = (
                f"VV backscatter decreased by {vv_change:.6f}, indicating potential physical change"
                if vv_evidence
                else f"VV backscatter stable ({vv_change:.6f}), suggesting no significant backscatter change"
            )
            
            evidence_items.append(EvidenceItem(
                evidence_id=f"reg{region_id}_vv",
                type="SAR_VV",
                source=vv_source,
                region=region_id,
                value=vv_change,
                quality="PRESENT" if vv_evidence else "WEAK",
                usable_pixel_fraction=usable_fraction,
                registration_ok=True,
                provenance={
                    "sensor": vv_source,
                    "source_raster": "vv_change.tif",
                    "before_year": before_year,
                    "after_year": after_year,
                    "processing_method": "VV change",
                    "threshold": -0.07333232,
                    "interpretation": vv_interpretation,
                    "supporting_pixels": int(region.get("sar_evidence", {}).get("vv_supporting_pixels", 0)),
                    "note": "SAR change alone does not prove construction/destruction; supporting evidence required"
                }
            ))
        
        # SAR VH Evidence
        if vh_change is not None:
            vh_evidence = region.get("evidence", {}).get("vh_decrease", False)
            vh_interpretation = (
                f"VH backscatter decreased by {vh_change:.6f}, indicating potential structural change"
                if vh_evidence
                else f"VH backscatter stable ({vh_change:.6f}), suggesting no significant change"
            )
            
            evidence_items.append(EvidenceItem(
                evidence_id=f"reg{region_id}_vh",
                type="SAR_VH",
                source=vh_source,
                region=region_id,
                value=vh_change,
                quality="PRESENT" if vh_evidence else "WEAK",
                usable_pixel_fraction=usable_fraction,
                registration_ok=True,
                provenance={
                    "sensor": vh_source,
                    "source_raster": "vh_change.tif",
                    "before_year": before_year,
                    "after_year": after_year,
                    "processing_method": "VH change",
                    "threshold": -0.02110010,
                    "interpretation": vh_interpretation,
                    "supporting_pixels": int(region.get("sar_evidence", {}).get("vh_supporting_pixels", 0)),
                    "note": "VH cross-polarization sensitive to vegetation and structural changes"
                }
            ))

    record = {
        "record_type": "satquery_change_evidence",

        "period": {
            "before": before_year,
            "after": after_year,
        },

        "spatial_reference": {
            "crs": summary["crs"],
            "width": summary["width"],
            "height": summary["height"],
            "resolution": summary["resolution"]
        },

        "evidence": {
            "valid_pixels": summary["valid_pixels"],
            "nodata_pixels": summary["nodata_pixels"],
            "score_counts": summary["score_counts"],
            "strong_evidence": summary["strong_evidence"]
        },

        "classification": {
            "strong": summary.get("classification", {}).get("strong", 0),
            "uncertain": summary.get("classification", {}).get("uncertain", 0),
            "other": summary.get("classification", {}).get("other", 0)
        },

        "regions": {
            "total": len(regions_list),
            "strong_evidence_regions": strong_regions
        },

        "provenance": {
            **summary.get("provenance", {}),
            "sensors": [
                ndvi_source,
                ndbi_source,
                vv_source,
                vh_source
            ],
            "indicators": [
                "NDVI",
                "NDBI",
                "VV",
                "VH"
            ],
            "thresholds": {
                "ndvi_decrease": -0.20,
                "ndbi_increase": 0.20,
                "vv_decrease": -0.07333232,
                "vh_decrease": -0.02110010
            },
            "scientific_basis": {
                "ndvi": "Normalized Difference Vegetation Index; negative change indicates vegetation loss",
                "ndbi": "Normalized Difference Built-up Index; positive change indicates urban/built-up area increase",
                "sar_vv": "SAR VV polarization backscatter; decrease may indicate structural change or vegetation loss",
                "sar_vh": "SAR VH cross-polarization; sensitive to vegetation and structural complexity changes"
            }
        },
        "evidence_items": [item.to_dict() for item in evidence_items]
    }

    with open(output_file, "w") as f:
        json.dump(record, f, indent=2)

    return record


if __name__ == "__main__":
    import sys
    print("create_evidence_record.py should be invoked via orchestration.")
    sys.exit(1)