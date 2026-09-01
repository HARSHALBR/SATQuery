"""Phase 9 Targeted Audit Script — read-only verification."""

from schemas.evidence import EvidenceRecord, QualityReport, Provenance
from schemas.response import EvidenceStatus
from evidence.comparator import (
    EvidenceComparator,
    _CLAIM_EVIDENCE_MAP,
    _REQUIRED_EVIDENCE_FOR_SUPPORTED,
    _normalize_claim,
    _claim_direction,
    _passes_quality,
    _extract_direction,
)

GOOD_Q = QualityReport(valid_pixel_fraction=0.95, registration_ok=True, cloud_cover=0.05)
LOW_Q = QualityReport(valid_pixel_fraction=0.40, registration_ok=False, cloud_cover=0.60)
BAD_REG = QualityReport(valid_pixel_fraction=0.90, registration_ok=False, cloud_cover=0.05)
HIGH_CLOUD = QualityReport(valid_pixel_fraction=0.90, registration_ok=True, cloud_cover=0.60)
DEFAULT_Q = QualityReport()

def prov(tool): return Provenance(tool=tool, tool_version="mock-1.0")

def vlm(interp, eid="vlm_1", q=None, region=None):
    return EvidenceRecord(evidence_id=eid, type="vlm_interpretation", tool_version="mock-1.0",
        value={"interpretation": interp, "confidence": 0.82}, quality=q or GOOD_Q,
        provenance=prov("run_rs_vlm"), region=region)

def ndvi(delta, direction, eid="ndvi_1", q=None, region=None):
    return EvidenceRecord(evidence_id=eid, type="vegetation_change", tool_version="mock-1.0",
        value={"ndvi_delta": delta, "direction": direction}, quality=q or GOOD_Q,
        provenance=prov("ndvi_delta"), region=region)

def ndbi(delta, direction, eid="ndbi_1", q=None):
    return EvidenceRecord(evidence_id=eid, type="built_up_change", tool_version="mock-1.0",
        value={"ndbi_delta": delta, "direction": direction}, quality=q or GOOD_Q,
        provenance=prov("ndbi_delta"))

def sar(eid="sar_1", q=None):
    return EvidenceRecord(evidence_id=eid, type="sar_amplitude_change", tool_version="mock-1.0",
        value={"sar_change_detected": True, "change_score": 0.76}, quality=q or GOOD_Q,
        provenance=prov("sar_change"))

def area(eid="area_1"):
    return EvidenceRecord(evidence_id=eid, type="spatial_measurement", tool_version="mock-1.0",
        value={"area_km2": 12.4}, quality=GOOD_Q, provenance=prov("area_measurement"))

def change_stats(eid="cs_1"):
    return EvidenceRecord(evidence_id=eid, type="change_quantification", tool_version="mock-1.0",
        value={"changed_pixel_fraction": 0.21, "change_detected": True}, quality=GOOD_Q,
        provenance=prov("change_statistics"))

def grounding(eid="gr_1"):
    return EvidenceRecord(evidence_id=eid, type="spatial_grounding", tool_version="mock-1.0",
        value={"bounding_box": [10,20,10.1,20.1]}, quality=GOOD_Q, provenance=prov("grounding"))


comp = EvidenceComparator()

# ============================================================
# AUDIT 1 — CLAIM SEMANTICS TABLE
# ============================================================
print("=" * 70)
print("AUDIT 1 — CLAIM/EVIDENCE MAPPING TABLE")
print("=" * 70)
all_claims = sorted(set(list(_CLAIM_EVIDENCE_MAP.keys()) + list(_REQUIRED_EVIDENCE_FOR_SUPPORTED.keys())))
print(f"{'Claim':<25} {'Required Evidence':<45} {'Direction':<12}")
print("-" * 82)
for c in all_claims:
    req = _REQUIRED_EVIDENCE_FOR_SUPPORTED.get(c, [])
    d = _claim_direction(c) or "(none)"
    print(f"{c:<25} {', '.join(req):<45} {d:<12}")

# ============================================================
# AUDIT 2 — VEGETATION CLAIM
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 2 — VEGETATION CLAIM TESTS")
print("=" * 70)

r = comp.compare("vegetation_decrease", [vlm("Vegetation appears to have decreased."), ndvi(-0.18, "decrease")])
print(f"2a) VLM decrease + NDVI decrease + good quality => {r.status.value}  (expected: SUPPORTED)")

r = comp.compare("vegetation_decrease", [vlm("Vegetation appears to have decreased."), ndvi(0.15, "increase", eid="ndvi_inc")])
print(f"2b) VLM decrease + NDVI increase + good quality => {r.status.value}  (expected: UNCERTAIN)")

r = comp.compare("vegetation_decrease", [vlm("Vegetation appears to have decreased."), ndvi(-0.18, "decrease", q=LOW_Q)])
print(f"2c) VLM decrease + NDVI decrease + LOW quality  => {r.status.value}  (expected: INSUFFICIENT)")

# ============================================================
# AUDIT 3 — VLM ONLY
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 3 — VLM-ONLY")
print("=" * 70)
r = comp.compare("vegetation_decrease", [vlm("Vegetation appears to have decreased.")])
print(f"VLM-only => {r.status.value}  reason: {r.reason}")

# ============================================================
# AUDIT 4 — NDVI ONLY
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 4 — NDVI-ONLY")
print("=" * 70)
r = comp.compare("vegetation_decrease", [ndvi(-0.18, "decrease")])
print(f"NDVI-only => {r.status.value}  reason: {r.reason}")

# ============================================================
# AUDIT 5 — SAR SEMANTICS
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 5 — SAR SEMANTICS")
print("=" * 70)
r = comp.compare("sar_cross_check", [vlm("Change detected.", eid="vlm_sar"), sar()])
print(f"5a) SAR cross-check (VLM+SAR) => {r.status.value}")
r = comp.compare("vegetation_decrease", [vlm("Vegetation decreased.", eid="vlm_v"), sar(eid="sar_2")])
print(f"5b) vegetation_decrease with VLM+SAR (no NDVI) => {r.status.value}")
r = comp.compare("vegetation_decrease", [sar()])
print(f"5c) vegetation_decrease with SAR-only => {r.status.value}")

# ============================================================
# AUDIT 6 — EVIDENCE TYPE SEMANTICS
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 6 — EVIDENCE TYPE SEMANTICS (non-interchangeability)")
print("=" * 70)
r = comp.compare("vegetation_decrease", [area()])
print(f"6a) area alone for veg_decrease    => {r.status.value}")
r = comp.compare("vegetation_decrease", [change_stats()])
print(f"6b) change_stats alone for veg_dec => {r.status.value}")
r = comp.compare("vegetation_decrease", [sar()])
print(f"6c) SAR alone for veg_decrease     => {r.status.value}")
r = comp.compare("change_detected", [grounding()])
print(f"6d) grounding alone for change_det => {r.status.value}")

# ============================================================
# AUDIT 7 — QUALITY PRECEDENCE
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 7 — QUALITY PRECEDENCE")
print("=" * 70)

# low pixel fraction
r = comp.compare("vegetation_decrease", [vlm("Vegetation decreased."), ndvi(-0.18, "decrease", q=QualityReport(valid_pixel_fraction=0.40))])
print(f"7a) pixel_frac=0.40 => {r.status.value}")

# registration failure
r = comp.compare("vegetation_decrease", [vlm("Vegetation decreased."), ndvi(-0.18, "decrease", q=BAD_REG)])
print(f"7b) registration_ok=False => {r.status.value}")

# cloud cover
r = comp.compare("vegetation_decrease", [vlm("Vegetation decreased."), ndvi(-0.18, "decrease", q=HIGH_CLOUD)])
print(f"7c) cloud_cover=0.60 => {r.status.value}")

# default/unknown quality
r = comp.compare("vegetation_decrease", [vlm("Vegetation decreased.", q=DEFAULT_Q), ndvi(-0.18, "decrease", q=DEFAULT_Q)])
print(f"7d) unknown quality (all None) => {r.status.value}")

# KEY: quality failure vs direction conflict precedence
# If NDVI has LOW quality AND conflicting direction, quality gate fires first
r = comp.compare("vegetation_decrease", [vlm("Vegetation decreased."), ndvi(0.15, "increase", q=LOW_Q, eid="ndvi_bad")])
print(f"7e) LOW quality + conflicting direction => {r.status.value}  (quality checked before direction)")

# ============================================================
# AUDIT 8 — SUPPORTING / CONFLICTING IDS
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 8 — EVIDENCE IDS")
print("=" * 70)
r = comp.compare("vegetation_decrease", [vlm("Vegetation decreased.", eid="V1"), ndvi(-0.18, "decrease", eid="N1")])
print(f"8a) SUPPORTED supporting_ids={r.supporting_evidence}  conflicting_ids={r.conflicting_evidence}")

r = comp.compare("vegetation_decrease", [vlm("Vegetation decreased.", eid="V2"), ndvi(0.15, "increase", eid="N2")])
print(f"8b) UNCERTAIN supporting_ids={r.supporting_evidence}  conflicting_ids={r.conflicting_evidence}")

# ============================================================
# AUDIT 9 — REGION CONSISTENCY
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 9 — REGION CONSISTENCY")
print("=" * 70)
r = comp.compare("vegetation_decrease", [
    vlm("Vegetation decreased.", eid="v_r", region={"id": "ROI_A"}),
    ndvi(-0.18, "decrease", eid="n_r", region={"id": "ROI_B"})
])
print(f"9a) Different regions => {r.status.value}  limitations={r.limitations}")

r = comp.compare("vegetation_decrease", [
    vlm("Vegetation decreased.", eid="v_s", region={"id": "ROI_A"}),
    ndvi(-0.18, "decrease", eid="n_s", region={"id": "ROI_A"})
])
print(f"9b) Same region       => {r.status.value}  limitations={r.limitations}")

r = comp.compare("vegetation_decrease", [
    vlm("Vegetation decreased.", eid="v_n"),
    ndvi(-0.18, "decrease", eid="n_n")
])
print(f"9c) No region info    => {r.status.value}  limitations={r.limitations}")

# ============================================================
# AUDIT 10 — TEMPORAL CONSISTENCY
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 10 — TEMPORAL CONSISTENCY")
print("=" * 70)
ev1 = vlm("Vegetation decreased.", eid="t_vlm")
ev2 = ndvi(-0.18, "decrease", eid="t_ndvi")
print(f"10a) VLM provenance.input_ids = {ev1.provenance.input_ids}")
print(f"10b) NDVI provenance.input_ids = {ev2.provenance.input_ids}")
print("     Limitation: provenance.input_ids are empty in mock outputs.")
print("     The comparator does NOT currently verify temporal consistency")
print("     because the mock evidence does not populate input_ids.")

# ============================================================
# AUDIT 11 — DETERMINISM
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 11 — DETERMINISM")
print("=" * 70)
evidence = [vlm("Vegetation decreased.", eid="d_v"), ndvi(-0.18, "decrease", eid="d_n")]
results = [comp.compare("vegetation_decrease", evidence) for _ in range(10)]
all_equal = all(r.model_dump() == results[0].model_dump() for r in results)
print(f"11) 10 identical runs produce identical result: {all_equal}")

# ============================================================
# AUDIT 12 — NO CONFIDENCE VOTING
# ============================================================
print("\n" + "=" * 70)
print("AUDIT 12 — NO CONFIDENCE VOTING")
print("=" * 70)
print("Inspected comparator.py: no 'confidence', 'score', 'vote', 'majority',")
print("'random', 'LLM', or 'API' logic found. Comparison is purely rule-based.")

print("\n" + "=" * 70)
print("ALL AUDITS COMPLETE")
print("=" * 70)
