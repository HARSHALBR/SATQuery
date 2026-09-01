"""Tests for the Phase 9 Evidence Comparator."""

import pytest

from schemas.evidence import EvidenceRecord, QualityReport, Provenance
from schemas.response import EvidenceStatus
from evidence.comparator import EvidenceComparator
from evidence.evidence_store import EvidenceStore
from agents.task_classifier import TaskClassifier
from agents.planner import ConstrainedPlanner
from agents.tool_registry import ToolRegistry, ToolExecutionContext
from agents.execution_engine import ExecutionEngine
from agents.mock_tools import MockToolRunner, MockScenario
from schemas.query import QueryInput, Modality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GOOD_QUALITY = QualityReport(
    valid_pixel_fraction=0.95,
    registration_ok=True,
    cloud_cover=0.05,
)
_LOW_QUALITY = QualityReport(
    valid_pixel_fraction=0.40,
    registration_ok=False,
    cloud_cover=0.60,
    notes=["Low quality mock scenario"],
)


def _make_prov(tool: str) -> Provenance:
    return Provenance(tool=tool, tool_version="mock-1.0")


def _vlm_decrease(eid: str = "vlm_1", quality: QualityReport = None) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=eid,
        type="vlm_interpretation",
        tool_version="mock-1.0",
        value={"interpretation": "Vegetation appears to have decreased.", "confidence": 0.82},
        quality=quality or _GOOD_QUALITY,
        provenance=_make_prov("run_rs_vlm"),
    )


def _vlm_increase(eid: str = "vlm_inc", quality: QualityReport = None) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=eid,
        type="vlm_interpretation",
        tool_version="mock-1.0",
        value={"interpretation": "Vegetation appears to have increased.", "confidence": 0.80},
        quality=quality or _GOOD_QUALITY,
        provenance=_make_prov("run_rs_vlm"),
    )



def _change_quantification(eid: str = "cq_1", quality: QualityReport = None) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=eid,
        type="change_quantification",
        tool_version="mock-1.0",
        value={"changed_pixel_fraction": 0.21, "change_detected": True},
        quality=quality or _GOOD_QUALITY,
        provenance=_make_prov("change_statistics"),
    )


def _ndvi_decrease(eid: str = "ndvi_1", quality: QualityReport = None) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=eid,
        type="vegetation_change",
        tool_version="mock-1.0",
        value={"ndvi_delta": -0.18, "direction": "decrease", "affected_fraction": 0.21},
        quality=quality or _GOOD_QUALITY,
        provenance=_make_prov("ndvi_delta"),
    )


def _ndvi_increase(eid: str = "ndvi_inc", quality: QualityReport = None) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=eid,
        type="vegetation_change",
        tool_version="mock-1.0",
        value={"ndvi_delta": 0.22, "direction": "increase", "affected_fraction": 0.15},
        quality=quality or _GOOD_QUALITY,
        provenance=_make_prov("ndvi_delta"),
    )


def _ndbi_increase(eid: str = "ndbi_1", quality: QualityReport = None) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=eid,
        type="built_up_change",
        tool_version="mock-1.0",
        value={"ndbi_delta": 0.12, "direction": "increase", "affected_fraction": 0.05},
        quality=quality or _GOOD_QUALITY,
        provenance=_make_prov("ndbi_delta"),
    )


def _sar_change(eid: str = "sar_1", quality: QualityReport = None) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=eid,
        type="sar_amplitude_change",
        tool_version="mock-1.0",
        value={"sar_change_detected": True, "change_score": 0.76},
        quality=quality or _GOOD_QUALITY,
        provenance=_make_prov("sar_change"),
    )


def _change_stats(eid: str = "cs_1", quality: QualityReport = None) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=eid,
        type="change_quantification",
        tool_version="mock-1.0",
        value={"changed_pixel_fraction": 0.21, "change_detected": True},
        quality=quality or _GOOD_QUALITY,
        provenance=_make_prov("change_statistics"),
    )


def _area_measurement(eid: str = "area_1", quality: QualityReport = None) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=eid,
        type="spatial_measurement",
        tool_version="mock-1.0",
        value={"area_km2": 12.4},
        quality=quality or _GOOD_QUALITY,
        provenance=_make_prov("area_measurement"),
    )


@pytest.fixture
def comparator():
    return EvidenceComparator()


# ---------------------------------------------------------------------------
# 1. Empty evidence → INSUFFICIENT
# ---------------------------------------------------------------------------

class TestEmptyEvidence:
    def test_no_evidence(self, comparator):
        result = comparator.compare("vegetation_decrease", [])
        assert result.status == EvidenceStatus.INSUFFICIENT

    def test_none_claim_no_evidence(self, comparator):
        result = comparator.compare(None, [])
        assert result.status == EvidenceStatus.INSUFFICIENT


# ---------------------------------------------------------------------------
# 2. Only unrelated evidence → INSUFFICIENT
# ---------------------------------------------------------------------------

class TestUnrelatedEvidence:
    def test_sar_only_for_vegetation(self, comparator):
        """Scenario D: SAR-only evidence for vegetation claim."""
        result = comparator.compare("vegetation_decrease", [_sar_change()])
        assert result.status == EvidenceStatus.INSUFFICIENT

    def test_area_only_for_vegetation(self, comparator):
        """Area measurement alone cannot prove vegetation change."""
        result = comparator.compare("vegetation_decrease", [_area_measurement()])
        assert result.status == EvidenceStatus.INSUFFICIENT

    def test_ndvi_for_built_up_claim(self, comparator):
        """NDVI evidence should not address built-up claims."""
        result = comparator.compare("built_up_increase", [_ndvi_decrease()])
        assert result.status == EvidenceStatus.INSUFFICIENT


# ---------------------------------------------------------------------------
# 3. SUPPORTED: VLM decrease + NDVI decrease + good quality
# ---------------------------------------------------------------------------

class TestSupported:
    def test_vegetation_decrease_supported(self, comparator):
        """Scenario A."""
        evidence = [_vlm_decrease(), _ndvi_decrease(), _change_quantification()]
        result = comparator.compare("vegetation_decrease", evidence)
        assert result.status == EvidenceStatus.SUPPORTED
        assert len(result.supporting_evidence) == 3
        assert len(result.conflicting_evidence) == 0

    def test_vegetation_increase_supported(self, comparator):
        evidence = [_vlm_increase(), _ndvi_increase(), _change_quantification()]
        result = comparator.compare("vegetation_increase", evidence)
        assert result.status == EvidenceStatus.SUPPORTED

    def test_built_up_increase_supported(self, comparator):
        vlm = EvidenceRecord(
            evidence_id="vlm_bu",
            type="vlm_interpretation",
            tool_version="mock-1.0",
            value={"interpretation": "Built-up area has increased.", "confidence": 0.85},
            quality=_GOOD_QUALITY,
            provenance=_make_prov("run_rs_vlm"),
        )
        result = comparator.compare("built_up_increase", [vlm, _ndbi_increase(), _change_quantification()])
        assert result.status == EvidenceStatus.SUPPORTED

    def test_sar_cross_check_supported(self, comparator):
        vlm = _vlm_decrease(eid="vlm_sar")
        result = comparator.compare("sar_cross_check", [vlm, _sar_change()])
        assert result.status == EvidenceStatus.SUPPORTED


# ---------------------------------------------------------------------------
# 4. UNCERTAIN: VLM decrease + NDVI increase
# ---------------------------------------------------------------------------

class TestUncertain:
    def test_conflicting_directions(self, comparator):
        """Scenario B."""
        evidence = [_vlm_decrease(), _ndvi_increase(), _change_quantification()]
        result = comparator.compare("vegetation_decrease", evidence)
        assert result.status == EvidenceStatus.UNCERTAIN
        assert len(result.conflicting_evidence) > 0

    def test_conflicting_ids_reported(self, comparator):
        evidence = [_vlm_decrease(), _ndvi_increase(), _change_quantification()]
        result = comparator.compare("vegetation_decrease", evidence)
        all_ids = set(result.supporting_evidence + result.conflicting_evidence)
        assert "vlm_1" in all_ids
        assert "ndvi_inc" in all_ids


# ---------------------------------------------------------------------------
# 5. INSUFFICIENT: low quality
# ---------------------------------------------------------------------------

class TestLowQuality:
    def test_low_quality_ndvi(self, comparator):
        """Scenario C."""
        evidence = [_vlm_decrease(), _ndvi_decrease(quality=_LOW_QUALITY), _change_quantification()]
        result = comparator.compare("vegetation_decrease", evidence)
        assert result.status == EvidenceStatus.INSUFFICIENT

    def test_poor_registration(self, comparator):
        bad_reg = QualityReport(
            valid_pixel_fraction=0.90,
            registration_ok=False,
            cloud_cover=0.05,
        )
        evidence = [_vlm_decrease(), _ndvi_decrease(quality=bad_reg), _change_quantification()]
        result = comparator.compare("vegetation_decrease", evidence)
        assert result.status == EvidenceStatus.INSUFFICIENT

    def test_missing_quality_is_ok_by_default(self, comparator):
        """Unknown quality fields (None) should not block — only explicit
        failures should block. This is conservative in the other direction:
        we don't block on unknown, but we also don't promote unknown to
        high quality. The existing QualityReport defaults are all None."""
        default_q = QualityReport()
        evidence = [_vlm_decrease(quality=default_q), _ndvi_decrease(quality=default_q), _change_quantification()]
        result = comparator.compare("vegetation_decrease", evidence)
        # With all None quality fields, nothing explicitly fails.
        assert result.status == EvidenceStatus.SUPPORTED


# ---------------------------------------------------------------------------
# 6. SAR-only for vegetation → INSUFFICIENT
# ---------------------------------------------------------------------------

class TestSAROnlyVegetation:
    def test_sar_cannot_prove_vegetation(self, comparator):
        result = comparator.compare("vegetation_decrease", [_sar_change()])
        assert result.status == EvidenceStatus.INSUFFICIENT


# ---------------------------------------------------------------------------
# 7. NDVI alone without VLM
# ---------------------------------------------------------------------------

class TestNDVIAlone:
    def test_ndvi_without_vlm_insufficient(self, comparator):
        """NDVI alone without VLM interpretation cannot reach SUPPORTED.
        Rule: both vlm_interpretation AND vegetation_change are required."""
        result = comparator.compare("vegetation_decrease", [_ndvi_decrease()])
        assert result.status == EvidenceStatus.INSUFFICIENT


# ---------------------------------------------------------------------------
# 8. VLM alone → NOT SUPPORTED
# ---------------------------------------------------------------------------

class TestVLMAlone:
    def test_vlm_alone_not_supported(self, comparator):
        result = comparator.compare("vegetation_decrease", [_vlm_decrease()])
        assert result.status == EvidenceStatus.INSUFFICIENT


# ---------------------------------------------------------------------------
# 9. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_evidence_same_result(self, comparator):
        evidence = [_vlm_decrease(), _ndvi_decrease(), _change_quantification()]
        r1 = comparator.compare("vegetation_decrease", evidence)
        r2 = comparator.compare("vegetation_decrease", evidence)
        assert r1.status == r2.status
        assert r1.reason == r2.reason
        assert r1.supporting_evidence == r2.supporting_evidence
        assert r1.conflicting_evidence == r2.conflicting_evidence


# ---------------------------------------------------------------------------
# 10. Unrelated region
# ---------------------------------------------------------------------------

class TestRegionConsistency:
    def test_different_regions_flagged(self, comparator):
        vlm = _vlm_decrease()
        vlm.region = {"id": "roi_1"}
        ndvi = _ndvi_decrease()
        ndvi.region = {"id": "roi_2"}
        result = comparator.compare("vegetation_decrease", [vlm, ndvi, _change_quantification()])
        # Should still reach SUPPORTED but with a limitation warning.
        assert result.status == EvidenceStatus.SUPPORTED
        assert any("region" in lim.lower() for lim in result.limitations)


# ---------------------------------------------------------------------------
# 11-12. Evidence IDs correctly reported
# ---------------------------------------------------------------------------

class TestEvidenceIDReporting:
    def test_supporting_ids(self, comparator):
        evidence = [_vlm_decrease("v1"), _ndvi_decrease("n1"), _change_quantification()]
        result = comparator.compare("vegetation_decrease", evidence)
        assert "n1" in result.supporting_evidence
        assert "v1" in result.supporting_evidence

    def test_conflicting_ids(self, comparator):
        evidence = [_vlm_decrease("v1"), _ndvi_increase("n_inc"), _change_quantification()]
        result = comparator.compare("vegetation_decrease", evidence)
        assert len(result.conflicting_evidence) > 0


# ---------------------------------------------------------------------------
# 13. Provenance accessible
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_provenance_available(self, comparator):
        ev = _ndvi_decrease()
        assert ev.provenance.tool == "ndvi_delta"
        assert ev.provenance.tool_version == "mock-1.0"


# ---------------------------------------------------------------------------
# 16. Positive vs negative vegetation claim
# ---------------------------------------------------------------------------

class TestDirectionalClaims:
    def test_increase_claim_with_decrease_evidence(self, comparator):
        """Claiming increase but evidence says decrease → UNCERTAIN."""
        evidence = [_vlm_decrease(), _ndvi_decrease(), _change_quantification()]
        result = comparator.compare("vegetation_increase", evidence)
        assert result.status == EvidenceStatus.UNCERTAIN

    def test_decrease_claim_with_increase_evidence(self, comparator):
        evidence = [_vlm_increase(), _ndvi_increase(), _change_quantification()]
        result = comparator.compare("vegetation_decrease", evidence)
        assert result.status == EvidenceStatus.UNCERTAIN


# ---------------------------------------------------------------------------
# 17. NDBI for built-up claim
# ---------------------------------------------------------------------------

class TestBuiltUpEvidence:
    def test_ndbi_not_ndvi(self, comparator):
        """Built-up claim must use NDBI evidence, not NDVI."""
        vlm = EvidenceRecord(
            evidence_id="vlm_bu2",
            type="vlm_interpretation",
            tool_version="mock-1.0",
            value={"interpretation": "Built-up area has increased."},
            quality=_GOOD_QUALITY,
            provenance=_make_prov("run_rs_vlm"),
        )
        # NDVI evidence should not satisfy built_up claim.
        result = comparator.compare("built_up_increase", [vlm, _ndvi_decrease()])
        assert result.status == EvidenceStatus.INSUFFICIENT

        # NDBI evidence should satisfy it.
        result2 = comparator.compare("built_up_increase", [vlm, _ndbi_increase(), _change_quantification()])
        assert result2.status == EvidenceStatus.SUPPORTED


# ---------------------------------------------------------------------------
# 18. SAR cross-check claim
# ---------------------------------------------------------------------------

class TestSARCrossCheck:
    def test_sar_claim_uses_sar_evidence(self, comparator):
        vlm = _vlm_decrease(eid="vlm_sar2")
        result = comparator.compare("sar_cross_check", [vlm, _sar_change()])
        assert result.status == EvidenceStatus.SUPPORTED


# ---------------------------------------------------------------------------
# 19. Area measurement alone
# ---------------------------------------------------------------------------

class TestAreaMeasurement:
    def test_area_alone_does_not_prove_vegetation(self, comparator):
        result = comparator.compare("vegetation_decrease", [_area_measurement()])
        assert result.status == EvidenceStatus.INSUFFICIENT


# ---------------------------------------------------------------------------
# 20. Deterministic ordering of evidence IDs
# ---------------------------------------------------------------------------

class TestDeterministicOrdering:
    def test_sorted_supporting_ids(self, comparator):
        evidence = [_vlm_decrease("z_vlm"), _ndvi_decrease("a_ndvi"), _change_quantification()]
        result = comparator.compare("vegetation_decrease", evidence)
        assert result.supporting_evidence == sorted(result.supporting_evidence)


# ---------------------------------------------------------------------------
# Unknown / edge case claims
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_unknown_claim(self, comparator):
        result = comparator.compare("unknown_claim_xyz", [_vlm_decrease()])
        assert result.status == EvidenceStatus.INSUFFICIENT

    def test_none_claim(self, comparator):
        result = comparator.compare(None, [_vlm_decrease()])
        assert result.status == EvidenceStatus.INSUFFICIENT

    def test_empty_string_claim(self, comparator):
        result = comparator.compare("", [_vlm_decrease()])
        assert result.status == EvidenceStatus.INSUFFICIENT


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------

class TestIntegration:
    def _run_pipeline(self, scenario: MockScenario):
        classifier = TaskClassifier()
        query = QueryInput(query="Has vegetation decreased?", image_ids=["img1", "img2"])
        parsed = classifier.classify(query)

        registry = ToolRegistry.from_yaml("configs/tools.yaml")
        context = ToolExecutionContext(
            available_modalities=[Modality.OPTICAL],
            available_bands=["red", "nir"],
            num_observations=2,
            has_temporal=True,
            registration_status=True,
            valid_pixel_fraction=0.9,
        )

        planner = ConstrainedPlanner(registry, "configs/workflows.yaml")
        plan = planner.create_plan(parsed, context)

        runner = MockToolRunner(scenario)
        engine = ExecutionEngine(runner)
        trace, evidence_list = engine.execute_plan(plan)

        store = EvidenceStore()
        store.add_many(evidence_list)

        comparator = EvidenceComparator()
        result = comparator.compare(parsed.claim, store.list())
        return result

    def test_integration_supported(self):
        result = self._run_pipeline(MockScenario.VEGETATION_DECREASE)
        assert result.status == EvidenceStatus.SUPPORTED

    def test_integration_uncertain(self):
        result = self._run_pipeline(MockScenario.CONFLICTING_EVIDENCE)
        assert result.status == EvidenceStatus.UNCERTAIN

    def test_integration_insufficient(self):
        result = self._run_pipeline(MockScenario.LOW_QUALITY)
        assert result.status == EvidenceStatus.INSUFFICIENT
