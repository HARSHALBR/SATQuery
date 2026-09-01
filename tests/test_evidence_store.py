"""Tests for the Phase 8 Evidence Store."""

import pytest
from datetime import datetime, timezone

from schemas.evidence import EvidenceRecord, QualityReport, Provenance
from evidence.evidence_store import EvidenceStore, DuplicateEvidenceError
from agents.task_classifier import TaskClassifier
from agents.planner import ConstrainedPlanner
from agents.tool_registry import ToolRegistry, ToolExecutionContext
from agents.execution_engine import ExecutionEngine
from agents.mock_tools import MockToolRunner, MockScenario
from schemas.query import QueryInput, Modality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evidence(
    eid: str,
    etype: str = "test",
    tool: str = "test_tool",
    tool_version: str = "1.0",
    value=None,
    region=None,
    quality=None,
) -> EvidenceRecord:
    """Create a minimal EvidenceRecord for testing."""
    return EvidenceRecord(
        evidence_id=eid,
        type=etype,
        tool_version=tool_version,
        value=value,
        region=region,
        quality=quality or QualityReport(),
        provenance=Provenance(tool=tool, tool_version=tool_version),
    )


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


class TestBasicCRUD:
    def test_empty_store(self):
        store = EvidenceStore()
        assert len(store) == 0
        assert store.list() == []

    def test_add_one(self):
        store = EvidenceStore()
        ev = _make_evidence("ev1")
        store.add(ev)
        assert len(store) == 1
        assert "ev1" in store

    def test_get_by_id(self):
        store = EvidenceStore()
        ev = _make_evidence("ev1", value={"x": 42})
        store.add(ev)
        retrieved = store.get("ev1")
        assert retrieved is not None
        assert retrieved.evidence_id == "ev1"
        assert retrieved.value == {"x": 42}

    def test_list_evidence(self):
        store = EvidenceStore()
        store.add(_make_evidence("a"))
        store.add(_make_evidence("b"))
        store.add(_make_evidence("c"))
        items = store.list()
        assert len(items) == 3
        assert [i.evidence_id for i in items] == ["a", "b", "c"]

    def test_add_many(self):
        store = EvidenceStore()
        evs = [_make_evidence("m1"), _make_evidence("m2"), _make_evidence("m3")]
        store.add_many(evs)
        assert len(store) == 3

    def test_clear(self):
        store = EvidenceStore()
        store.add(_make_evidence("ev1"))
        store.add(_make_evidence("ev2"))
        assert len(store) == 2
        store.clear()
        assert len(store) == 0
        assert store.list() == []

    def test_to_list_alias(self):
        store = EvidenceStore()
        store.add(_make_evidence("ev1"))
        assert store.to_list() == store.list()


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_get_by_tool(self):
        store = EvidenceStore()
        store.add(_make_evidence("a", tool="run_rs_vlm"))
        store.add(_make_evidence("b", tool="ndvi_delta"))
        store.add(_make_evidence("c", tool="run_rs_vlm"))
        results = store.get_by_tool("run_rs_vlm")
        assert len(results) == 2
        assert all(r.provenance.tool == "run_rs_vlm" for r in results)

    def test_get_by_type(self):
        store = EvidenceStore()
        store.add(_make_evidence("a", etype="vegetation_change"))
        store.add(_make_evidence("b", etype="built_up_change"))
        store.add(_make_evidence("c", etype="vegetation_change"))
        results = store.get_by_type("vegetation_change")
        assert len(results) == 2

    def test_get_by_region(self):
        store = EvidenceStore()
        store.add(_make_evidence("a", region={"id": "roi_1"}))
        store.add(_make_evidence("b", region={"id": "roi_2"}))
        store.add(_make_evidence("c", region={"id": "roi_1"}))
        results = store.get_by_region("roi_1")
        assert len(results) == 2

    def test_missing_id_returns_none(self):
        store = EvidenceStore()
        assert store.get("nonexistent") is None

    def test_empty_tool_filter(self):
        store = EvidenceStore()
        store.add(_make_evidence("a", tool="run_rs_vlm"))
        assert store.get_by_tool("no_such_tool") == []

    def test_empty_type_filter(self):
        store = EvidenceStore()
        store.add(_make_evidence("a", etype="foo"))
        assert store.get_by_type("bar") == []

    def test_empty_region_filter(self):
        store = EvidenceStore()
        store.add(_make_evidence("a", region={"id": "roi_1"}))
        assert store.get_by_region("roi_99") == []


# ---------------------------------------------------------------------------
# Duplicate / Identity
# ---------------------------------------------------------------------------


class TestDuplicateGuard:
    def test_duplicate_id_rejected(self):
        store = EvidenceStore()
        store.add(_make_evidence("dup"))
        with pytest.raises(DuplicateEvidenceError):
            store.add(_make_evidence("dup"))

    def test_add_many_duplicate_in_batch(self):
        store = EvidenceStore()
        with pytest.raises(DuplicateEvidenceError):
            store.add_many([_make_evidence("x"), _make_evidence("x")])
        # Atomicity: no partial inserts.
        assert len(store) == 0

    def test_add_many_duplicate_with_existing(self):
        store = EvidenceStore()
        store.add(_make_evidence("existing"))
        with pytest.raises(DuplicateEvidenceError):
            store.add_many([_make_evidence("new1"), _make_evidence("existing")])
        # Atomicity: the batch should not have partially committed.
        assert len(store) == 1


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


class TestOrdering:
    def test_insertion_order_preserved(self):
        store = EvidenceStore()
        for eid in ["z", "a", "m", "b"]:
            store.add(_make_evidence(eid))
        ids = [r.evidence_id for r in store.list()]
        assert ids == ["z", "a", "m", "b"]


# ---------------------------------------------------------------------------
# Preservation / Integrity
# ---------------------------------------------------------------------------


class TestPreservation:
    def test_serialization_intact(self):
        store = EvidenceStore()
        ev = _make_evidence("ev1", value={"delta": -0.18})
        store.add(ev)
        got = store.get("ev1")
        assert got.model_dump() == ev.model_dump()

    def test_provenance_preserved(self):
        store = EvidenceStore()
        prov = Provenance(
            input_ids=["img1", "img2"],
            tool="ndvi_delta",
            tool_version="0.2.0",
        )
        ev = EvidenceRecord(
            evidence_id="prov_test",
            type="vegetation_change",
            tool_version="0.2.0",
            provenance=prov,
        )
        store.add(ev)
        got = store.get("prov_test")
        assert got.provenance.input_ids == ["img1", "img2"]
        assert got.provenance.tool == "ndvi_delta"
        assert got.provenance.tool_version == "0.2.0"

    def test_quality_preserved(self):
        store = EvidenceStore()
        q = QualityReport(
            valid_pixel_fraction=0.85,
            registration_ok=True,
            cloud_cover=0.12,
            notes=["some note"],
        )
        ev = _make_evidence("q_test", quality=q)
        store.add(ev)
        got = store.get("q_test")
        assert got.quality.valid_pixel_fraction == 0.85
        assert got.quality.registration_ok is True
        assert got.quality.cloud_cover == 0.12
        assert got.quality.notes == ["some note"]

    def test_mutation_safety(self):
        """Mutating a returned record must not affect the store."""
        store = EvidenceStore()
        store.add(_make_evidence("safe", value={"k": 1}))
        got = store.get("safe")
        got.value["k"] = 999
        original = store.get("safe")
        assert original.value["k"] == 1


# ---------------------------------------------------------------------------
# Independence — Critical Regression
# ---------------------------------------------------------------------------


class TestIndependence:
    def test_same_type_different_tools(self):
        """Two tools producing the same evidence type must remain independent."""
        store = EvidenceStore()
        ev_a = _make_evidence("A", etype="region_mask", tool="run_rs_vlm", value={"mask": "A"})
        ev_b = _make_evidence("B", etype="region_mask", tool="grounding", value={"mask": "B"})
        store.add(ev_a)
        store.add(ev_b)
        assert len(store) == 2
        assert store.get("A").value == {"mask": "A"}
        assert store.get("B").value == {"mask": "B"}
        by_type = store.get_by_type("region_mask")
        assert len(by_type) == 2

    def test_vlm_and_ndvi_independent(self):
        store = EvidenceStore()
        store.add(_make_evidence("vlm_1", etype="vlm_interpretation", tool="run_rs_vlm"))
        store.add(_make_evidence("ndvi_1", etype="vegetation_change", tool="ndvi_delta"))
        assert len(store) == 2
        assert len(store.get_by_tool("run_rs_vlm")) == 1
        assert len(store.get_by_tool("ndvi_delta")) == 1

    def test_same_region_multiple_records(self):
        store = EvidenceStore()
        store.add(_make_evidence("r1", region={"id": "roi_1"}, tool="run_rs_vlm"))
        store.add(_make_evidence("r2", region={"id": "roi_1"}, tool="ndvi_delta"))
        results = store.get_by_region("roi_1")
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Integration with MockToolRunner pipeline
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_pipeline_evidence_into_store(self):
        """Classifier → Planner → Engine → MockRunner → EvidenceStore."""
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

        runner = MockToolRunner(MockScenario.VEGETATION_DECREASE)
        engine = ExecutionEngine(runner)
        trace, evidence_list = engine.execute_plan(plan)

        # Insert collected evidence into the store.
        store = EvidenceStore()
        store.add_many(evidence_list)

        assert len(store) == len(evidence_list)
        assert len(store) > 0

        # NDVI evidence exists.
        ndvi = store.get_by_type("vegetation_change")
        assert len(ndvi) >= 1

        # Change statistics evidence exists.
        cs = store.get_by_type("change_quantification")
        assert len(cs) >= 1

        # Each has its own evidence_id.
        all_ids = [r.evidence_id for r in store.list()]
        assert len(all_ids) == len(set(all_ids))

        # Provenance is preserved for each record.
        for record in store.list():
            assert record.provenance.tool != ""
            assert record.provenance.tool_version == "mock-1.0"

        # Quality is preserved.
        for record in store.list():
            assert record.quality is not None
