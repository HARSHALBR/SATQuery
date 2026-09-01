"""Tests for the SATQuery AI task classifier.

Covers all 9 task types, modality detection, temporal/spatial/quantification
requirements, claim extraction, disambiguation priority, observation
awareness, sensor-to-modality mapping, and edge cases.
"""

import pytest

from agents.task_classifier import TaskClassifier
from schemas.query import (
    ImageMetadata,
    Modality,
    ObservationInput,
    ObservationRole,
    ParsedQuery,
    QueryInput,
    TaskType,
)


# ===================================================================
# Helpers
# ===================================================================


def _qi(query: str, observations: list[ObservationInput] | None = None) -> QueryInput:
    """Shorthand to build a QueryInput."""
    return QueryInput(query=query, observations=observations or [])


def _optical_obs(role: ObservationRole = ObservationRole.SINGLE, obs_id: str = "obs_1") -> ObservationInput:
    return ObservationInput(
        observation_id=obs_id,
        image_path=f"/data/{obs_id}.tif",
        role=role,
        metadata=ImageMetadata(
            modality=Modality.OPTICAL,
            bands=["red", "nir", "blue", "green"],
            sensor="Sentinel-2",
        ),
    )


def _sar_obs(role: ObservationRole = ObservationRole.SINGLE, obs_id: str = "sar_1") -> ObservationInput:
    return ObservationInput(
        observation_id=obs_id,
        image_path=f"/data/{obs_id}.tif",
        role=role,
        metadata=ImageMetadata(
            modality=Modality.SAR,
            sensor="Sentinel-1",
        ),
    )


@pytest.fixture
def classifier() -> TaskClassifier:
    return TaskClassifier()


# ===================================================================
# 1. Single-image VQA
# ===================================================================


class TestSingleImageVQA:
    def test_what_is_visible(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What is visible in this satellite image?"))
        assert result.task == TaskType.SINGLE_IMAGE_VQA

    def test_what_objects(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What objects are present in the image?"))
        assert result.task == TaskType.SINGLE_IMAGE_VQA

    def test_what_land_cover(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What type of land cover is shown?"))
        assert result.task == TaskType.SINGLE_IMAGE_VQA

    def test_observations_is_one(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What is visible?"))
        assert result.observations == 1

    def test_temporal_false(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What is visible?"))
        assert result.temporal is False


# ===================================================================
# 2. Captioning
# ===================================================================


class TestCaptioning:
    def test_describe_image(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Describe this satellite image."))
        assert result.task == TaskType.CAPTIONING

    def test_generate_caption(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Generate a caption for this image."))
        assert result.task == TaskType.CAPTIONING

    def test_tell_me_about(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Tell me about this satellite image."))
        assert result.task == TaskType.CAPTIONING


# ===================================================================
# 3. Grounding
# ===================================================================


class TestGrounding:
    def test_where_is_building(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Where is the building?"))
        assert result.task == TaskType.GROUNDING

    def test_locate_river(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Locate the river in this image."))
        assert result.task == TaskType.GROUNDING

    def test_spatial_true(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Where is the airport?"))
        assert result.spatial is True

    def test_claim_extraction(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Where is the new bridge?"))
        assert result.claim is not None
        assert "bridge" in result.claim.lower() or "location" in result.claim.lower()


# ===================================================================
# 4. Bi-temporal change
# ===================================================================


class TestBiTemporalChange:
    def test_what_changed_between(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What changed between these two satellite images?"))
        assert result.task == TaskType.BI_TEMPORAL_CHANGE

    def test_what_is_different(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What is different between 2019 and 2024?"))
        assert result.task == TaskType.BI_TEMPORAL_CHANGE

    def test_detect_changes(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Detect changes between these observations."))
        assert result.task == TaskType.BI_TEMPORAL_CHANGE

    def test_temporal_true(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What changed between these two images?"))
        assert result.temporal is True

    def test_two_observations(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What changed between these two images?"))
        assert result.observations == 2

    def test_change_statistics_in_evidence(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What changed between these two images?"))
        assert "change_statistics" in result.required_evidence


# ===================================================================
# 5. Vegetation change
# ===================================================================


class TestVegetationChange:
    def test_vegetation_decreased(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Has vegetation decreased between these two images?"))
        assert result.task == TaskType.VEGETATION_CHANGE

    def test_vegetation_increased(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Did vegetation increase in this region?"))
        assert result.task == TaskType.VEGETATION_CHANGE

    def test_forest_loss(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Where has deforestation occurred?"))
        assert result.task == TaskType.VEGETATION_CHANGE

    def test_vegetation_cover_changed(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Has the vegetation cover changed over time?"))
        assert result.task == TaskType.VEGETATION_CHANGE

    def test_temporal_true(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Has vegetation decreased?"))
        assert result.temporal is True

    def test_ndvi_in_evidence(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Has vegetation decreased?"))
        assert "ndvi_delta" in result.required_evidence
        assert "change_statistics" in result.required_evidence

    def test_claim_decrease(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Has vegetation decreased?"))
        assert result.claim == "vegetation_decrease"

    def test_claim_increase(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Did vegetation increase?"))
        assert result.claim == "vegetation_increase"

    def test_candidate_tools(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Has vegetation decreased?"))
        assert "validate_inputs" in result.candidate_tools
        assert "run_rs_vlm" in result.candidate_tools
        assert "ndvi_delta" in result.candidate_tools
        assert "compare_evidence" in result.candidate_tools
        assert "generate_response" in result.candidate_tools


# ===================================================================
# 6. Built-up change
# ===================================================================


class TestBuiltUpChange:
    def test_built_up_increased(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Has built-up area increased?"))
        assert result.task == TaskType.BUILT_UP_CHANGE

    def test_urban_development(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Where has urban development occurred?"))
        assert result.task == TaskType.BUILT_UP_CHANGE

    def test_construction_increase(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Did construction increase in this area?"))
        assert result.task == TaskType.BUILT_UP_CHANGE

    def test_settlement_expanded(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Has the settlement expanded?"))
        assert result.task == TaskType.BUILT_UP_CHANGE

    def test_ndbi_in_evidence(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Has built-up area increased?"))
        assert "ndbi_delta" in result.required_evidence

    def test_claim_increase(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Has built-up area increased?"))
        assert result.claim == "built_up_increase"


# ===================================================================
# 7. Optical + SAR cross-check
# ===================================================================


class TestOpticalSARCrossCheck:
    def test_sar_support(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Does SAR support the suspected change?"))
        assert result.task == TaskType.OPTICAL_SAR_CROSS_CHECK

    def test_sar_confirm(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Can SAR confirm this change?"))
        assert result.task == TaskType.OPTICAL_SAR_CROSS_CHECK

    def test_radar_support(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Does radar imagery support this interpretation?"))
        assert result.task == TaskType.OPTICAL_SAR_CROSS_CHECK

    def test_optical_sar_visible(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Is the optical change also visible in SAR?"))
        assert result.task == TaskType.OPTICAL_SAR_CROSS_CHECK

    def test_sar_modality_detected(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Does SAR support the suspected change?"))
        assert Modality.SAR in result.modalities

    def test_temporal_true(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Does SAR support the suspected change?"))
        assert result.temporal is True


# ===================================================================
# 8. Spatial measurement
# ===================================================================


class TestSpatialMeasurement:
    def test_area_of_changed_region(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What is the area of the changed region?"))
        assert result.task == TaskType.SPATIAL_MEASUREMENT

    def test_square_kilometers(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("How many square kilometers changed?"))
        assert result.task == TaskType.SPATIAL_MEASUREMENT

    def test_size_of_region(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What is the size of the affected region?"))
        assert result.task == TaskType.SPATIAL_MEASUREMENT

    def test_quantification_true(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What is the area of the changed region?"))
        assert result.quantification is True

    def test_spatial_true(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What is the area of the changed region?"))
        assert result.spatial is True


# ===================================================================
# 9. Unsupported / insufficient capability
# ===================================================================


class TestInsufficientCapability:
    def test_random_nonsense(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What is the meaning of life?"))
        assert result.task == TaskType.INSUFFICIENT_CAPABILITY

    def test_unrelated_question(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("How do I cook pasta?"))
        assert result.task == TaskType.INSUFFICIENT_CAPABILITY

    def test_intent_is_unsupported(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Something completely random XYZ123"))
        assert result.intent == "unsupported query"

    def test_claim_is_none(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Play me some music"))
        assert result.claim is None


# ===================================================================
# 10. Ambiguous queries — disambiguation priority
# ===================================================================


class TestDisambiguation:
    def test_vegetation_grounding_prefers_vegetation(self, classifier: TaskClassifier):
        """'Where has vegetation loss occurred?' → vegetation_change, not grounding."""
        result = classifier.classify(_qi("Where has vegetation loss occurred?"))
        assert result.task == TaskType.VEGETATION_CHANGE
        assert result.spatial is True  # spatial flag still set

    def test_built_up_grounding_prefers_built_up(self, classifier: TaskClassifier):
        """'Where did new construction occur?' → built_up_change, not grounding."""
        result = classifier.classify(_qi("Where did new construction occur?"))
        assert result.task == TaskType.BUILT_UP_CHANGE
        assert result.spatial is True

    def test_interesting_image_not_change(self, classifier: TaskClassifier):
        """'Tell me something interesting about this satellite image.'
        should not force a change task."""
        result = classifier.classify(
            _qi("Tell me something interesting about this satellite image.")
        )
        assert result.task in (
            TaskType.CAPTIONING,
            TaskType.SINGLE_IMAGE_VQA,
            TaskType.INSUFFICIENT_CAPABILITY,
        )
        assert result.task != TaskType.BI_TEMPORAL_CHANGE
        assert result.task != TaskType.VEGETATION_CHANGE


# ===================================================================
# 11. Modality detection
# ===================================================================


class TestModalityDetection:
    def test_explicit_optical_keyword(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Analyze this optical satellite image"))
        assert Modality.OPTICAL in result.modalities

    def test_explicit_sar_keyword(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What does the SAR image show?"))
        assert Modality.SAR in result.modalities

    def test_sentinel1_maps_to_sar(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Analyze this Sentinel-1 image"))
        assert Modality.SAR in result.modalities

    def test_sentinel2_maps_to_optical(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Analyze this Sentinel-2 image"))
        assert Modality.OPTICAL in result.modalities

    def test_modality_from_observation_metadata(self, classifier: TaskClassifier):
        qi = _qi(
            "What is visible?",
            observations=[_optical_obs()],
        )
        result = classifier.classify(qi)
        assert Modality.OPTICAL in result.modalities

    def test_sar_modality_from_observation(self, classifier: TaskClassifier):
        qi = _qi(
            "What is visible?",
            observations=[_sar_obs()],
        )
        result = classifier.classify(qi)
        assert Modality.SAR in result.modalities

    def test_multi_modality_from_mixed_observations(self, classifier: TaskClassifier):
        qi = _qi(
            "Compare the images",
            observations=[
                _optical_obs(ObservationRole.T1, "opt_1"),
                _sar_obs(ObservationRole.SAR_T1, "sar_1"),
            ],
        )
        result = classifier.classify(qi)
        assert Modality.OPTICAL in result.modalities
        assert Modality.SAR in result.modalities


# ===================================================================
# 12. Observation awareness
# ===================================================================


class TestObservationAwareness:
    def test_task_unchanged_with_single_observation(self, classifier: TaskClassifier):
        """Vegetation change is still vegetation_change even if only 1 image provided.
        TASK ≠ CURRENT CAPABILITY."""
        qi = _qi(
            "Has vegetation decreased?",
            observations=[_optical_obs(ObservationRole.SINGLE)],
        )
        result = classifier.classify(qi)
        assert result.task == TaskType.VEGETATION_CHANGE
        assert result.observations == 2  # semantic requirement

    def test_missing_second_observation_still_temporal(self, classifier: TaskClassifier):
        qi = _qi(
            "What changed between these two images?",
            observations=[_optical_obs(ObservationRole.T1)],
        )
        result = classifier.classify(qi)
        assert result.task == TaskType.BI_TEMPORAL_CHANGE
        assert result.temporal is True
        assert result.observations == 2


# ===================================================================
# 13. Temporal detection
# ===================================================================


class TestTemporalDetection:
    def test_explicit_year_range(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What changed from 2019 to 2024?"))
        assert result.temporal is True

    def test_before_and_after(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Show before and after comparison"))
        assert result.temporal is True

    def test_vqa_not_temporal(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What is visible in this image?"))
        assert result.temporal is False


# ===================================================================
# 14. Quantification detection
# ===================================================================


class TestQuantificationDetection:
    def test_how_much(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("How much area has changed?"))
        assert result.quantification is True

    def test_percentage(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What percent of the region is vegetation?"))
        assert result.quantification is True

    def test_vqa_not_quantified(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What is visible?"))
        assert result.quantification is False


# ===================================================================
# 15. Claim extraction
# ===================================================================


class TestClaimExtraction:
    def test_general_change_claim(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What changed between these two images?"))
        assert result.claim is not None
        assert "change" in result.claim.lower()

    def test_vegetation_decrease_claim(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Has vegetation decreased?"))
        assert result.claim == "vegetation_decrease"

    def test_built_up_increase_claim(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Has built-up area increased?"))
        assert result.claim == "built_up_increase"

    def test_sar_cross_check_claim(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Does SAR support the suspected change?"))
        assert result.claim == "sar_cross_check"

    def test_area_measurement_claim(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("What is the area of the changed region?"))
        assert result.claim == "area_measurement"


# ===================================================================
# 16. Intent extraction
# ===================================================================


class TestIntentExtraction:
    def test_vegetation_intent(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Has vegetation decreased?"))
        assert "vegetation" in result.intent.lower()

    def test_built_up_intent(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Has built-up area increased?"))
        assert "built-up" in result.intent.lower() or "built_up" in result.intent.lower()


# ===================================================================
# 17. ParsedQuery contract
# ===================================================================


class TestParsedQueryContract:
    def test_returns_parsed_query_type(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Has vegetation decreased?"))
        assert isinstance(result, ParsedQuery)

    def test_raw_query_preserved(self, classifier: TaskClassifier):
        original = "Has vegetation decreased between 2020 and 2024?"
        result = classifier.classify(_qi(original))
        assert result.raw_query == original

    def test_serialization_works(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Has vegetation decreased?"))
        data = result.model_dump()
        reconstructed = ParsedQuery(**data)
        assert reconstructed == result

    def test_candidate_tools_not_empty_for_real_task(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Has vegetation decreased?"))
        assert len(result.candidate_tools) > 0

    def test_candidate_tools_minimal_for_insufficient(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Cook me some dinner"))
        assert result.candidate_tools == ["generate_response"]


# ===================================================================
# 18. Edge cases
# ===================================================================


class TestEdgeCases:
    def test_empty_query(self, classifier: TaskClassifier):
        result = classifier.classify(_qi(""))
        assert result.task == TaskType.INSUFFICIENT_CAPABILITY

    def test_whitespace_only(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("   "))
        assert result.task == TaskType.INSUFFICIENT_CAPABILITY

    def test_case_insensitivity(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("HAS VEGETATION DECREASED?"))
        assert result.task == TaskType.VEGETATION_CHANGE

    def test_extra_whitespace(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("  Has   vegetation   decreased  ?  "))
        assert result.task == TaskType.VEGETATION_CHANGE

    def test_cropland_as_vegetation(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Has cropland area decreased?"))
        assert result.task == TaskType.VEGETATION_CHANGE

    def test_road_as_built_up(self, classifier: TaskClassifier):
        result = classifier.classify(_qi("Has the road network expanded?"))
        assert result.task == TaskType.BUILT_UP_CHANGE
