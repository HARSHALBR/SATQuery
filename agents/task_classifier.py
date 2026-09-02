"""Task classifier for SATQuery AI.

Converts a natural-language satellite-imagery query into a structured
ParsedQuery object.  The classifier is deterministic — it uses keyword
patterns and observation metadata to identify the primary task, claim,
intent, modality requirements, expected evidence, and candidate tools.

Design principles:
    * No LLM, no embeddings, no external API calls.
    * TASK ≠ CURRENT CAPABILITY — the classifier reports *what kind of
      task this is*, not whether the current inputs can satisfy it.
    * The planner (Phase 5) decides the actual workflow.
    * Tool applicability remains the ToolRegistry's responsibility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from schemas.query import (
    Modality,
    ObservationInput,
    ParsedQuery,
    QueryInput,
    TaskType,
)


# ---------------------------------------------------------------------------
# Internal: keyword groups
# ---------------------------------------------------------------------------

# Each group is a compiled regex that matches against a lowercased query.
# Order of evaluation matters — see _detect_task() below.
#
# NOTE on word boundaries:
#   Use \b only at the START of each alternative.  Many alternatives are
#   word *stems* (e.g. "decreas", "descri", "compar") which must match
#   full words such as "decreased", "describe", "compare".  A trailing
#   \b would prevent those matches because there is no word-boundary
#   between the stem and its suffix.

_RE_VEGETATION = re.compile(
    r"\b("
    r"vegetation|ndvi|greenery|green\s*cover"
    r"|forest|deforest|afforest"
    r"|crop|cropland|farmland|agricultural"
    r"|tree|trees|canopy"
    r")",
    re.IGNORECASE,
)

_RE_BUILT_UP = re.compile(
    r"\b("
    r"built[\s-]?up|urban|settlement|city|town"
    r"|construction|building|buildings|infrastructure"
    r"|ndbi|impervious"
    r"|road|roads|highway|bridge"
    r"|residential|commercial|industrial"
    r")",
    re.IGNORECASE,
)

_RE_SAR = re.compile(
    r"\b("
    r"sar\b|radar|backscatter|sentinel[\s-]?1"
    r"|synthetic\s*aperture"
    r"|c[\s-]?band|x[\s-]?band"
    r")",
    re.IGNORECASE,
)

_RE_CHANGE = re.compile(
    r"\b("
    r"chang|differ|compar|before\s+and\s+after"
    r"|increas|decreas|expand|shrink|shrunk|reduc"
    r"|grow|grown|loss|lost"
    r"|develop|spread|encroach|occur"
    r")",
    re.IGNORECASE,
)

_RE_TEMPORAL = re.compile(
    r"\b("
    r"between|over\s+time|temporal|time\s*series"
    r"|before\s+and\s+after|from\s+\d{4}|since\s+\d{4}"
    r"|two\s+images|two\s+observations|bi[\s-]?temporal"
    r"|\d{4}\s*(and|to|vs\.?|versus)\s*\d{4}"
    r"|year|years|month|months|period"
    r")",
    re.IGNORECASE,
)

# _RE_SPATIAL detects spatial/locational intent.
# "area" is intentionally EXCLUDED here — it is too ambiguous
# (e.g. "built-up area" is a subject, not a spatial request).
_RE_SPATIAL = re.compile(
    r"\b("
    r"where|locat|region|zone|boundar"
    r"|spatial|extent|polygon"
    r"|show\s+me|point\s+out|highlight"
    r")",
    re.IGNORECASE,
)

# _RE_QUANTIFY detects requests for numeric measurement.
# "area" is handled carefully: only match "area of" or "area.*changed"
# to avoid false triggers on "built-up area increased".
_RE_QUANTIFY = re.compile(
    r"\b("
    r"hectare|square|sq\s*km|sq\s*m|km²|m²"
    r"|how\s+(?:much|many|big|large|small)"
    r"|measure|size|quantif|percent|fraction|proportion"
    r"|area\s+of"
    r")",
    re.IGNORECASE,
)

_RE_CAPTION = re.compile(
    r"\b("
    r"descri|caption|summar|explain\s+this\s+image"
    r"|tell\s+me\s+about\s+this"
    r"|what\s+does\s+this\s+image\s+show"
    r"|generate\s+a\s+description"
    r")",
    re.IGNORECASE,
)

_RE_VQA = re.compile(
    r"\b("
    r"what\s+is\s+visible|what\s+objects|what\s+type"
    r"|what\s+can\s+you\s+see|identify|recogni"
    r"|what\s+is\s+in\s+this|what\s+is\s+shown"
    r"|what\s+is\s+this|what\s+are\s+the"
    r"|what\s+land\s*cover|land[\s-]?use"
    r")",
    re.IGNORECASE,
)

_RE_GROUNDING = re.compile(
    r"\b("
    r"where\s+is|locate|find|show\s+me\s+where"
    r"|point\s+out|bounding|pinpoint"
    r")",
    re.IGNORECASE,
)

_RE_OPTICAL = re.compile(
    r"\b("
    r"optical|sentinel[\s-]?2|landsat|rgb|multispectral"
    r"|red\s+band|nir\s+band|swir\s+band|blue\s+band|green\s+band"
    r")",
    re.IGNORECASE,
)

_RE_CROSS_CHECK = re.compile(
    r"\b("
    r"cross[\s-]?check|confirm|support|corroborat"
    r"|agree|consistent|match"
    r"|also\s+visible"
    r")",
    re.IGNORECASE,
)

# Sensor → modality mapping
_SENSOR_MODALITY: dict[re.Pattern, Modality] = {
    re.compile(r"sentinel[\s-]?1", re.IGNORECASE): Modality.SAR,
    re.compile(r"sentinel[\s-]?2", re.IGNORECASE): Modality.OPTICAL,
    re.compile(r"landsat", re.IGNORECASE): Modality.OPTICAL,
    re.compile(r"modis", re.IGNORECASE): Modality.OPTICAL,
}


# ---------------------------------------------------------------------------
# Internal: task → evidence / candidate-tool mappings
# ---------------------------------------------------------------------------

_TASK_EVIDENCE: dict[TaskType, list[str]] = {
    TaskType.SINGLE_IMAGE_VQA: [],
    TaskType.CAPTIONING: [],
    TaskType.GROUNDING: [],
    TaskType.BI_TEMPORAL_CHANGE: ["change_statistics"],
    TaskType.VEGETATION_CHANGE: ["ndvi_delta", "change_statistics"],
    TaskType.BUILT_UP_CHANGE: ["ndbi_delta", "change_statistics"],
    TaskType.OPTICAL_SAR_CROSS_CHECK: ["change_statistics", "sar_change"],
    TaskType.SPATIAL_MEASUREMENT: ["area_measurement"],
    TaskType.INSUFFICIENT_CAPABILITY: [],
}

_TASK_CANDIDATE_TOOLS: dict[TaskType, list[str]] = {
    TaskType.SINGLE_IMAGE_VQA: [
        "validate_inputs",
        "run_rs_vlm",
        "compare_evidence",
        "generate_response",
    ],
    TaskType.CAPTIONING: [
        "validate_inputs",
        "run_rs_vlm",
        "generate_response",
    ],
    TaskType.GROUNDING: [
        "validate_inputs",
        "run_rs_vlm",
        "grounding",
        "generate_response",
    ],
    TaskType.BI_TEMPORAL_CHANGE: [
        "validate_inputs",
        "run_rs_vlm",
        "change_statistics",
        "compare_evidence",
        "generate_response",
    ],
    TaskType.VEGETATION_CHANGE: [
        "validate_inputs",
        "run_rs_vlm",
        "ndvi_delta",
        "change_statistics",
        "compare_evidence",
        "generate_response",
    ],
    TaskType.BUILT_UP_CHANGE: [
        "validate_inputs",
        "run_rs_vlm",
        "ndbi_delta",
        "change_statistics",
        "compare_evidence",
        "generate_response",
    ],
    TaskType.OPTICAL_SAR_CROSS_CHECK: [
        "validate_inputs",
        "run_rs_vlm",
        "change_statistics",
        "sar_change",
        "compare_evidence",
        "generate_response",
    ],
    TaskType.SPATIAL_MEASUREMENT: [
        "validate_inputs",
        "change_statistics",
        "area_measurement",
        "generate_response",
    ],
    TaskType.INSUFFICIENT_CAPABILITY: [
        "generate_response",
    ],
}


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TaskMatch:
    """A candidate task match with a confidence weight."""

    task: TaskType
    weight: float
    intent: str
    claim: str


# ---------------------------------------------------------------------------
# TaskClassifier
# ---------------------------------------------------------------------------


class TaskClassifier:
    """Deterministic task classifier for satellite-imagery queries.

    Converts a ``QueryInput`` into a ``ParsedQuery`` using keyword
    pattern matching and observation metadata.  No LLM calls, no
    network access, fully offline.

    Usage::

        classifier = TaskClassifier()
        parsed = classifier.classify(query_input)
    """

    # -- Public API ---------------------------------------------------------

    def classify(self, query_input: QueryInput) -> ParsedQuery:
        """Classify a user query into a structured ``ParsedQuery``.

        Args:
            query_input: Raw query string plus uploaded observations.

        Returns:
            A fully-populated ``ParsedQuery``.
        """
        q = self._normalize_query(query_input.query)

        task_match = self._detect_task(q)
        modalities = self._detect_modalities(q, query_input.observations)
        temporal = self._detect_temporal(q, task_match.task)
        spatial = self._detect_spatial(q, task_match.task)
        quantification = self._detect_quantification(q, task_match.task)
        observations = self._count_required_observations(task_match.task)
        required_evidence = _TASK_EVIDENCE.get(task_match.task, [])
        candidate_tools = _TASK_CANDIDATE_TOOLS.get(task_match.task, [])

        return ParsedQuery(
            raw_query=query_input.query,
            task=task_match.task,
            intent=task_match.intent,
            claim=task_match.claim if task_match.claim else None,
            observations=observations,
            temporal=temporal,
            spatial=spatial,
            quantification=quantification,
            modalities=modalities,
            required_evidence=list(required_evidence),
            candidate_tools=list(candidate_tools),
        )

    # -- Normalisation ------------------------------------------------------

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Lowercase, collapse whitespace, strip punctuation edges."""
        q = query.lower().strip()
        q = re.sub(r"\s+", " ", q)
        return q

    # -- Task detection -----------------------------------------------------

    def _detect_task(self, q: str) -> _TaskMatch:
        """Determine the primary task from the normalised query.

        Evaluation order implements the disambiguation priority:
            1. Optical-SAR cross-check (most specific multimodal)
            2. Spatial measurement (quantification-first)
            3. Vegetation change (domain-specific change)
            4. Built-up change (domain-specific change)
            5. Bi-temporal change (generic change)
            6. Grounding (spatial-only)
            7. Captioning
            8. Single-image VQA
            9. Insufficient capability (fallback)
        """
        candidates: list[_TaskMatch] = []

        # 1. Optical-SAR cross-check
        if _RE_SAR.search(q) and _RE_CROSS_CHECK.search(q):
            candidates.append(_TaskMatch(
                task=TaskType.OPTICAL_SAR_CROSS_CHECK,
                weight=100,
                intent="cross-check optical interpretation with SAR",
                claim=self._extract_claim(q, "optical_sar_cross_check"),
            ))

        # 2. Spatial measurement
        if _RE_QUANTIFY.search(q):
            # Only if the primary focus is measuring, not detecting domain change
            if not (_RE_VEGETATION.search(q) or _RE_BUILT_UP.search(q)):
                candidates.append(_TaskMatch(
                    task=TaskType.SPATIAL_MEASUREMENT,
                    weight=90,
                    intent="measure changed area",
                    claim=self._extract_claim(q, "spatial_measurement"),
                ))

        # 3. Vegetation change
        if _RE_VEGETATION.search(q) and _RE_CHANGE.search(q):
            candidates.append(_TaskMatch(
                task=TaskType.VEGETATION_CHANGE,
                weight=85,
                intent="detect vegetation change",
                claim=self._extract_claim(q, "vegetation_change"),
            ))

        # 4. Built-up change
        if _RE_BUILT_UP.search(q) and _RE_CHANGE.search(q):
            candidates.append(_TaskMatch(
                task=TaskType.BUILT_UP_CHANGE,
                weight=85,
                intent="detect built-up area change",
                claim=self._extract_claim(q, "built_up_change"),
            ))

        # 5. Bi-temporal change (generic)
        if _RE_CHANGE.search(q) and _RE_TEMPORAL.search(q):
            candidates.append(_TaskMatch(
                task=TaskType.BI_TEMPORAL_CHANGE,
                weight=70,
                intent="identify temporal change",
                claim=self._extract_claim(q, "bi_temporal_change"),
            ))

        # 6. Grounding (pure spatial localisation, not change)
        if _RE_GROUNDING.search(q) and not _RE_CHANGE.search(q):
            candidates.append(_TaskMatch(
                task=TaskType.GROUNDING,
                weight=60,
                intent="locate feature in image",
                claim=self._extract_claim(q, "grounding"),
            ))

        # 7. Captioning
        if _RE_CAPTION.search(q) and not _RE_CHANGE.search(q):
            candidates.append(_TaskMatch(
                task=TaskType.CAPTIONING,
                weight=55,
                intent="generate image description",
                claim=None,
            ))

        # 8. Single-image VQA
        if _RE_VQA.search(q) and not _RE_CHANGE.search(q):
            candidates.append(_TaskMatch(
                task=TaskType.SINGLE_IMAGE_VQA,
                weight=50,
                intent="answer question about image",
                claim=None,
            ))

        # Pick highest-weight candidate
        if candidates:
            candidates.sort(key=lambda m: m.weight, reverse=True)
            return candidates[0]

        # ---- Broad fallbacks (weaker signals) ----

        # Change keywords alone (without explicit temporal markers)
        if _RE_CHANGE.search(q):
            # Check for domain-specific subject
            if _RE_VEGETATION.search(q):
                return _TaskMatch(
                    task=TaskType.VEGETATION_CHANGE,
                    weight=60,
                    intent="detect vegetation change",
                    claim=self._extract_claim(q, "vegetation_change"),
                )
            if _RE_BUILT_UP.search(q):
                return _TaskMatch(
                    task=TaskType.BUILT_UP_CHANGE,
                    weight=60,
                    intent="detect built-up area change",
                    claim=self._extract_claim(q, "built_up_change"),
                )
            return _TaskMatch(
                task=TaskType.BI_TEMPORAL_CHANGE,
                weight=50,
                intent="identify temporal change",
                claim=self._extract_claim(q, "bi_temporal_change"),
            )

        # Spatial measurement without change context
        if _RE_QUANTIFY.search(q):
            return _TaskMatch(
                task=TaskType.SPATIAL_MEASUREMENT,
                weight=40,
                intent="measure area",
                claim=self._extract_claim(q, "spatial_measurement"),
            )

        # Pure grounding
        if _RE_GROUNDING.search(q):
            return _TaskMatch(
                task=TaskType.GROUNDING,
                weight=40,
                intent="locate feature in image",
                claim=self._extract_claim(q, "grounding"),
            )

        # Captioning-like
        if _RE_CAPTION.search(q):
            return _TaskMatch(
                task=TaskType.CAPTIONING,
                weight=35,
                intent="generate image description",
                claim=None,
            )

        # VQA-like
        if _RE_VQA.search(q):
            return _TaskMatch(
                task=TaskType.SINGLE_IMAGE_VQA,
                weight=30,
                intent="answer question about image",
                claim=None,
            )

        # ---- SAR-specific without cross-check ----
        if _RE_SAR.search(q) and _RE_CHANGE.search(q):
            return _TaskMatch(
                task=TaskType.OPTICAL_SAR_CROSS_CHECK,
                weight=45,
                intent="cross-check optical interpretation with SAR",
                claim=self._extract_claim(q, "optical_sar_cross_check"),
            )

        # Fallback
        return _TaskMatch(
            task=TaskType.INSUFFICIENT_CAPABILITY,
            weight=0,
            intent="unsupported query",
            claim=None,
        )

    # -- Modality detection -------------------------------------------------

    def _detect_modalities(
        self,
        q: str,
        observations: list[ObservationInput],
    ) -> list[Modality]:
        """Detect required/available modalities from query text and observations."""
        modalities: set[Modality] = set()

        # From query keywords
        if _RE_OPTICAL.search(q):
            modalities.add(Modality.OPTICAL)
        if _RE_SAR.search(q):
            modalities.add(Modality.SAR)

        # From sensor names in query
        for pattern, mod in _SENSOR_MODALITY.items():
            if pattern.search(q):
                modalities.add(mod)

        # From observation metadata
        for obs in observations:
            modalities.add(obs.metadata.modality)
            # Also check sensor field
            if obs.metadata.sensor:
                for pattern, mod in _SENSOR_MODALITY.items():
                    if pattern.search(obs.metadata.sensor):
                        modalities.add(mod)

        return sorted(modalities, key=lambda m: m.value)

    # -- Temporal detection -------------------------------------------------

    def _detect_temporal(self, q: str, task: TaskType) -> bool:
        """Determine whether the query requires temporal comparison."""
        # Some tasks are inherently temporal
        inherently_temporal = {
            TaskType.BI_TEMPORAL_CHANGE,
            TaskType.VEGETATION_CHANGE,
            TaskType.BUILT_UP_CHANGE,
            TaskType.OPTICAL_SAR_CROSS_CHECK,
        }
        if task in inherently_temporal:
            return True
        if _RE_TEMPORAL.search(q):
            return True
        return False

    # -- Spatial detection --------------------------------------------------

    def _detect_spatial(self, q: str, task: TaskType) -> bool:
        """Determine whether the query has a spatial component."""
        inherently_spatial = {
            TaskType.GROUNDING,
            TaskType.SPATIAL_MEASUREMENT,
        }
        if task in inherently_spatial:
            return True
        if _RE_SPATIAL.search(q):
            return True
        return False

    # -- Quantification detection -------------------------------------------

    def _detect_quantification(self, q: str, task: TaskType) -> bool:
        """Determine whether the query requests numeric quantification."""
        if task == TaskType.SPATIAL_MEASUREMENT:
            return True
        if _RE_QUANTIFY.search(q):
            return True
        return False

    # -- Observation counting -----------------------------------------------

    @staticmethod
    def _count_required_observations(task: TaskType) -> int:
        """Return the number of observations semantically required by a task."""
        multi_observation = {
            TaskType.BI_TEMPORAL_CHANGE,
            TaskType.VEGETATION_CHANGE,
            TaskType.BUILT_UP_CHANGE,
            TaskType.OPTICAL_SAR_CROSS_CHECK,
        }
        if task in multi_observation:
            return 2
        return 1

    # -- Claim extraction ---------------------------------------------------

    @staticmethod
    def _extract_claim(q: str, task_key: str) -> str:
        """Extract a concise semantic claim from the normalised query.

        The claim is a short structured label, not a sentence.
        """
        # Directional change claims
        if task_key in ("vegetation_change", "built_up_change", "bi_temporal_change"):
            subject = {
                "vegetation_change": "vegetation",
                "built_up_change": "built_up",
                "bi_temporal_change": "general",
            }[task_key]

            if re.search(r"\b(decreas|loss|lost|reduc|shrink|shrunk|deforest)", q, re.IGNORECASE):
                return f"{subject}_decrease"
            if re.search(r"\b(increas|grow|grown|expand|develop|spread|encroach|afforest)", q, re.IGNORECASE):
                return f"{subject}_increase"
            return f"{subject}_change"

        if task_key == "optical_sar_cross_check":
            return "sar_cross_check"

        if task_key == "spatial_measurement":
            return "area_measurement"

        if task_key == "grounding":
            # Try to capture the object being located
            m = re.search(
                r"(?:where\s+is\s+(?:the\s+)?|locate\s+(?:the\s+)?|find\s+(?:the\s+)?)"
                r"([\w\s]+?)(?:\?|$|\.)",
                q,
            )
            if m:
                obj = m.group(1).strip()
                return f"{obj.replace(' ', '_')}_location"
            return "feature_location"

        return "general_query"
