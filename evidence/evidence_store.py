"""Evidence Store for SATQuery AI.

Provides an authoritative in-memory store for EvidenceRecord objects
produced by the execution layer. The store organizes, preserves,
retrieves, and queries evidence.

The EvidenceStore does NOT determine whether evidence is true.
It does NOT produce SUPPORTED / UNCERTAIN / INSUFFICIENT.
Evidence comparison belongs to Phase 9 (EvidenceComparator).
"""

from __future__ import annotations

from typing import Optional

from schemas.evidence import EvidenceRecord


class DuplicateEvidenceError(ValueError):
    """Raised when attempting to add an EvidenceRecord with an ID that already exists."""
    pass


class EvidenceStore:
    """In-memory store for EvidenceRecord objects.

    Evidence records are stored independently and keyed by evidence_id.
    Insertion order is preserved for deterministic listing.

    Returned records are Pydantic model copies to avoid accidental
    mutation of stored state. Callers receive independent objects.
    """

    def __init__(self) -> None:
        self._records: dict[str, EvidenceRecord] = {}

    # -- Mutation ------------------------------------------------------------

    def add(self, evidence: EvidenceRecord) -> None:
        """Add a single evidence record.

        Raises:
            DuplicateEvidenceError: If evidence_id already exists in the store.
        """
        if evidence.evidence_id in self._records:
            raise DuplicateEvidenceError(
                f"Evidence '{evidence.evidence_id}' already exists in the store."
            )
        self._records[evidence.evidence_id] = evidence.model_copy(deep=True)

    def add_many(self, evidence: list[EvidenceRecord]) -> None:
        """Add multiple evidence records.

        Raises:
            DuplicateEvidenceError: If any evidence_id already exists.
                No records from the batch are added on failure.
        """
        # Pre-validate the entire batch before mutating state.
        for ev in evidence:
            if ev.evidence_id in self._records:
                raise DuplicateEvidenceError(
                    f"Evidence '{ev.evidence_id}' already exists in the store."
                )
        # Check for duplicates within the batch itself.
        ids_in_batch: set[str] = set()
        for ev in evidence:
            if ev.evidence_id in ids_in_batch:
                raise DuplicateEvidenceError(
                    f"Duplicate evidence_id '{ev.evidence_id}' within the batch."
                )
            ids_in_batch.add(ev.evidence_id)
        # Commit.
        for ev in evidence:
            self._records[ev.evidence_id] = ev.model_copy(deep=True)

    def clear(self) -> None:
        """Remove all evidence records from the store."""
        self._records.clear()

    # -- Retrieval -----------------------------------------------------------

    def get(self, evidence_id: str) -> Optional[EvidenceRecord]:
        """Retrieve a single evidence record by its ID.

        Returns:
            The EvidenceRecord if found, otherwise None.
        """
        record = self._records.get(evidence_id)
        if record is None:
            return None
        return record.model_copy(deep=True)

    def list(self) -> list[EvidenceRecord]:
        """Return all evidence records in insertion order."""
        return [r.model_copy(deep=True) for r in self._records.values()]

    def to_list(self) -> list[EvidenceRecord]:
        """Alias for list(). Return all evidence records in insertion order."""
        return self.list()

    # -- Filtering -----------------------------------------------------------

    def get_by_tool(self, tool_name: str) -> list[EvidenceRecord]:
        """Return all evidence records produced by a specific tool."""
        return [
            r.model_copy(deep=True)
            for r in self._records.values()
            if r.provenance.tool == tool_name
        ]

    def get_by_type(self, evidence_type: str) -> list[EvidenceRecord]:
        """Return all evidence records of a specific type."""
        return [
            r.model_copy(deep=True)
            for r in self._records.values()
            if r.type == evidence_type
        ]

    def get_by_region(self, region_id: str) -> list[EvidenceRecord]:
        """Return all evidence records associated with a specific region.

        Matches when the record's region dict contains a key 'id'
        equal to region_id, or when the entire region dict is
        {"id": region_id}.
        """
        results: list[EvidenceRecord] = []
        for r in self._records.values():
            if r.region is not None and r.region.get("id") == region_id:
                results.append(r.model_copy(deep=True))
        return results

    # -- Introspection -------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, evidence_id: str) -> bool:
        return evidence_id in self._records
