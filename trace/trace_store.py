"""Trace Store for SATQuery AI.

Provides an authoritative in-memory store for ExecutionTrace objects
produced by the ExecutionEngine.

The TraceStore is responsible only for storage and retrieval of completed
traces. It does not execute tools, modify plans, or perform classification.
Stored traces are deep-copied to maintain immutability.
"""

from __future__ import annotations

from typing import Optional

from schemas.trace import ExecutionTrace


class DuplicateTraceError(ValueError):
    """Raised when attempting to add an ExecutionTrace with a trace_id that already exists."""
    pass


class TraceStore:
    """In-memory store for ExecutionTrace objects.

    Traces are keyed by trace_id.
    Insertion order is preserved for deterministic listing.
    All returned traces are deep copies of the stored data.
    """

    def __init__(self) -> None:
        self._traces: dict[str, ExecutionTrace] = {}

    def add(self, trace: ExecutionTrace) -> None:
        """Add an execution trace to the store.

        Args:
            trace: The ExecutionTrace to add.

        Raises:
            DuplicateTraceError: If the trace_id already exists.
        """
        if trace.trace_id in self._traces:
            raise DuplicateTraceError(
                f"Trace with ID '{trace.trace_id}' already exists in the store."
            )
        self._traces[trace.trace_id] = trace.model_copy(deep=True)

    def get(self, trace_id: str) -> Optional[ExecutionTrace]:
        """Retrieve a single execution trace by its ID.

        Returns:
            The ExecutionTrace if found, otherwise None.
        """
        record = self._traces.get(trace_id)
        if record is None:
            return None
        return record.model_copy(deep=True)

    def list(self) -> list[ExecutionTrace]:
        """Return all traces in insertion order."""
        return [t.model_copy(deep=True) for t in self._traces.values()]

    def delete(self, trace_id: str) -> None:
        """Delete a trace by its ID. Does nothing if ID not found."""
        self._traces.pop(trace_id, None)

    def clear(self) -> None:
        """Remove all traces from the store."""
        self._traces.clear()

    def get_by_workflow(self, workflow_id: str) -> list[ExecutionTrace]:
        """Return all execution traces associated with a specific workflow_id."""
        return [
            t.model_copy(deep=True)
            for t in self._traces.values()
            if t.workflow_id == workflow_id
        ]
