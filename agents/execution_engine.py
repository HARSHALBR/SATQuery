"""Execution Engine for GeoVision.

Consumes a WorkflowPlan from the Constrained Planner and executes its
DAG of WorkflowSteps using a dependency-injected ToolRunner interface.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Protocol
from datetime import datetime, timezone

from schemas.workflow import WorkflowPlan, WorkflowStatus
from schemas.tools import ToolResult, ToolStatus
from schemas.trace import TraceStep, ExecutionTrace
from schemas.evidence import EvidenceRecord

def _utcnow() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)

class ToolRunner(Protocol):
    """Clean abstraction for tool execution."""
    def execute(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        ...

class ExecutionEngine:
    """Orchestrates the execution of a WorkflowPlan."""
    
    def __init__(self, runner: ToolRunner):
        self.runner = runner

    def execute_plan(self, plan: WorkflowPlan) -> tuple[ExecutionTrace, List[EvidenceRecord]]:
        """Executes the workflow plan and collects traces and evidence.
        
        Args:
            plan: The WorkflowPlan produced by the Constrained Planner.
            
        Returns:
            A tuple containing the ExecutionTrace and a list of EvidenceRecords.
        """
        plan.status = WorkflowStatus.EXECUTING
        
        started_at = _utcnow()
        trace = ExecutionTrace(
            trace_id=str(uuid.uuid4()),
            workflow_id=plan.workflow_id,
            started_at=started_at
        )
        
        evidence_records: List[EvidenceRecord] = []
        step_statuses: Dict[int, ToolStatus] = {}
        trace_steps: List[TraceStep] = []
        step_outputs: Dict[int, Dict[str, Any]] = {}
        
        workflow_failed = False
        
        # Since the planner validates that for any step i, all dep_idx < i,
        # sequential iteration guarantees topological execution order.
        for i, step in enumerate(plan.steps):
            # 1. Check dependencies
            deps_failed = False
            for dep_idx in step.depends_on:
                dep_status = step_statuses.get(dep_idx)
                if dep_status == ToolStatus.ERROR:
                    deps_failed = True
                    break
                elif dep_status in [ToolStatus.UNAVAILABLE, ToolStatus.SKIPPED]:
                    # Terminal synthesis tools (compare_evidence, generate_response) proceed
                    # if other parallel tools were able to run.
                    if step.tool in ["compare_evidence", "generate_response"]:
                        continue
                    deps_failed = True
                    break
            
            if deps_failed:
                step_statuses[i] = ToolStatus.SKIPPED
                trace_steps.append(TraceStep(
                    step=i,
                    tool=step.tool,
                    status=ToolStatus.SKIPPED,
                    error="Skipped due to dependency failure."
                ))
                continue
            
            # 2. Prepare parameters
            merged_params = dict(step.parameters)
            for dep_idx in step.depends_on:
                dep_output = step_outputs.get(dep_idx)
                if dep_output:
                    merged_params.update(dep_output)
            
            # 3. Execute with error boundary
            start_ms = time.perf_counter()
            try:
                result = self.runner.execute(step.tool, merged_params)
            except Exception as e:
                # Catch unhandled runtime exceptions gracefully
                result = ToolResult(
                    tool=step.tool,
                    status=ToolStatus.ERROR,
                    error=str(e)
                )
            end_ms = time.perf_counter()
            duration_ms = result.duration_ms if result.duration_ms is not None else int((end_ms - start_ms) * 1000)
            
            step_statuses[i] = result.status
            if result.status == ToolStatus.SUCCESS and result.output:
                step_outputs[i] = result.output
                
            # 4. Collect Evidence
            if result.output and "evidence" in result.output:
                ev_data = result.output["evidence"]
                if isinstance(ev_data, list):
                    for item in ev_data:
                        if isinstance(item, EvidenceRecord):
                            evidence_records.append(item)
                        elif isinstance(item, dict):
                            evidence_records.append(EvidenceRecord(**item))
                elif isinstance(ev_data, EvidenceRecord):
                    evidence_records.append(ev_data)
                elif isinstance(ev_data, dict):
                    evidence_records.append(EvidenceRecord(**ev_data))
                    
            # 5. Record TraceStep
            trace_steps.append(TraceStep(
                step=i,
                tool=step.tool,
                status=result.status,
                duration_ms=duration_ms,
                error=result.error,
                input_summary=str(merged_params) if merged_params else None,
                output_summary=str(result.output) if result.output else None
            ))
            
            if result.status == ToolStatus.ERROR:
                workflow_failed = True
        
        # 6. Finalize Trace and Plan
        trace.steps = trace_steps
        trace.completed_at = _utcnow()
        trace.total_duration_ms = int((trace.completed_at - started_at).total_seconds() * 1000)
        
        has_any_success = any(s.status == ToolStatus.SUCCESS for s in trace_steps)
        if workflow_failed or not has_any_success:
            plan.status = WorkflowStatus.FAILED
        else:
            plan.status = WorkflowStatus.COMPLETED
            
        return trace, evidence_records
