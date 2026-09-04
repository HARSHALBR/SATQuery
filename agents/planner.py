import uuid
import yaml
from copy import deepcopy
from typing import List, Dict, Any, Optional

from schemas.query import ParsedQuery, TaskType
from schemas.workflow import WorkflowPlan, WorkflowStep, WorkflowStatus
from schemas.tools import ToolApplicability
from agents.tool_registry import ToolRegistry, ToolExecutionContext

class ConstrainedPlanner:
    def __init__(self, registry, config_path: str):
        self.registry = registry
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.workflows = self.config.get('workflows', {})

    def _validate_dag(self, steps: List[WorkflowStep]) -> bool:
        """Validate no forward references, no cycles, valid bounds."""
        for i, step in enumerate(steps):
            for dep in step.depends_on:
                if dep >= i or dep < 0:
                    return False
        return True

    def create_plan(self, query, context) -> WorkflowPlan:
        task_name = getattr(query.task, "name", str(query.task)).lower()
        
        if task_name == "insufficient_capability":
            return WorkflowPlan(
                workflow_id=str(uuid.uuid4()),
                task=query.task,
                steps=[],
                status=WorkflowStatus.FAILED,
                fallback="Insufficient capability to fulfill query."
            )

        if task_name not in self.workflows:
            return WorkflowPlan(
                workflow_id=str(uuid.uuid4()),
                task=query.task,
                steps=[],
                status=WorkflowStatus.FAILED,
                fallback=f"No workflow template found for {task_name}"
            )
            
        template = deepcopy(self.workflows[task_name])
        
        # Dynamic filtering for spatial_measurement based on intents or claims
        if task_name == "spatial_measurement":
            needs_grounding = (hasattr(query, 'intent') and query.intent in ['locate', 'find']) or (hasattr(query, 'claim') and query.claim)
            if needs_grounding:
                template = [s for s in template if s['tool'] != 'change_statistics']
            else:
                template = [s for s in template if s['tool'] != 'grounding']
            
            # Ensure area_measurement dependency is sound after filtering
            available_tools = {s['tool'] for s in template}
            for step in template:
                if step['tool'] == 'area_measurement':
                    step['depends_on'] = [dep for dep in step['depends_on'] if dep in available_tools]

        sim_ctx = context.model_copy(deep=True)
        planned_tools = []
        workflow_steps = []
        tool_to_idx = {}
        
        for i, step_dict in enumerate(template):
            tool_name = step_dict['tool']
            depends_on_names = step_dict.get('depends_on', [])
            
            tool_def = self.registry.get_tool(tool_name)
            if not tool_def:
                return WorkflowPlan(
                    workflow_id=str(uuid.uuid4()),
                    task=query.task,
                    steps=[],
                    status=WorkflowStatus.FAILED,
                    fallback=f"Unknown tool in workflow: {tool_name}"
                )
            
            sim_ctx.completed_tools = list(planned_tools)
            
            # Applicability check
            is_applicable, reason = self.registry.check_applicability(tool_def, sim_ctx)
            
            if not is_applicable:
                return WorkflowPlan(
                    workflow_id=str(uuid.uuid4()),
                    task=query.task,
                    steps=[],
                    status=WorkflowStatus.FAILED,
                    fallback=reason
                )
                
            depends_on_indices = []
            for dep_name in depends_on_names:
                if dep_name not in tool_to_idx:
                    return WorkflowPlan(
                        workflow_id=str(uuid.uuid4()),
                        task=query.task,
                        steps=[],
                        status=WorkflowStatus.FAILED,
                        fallback=f"Missing dependency: {dep_name}"
                    )
                depends_on_indices.append(tool_to_idx[dep_name])
                
            step = WorkflowStep(
                tool=tool_name,
                depends_on=depends_on_indices
            )
            workflow_steps.append(step)
            planned_tools.append(tool_name)
            tool_to_idx[tool_name] = len(workflow_steps) - 1

        if not self._validate_dag(workflow_steps):
            return WorkflowPlan(
                workflow_id=str(uuid.uuid4()),
                task=query.task,
                steps=[],
                status=WorkflowStatus.FAILED,
                fallback="Invalid DAG generated."
            )

        return WorkflowPlan(
            workflow_id=str(uuid.uuid4()),
            task=query.task,
            steps=workflow_steps,
            status=WorkflowStatus.PLANNED
        )
