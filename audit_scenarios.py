import json
from pathlib import Path

from schemas.query import ParsedQuery, QueryInput, ObservationInput, ObservationRole, ImageMetadata, Modality, TaskType
from agents.tool_registry import ToolRegistry, ToolExecutionContext
from agents.planner import ConstrainedPlanner
from agents.task_classifier import TaskClassifier

def print_plan(name, plan):
    print(f"--- {name} ---")
    print(f"Task: {plan.task}")
    print(f"Status: {plan.status}")
    if plan.fallback:
        print(f"Fallback: {plan.fallback}")
    print("Steps:")
    for i, step in enumerate(plan.steps):
        print(f"  [{i}] {step.tool} (depends_on: {step.depends_on})")
    print("\n")

def run_audit():
    classifier = TaskClassifier()
    registry = ToolRegistry()
    registry = ToolRegistry.from_yaml(Path("configs/tools.yaml"))
    planner = ConstrainedPlanner(registry, Path("configs/workflows.yaml"))

    # A. Vegetation decreased + Optical
    q_a = "Has vegetation decreased?"
    obs_a = [
        ObservationInput(observation_id="o1", image_path="fake/path1.tif", role=ObservationRole.T1, metadata=ImageMetadata(modality=Modality.OPTICAL, bands=["red", "nir", "green", "blue"], sensor="Sentinel-2", registered=True)),
        ObservationInput(observation_id="o2", image_path="fake/path2.tif", role=ObservationRole.T2, metadata=ImageMetadata(modality=Modality.OPTICAL, bands=["red", "nir", "green", "blue"], sensor="Sentinel-2", registered=True))
    ]
    parsed_a = classifier.classify(QueryInput(query=q_a, observations=obs_a))
    ctx_a = ToolExecutionContext(query=parsed_a, observations=obs_a)
    plan_a = planner.create_plan(parsed_a, ctx_a)
    print_plan("Scenario A", plan_a)

    # B. Vegetation decreased + SAR only
    obs_b = [
        ObservationInput(observation_id="s1", image_path="fake/path3.tif", role=ObservationRole.SAR_T1, metadata=ImageMetadata(modality=Modality.SAR, sensor="Sentinel-1", registered=True)),
        ObservationInput(observation_id="s2", image_path="fake/path4.tif", role=ObservationRole.SAR_T2, metadata=ImageMetadata(modality=Modality.SAR, sensor="Sentinel-1", registered=True))
    ]
    parsed_b = classifier.classify(QueryInput(query=q_a, observations=obs_b))
    ctx_b = ToolExecutionContext(query=parsed_b, observations=obs_b)
    plan_b = planner.create_plan(parsed_b, ctx_b)
    print_plan("Scenario B", plan_b)

    # C. Generic change + Optical
    q_c = "What changed between these two images?"
    parsed_c = classifier.classify(QueryInput(query=q_c, observations=obs_a))
    ctx_c = ToolExecutionContext(query=parsed_c, observations=obs_a)
    plan_c = planner.create_plan(parsed_c, ctx_c)
    print_plan("Scenario C", plan_c)

    # D. SAR support + Optical/SAR
    q_d = "Does SAR support the suspected change?"
    obs_d = obs_a + obs_b
    parsed_d = classifier.classify(QueryInput(query=q_d, observations=obs_d))
    ctx_d = ToolExecutionContext(query=parsed_d, observations=obs_d)
    plan_d = planner.create_plan(parsed_d, ctx_d)
    print_plan("Scenario D", plan_d)
    
    # 5. Verify A and D are different
    print("--- Verify A != D ---")
    print(f"Different tasks: {plan_a.task != plan_d.task}")
    print(f"Different steps len: {len(plan_a.steps)} vs {len(plan_d.steps)}")
    steps_a = [s.tool for s in plan_a.steps]
    steps_d = [s.tool for s in plan_d.steps]
    print(f"Steps A: {steps_a}")
    print(f"Steps D: {steps_d}")
    print("\n")
    
    # 6. Verify Determinism
    print("--- Verify Determinism (Scenario A x 3) ---")
    plan_a1 = planner.create_plan(parsed_a, ctx_a)
    plan_a2 = planner.create_plan(parsed_a, ctx_a)
    plan_a3 = planner.create_plan(parsed_a, ctx_a)
    # Check if they are identical (ignoring the generated UUID workflow_id)
    def to_dict(p): 
        d = p.model_dump() if hasattr(p, 'model_dump') else p.dict()
        if 'workflow_id' in d:
            del d['workflow_id']
        return d
    print(f"Run 1 == Run 2: {to_dict(plan_a1) == to_dict(plan_a2)}")
    print(f"Run 2 == Run 3: {to_dict(plan_a2) == to_dict(plan_a3)}")

if __name__ == "__main__":
    run_audit()
