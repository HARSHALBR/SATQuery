import re
path = r'd:\Sem 5\SIH\satquery-ai\tests\test_planner.py'
with open(path, 'r') as f: content = f.read()

# Fix test_dep_unavailable_tool
content = content.replace(
    'mock_registry.check_applicability.return_value = (True, "")\n    planner = ConstrainedPlanner(mock_registry, temp_config)\n    q = MagicMock(); q.task = "single_image_vqa"\n    assert planner.create_plan(q, mock_context).status == WorkflowStatus.FAILED',
    'mock_registry.check_applicability.return_value = (False, "Unavailable")\n    planner = ConstrainedPlanner(mock_registry, temp_config)\n    q = MagicMock(); q.task = "single_image_vqa"\n    assert planner.create_plan(q, mock_context).status == WorkflowStatus.FAILED'
)

# Fix test_beh_fallback_on_tool_failure
content = content.replace(
    'mock_registry.check_applicability.return_value = (True, "")\n    planner = ConstrainedPlanner(mock_registry, temp_config)\n    q = MagicMock(); q.task = "single_image_vqa"\n    plan = planner.create_plan(q, mock_context)\n    assert plan.status == WorkflowStatus.FAILED',
    'mock_registry.check_applicability.return_value = (False, "Fallback reason test")\n    planner = ConstrainedPlanner(mock_registry, temp_config)\n    q = MagicMock(); q.task = "single_image_vqa"\n    plan = planner.create_plan(q, mock_context)\n    assert plan.status == WorkflowStatus.FAILED'
)

# Fix test_dep_unknown_tool
content = content.replace(
    '        with open(temp_config, "w") as f: yaml.dump({"workflows": {"single_image_vqa": [{"tool": "unknown_tool", "depends_on": []}]}}, f)\n        planner = ConstrainedPlanner(mock_registry, temp_config)\n        q = MagicMock(); q.task = "single_image_vqa"\n        assert planner.create_plan(q, mock_context).status == WorkflowStatus.FAILED',
    '        with open(temp_config, "w") as f: yaml.dump({"workflows": {"single_image_vqa": [{"tool": "unknown_tool", "depends_on": []}]}}, f)\n        mock_registry.get_tool.side_effect = lambda n: None if n == "unknown_tool" else MagicMock(name=n)\n        planner = ConstrainedPlanner(mock_registry, temp_config)\n        q = MagicMock(); q.task = "single_image_vqa"\n        assert planner.create_plan(q, mock_context).status == WorkflowStatus.FAILED'
)

with open(path, 'w') as f: f.write(content)
