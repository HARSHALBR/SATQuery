# PHASE 13E — REALTOOLRUNNER REPORT

## A. Files created
- `agents/real_runner.py`: The `RealToolRunner` adapter.
- `tests/test_real_runner_integration.py`: Integration test suite for physical execution.

## B. Files modified
- `backend/services/orchestrator.py`: Modified `__init__` to accept an optional `runner` parameter, enabling Dependency Injection (DI) of the `RealToolRunner` without destroying `MockToolRunner` defaults for the existing suite.

## C. Dispatcher architecture
`RealToolRunner` acts as a pure in-memory state bridge. It satisfies the `ToolRunner` Protocol's `execute(tool_name, parameters)` contract. 
Instead of modifying `ExecutionEngine` to inject physical data files dynamically into the DAG parameters, `RealToolRunner` is initialized with the `ObservationInput` list. 
It uses an `array_store` memory mapping to pass heavy NumPy arrays (like the intermediate `delta` and `final_mask`) between `ndvi_delta` and `change_statistics` steps via dynamically generated UUID string keys natively supported by the string-based planner contract.

## D. Tool mappings
1. **`validate_inputs`** -> `tools.rs.validation.validate_observations`
2. **`ndvi_delta`** -> Coordinates sequential execution of:
   - `tools.rs.alignment.align_rasters`
   - `tools.rs.masking.combined_valid_mask`
   - `tools.rs.ndvi.compute_ndvi_delta`
3. **`change_statistics`** -> `tools.rs.statistics.compute_change_statistics`

## E. ToolResult conversion
Each native math call receives Python types (dicts, strings, datetimes) and outputs NumPy arrays/floats. The runner wraps these natively into standard `ToolResult` structures exactly mirroring `MockToolRunner`. 
`output` dictionaries are enriched with the requisite keys expected by downstream steps. Most critically, the runner constructs the `EvidenceRecord` embedded in `ToolResult.output["evidence"]` to satisfy `ExecutionEngine` evidence harvesting requirements.

## F. Provenance handling
The physical `ObservationInput.metadata.stac_item_id`s are collected during input validation and propagated forward through the `input_ids` parameter.
Whenever `ndvi_delta` or `change_statistics` instantiates an `EvidenceRecord`, it embeds `input_ids` into the `Provenance` block, fully securing the lineage back to the raw STAC identifiers (`S2A_10TFK_20210708_0_L2A`, etc).

## G. Case A result
- Tested via `test_real_tool_runner_case_a`.
- **Outcome:** SUCCESSFULLY dispatched through `ExecutionEngine`. 
- **Decrease fraction:** `> 0.40` (47% physically computed).
- Trace properly closed successfully.

## H. Case B result
- Tested via `test_real_tool_runner_case_b`.
- **Outcome:** SUCCESSFULLY dispatched through `ExecutionEngine`. 
- **Decrease fraction:** `< 0.05` (0.4% physically computed).

## I. Case C result
- Tested via `test_real_tool_runner_case_c`.
- **Outcome:** SUCCESSFULLY dispatched through `ExecutionEngine`. 
- **Decrease fraction:** `> 0.05` (5.8% physically computed).

## J. Failure handling
- Tested via `test_real_tool_runner_failure`. 
- Giving non-existent mock strings mapping to non-existent TIFF paths successfully triggered the Python OS/rasterio crash boundaries.
- **Outcome:** `validate_inputs` caught the error, returning `ToolStatus.ERROR`. The `ExecutionEngine` correctly cascaded `ToolStatus.SKIPPED` to `ndvi_delta` and `change_statistics`. 

## K. Tests added
1. `test_real_tool_runner_case_a`
2. `test_real_tool_runner_case_b`
3. `test_real_tool_runner_case_c`
4. `test_real_tool_runner_failure`
5. `test_real_tool_runner_orchestrator_e2e_case_a` (Orchestrator E2E proof preserving execution boundaries up to FinalResponse)

## L. Total tests
302

## M. Passed/failed/skipped
302 Passed / 0 Failed / 0 Skipped

## N. VLM/LLM confirmation
**Confirmed**: No API keys, language models, or HTTP inference calls were placed in the adapter. Tools returning `ToolStatus.UNAVAILABLE` successfully halt unsupported capabilities while maintaining RS rigor.

## O. Remaining limitations
The pipeline natively bridges strings and arrays, but memory limitations are theoretically bounded by `RealToolRunner.array_store` if processing hundreds of requests simultaneously since garbage collection mapping is not yet enforced at the workflow termination stage.
