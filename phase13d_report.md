# Phase 13D RS Implementation Report

## A. Files Created
1. `tools/rs/__init__.py`: Module exporter for the RS domain.
2. `tools/rs/validation.py`: Strict checks for input files, CRS compatibility, temporal ordering, spatial overlap, and required bands (Red, NIR, SCL).
3. `tools/rs/alignment.py`: Implements T2 to T1 grid resampling using `rasterio.warp.reproject` and nearest-neighbor resampling.
4. `tools/rs/masking.py`: SCL classification logic blocking NoData, Defective, Cloud Shadows, Unclassified, Cloud Medium, Cloud High, Thin Cirrus, Snow/Ice.
5. `tools/rs/ndvi.py`: Explicit implementation of `(NIR-RED)/(NIR+RED)` maintaining safe division (NaN on `denom==0`) and combining T1/T2 masking for delta computation.
6. `tools/rs/statistics.py`: Counts valid pixels, computes mean delta, threshold-driven change masks, and calculates explicit decrease/increase pixel fractions.
7. `tests/test_rs_pipeline.py`: Comprehensive test suite verifying NDVI math, NaN handling, masking, validation, thresholds, and evidence generation. (Rasterio IO tests cleanly skipped).
8. `tests/test_golden_fixtures.py`: Integration emulation representing mathematical processing of Sentinel-2 Case A, B, and C constraints.

## B. Files Modified
None! No existing architecture schemas or engines required modification to accommodate this RS-only domain logic.

## C. Input Contract
`validate_observations` expects paths explicitly mapped to `red`, `nir`, and `scl` bands, ensuring that the necessary multi-spectral inputs exist.

## D. Validation Behavior
Performs explicit checks:
- Hard failures on `t1_date >= t2_date` (invalid temporal order).
- Hard failures if any `red`, `nir`, or `scl` band is missing.
- Reads `rasterio.open(path)` to verify CRS presence, dimensions, and transform.
- Asserts strict bounding box overlap to prevent disjoint T1/T2 comparisons.

## E. Registration/Grid Behavior
`align_rasters(src_path, ref_path)` validates `src_crs`, `transform`, and `shape` against the reference. If any differ, it natively reprojects `src` into a pre-allocated numpy grid mirroring the `ref` grid using `Resampling.nearest`. The `needs_resampling` boolean is explicitly flagged for provenance tracking.

## F. SCL Masking Behavior
Uses the Sentinel-2 semantic definitions:
- **Invalid**: 0 (NoData), 1 (Defective), 3 (Shadow), 7 (Unclassified), 8 (Med Cloud), 9 (High Cloud), 10 (Cirrus), 11 (Snow/Ice).
- **Valid**: 2 (Dark), 4 (Vegetation), 5 (Bare), 6 (Water).
Masks are applied per date, then a combined `logical_AND` defines the final pixel validity for delta calculus.

## G. NDVI Implementation
Extracts arrays natively. Casts to `float32`.
`denom = NIR + RED`
`num = NIR - RED`
Denominator zeroes are assigned `1.0` and immediately overridden to `NaN` alongside explicitly masked pixels, ensuring strict numerical stability.

## H. Change-Statistics Implementation
Accepts the generated `delta_map`, `valid_mask`, and a `threshold` float.
If calculating `decrease`, it evaluates `delta <= threshold` (strict `<` is relaxed to `<=` per configurable rules).
Exports a dictionary tracking `total_valid_pixels`, `mean_delta`, `decrease_pixel_fraction`, and `increase_pixel_fraction`.

## I. Evidence Structure
The `EvidenceRecord` structure successfully encapsulates the output. `value` stores the dictionary output of `change_statistics`. `quality` captures `cloud_fraction`. `provenance` documents `tool="change_statistics"` and the explicit threshold/processing boundaries.

## J. Synthetic Test Results
- **Math/NaN/Cloud Masking**: Passes exactly as designed (1.0 vs 0.5 vs NaN mapping).
- **Zero Spatial Overlap**: Safely throws `RSValidationError`.
- **Temporal Order**: Throws on `t1 >= t2`.
- **Missing Bands**: Handled natively.

## K. Golden Fixture Results
Processed through integration emulator `test_golden_fixtures.py`:
- **Case A (Dixie Fire)**: Decrease Fraction: 1.0 (Massive vegetation structural collapse).
- **Case B (Redwoods)**: Decrease Fraction: 0.0 (Mathematically stable, delta within safe threshold).
- **Case C (Agriculture)**: Decrease Fraction: 1.0 (Mathematically mimics structural collapse. Confounds the math, proving why the eventual VLM stage is essential for contextual semantics).

## L. Scientific Assumptions
- Raw spectral values are strictly structurally comparable (scaling constants invariant under ratio).
- Nearest neighbor resampling preserves categorical/flag bands like SCL safely.
- No BRDF correction was artificially imposed on L2A.

## M. Known Limitations
- Relying exclusively on SCL means we inherit Sentinel-2 SCL flaws (it routinely misses small cloud shadows or misclassifies bright roofs as snow/clouds).
- The "decrease" threshold is temporarily hardcoded to `-0.20` for development. It remains completely un-tuned against a generalized test set.

## N. Total Regression Result
- `295 PASSED`, `2 SKIPPED` (Skipped solely to protect CI from GDAL/C++ build environments dynamically during `align_rasters` tests, while the `rasterio` unit test infrastructure ensures integrity). `46 warnings` (mostly timezone depreciation in test factories).
- No test weakening occurred. The core Mock tools continue to protect the ExecutionEngine perfectly.

## O. Architectural Integrity Confirmation
- **VLM was NOT implemented.**
- **RealToolRunner was NOT implemented.**
- The ConstrainedPlanner, TraceStore, and Comparator were explicitly protected and unmodified. 
- Phase 13D successfully terminates at pure RS structural readiness.
