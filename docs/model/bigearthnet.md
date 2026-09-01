# BigEarthNet.txt Multi-Modal Data Engineering Pipeline

**SATQuery AI — SIH 2026 Problem Statement 167 (Member 1: Core VLM & Data Engineering)**

---

## 1. Official Dataset Source & References

`BigEarthNet.txt` is an Earth observation instruction dataset and multi-modal benchmark curated by the Remote Sensing Image Analysis (RSiM) Group at TU Berlin and the Berlin Institute for the Foundations of Learning and Data (BIFOLD).

- **Official Paper**: *"BigEarthNet.txt: A Large-Scale Multi-Sensor Image-Text Dataset and Benchmark for Earth Observation"*, ArXiv:2603.29630 (2026).
- **Project Website**: [https://txt.bigearth.net](https://txt.bigearth.net)
- **Hugging Face Hub**: [`BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt)
- **Image Archive (BigEarthNet v2.0 / reBEN)**: [https://bigearth.net](https://bigearth.net)
- **License**: Community Data License Agreement – Permissive – Version 1.0 (CDLA-Permissive-1.0)

---

## 2. Dataset Hierarchy & Storage Structure

All raw dataset files must reside under `data/bigearthnet_txt/`. Dataset files are ignored by git (`.gitignore` contains `data/`).

```text
data/bigearthnet_txt/
├── BigEarthNet.txt.parquet              # Text instructions, queries, ground truths, & metadata
├── images_s1/                           # Sentinel-1 SAR patch folders
│   └── S1A_IW_GRDH_1SDV_.../
│       ├── S1A_..._VV.tif               # Vertical transmit / Vertical receive (10m resolution)
│       └── S1A_..._VH.tif               # Vertical transmit / Horizontal receive (10m resolution)
└── images_s2/                           # Sentinel-2 Multispectral patch folders
    └── S2A_MSIL2A_.../
        ├── S2A_..._B01.tif              # Coastal aerosol (60m)
        ├── S2A_..._B02.tif              # Blue (10m)
        ├── S2A_..._B03.tif              # Green (10m)
        ├── S2A_..._B04.tif              # Red (10m)
        ├── S2A_..._B05.tif              # Vegetation red edge 1 (20m)
        ├── S2A_..._B06.tif              # Vegetation red edge 2 (20m)
        ├── S2A_..._B07.tif              # Vegetation red edge 3 (20m)
        ├── S2A_..._B08.tif              # NIR broad (10m)
        ├── S2A_..._B8A.tif              # Narrow NIR (20m)
        ├── S2A_..._B09.tif              # Water vapour (60m)
        ├── S2A_..._B11.tif              # SWIR 1 (20m)
        └── S2A_..._B12.tif              # SWIR 2 (20m)
```

---

## 3. Band Characteristics & Normalization Statistics

Each optical and SAR band is standardized to a uniform spatial grid ($120 \times 120$ pixels for the native $1.2\,\text{km} \times 1.2\,\text{km}$ patch) and normalized using the official dataset statistics:

| Sensor | Band | Description | Native Res. | Native Shape | Official Mean | Official Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1 SAR** | `VV` | Dual-pol Vertical | 10m | $120 \times 120$ | `-12.64386` | `5.13349` |
| **S1 SAR** | `VH` | Cross-pol Vertical/Horiz. | 10m | $120 \times 120$ | `-19.35256` | `5.59051` |
| **S2 Optical** | `B01` | Coastal Aerosol | 60m | $20 \times 20$ | `361.07678` | `575.06873` |
| **S2 Optical** | `B02` | Blue | 10m | $120 \times 120$ | `438.37207` | `607.02686` |
| **S2 Optical** | `B03` | Green | 10m | $120 \times 120$ | `614.05566` | `603.29681` |
| **S2 Optical** | `B04` | Red | 10m | $120 \times 120$ | `588.40961` | `684.56885` |
| **S2 Optical** | `B05` | Red Edge 1 | 20m | $60 \times 60$ | `942.84332` | `738.43268` |
| **S2 Optical** | `B06` | Red Edge 2 | 20m | $60 \times 60$ | `1769.93164` | `1100.45605` |
| **S2 Optical** | `B07` | Red Edge 3 | 20m | $60 \times 60$ | `2049.55151` | `1275.80542` |
| **S2 Optical** | `B08` | NIR Broad | 10m | $120 \times 120$ | `2193.29199` | `1369.37170` |
| **S2 Optical** | `B8A` | Narrow NIR | 20m | $60 \times 60$ | `2235.55664` | `1356.54407` |
| **S2 Optical** | `B09` | Water Vapour | 60m | $20 \times 20$ | `2241.45532` | `1316.39331` |
| **S2 Optical** | `B11` | SWIR 1 | 20m | $60 \times 60$ | `1568.22681` | `1070.16125` |
| **S2 Optical** | `B12` | SWIR 2 | 20m | $60 \times 60$ | `997.73248` | `813.52765` |

---

## 4. Manifest Schema & Generation

Manifests are stored in `data/manifests/` in line-delimited JSON (`.jsonl`) format:

- `manifest_full.jsonl`: Complete index of all dataset samples.
- `manifest_train.jsonl`: Training split partition.
- `manifest_validation.jsonl`: Validation split partition.
- `manifest_test.jsonl`: Test split partition.
- `manifest_summary.json`: Detailed audit counts, distribution metrics, and execution metadata.

### Sample Manifest Line (JSON Schema)

```json
{
  "sample_id": "ben_txt_000001",
  "image_id": "S2A_MSIL2A_20170613T101031_N0205_R022_T32ULD_22_38",
  "s1_name": "S1A_IW_GRDH_1SDV_20170613T165043_33N_53E_6367f0",
  "s1_path": "images_s1/S1A_IW_GRDH_1SDV_20170613T165043_33N_53E_6367f0",
  "s2_path": "images_s2/S2A_MSIL2A_20170613T101031_N0205_R022_T32ULD_22_38",
  "text_input": "Is coniferous forest present in this region?",
  "text_output": "Yes, coniferous forest is present.",
  "task_type": "binary",
  "task_category": "presence",
  "split": "train",
  "metadata": {
    "latitude": 45.1234,
    "longitude": 12.5678,
    "country": "Austria",
    "season": "Summer",
    "climate_zone": "Cfb",
    "patch_id": "S2A_MSIL2A_20170613T101031_N0205_R022_T32ULD_22_38",
    "s1_name": "S1A_IW_GRDH_1SDV_20170613T165043_33N_53E_6367f0",
    "split": "train"
  },
  "is_valid": true,
  "validation_errors": []
}
```

---

## 5. Deterministic Dataset Splitting Strategy

1. **Patch-Level Grouping**: Splitting is performed on the distinct `patch_id` set rather than individual question rows. This ensures that all instructions and questions derived from the same spatial area remain strictly within one split, eliminating spatial and multi-modal data leakage between training and testing.
2. **Fixed Random Seed**: Splitting utilizes a deterministic NumPy generator with configurable seed (default: `42`). Repeated runs produce byte-for-byte identical manifests.
3. **Zero Overlap Guarantee**: Automated unit and pipeline tests enforce that $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, and $\text{Val} \cap \text{Test} = \emptyset$.
4. **Preservation of Official Benchmark**: If official split columns are provided in the source metadata, they are preserved by default to enable direct benchmarking against published literature.

---

## 6. Validation Rules & Corruption Auditing

The validation engine (`scripts/validate_bigearthnet.py`) evaluates every sample against six criteria:

1. **S1 Band Completeness**: Checks presence and readability of both `VV` and `VH` GeoTIFFs.
2. **S2 Band Completeness**: Checks presence and readability of all 12 spectral bands (`B01` through `B12`).
3. **Raster Data Integrity**: Inspects TIFF headers, verifies array dimensions match native/target shapes, and checks for corrupted 0-byte or all-NaN/all-Inf arrays.
4. **Multi-Sensor Pairing**: Verifies co-registration identifier naming conventions and asserts that S1 and S2 spatial footprints match within valid tolerance.
5. **Annotation Non-Emptiness**: Validates that natural language queries (`input`) and answers (`output`) are non-empty.
6. **Geographic Coordinates**: Asserts $-90.0 \le \text{lat} \le 90.0$ and $-180.0 \le \text{lon} \le 180.0$.

Invalid samples are never dropped silently; they are recorded with error reason codes in `reports/invalid_samples.jsonl` and summarized in `reports/validation_report.json`.

---

## 7. PyTorch Dataset Interface

### Usage Example

```python
from torch.utils.data import DataLoader
from data.bigearthnet_txt import BigEarthNetDataset, collate_bigearthnet

# Initialize dataset
dataset = BigEarthNetDataset(
    manifest_path="data/manifests/manifest_train.jsonl",
    data_root="data/bigearthnet_txt",
    s1_bands=["VV", "VH"],
    s2_bands="all",           # or "RGB", "S2-10m20m"
    img_size=120,             # or 224 / 448 for vision-language backbones
    is_training=True,
    normalize=True,
)

# DataLoader
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_bigearthnet,
)

for batch in loader:
    s1 = batch["image_s1"]           # Tensor shape: [16, 2, 120, 120], float32
    s2 = batch["image_s2"]           # Tensor shape: [16, 12, 120, 120], float32
    prompts = batch["text"]          # List of 16 query strings
    targets = batch["target_text"]   # List of 16 answer strings
    metadata = batch["metadata"]     # List of 16 metadata dicts
    break
```

---

## 8. CLI Commands & Execution Guide

### Manifest Generation Command
```bash
python scripts/build_bigearthnet_manifest.py \
    --data-root data/bigearthnet_txt \
    --output-dir data/manifests \
    --seed 42 \
    --train-ratio 0.70 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

### Dataset Validation Command
```bash
python scripts/validate_bigearthnet.py \
    --manifest data/manifests/manifest_full.jsonl \
    --data-root data/bigearthnet_txt \
    --report-output reports/validation_report.json \
    --invalid-log reports/invalid_samples.jsonl
```

### Running Test Suite
```bash
pytest tests/ -v
```

---

## 8. Sentinel-2 Raw vs. Model Band Mapping

BigEarthNet v2.0 S2 patches ship **12 L2A bands** per tile. The RS-InternVL `S2Encoder` is designed to receive only **10 bands**, excluding the two 60 m coarse-resolution bands that carry negligible discriminative spectral information at the 120 px patch scale.

### Raw Tile Bands (12 total — what lives on disk)

| Raw Idx | Band | Description          | Resolution | Model? |
|---------|------|----------------------|------------|--------|
| 0       | B01  | Coastal aerosol      | 60 m       | ❌ Excluded |
| 1       | B02  | Blue                 | 10 m       | ✅ ch 0 |
| 2       | B03  | Green                | 10 m       | ✅ ch 1 |
| 3       | B04  | Red                  | 10 m       | ✅ ch 2 |
| 4       | B05  | Red Edge 1           | 20 m       | ✅ ch 3 |
| 5       | B06  | Red Edge 2           | 20 m       | ✅ ch 4 |
| 6       | B07  | Red Edge 3           | 20 m       | ✅ ch 5 |
| 7       | B08  | NIR Broad            | 10 m       | ✅ ch 6 |
| 8       | B8A  | Narrow NIR           | 20 m       | ✅ ch 7 |
| 9       | B09  | Water vapour         | 60 m       | ❌ Excluded |
| 10      | B11  | SWIR 1               | 20 m       | ✅ ch 8 |
| 11      | B12  | SWIR 2               | 20 m       | ✅ ch 9 |

### Single Source of Truth

The 10-band list is defined **once** in:
```
data/bigearthnet_txt/constants.py  →  MODEL_S2_BANDS
```

`RSInternVLConfig` (in `models/rs_internvl/config.py`) imports `MODEL_S2_BANDS` from the data package, so the model and the dataset always agree.

### Dataset API

```python
from data.bigearthnet_txt import BigEarthNetDataset

# Default (s2_bands=None) → MODEL_S2_BANDS: 10-channel tensor [B, 10, H, W]
ds = BigEarthNetDataset(manifest_path=..., data_root=...)

# Explicit 10-band model input (equivalent)
ds = BigEarthNetDataset(..., s2_bands="S2-10m20m")

# Raw 12-band tensor (for analysis only — NOT model-ready)
ds_raw = BigEarthNetDataset(..., s2_bands="S2-all")
```

> **Important:** Passing a 12-channel S2 tensor to `RSInternVL.encode_vision()` will raise a `ValueError` with a clear diagnostic message. Never pass raw 12-band tensors directly to the model.
