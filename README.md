# 🛰️ GeoVision — Evidence-First Agentic Geospatial Intelligence

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Remote Sensing](https://img.shields.io/badge/GeoTIFF-Rasterio%20%7C%20Shapely-green.svg)](https://rasterio.readthedocs.io/)
[![VLM](https://img.shields.io/badge/VLM-RS--InternVL%20%7C%20Gemini-purple.svg)](https://github.com/OpenGVLab/InternVL)
[![License](https://img.shields.io/badge/License-MIT-amber.svg)](LICENSE)

**GeoVision** is an evidence-first, multi-agent geospatial reasoning system designed to answer natural language questions about satellite imagery with mathematical rigor and auditability.

Unlike conventional Vision-Language Models (VLMs) that may hallucinate visual changes or misinterpret seasonal variations as structural alterations, GeoVision enforces an **independent dual-path verification protocol**: semantic observations from multimodal models must converge with physical, quantitative remote sensing algorithms before any claim is accepted as verified truth.

---

## 🌟 Key Architecture & Highlights

```
                                 [ Natural Language Query ]
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │    01 QUERY UNDERSTANDING     │
                             │  Task, Target, Temporal Roles │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │     02 DATA QUALITY GATE      │
                             │   Band, CRS, Alignment Checks │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │       03 AGENT PLANNER        │
                             │   Dual-Path Execution Plan    │
                             └───────┬───────────────┬───────┘
                                     │               │
                     ┌───────────────┘               └───────────────┐
                     ▼                                               ▼
      ┌─────────────────────────────┐                 ┌─────────────────────────────┐
      │     PATH A · SEMANTIC       │                 │   PATH B · QUANTITATIVE     │
      │   RS-InternVL / Gemini      │                 │  Deterministic Calculations │
      │  Temporal Visual Reasoning  │                 │   NDVI / NDBI / SAR Deltas  │
      └──────────────┬──────────────┘                 └──────────────┬──────────────┘
                     │                                               │
                     └───────────────┐               ┌───────────────┘
                                     ▼               ▼
                             ┌───────────────────────────────┐
                             │     05 EVIDENCE COMPARATOR    │
                             │  Contradiction & Convergence  │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │   06 VERIFIED VERDICT & MAP   │
                             │  SUPPORTED / UNCERTAIN / FAIL │
                             │  Interactive Spatial GeoJSON  │
                             └───────────────────────────────┘
```

### 1. Dual-Path Verification
* **Path A (Semantic Analysis)**: Employs Vision-Language Models (RS-InternVL dual-encoder fusion for Sentinel-1 SAR + Sentinel-2 optical, or Gemini 3.6 Flash) to interpret visual features, land cover changes, and spatial context.
* **Path B (Physical Remote Sensing)**: Executes deterministic raster processing via Rasterio, Shapely, and NumPy—performing pixel-level alignment, cloud/shadow masking via Scene Classification Layers (SCL), radiometric index calculations (NDVI, NDBI), and change-clustering statistics.

### 2. Evidence Comparator & Agreement Gate
Independent findings from Path A and Path B feed into the Evidence Comparator:
* **`SUPPORTED`**: Both semantic reasoning and physical sensor calculations agree beyond the confidence threshold.
* **`UNCERTAIN / CONFLICTING`**: Semantic claims contradict physical measurements (e.g., visual greening caused by seasonal grass vs. actual lack of forest recovery).
* **`INSUFFICIENT`**: Missing bands, mismatched coordinate systems, excessive cloud cover, or poor raster readability fail the pre-flight Data Quality Gate.

### 3. Interactive Tabbed Workspace & Spatial Map
* **Tabbed Interface**: Query, Observations, Validation, Analysis, Evidence, Change Map, and Final Answer tabs provide instant, non-reloading inspection.
* **Full-Width Leaflet Map**: Displays the raster bounding box, georeferenced extents, and highlighted change polygons with instant auto-fitting.
* **Evidence Ledger**: Full audit trail recording tool runtimes, parameters, confidence scores, and limitation caveats.

---

## 📁 Repository Structure

```text
GeoVision/
├── agents/                       # Agentic execution and orchestration
│   ├── execution_engine.py       # Plan executor and context coordinator
│   ├── planner.py                # Constrained execution planner (Path A vs B)
│   ├── task_classifier.py        # Natural language intent & temporal parser
│   ├── tool_registry.py          # Tool registry and schema validation
│   ├── real_runner.py            # Live execution engine for real rasters
│   └── mock_tools.py             # Test harness for offline simulation
├── backend/                      # FastAPI Web Application
│   ├── main.py                   # FastAPI server entrypoint & static mount
│   ├── routes/
│   │   ├── analyze.py            # POST /api/v1/analyze endpoint
│   │   ├── upload.py             # POST /api/v1/upload (ZIP / TIFF intake)
│   │   └── health.py             # GET /health healthcheck
│   └── services/
│       └── orchestrator.py       # Pipeline workflow orchestrator
├── tools/                        # Analytical tool suite
│   ├── rs/                       # Remote sensing raster processing
│   │   ├── alignment.py          # Spatial reprojection & grid alignment
│   │   ├── masking.py            # SCL cloud, shadow, and water masking
│   │   ├── ndvi.py               # NDVI difference & delta computations
│   │   ├── statistics.py         # Affected pixel fractions & metrics
│   │   └── validation.py         # Input file readability & band verification
│   └── vlm/                      # Multimodal Vision-Language tools
│       ├── client.py             # Abstract VLM client interface
│       ├── gemini_client.py      # Google Gemini 3.6 Flash structured provider
│       ├── rs_internvl_client.py # Local RS-InternVL client
│       ├── satellite_tensors.py  # Sentinel-1 & 2 tensor pre-processors
│       ├── image_utils.py        # Composite RGB image preparation
│       └── prompt.py             # Structured prompt templates & schema guards
├── evidence/                     # Evidence evaluation & audit store
│   ├── comparator.py             # Cross-modal contradiction & agreement engine
│   └── evidence_store.py         # In-memory and persisted evidence storage
├── models/                       # Deep learning model architectures
│   └── rs_internvl/              # RS-InternVL dual-encoder fusion (SAR + Optical)
├── schemas/                      # Pydantic data contracts
│   ├── query.py                  # User query and observation requests
│   ├── response.py               # API responses, verdicts, and spatial regions
│   ├── evidence.py               # Evidence items and status enums
│   ├── tools.py                  # Tool inputs, outputs, and parameters
│   └── trace.py                  # Pipeline execution trace steps
├── configs/                      # Pipeline configuration files
│   ├── tools.yaml                # Tool registry definitions
│   ├── workflows.yaml            # Multi-agent workflow plans
│   └── model/                    # Model hyper-parameters & LoRA manifests
├── trace/                        # Auditability and tracing ledger
│   └── trace_store.py            # Latency and step execution tracker
├── training/                     # Fine-tuning and adaptation
│   ├── lora.py                   # LoRA adaptation for RS-InternVL
│   └── tokenizer.py              # Multimodal tokenizer & special tokens
├── Frontend/                     # User Interface
│   ├── index.html                # Tabbed responsive web dashboard
│   └── data/demo_scenarios/      # Pre-packaged Sentinel-2 GeoTIFF test sets
├── test_zips/                    # Sample ZIP archives for upload testing
├── requirements.txt              # Core Python dependencies
├── .env.example                  # Environment variable template
└── README.md                     # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
* Python 3.10, 3.11, 3.12, or 3.13
* GDAL / PROJ (standard with `rasterio` binary wheels)
* A modern web browser

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/HARSHALBR/SATQuery.git
cd GeoVision
git checkout demo

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the sample environment file:

```bash
cp .env.example .env
```

Key environment variables:
* `MOCK_RS_TOOLS=false`: Runs real mathematical raster calculations on GeoTIFFs using Rasterio and NumPy.
* `MOCK_VLM=true`: Uses deterministic offline VLM simulation (set to `false` and configure `GEMINI_API_KEY` to query live Google Gemini 3.6 Flash).

### 4. Running the Application

Launch the FastAPI backend server:

```bash
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload
```

Open your browser and navigate to:
👉 **[http://localhost:8001](http://localhost:8001)**

---

## 🎯 Pre-Loaded Demo Scenarios

The web interface includes three pre-configured satellite change scenarios ready for one-click demonstration:

| Scenario | Claim | Path A (VLM) | Path B (RS) | Final Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **01 Vegetation Increase** | Wildfire recovery; vegetation increased | Vegetation recovery detected | $\Delta\text{NDVI} > +0.24$; healthy regrowth | 🟢 **`SUPPORTED`** |
| **02 Built-Up Area Decrease** | Demolition; built-up decreased | Visual clearing inferred | $\Delta\text{NDBI} \approx 0$; soil exposed, not demolished | 🟡 **`UNCERTAIN / CONFLICTING`** |
| **03 Water Body Change** | Flood extent change | Indeterminate | Cloud cover $> 65\%$; SCL invalid | 🔴 **`INSUFFICIENT`** |

---

## 🔌 API Endpoints

### `GET /health`
Returns the operational health of the GeoVision service.

### `POST /api/v1/analyze`
Executes the evidence-first verification pipeline on paired satellite observations.

**Payload:**
```json
{
  "query": "Has vegetation decreased in this region?",
  "observations": [
    {
      "observation_id": "t1_veg",
      "image_path": "Frontend/data/demo_scenarios/veg_increase/t1/t1",
      "role": "t1",
      "metadata": { "modality": "optical", "bands": ["red", "nir", "scl"] }
    },
    {
      "observation_id": "t2_veg",
      "image_path": "Frontend/data/demo_scenarios/veg_increase/t2/t2",
      "role": "t2",
      "metadata": { "modality": "optical", "bands": ["red", "nir", "scl"] }
    }
  ]
}
```

### `POST /api/v1/upload`
Accepts a ZIP archive containing multi-spectral bands (`red`, `nir`, `scl`, `swir`) and validates readability and alignment before analysis.

---

## 🛡️ License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.