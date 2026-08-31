# SATQuery AI - Member 1: Core VLM & Data Engineering Module

**SIH 2026 - Problem Statement 167 (SIH26167)**

## Overview

This repository section represents the core data engineering, VLM architecture, training, and evaluation pipeline managed by **Member 1** for the **SATQuery AI** system.

### Scope & Responsibilities (Member 1)
- **Data Engineering**: BigEarthNet.txt processing and Sentinel-1 (SAR) + Sentinel-2 (Multispectral) data handling.
- **Manifests & Splits**: Creation of deterministic dataset manifests and training/validation/test split logic.
- **Model Backbone**: RS-InternVL-style multimodal architecture built on an InternVL3-1B-class foundation.
- **Efficient Fine-Tuning**: PEFT/LoRA adaptation for satellite domain alignment.
- **Training Pipeline**: Multi-modal training, checkpoint management, and validation hooks.
- **Evaluation & Inference**: VLM evaluation metrics and structured JSON/schema model output generation.

---

## Directory Structure

```text
.
├── configs/
│   └── model/              # Model and hyperparameters configurations
├── data/
│   ├── bigearthnet_txt/    # BigEarthNet data raw files and text metadata
│   └── manifests/          # Deterministic split manifests (train/val/test)
├── models/
│   └── rs_internvl/        # RS-InternVL model architecture and wrapper code
├── training/               # Training loop and trainer execution modules
├── evaluation/
│   └── vlm/                # VLM evaluation metrics and benchmarks
├── docs/
│   └── model/              # Model documentation and technical specs
├── scripts/                # Execution & data processing scripts
├── tests/                  # Unit and integration tests
├── notebooks/              # Analysis and experimentation notebooks
├── checkpoints/            # Model checkpoints storage
└── outputs/                # Evaluation and inference outputs
```

---

## Environment Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate the virtual environment:
   - **Windows**: `.venv\Scripts\activate`
   - **Linux/macOS**: `source .venv/bin/activate`
3. Install baseline dependencies:
   ```bash
   pip install -r requirements.txt
   ```
