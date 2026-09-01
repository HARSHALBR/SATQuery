# SATQuery AI

**Evidence-first agentic geospatial intelligence system.**

SATQuery AI turns natural-language questions about satellite imagery into
auditable, multimodal remote-sensing investigations. It decides how to
investigate the query, checks the AI interpretation against independent
remote-sensing evidence, and shows why the result should — or should not —
be trusted.

> **SIH 2026 — PS 167 / SIH26167**

---

## Architecture

```
USER
  │
  ▼
QUERY + IMAGE(S)
  │
  ▼
┌──────────────────────────────┐
│  1. Query Understanding      │
│  2. Input Validation         │
│  3. Constrained Planner      │
│     ┌───────────┬──────────┐ │
│     │ RS-VLM    │ RS Tools │ │
│     └─────┬─────┴────┬─────┘ │
│           ▼          ▼       │
│  4. Evidence Graph           │
│  5. Evidence Comparator      │
│     ┌────────┬────────┐      │
│     │SUPPORT │UNCERT. │INSUF.│
│     └────────┴────────┘      │
│  6. Final Response           │
└──────────────────────────────┘
```

The system produces three possible evidence statuses:

| Status | Meaning |
|---|---|
| **SUPPORTED** | Evidence is applicable, quality checks pass, evidence supports the interpretation. |
| **UNCERTAIN** | Evidence is weak, contradictory, or has quality issues. |
| **INSUFFICIENT** | Required evidence cannot be produced — data or tool capability is missing. |

---

## Repository Structure

```
satquery-ai/
├── agents/          # Planner, router, tool registry, execution engine
├── schemas/         # Pydantic models: query, tools, evidence, workflow, trace, response
├── backend/         # FastAPI application
│   ├── routes/
│   └── services/
├── integrations/    # VLM client, verification client, frontend client
├── evidence/        # Evidence store, comparator, provenance
├── configs/         # YAML tool definitions, workflow templates, settings
├── tests/           # pytest test suite
├── frontend/        # (future) Streamlit / React frontend
├── evaluation/      # (future) Evaluation datasets and benchmarks
└── docs/            # Architecture, API, and workflow documentation
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- pip

### 2. Install Dependencies

**For agent/backend development (no GDAL required):**

```bash
pip install -r requirements-core.txt
```

**For full installation including remote-sensing tools:**

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env as needed — mock mode is enabled by default
```

### 4. Run Tests

```bash
cd satquery-ai
python -m pytest tests/ -v
```

### 5. Start the Backend (after Phase 11)

```bash
uvicorn backend.main:app --reload
```

---

## Development

### Branching Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, demo-ready code |
| `develop` | Integration branch |
| `member1/*` | VLM development |
| `member2/*` | Agent/backend development |
| `member3/*` | RS tools / evidence / frontend |

### Commit Prefixes

```
feat:     new feature
fix:      bug fix
test:     test addition or modification
docs:     documentation
refactor: code restructuring
```

### Mock Mode

Set `MOCK_VLM=true` and `MOCK_RS_TOOLS=true` in `.env` to run the full
pipeline without real AI models. This allows end-to-end integration
testing before Member 1 (VLM) and Member 3 (RS tools) deliver their
implementations.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Pydantic, Uvicorn |
| Agent | Custom constrained planner, structured JSON tool registry |
| RS Tools | Rasterio, GDAL, NumPy, GeoPandas, Shapely, SciPy, scikit-image |
| Storage | SQLite (initial) |
| Testing | pytest |
| Configuration | YAML, environment variables |
| Packaging | Docker |

---

## Team

| Member | Responsibility |
|---|---|
| Member 1 | RS-adapted VLM |
| **Member 2** | **Agent / Backend / Integration** |
| Member 3 | Remote-sensing tools / Evidence / Frontend |

---

## License

Internal — SIH 2026 competition submission.
