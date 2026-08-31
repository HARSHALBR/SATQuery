# VLM Benchmark Evaluation Interfaces (`evaluation/vlm/`)

This directory provides standardized evaluation interfaces and contracts for benchmarking **RS-InternVL** on standard Earth observation visual question answering benchmarks:

1. **BigEarthNet Multimodal VQA** ([`scripts/pretrained_full_manifest_training.py`](../../scripts/pretrained_full_manifest_training.py))
2. **VRSBench Interface** ([`vrsbench_interface.py`](vrsbench_interface.py))
3. **RSVQA Interface** ([`rsvqa_interface.py`](rsvqa_interface.py))

> [!NOTE]
> **Status:** These benchmark adapters are implemented as structured evaluation interfaces and schema contracts for future full-scale evaluation. Benchmark scores are strictly reported for the currently trained BigEarthNet dataset in [`docs/model/model_card.md`](../../docs/model/model_card.md). No external benchmark scores are fabricated.

## Input / Output Contract

All benchmark interfaces adhere to the standard RS-InternVL prediction output schema:
```json
{
  "question_id": "vrs_001",
  "answer": "Yes, broad-leaved forest is present.",
  "claim_type": "presence_verification",
  "model_score": 0.942,
  "model_version": "RS-InternVL-Step10",
  "is_correct": true
}
```
