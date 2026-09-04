"""
pipeline_config.py — Central configuration for the SatQuery pipeline.
Defines shared constants used across all pipeline modules.
"""
from pathlib import Path

# Temporal range for the demo dataset
BEFORE_YEAR: int = 2017
AFTER_YEAR: int = 2018

# Root data directory (relative to this file, i.e. Frontend/data)
DATA_DIR: Path = Path(__file__).parent / "data" / "sample"
