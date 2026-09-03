"""
workspace.py — Analysis workspace manager for SatQuery.

Creates a temp directory structure for a single analysis run and
manages input file validation and metadata extraction.
"""
from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import rasterio


class AnalysisWorkspace:
    """
    Manages a single analysis run's directory structure.

    Directory layout (all inside a system tempdir):
        <analysis_id>/
            metadata/
            derived/
            aligned/
            regions/
            evidence/
    """

    SUBDIRS = ["metadata", "derived", "aligned", "regions", "evidence"]

    def __init__(self, base_dir: Optional[Path] = None):
        self.analysis_id = str(uuid.uuid4())[:8]
        if base_dir is None:
            self._tmpdir = tempfile.mkdtemp(prefix=f"satquery_{self.analysis_id}_")
            self.root = Path(self._tmpdir)
        else:
            self.root = Path(base_dir) / self.analysis_id
            self.root.mkdir(parents=True, exist_ok=True)
            self._tmpdir = None

        # Create all subdirectories up front
        for sub in self.SUBDIRS:
            (self.root / sub).mkdir(parents=True, exist_ok=True)

        self._input_files: list[dict] = []

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def get_dir(self, name: str) -> Path:
        """Return (and ensure) a named sub-directory inside the workspace root."""
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Input file validation
    # ------------------------------------------------------------------

    def add_input_file(self, label: str, path: Path) -> dict:
        """
        Validate and register an input GeoTIFF.

        Returns a metadata dict with 'path', 'bands', 'crs', 'shape', etc.
        Raises ValueError if the file is not a valid raster.
        """
        path = Path(path)
        if not path.exists():
            raise ValueError(f"File does not exist: {path}")

        try:
            with rasterio.open(path) as src:
                crs = str(src.crs) if src.crs else "unknown"
                bands = src.count
                width = src.width
                height = src.height
                transform = list(src.transform)
                band_names = list(src.descriptions) if src.descriptions else [f"band_{i+1}" for i in range(bands)]
                dtype = src.dtypes[0] if src.dtypes else "unknown"
                nodata = src.nodata
        except Exception as exc:
            raise ValueError(f"Cannot open {path.name} as a raster: {exc}") from exc

        entry = {
            "label": label,
            "path": str(path),
            "filename": path.name,
            "bands": band_names,
            "band_count": bands,
            "crs": crs,
            "width": width,
            "height": height,
            "transform": transform,
            "dtype": dtype,
            "nodata": nodata,
            "status": "success",
            # observation_id for API response compatibility
            "observation_id": f"{label}_{self.analysis_id}",
            # dev_scenario placeholder
            "metadata": {
                "crs": crs,
                "dev_scenario": None,
            },
        }
        self._input_files.append(entry)
        return entry

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_manifest(self) -> Path:
        """Write a JSON manifest of all registered input files."""
        manifest_path = self.get_dir("metadata") / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(
                {
                    "analysis_id": self.analysis_id,
                    "root": str(self.root),
                    "input_files": self._input_files,
                },
                f,
                indent=2,
            )
        return manifest_path

    def __repr__(self) -> str:
        return f"<AnalysisWorkspace id={self.analysis_id} root={self.root}>"
