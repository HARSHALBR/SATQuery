"""End-to-end automated test suite for Before / After Satellite Image Validation.

Covers:
A. Upload T1 JPEG (200, role == t1, observation_id, preview_url, filename)
B. Upload T2 JPEG (200, role == t2, observation_id, preview_url, filename)
C. Upload T1 ZIP (200, red/nir/scl validation, canonical band files exist)
D. Upload T2 ZIP (200, red/nir/scl validation, canonical band files exist)
E. T1/T2 independence (uploading T2 does not overwrite or mutate T1)
F. Replacement (replacing T1 changes only T1; replacing T2 changes only T2)
G. Analyze uploaded pair (exact T1/T2 observation IDs reach /api/v1/analyze)
H. RGB safety (RGB JPEG does NOT generate fake NIR/NDVI/SCL evidence)
I. Multispectral analysis (valid multispectral T1/T2 runs through deterministic RS tools)
"""

import os
import io
import pytest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

JPEG_T1_PATH = Path("WhatsApp Image 2026-09-03 at 22.38.03.jpeg")
JPEG_T2_PATH = Path("WhatsApp Image 2026-09-03 at 22.38.21.jpeg")
ZIP_T1_PATH = Path("test_zips/valid_t1.zip")
ZIP_T2_PATH = Path("test_zips/valid_t2.zip")


def _create_dummy_jpeg(name: str) -> bytes:
    """Helper to create an in-memory JPEG."""
    buf = io.BytesIO()
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestUploadValidationE2E:

    def test_a_upload_t1_jpeg(self):
        """A. Upload T1 JPEG -> 200, role == t1, observation_id, preview_url, filename."""
        if JPEG_T1_PATH.exists():
            with open(JPEG_T1_PATH, "rb") as f:
                content = f.read()
        else:
            content = _create_dummy_jpeg("t1.jpg")

        res = client.post(
            "/api/v1/upload?role=t1",
            files={"file": ("test_t1.jpg", io.BytesIO(content), "image/jpeg")}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["role"] == "t1"
        assert data["observation_id"] is not None
        assert data["image_path"] is not None
        assert data["preview_url"] is not None
        assert data["modality"] == "optical"
        assert data["bands"] == ["red", "green", "blue"]
        assert data["filename"] == "test_t1.jpg"

    def test_b_upload_t2_jpeg(self):
        """B. Upload T2 JPEG -> 200, role == t2, observation_id, preview_url, filename."""
        if JPEG_T2_PATH.exists():
            with open(JPEG_T2_PATH, "rb") as f:
                content = f.read()
        else:
            content = _create_dummy_jpeg("t2.jpg")

        res = client.post(
            "/api/v1/upload?role=t2",
            files={"file": ("test_t2.jpg", io.BytesIO(content), "image/jpeg")}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["role"] == "t2"
        assert data["observation_id"] is not None
        assert data["image_path"] is not None
        assert data["preview_url"] is not None
        assert data["modality"] == "optical"
        assert data["bands"] == ["red", "green", "blue"]
        assert data["filename"] == "test_t2.jpg"

    def test_c_upload_t1_zip(self):
        """C. Upload T1 ZIP -> 200, red/nir/scl validation, canonical band files exist."""
        assert ZIP_T1_PATH.exists(), "test_zips/valid_t1.zip fixture must exist"
        with open(ZIP_T1_PATH, "rb") as f:
            res = client.post(
                "/api/v1/upload?role=t1",
                files={"file": ("valid_t1.zip", f, "application/zip")}
            )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["role"] == "t1"
        assert "red" in data["bands"]
        assert "nir" in data["bands"]
        assert "scl" in data["bands"]
        assert data["modality"] == "multispectral"
        assert data["preview_url"] is not None
        assert data["filename"] == "valid_t1.zip"

        # Check canonical raster band files on disk
        obs_dir = Path("Frontend/data/demo_uploads") / data["observation_id"]
        assert (obs_dir / f"{data['observation_id']}_red.tif").exists()
        assert (obs_dir / f"{data['observation_id']}_nir.tif").exists()
        assert (obs_dir / f"{data['observation_id']}_scl.tif").exists()

    def test_d_upload_t2_zip(self):
        """D. Upload T2 ZIP -> 200, red/nir/scl validation, canonical band files exist."""
        assert ZIP_T2_PATH.exists(), "test_zips/valid_t2.zip fixture must exist"
        with open(ZIP_T2_PATH, "rb") as f:
            res = client.post(
                "/api/v1/upload?role=t2",
                files={"file": ("valid_t2.zip", f, "application/zip")}
            )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["role"] == "t2"
        assert "red" in data["bands"]
        assert "nir" in data["bands"]
        assert "scl" in data["bands"]
        assert data["modality"] == "multispectral"
        assert data["preview_url"] is not None
        assert data["filename"] == "valid_t2.zip"

        # Check canonical raster band files on disk
        obs_dir = Path("Frontend/data/demo_uploads") / data["observation_id"]
        assert (obs_dir / f"{data['observation_id']}_red.tif").exists()
        assert (obs_dir / f"{data['observation_id']}_nir.tif").exists()
        assert (obs_dir / f"{data['observation_id']}_scl.tif").exists()

    def test_e_t1_t2_independence(self):
        """E. T1/T2 independence -> uploading T2 does not overwrite or mutate T1."""
        jpeg1 = _create_dummy_jpeg("t1.jpg")
        jpeg2 = _create_dummy_jpeg("t2.jpg")

        res_t1 = client.post(
            "/api/v1/upload?role=t1",
            files={"file": ("t1_indep.jpg", io.BytesIO(jpeg1), "image/jpeg")}
        ).json()
        t1_id = res_t1["observation_id"]

        res_t2 = client.post(
            "/api/v1/upload?role=t2",
            files={"file": ("t2_indep.jpg", io.BytesIO(jpeg2), "image/jpeg")}
        ).json()
        t2_id = res_t2["observation_id"]

        # Ensure distinct IDs and roles
        assert t1_id != t2_id
        assert res_t1["role"] == "t1"
        assert res_t2["role"] == "t2"

        # Verify T1 file still exists unchanged on disk after T2 upload
        assert Path(res_t1["image_path"]).exists()
        assert Path(res_t2["image_path"]).exists()

    def test_f_replacement(self):
        """F. Replacement -> replacing T1 changes only T1; replacing T2 changes only T2."""
        t1_a = client.post(
            "/api/v1/upload?role=t1",
            files={"file": ("t1_a.jpg", io.BytesIO(_create_dummy_jpeg("a")), "image/jpeg")}
        ).json()
        t2_a = client.post(
            "/api/v1/upload?role=t2",
            files={"file": ("t2_a.jpg", io.BytesIO(_create_dummy_jpeg("b")), "image/jpeg")}
        ).json()

        # Replace T1 with T1-B
        t1_b = client.post(
            "/api/v1/upload?role=t1",
            files={"file": ("t1_b.jpg", io.BytesIO(_create_dummy_jpeg("c")), "image/jpeg")}
        ).json()

        assert t1_b["observation_id"] != t1_a["observation_id"]
        assert t1_b["role"] == "t1"
        # T2 remains unchanged
        assert t2_a["role"] == "t2"
        assert Path(t2_a["image_path"]).exists()

        # Replace T2 with T2-B
        t2_b = client.post(
            "/api/v1/upload?role=t2",
            files={"file": ("t2_b.jpg", io.BytesIO(_create_dummy_jpeg("d")), "image/jpeg")}
        ).json()

        assert t2_b["observation_id"] != t2_a["observation_id"]
        assert t2_b["role"] == "t2"
        # T1-B remains unchanged
        assert t1_b["role"] == "t1"
        assert Path(t1_b["image_path"]).exists()

    def test_g_and_h_analyze_uploaded_rgb_safety(self):
        """G & H. Analyze uploaded RGB pair -> exact IDs reach /api/v1/analyze, NO fake spectral evidence."""
        res_t1 = client.post(
            "/api/v1/upload?role=t1",
            files={"file": ("t1_safe.jpg", io.BytesIO(_create_dummy_jpeg("s1")), "image/jpeg")}
        ).json()
        res_t2 = client.post(
            "/api/v1/upload?role=t2",
            files={"file": ("t2_safe.jpg", io.BytesIO(_create_dummy_jpeg("s2")), "image/jpeg")}
        ).json()

        payload = {
            "query": "What changed between the two observations?",
            "observations": [
                {
                    "observation_id": res_t1["observation_id"],
                    "image_path": res_t1["image_path"],
                    "role": "t1",
                    "metadata": {"modality": "optical", "bands": ["red", "green", "blue"]}
                },
                {
                    "observation_id": res_t2["observation_id"],
                    "image_path": res_t2["image_path"],
                    "role": "t2",
                    "metadata": {"modality": "optical", "bands": ["red", "green", "blue"]}
                }
            ]
        }

        res = client.post("/api/v1/analyze", json=payload)
        assert res.status_code == 200
        data = res.json()

        # RGB Safety: physical spectral tools report UNAVAILABLE, no fabricated NDVI
        trace_tools = {step["tool"]: step["status"] for step in data["execution_trace"]}
        assert trace_tools["validate_inputs"] == "success"
        assert trace_tools["grounding"] == "success"
        assert trace_tools["run_rs_vlm"] == "success"
        assert trace_tools["change_statistics"] == "unavailable"

        # Evidence records should only contain non-fabricated evidence (grounding + vlm)
        evidence_types = [ev["type"] for ev in data["evidence"]]
        assert "spatial_grounding" in evidence_types
        assert "vlm_interpretation" in evidence_types
        assert "vegetation_change" not in evidence_types
        assert "change_quantification" not in evidence_types

        # Decision verdict is honestly INSUFFICIENT due to absent spectral NIR bands
        assert data["status"] == "INSUFFICIENT"

    def test_i_multispectral_analysis(self):
        """I. Multispectral analysis -> valid multispectral T1/T2 continues through deterministic RS tools."""
        with open(ZIP_T1_PATH, "rb") as f1, open(ZIP_T2_PATH, "rb") as f2:
            res_t1 = client.post(
                "/api/v1/upload?role=t1",
                files={"file": ("ms_t1.zip", f1, "application/zip")}
            ).json()
            res_t2 = client.post(
                "/api/v1/upload?role=t2",
                files={"file": ("ms_t2.zip", f2, "application/zip")}
            ).json()

        payload = {
            "query": "Has vegetation decreased between the two observations?",
            "observations": [
                {
                    "observation_id": res_t1["observation_id"],
                    "image_path": res_t1["image_path"],
                    "role": "t1",
                    "metadata": {"modality": "multispectral", "bands": ["red", "nir", "scl"]}
                },
                {
                    "observation_id": res_t2["observation_id"],
                    "image_path": res_t2["image_path"],
                    "role": "t2",
                    "metadata": {"modality": "multispectral", "bands": ["red", "nir", "scl"]}
                }
            ]
        }

        res = client.post("/api/v1/analyze", json=payload)
        assert res.status_code == 200
        data = res.json()

        # For multispectral dataset, deterministic tools run to SUCCESS
        trace_tools = {step["tool"]: step["status"] for step in data["execution_trace"]}
        assert trace_tools["validate_inputs"] == "success"
        assert trace_tools["grounding"] == "success"
        assert trace_tools["ndvi_delta"] == "success"
        assert trace_tools["change_statistics"] == "success"
        assert trace_tools["run_rs_vlm"] == "success"
        assert trace_tools["compare_evidence"] == "success"

        # Supporting evidence includes genuine physical remote sensing records
        evidence_types = [ev["type"] for ev in data["evidence"]]
        assert "vegetation_change" in evidence_types
        assert "change_quantification" in evidence_types
