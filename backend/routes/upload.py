import os
import shutil
import zipfile
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import rasterio

router = APIRouter()

UPLOAD_DIR = Path("Frontend/data/demo_uploads")

class UploadResponse(BaseModel):
    status: str
    message: str
    observation_id: str | None = None
    image_path: str | None = None
    bands: list[str] | None = None
    metadata: dict | None = None

def normalize_band_name(filename: str) -> str | None:
    lower_name = filename.lower()
    if "red" in lower_name: return "red"
    if "nir" in lower_name: return "nir"
    if "scl" in lower_name: return "scl"
    if "vv" in lower_name: return "vv"
    if "vh" in lower_name: return "vh"
    return None

@router.post("/upload", response_model=UploadResponse)
async def upload_observation(file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        return UploadResponse(status="error", message="Only ZIP files are supported.")
        
    obs_id = str(uuid.uuid4())[:8]
    obs_dir = UPLOAD_DIR / obs_id
    obs_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = obs_dir / file.filename
    try:
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        return UploadResponse(status="error", message=f"Failed to save upload: {str(e)}")
        
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Prevent path traversal
            for member in zip_ref.namelist():
                if member.startswith("/") or ".." in member:
                    raise Exception("Invalid path in zip file.")
            zip_ref.extractall(obs_dir)
    except Exception as e:
        return UploadResponse(status="error", message=f"Failed to extract ZIP: {str(e)}")
        
    # Clean up the zip file itself
    zip_path.unlink(missing_ok=True)
    
    # Discover and normalize bands
    found_bands = {}
    for root, _, files in os.walk(obs_dir):
        for f in files:
            if not f.endswith((".tif", ".tiff")): continue
            band = normalize_band_name(f)
            if band:
                # rename to standard format base_path_band.tif
                old_path = Path(root) / f
                new_path = obs_dir / f"{obs_id}_{band}.tif"
                os.rename(old_path, new_path)
                found_bands[band] = new_path

    required_bands = {"red", "nir", "scl"}
    if not required_bands.issubset(found_bands.keys()):
        missing = required_bands - found_bands.keys()
        shutil.rmtree(obs_dir, ignore_errors=True)
        return UploadResponse(status="error", message=f"Missing required bands: {', '.join(missing)}")
        
    # Validate rasters
    raster_metadata = {}
    for band, path in found_bands.items():
        try:
            with rasterio.open(path) as src:
                raster_metadata[band] = {
                    "crs": str(src.crs),
                    "width": src.width,
                    "height": src.height,
                    "transform": src.transform
                }
        except Exception as e:
            shutil.rmtree(obs_dir, ignore_errors=True)
            return UploadResponse(status="error", message=f"Raster {band} is unreadable or invalid: {str(e)}")
            
    # Check alignment across the required bands
    ref_meta = raster_metadata["red"]
    for band in ["nir", "scl"]:
        meta = raster_metadata[band]
        if meta["crs"] != ref_meta["crs"] or meta["width"] != ref_meta["width"] or meta["height"] != ref_meta["height"]:
            shutil.rmtree(obs_dir, ignore_errors=True)
            return UploadResponse(status="error", message="Spatial metadata mismatch across bands. All bands must have identical CRS and dimensions.")

    # Base path compatible with RealToolRunner (it appends _red.tif)
    base_path = str(obs_dir / obs_id).replace("\\", "/")
    
    return UploadResponse(
        status="success", 
        message="Observation validated successfully.",
        observation_id=obs_id,
        image_path=base_path,
        bands=list(found_bands.keys()),
        metadata={"crs": ref_meta["crs"], "width": ref_meta["width"], "height": ref_meta["height"]}
    )
