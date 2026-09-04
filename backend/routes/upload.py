import os
import shutil
import zipfile
import uuid
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException
from pydantic import BaseModel
import numpy as np
import rasterio
from PIL import Image

router = APIRouter()

UPLOAD_DIR = Path("Frontend/data/demo_uploads")

class UploadResponse(BaseModel):
    status: str
    message: str
    observation_id: Optional[str] = None
    image_path: Optional[str] = None
    bands: Optional[List[str]] = None
    modality: Optional[str] = None  # "multispectral" or "optical"
    preview_url: Optional[str] = None
    role: Optional[str] = None  # "t1" or "t2"
    filename: Optional[str] = None
    metadata: Optional[dict] = None

def normalize_band_name(filename: str) -> Optional[str]:
    lower_name = filename.lower()
    if "red" in lower_name: return "red"
    if "nir" in lower_name: return "nir"
    if "scl" in lower_name: return "scl"
    if "vv" in lower_name: return "vv"
    if "vh" in lower_name: return "vh"
    return None

def _generate_raster_preview(red_path: Path, nir_path: Optional[Path], output_png: Path) -> None:
    """Generate a browser-viewable 8-bit PNG preview from raster bands."""
    try:
        with rasterio.open(red_path) as src:
            red = src.read(1).astype(np.float32)
        
        # Normalize to 0-255
        r_min, r_max = np.percentile(red, 2), np.percentile(red, 98)
        if r_max > r_min:
            red_norm = np.clip((red - r_min) / (r_max - r_min) * 255.0, 0, 255).astype(np.uint8)
        else:
            red_norm = np.clip(red, 0, 255).astype(np.uint8)

        if nir_path and nir_path.exists():
            with rasterio.open(nir_path) as src:
                nir = src.read(1).astype(np.float32)
            n_min, n_max = np.percentile(nir, 2), np.percentile(nir, 98)
            if n_max > n_min:
                nir_norm = np.clip((nir - n_min) / (n_max - n_min) * 255.0, 0, 255).astype(np.uint8)
            else:
                nir_norm = np.clip(nir, 0, 255).astype(np.uint8)
            # False color composite (NIR, Red, Red)
            rgb = np.stack([nir_norm, red_norm, red_norm], axis=-1)
            img = Image.fromarray(rgb)
        else:
            img = Image.fromarray(red_norm, mode='L').convert('RGB')

        img.thumbnail((1200, 1200))
        img.save(output_png, format="PNG", optimize=True)
    except Exception as e:
        # Fallback dummy image if rasterio fails
        img = Image.new("RGB", (300, 300), color=(70, 90, 120))
        img.save(output_png, format="PNG")

@router.post("/upload", response_model=UploadResponse)
async def upload_observation(
    file: UploadFile = File(...),
    role: Optional[str] = Query(None, description="Observation role: 't1' or 't2'")
):
    """
    Upload and validate a satellite observation.
    Supports:
    1. ZIP archives containing multispectral bands (red, nir, scl)
    2. Direct visual satellite images (.jpg, .jpeg, .png, .tif, .tiff, .webp)
    """
    obs_id = str(uuid.uuid4())[:8]
    obs_dir = UPLOAD_DIR / obs_id
    obs_dir.mkdir(parents=True, exist_ok=True)
    
    filename = file.filename or "upload.bin"
    ext = Path(filename).suffix.lower()

    # If role wasn't supplied in query, check if filename hints role or leave None
    assigned_role = role.lower() if role in ["t1", "t2"] else None

    # CASE A: ZIP MULTISPECTRAL ARCHIVE
    if ext == ".zip":
        zip_path = obs_dir / filename
        try:
            with open(zip_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            shutil.rmtree(obs_dir, ignore_errors=True)
            return UploadResponse(status="error", message=f"Failed to save upload: {str(e)}", role=assigned_role)
            
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    if member.startswith("/") or ".." in member:
                        raise Exception("Invalid path in zip file.")
                zip_ref.extractall(obs_dir)
        except Exception as e:
            shutil.rmtree(obs_dir, ignore_errors=True)
            return UploadResponse(status="error", message=f"Failed to extract ZIP: {str(e)}", role=assigned_role)
            
        zip_path.unlink(missing_ok=True)
        
        # Discover and normalize bands
        found_bands = {}
        for root, _, files in os.walk(obs_dir):
            for f in files:
                if not f.endswith((".tif", ".tiff")): continue
                band = normalize_band_name(f)
                if band:
                    old_path = Path(root) / f
                    new_path = obs_dir / f"{obs_id}_{band}.tif"
                    os.rename(old_path, new_path)
                    found_bands[band] = new_path

        required_bands = {"red", "nir", "scl"}
        if not required_bands.issubset(found_bands.keys()):
            missing = required_bands - found_bands.keys()
            shutil.rmtree(obs_dir, ignore_errors=True)
            return UploadResponse(
                status="error",
                message=f"Missing required multispectral bands: {', '.join(missing)}",
                role=assigned_role
            )
            
        # Validate rasters
        raster_metadata = {}
        for band, path in found_bands.items():
            try:
                with rasterio.open(path) as src:
                    raster_metadata[band] = {
                        "crs": str(src.crs),
                        "width": src.width,
                        "height": src.height,
                        "transform": [float(v) for v in src.transform]
                    }
            except Exception as e:
                shutil.rmtree(obs_dir, ignore_errors=True)
                return UploadResponse(status="error", message=f"Raster {band} is unreadable or invalid: {str(e)}", role=assigned_role)
                
        ref_meta = raster_metadata["red"]
        for band in ["nir", "scl"]:
            meta = raster_metadata[band]
            if meta["crs"] != ref_meta["crs"] or meta["width"] != ref_meta["width"] or meta["height"] != ref_meta["height"]:
                shutil.rmtree(obs_dir, ignore_errors=True)
                return UploadResponse(
                    status="error",
                    message="Spatial metadata mismatch across bands. All bands must have identical CRS and dimensions.",
                    role=assigned_role
                )

        # Generate browser preview
        preview_path = obs_dir / "preview.png"
        _generate_raster_preview(found_bands["red"], found_bands.get("nir"), preview_path)
        preview_url = f"/data/demo_uploads/{obs_id}/preview.png"
        base_path = str(obs_dir / obs_id).replace("\\", "/")
        
        return UploadResponse(
            status="success", 
            message="Multispectral observation validated successfully.",
            observation_id=obs_id,
            image_path=base_path,
            bands=list(found_bands.keys()),
            modality="multispectral",
            preview_url=preview_url,
            role=assigned_role,
            filename=file.filename,
            metadata={
                "crs": ref_meta["crs"],
                "width": ref_meta["width"],
                "height": ref_meta["height"],
                "type": "multispectral",
                "spectral_bands_present": True
            }
        )

    # CASE B: DIRECT VISUAL SATELLITE IMAGE (.jpg, .jpeg, .png, .tif, .tiff, .webp)
    elif ext in [".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".jp2"]:
        temp_img_path = obs_dir / f"uploaded{ext}"
        try:
            with open(temp_img_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            shutil.rmtree(obs_dir, ignore_errors=True)
            return UploadResponse(status="error", message=f"Failed to save image upload: {str(e)}", role=assigned_role)

        # Validate with Pillow
        try:
            with Image.open(temp_img_path) as img:
                img.verify()
            with Image.open(temp_img_path) as img:
                rgb_img = img.convert("RGB")
                width, height = rgb_img.size
                format_name = img.format or ext.replace(".", "").upper()
                
                # Save visual image in canonical location
                visual_path = obs_dir / f"{obs_id}_visual.png"
                rgb_img.save(visual_path, format="PNG")
                
                # Save optimized preview
                preview_path = obs_dir / "preview.png"
                preview_copy = rgb_img.copy()
                preview_copy.thumbnail((1200, 1200))
                preview_copy.save(preview_path, format="PNG", optimize=True)
        except Exception as e:
            shutil.rmtree(obs_dir, ignore_errors=True)
            return UploadResponse(status="error", message=f"Image file is unreadable or corrupt: {str(e)}", role=assigned_role)
        finally:
            temp_img_path.unlink(missing_ok=True)

        preview_url = f"/data/demo_uploads/{obs_id}/preview.png"
        image_path = str(visual_path).replace("\\", "/")

        return UploadResponse(
            status="success",
            message="Visual RGB satellite observation validated successfully.",
            observation_id=obs_id,
            image_path=image_path,
            bands=["red", "green", "blue"],
            modality="optical",
            preview_url=preview_url,
            role=assigned_role,
            filename=file.filename,
            metadata={
                "width": width,
                "height": height,
                "format": format_name,
                "type": "visual_rgb",
                "spectral_bands_present": False,
                "note": "RGB visual imagery. NIR/SCL spectral bands are not present in this dataset."
            }
        )

    else:
        shutil.rmtree(obs_dir, ignore_errors=True)
        return UploadResponse(
            status="error",
            message=f"Unsupported format '{ext}'. Please upload a ZIP dataset with bands or an image (.jpg, .png, .tif).",
            role=assigned_role
        )
