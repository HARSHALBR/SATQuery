import rasterio
from rasterio.enums import Resampling
import numpy as np
from pathlib import Path

from pipeline_config import DATA_DIR, BEFORE_YEAR


def calculate_ndbi_for_year(year, *, nir_input=None, swir_input=None, output_name=None, data_dir=DATA_DIR, is_absolute=False):
    """Calculate NDBI for a given year or reference dataset while preserving current output contracts."""
    if is_absolute:
        nir_file = Path(nir_input)
        swir_file = Path(swir_input)
        output_file = Path(output_name)
    else:
        folder = Path(data_dir)
        if nir_input is None:
            nir_input = f"S2_B08_{year}.tif"
        if swir_input is None:
            swir_input = f"S2_B11_{year}.tif"
        if output_name is None:
            output_name = f"S2_NDBI_{year}.tif"
    
        nir_file = folder / nir_input
        swir_file = folder / swir_input
        output_file = folder / output_name
    
    return calculate_ndbi(nir_file, swir_file, output_file)


def calculate_ndbi(nir_path: Path, swir_path: Path, output_path: Path) -> Path:
    """
    Calculate NDBI from NIR and SWIR inputs.
    Scientifically preserves resampling of SWIR to the NIR reference grid.
    
    Args:
        nir_path: Path to the NIR (e.g., B08) raster.
        swir_path: Path to the SWIR (e.g., B11) raster.
        output_path: Path where the NDBI output should be saved.
        
    Returns:
        Path to the generated NDBI raster.
    """

    with rasterio.open(nir_path) as nir_src, rasterio.open(swir_path) as swir_src:
        if nir_src.crs != swir_src.crs:
            raise ValueError(f"CRS mismatch: NIR {nir_src.crs} != SWIR {swir_src.crs}")

        nir = nir_src.read(1).astype(np.float32)

        swir = swir_src.read(
            1,
            out_shape=(nir_src.height, nir_src.width),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

        profile = nir_src.profile.copy()

        valid = (
            (nir != nir_src.nodata) &
            (swir != swir_src.nodata) &
            ((swir + nir) != 0)
        )

        ndbi = np.full(
            nir.shape,
            -9999.0,
            dtype=np.float32,
        )

        ndbi[valid] = (
            (swir[valid] - nir[valid]) /
            (swir[valid] + nir[valid])
        )

        profile.update(
            dtype="float32",
            nodata=-9999.0,
            count=1,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(ndbi, 1)

    values = ndbi[valid]
    print(f"Saved: {output_path}")
    print("Valid pixels:", valid.sum())
    if valid.sum() > 0:
        print("Min:", values.min())
        print("Max:", values.max())
        print("Mean:", values.mean())
        print("Median:", np.median(values))

    return output_path


if __name__ == "__main__":
    import sys
    print("calculate_ndbi.py should be invoked via orchestration, not as a standalone script using hardcoded paths.")
    sys.exit(1)