import rasterio
import numpy as np
from pathlib import Path

from pipeline_config import DATA_DIR, BEFORE_YEAR


def calculate_ndvi_for_year(year, *, red_input=None, nir_input=None, output_name=None, data_dir=DATA_DIR, is_absolute=False):
    """Calculate NDVI for a given year or reference dataset while preserving current output contracts."""
    if is_absolute:
        red_file = Path(red_input)
        nir_file = Path(nir_input)
        output_file = Path(output_name)
    else:
        folder = Path(data_dir)
        if red_input is None:
            red_input = f"S2_B04_{year}.tif"
        if nir_input is None:
            nir_input = f"S2_B08_{year}.tif"
        if output_name is None:
            output_name = f"S2_NDVI_{year}.tif"
    
        red_file = folder / red_input
        nir_file = folder / nir_input
        output_file = folder / output_name

    with rasterio.open(red_file) as red_src, rasterio.open(nir_file) as nir_src:
        red = red_src.read(1).astype(np.float32)
        nir = nir_src.read(1).astype(np.float32)

        profile = red_src.profile.copy()
        nodata = red_src.nodata

        valid = (
            (red != nodata) &
            (nir != nir_src.nodata) &
            ((nir + red) != 0)
        )

        ndvi = np.full(red.shape, -9999.0, dtype=np.float32)

        ndvi[valid] = (
            (nir[valid] - red[valid]) /
            (nir[valid] + red[valid])
        )

        profile.update(
            dtype="float32",
            nodata=-9999.0,
            count=1
        )

        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(ndvi, 1)

    valid_values = ndvi[valid]
    print("Saved:", output_file)
    print("Valid pixels:", valid.sum())
    print("Min:", valid_values.min())
    print("Max:", valid_values.max())
    print("Mean:", valid_values.mean())
    print("Median:", np.median(valid_values))

    return output_file


if __name__ == "__main__":
    calculate_ndvi_for_year(
        BEFORE_YEAR,
        red_input="S2_B04_sample.tif",
        nir_input="S2_B08_sample.tif",
        output_name="S2_NDVI_sample.tif",
        data_dir=DATA_DIR,
    )