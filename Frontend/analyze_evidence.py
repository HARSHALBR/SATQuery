import rasterio
import numpy as np

from pipeline_config import DATA_DIR, BEFORE_YEAR, AFTER_YEAR

period = f"{BEFORE_YEAR}_{AFTER_YEAR}"
folder = DATA_DIR

classification_file = folder / "S2_change_classification_patch.tif"
vv_file = folder / f"S1_vv_change_{period}.tif"
vh_file = folder / f"S1_vh_change_{period}.tif"

with rasterio.open(classification_file) as src:
    classification = src.read(1)

with rasterio.open(vv_file) as src:
    vv = src.read(1)
    vv_nodata = src.nodata

with rasterio.open(vh_file) as src:
    vh = src.read(1)
    vh_nodata = src.nodata


# Valid SAR pixels
valid = (
    (vv != vv_nodata) &
    (vh != vh_nodata) &
    np.isfinite(vv) &
    np.isfinite(vh)
)

print("Evidence analysis")
print("=================")

for cls in range(4):

    mask = valid & (classification == cls)

    count = mask.sum()

    print(f"\nClass {cls}")
    print("Pixels:", count)

    if count == 0:
        continue

    vv_values = vv[mask]
    vh_values = vh[mask]

    print(
        "VV change:",
        "mean =", vv_values.mean(),
        "median =", np.median(vv_values)
    )

    print(
        "VH change:",
        "mean =", vh_values.mean(),
        "median =", np.median(vh_values)
    )


# Specifically analyze the strongest optical candidates
candidate = valid & (classification == 3)

vv_candidate = vv[candidate]
vh_candidate = vh[candidate]

print("\n================================")
print("STRONG OPTICAL CANDIDATES")
print("NDVI decrease + NDBI increase")
print("================================")

print("Candidate pixels:", candidate.sum())

if candidate.sum() > 0:

    print(
        "VV:",
        "mean =", vv_candidate.mean(),
        "median =", np.median(vv_candidate),
        "std =", vv_candidate.std()
    )

    print(
        "VH:",
        "mean =", vh_candidate.mean(),
        "median =", np.median(vh_candidate),
        "std =", vh_candidate.std()
    )