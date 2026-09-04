import os
import zipfile
import numpy as np
import rasterio
from rasterio.transform import from_origin

def create_dummy_tif(path: str, crs: str = "EPSG:32610", width: int = 100, height: int = 100):
    transform = from_origin(500000, 4600000, 10, 10)
    data = np.random.randint(0, 255, (height, width), dtype=np.uint16)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(data, 1)

os.makedirs("test_zips", exist_ok=True)

# Valid T1
os.makedirs("test_zips/t1", exist_ok=True)
create_dummy_tif("test_zips/t1/t1_red.tif")
create_dummy_tif("test_zips/t1/t1_nir.tif")
create_dummy_tif("test_zips/t1/t1_scl.tif")

with zipfile.ZipFile("test_zips/valid_t1.zip", "w") as z:
    z.write("test_zips/t1/t1_red.tif", "t1_red.tif")
    z.write("test_zips/t1/t1_nir.tif", "t1_nir.tif")
    z.write("test_zips/t1/t1_scl.tif", "t1_scl.tif")

# Valid T2
os.makedirs("test_zips/t2", exist_ok=True)
create_dummy_tif("test_zips/t2/t2_red.tif")
create_dummy_tif("test_zips/t2/t2_nir.tif")
create_dummy_tif("test_zips/t2/t2_scl.tif")

with zipfile.ZipFile("test_zips/valid_t2.zip", "w") as z:
    z.write("test_zips/t2/t2_red.tif", "t2_red.tif")
    z.write("test_zips/t2/t2_nir.tif", "t2_nir.tif")
    z.write("test_zips/t2/t2_scl.tif", "t2_scl.tif")

# Invalid (Missing NIR)
os.makedirs("test_zips/t3", exist_ok=True)
create_dummy_tif("test_zips/t3/t3_red.tif")
create_dummy_tif("test_zips/t3/t3_scl.tif")
with zipfile.ZipFile("test_zips/invalid_missing_nir.zip", "w") as z:
    z.write("test_zips/t3/t3_red.tif", "t3_red.tif")
    z.write("test_zips/t3/t3_scl.tif", "t3_scl.tif")

# Mismatched T2
os.makedirs("test_zips/t4", exist_ok=True)
create_dummy_tif("test_zips/t4/t4_red.tif", width=200) # different size
create_dummy_tif("test_zips/t4/t4_nir.tif", width=200)
create_dummy_tif("test_zips/t4/t4_scl.tif", width=200)
with zipfile.ZipFile("test_zips/mismatched_t2.zip", "w") as z:
    z.write("test_zips/t4/t4_red.tif", "t4_red.tif")
    z.write("test_zips/t4/t4_nir.tif", "t4_nir.tif")
    z.write("test_zips/t4/t4_scl.tif", "t4_scl.tif")

print("Created test ZIPs in test_zips/")
