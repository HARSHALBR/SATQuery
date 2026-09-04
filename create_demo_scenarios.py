import os
import rasterio
import numpy as np
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

base_dir = "Frontend/data/demo_scenarios"

# Scenario 1: Vegetation Increased
os.makedirs(f"{base_dir}/veg_increase/t1", exist_ok=True)
create_dummy_tif(f"{base_dir}/veg_increase/t1/t1_red.tif")
create_dummy_tif(f"{base_dir}/veg_increase/t1/t1_nir.tif")
create_dummy_tif(f"{base_dir}/veg_increase/t1/t1_scl.tif")

os.makedirs(f"{base_dir}/veg_increase/t2", exist_ok=True)
create_dummy_tif(f"{base_dir}/veg_increase/t2/t2_red.tif")
create_dummy_tif(f"{base_dir}/veg_increase/t2/t2_nir.tif")
create_dummy_tif(f"{base_dir}/veg_increase/t2/t2_scl.tif")

# Scenario 2: Built-up Area Decrease
os.makedirs(f"{base_dir}/builtup_decrease/t1", exist_ok=True)
create_dummy_tif(f"{base_dir}/builtup_decrease/t1/t1_red.tif")
create_dummy_tif(f"{base_dir}/builtup_decrease/t1/t1_nir.tif")
create_dummy_tif(f"{base_dir}/builtup_decrease/t1/t1_scl.tif")

os.makedirs(f"{base_dir}/builtup_decrease/t2", exist_ok=True)
create_dummy_tif(f"{base_dir}/builtup_decrease/t2/t2_red.tif")
create_dummy_tif(f"{base_dir}/builtup_decrease/t2/t2_nir.tif")
create_dummy_tif(f"{base_dir}/builtup_decrease/t2/t2_scl.tif")

# Scenario 3: Flood / Water Body Change
os.makedirs(f"{base_dir}/flood_change/t1", exist_ok=True)
create_dummy_tif(f"{base_dir}/flood_change/t1/t1_red.tif")
create_dummy_tif(f"{base_dir}/flood_change/t1/t1_nir.tif")
create_dummy_tif(f"{base_dir}/flood_change/t1/t1_scl.tif")

os.makedirs(f"{base_dir}/flood_change/t2", exist_ok=True)
create_dummy_tif(f"{base_dir}/flood_change/t2/t2_red.tif")
create_dummy_tif(f"{base_dir}/flood_change/t2/t2_nir.tif")
create_dummy_tif(f"{base_dir}/flood_change/t2/t2_scl.tif")

print("Demo scenarios created.")
