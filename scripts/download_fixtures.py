import json
import os
import requests
import argparse
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

STAC_API = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/{item_id}"

def download_asset(url, dest_path, roi):
    print(f"Downloading ROI from {url} to {dest_path}")
    
    # roi is [min_lon, min_lat, max_lon, max_lat] in EPSG:4326
    # We will open the remote COG and read only the window intersecting the ROI
    try:
        with rasterio.open(url) as src:
            # Transform the 4326 ROI bounds into the source CRS
            src_crs = src.crs
            min_lon, min_lat, max_lon, max_lat = roi
            # transform_bounds(src_crs, dst_crs, left, bottom, right, top)
            # So here we transform from 4326 to src_crs
            roi_src_crs = transform_bounds("EPSG:4326", src_crs, min_lon, min_lat, max_lon, max_lat)
            
            # Create a window from these bounds
            window = from_bounds(*roi_src_crs, transform=src.transform)
            
            # Snap window to nearest integer pixels
            window = window.round_lengths().round_offsets()
            
            # Read data
            data = src.read(1, window=window)
            
            # Calculate new transform for the cropped window
            out_transform = src.window_transform(window)
            
            # Write out to local file
            profile = src.profile
            profile.update({
                'height': window.height,
                'width': window.width,
                'transform': out_transform,
                'blockxsize': min(window.width, 256),
                'blockysize': min(window.height, 256),
                'tiled': True,
                'compress': 'lzw'
            })
            
            with rasterio.open(dest_path, 'w', **profile) as dst:
                dst.write(data, 1)
                
        print(f"Successfully saved cropped ROI to {dest_path}")
    except Exception as e:
        print(f"Failed to download/crop {url}: {e}")
        raise e

def verify_file(path, expected_band):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")
    import rasterio
    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError(f"File {path} has no CRS.")
            if src.width <= 0 or src.height <= 0:
                raise ValueError(f"File {path} has invalid dimensions.")
    except Exception as e:
        raise ValueError(f"File {path} is not a valid readable raster: {e}")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument("--output-dir", default="datasets/golden_fixtures/raw", help="Output directory")
    args = parser.parse_args()
    
    with open(args.manifest, "r") as f:
        manifest = json.load(f)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    for pair in manifest["pairs"]:
        print(f"Processing pair: {pair['pair_id']}")
        roi = pair["roi"]
        
        for t_key in ["observation_t1", "observation_t2"]:
            obs = pair[t_key]
            item_id = obs["stac_item_id"]
            
            # 1. Fetch STAC item metadata
            meta_resp = requests.get(STAC_API.format(item_id=item_id))
            if meta_resp.status_code != 200:
                print(f"Could not find STAC item {item_id}")
                continue
                
            item = meta_resp.json()
            assets = item.get("assets", {})
            
            # 2. Download requested bands
            for band in ["red", "nir", "scl"]:
                if band in assets:
                    url = assets[band]["href"]
                    dest = os.path.join(args.output_dir, f"{item_id}_{band}.tif")
                    download_asset(url, dest, roi)
                    verify_file(dest, band)
                else:
                    print(f"Missing band {band} in item {item_id}")

if __name__ == "__main__":
    main()
