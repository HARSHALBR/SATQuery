import json
import rasterio
import numpy as np

from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from pyproj import Transformer


def convert_regions_to_geojson(
    input_json_path,
    cluster_file_path,
    output_geojson_path,
    target_crs="EPSG:4326"
):
    """
    Dynamically convert region report and clustering raster into a GeoJSON geometry file.
    """
    # Read exact cluster raster
    with rasterio.open(cluster_file_path) as src:
        clusters = src.read(1)
        raster_transform = src.transform
        raster_crs = str(src.crs)

    # Coordinate transformation
    transformer = Transformer.from_crs(
        raster_crs,
        target_crs,
        always_xy=True
    )

    # Read region report
    with open(input_json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    features = []

    # Extract exact geometry for every region
    for region in report["regions"]:
        region_id = int(region["region_id"])
        mask = (clusters == region_id).astype(np.uint8)

        if mask.sum() == 0:
            raise ValueError(f"Region {region_id} has no pixels in the cluster raster.")

        # Extract connected raster geometry
        geometries = []
        for geom, value in shapes(mask, mask=mask.astype(bool), transform=raster_transform):
            if value == 1:
                geometries.append(shape(geom))

        if not geometries:
            raise ValueError(f"Could not extract geometry for region {region_id}.")

        # Combine pieces if necessary
        exact_geometry = unary_union(geometries)

        # Convert to WGS84
        exact_geometry_wgs84 = shapely_transform(
            transformer.transform,
            exact_geometry
        )

        properties = {
            "region_id": region_id,
            "pixels": region["pixels"],
            "area_m2": region["area_m2"],
            "area_ha": region["area_ha"],
            "ndvi_change": region["changes"]["ndvi"],
            "ndbi_change": region["changes"]["ndbi"],
            "vv_change": region["changes"]["vv"],
            "vh_change": region["changes"]["vh"],
            "ndvi_decrease": region["evidence"]["ndvi_decrease"],
            "ndbi_increase": region["evidence"]["ndbi_increase"],
            "vv_decrease": region["evidence"]["vv_decrease"],
            "vh_decrease": region["evidence"]["vh_decrease"],
            "evidence_score": region["evidence_score"],
            "status": region.get("status", "UNCERTAIN"),
            "pixel_composition": region.get("pixel_composition", {}),
            "optical_evidence": region.get("optical_evidence", {}),
            "sar_evidence": region.get("sar_evidence", {})
        }

        features.append({
            "type": "Feature",
            "properties": properties,
            "geometry": mapping(exact_geometry_wgs84)
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(output_geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

    return geojson


if __name__ == "__main__":
    import sys
    print("convert_regions_to_geojson.py should be invoked via orchestration.")
    sys.exit(1)