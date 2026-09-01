import rasterio
import numpy as np


def analyze_clusters(
    cluster_file,
    ndvi_file,
    ndbi_file,
    vv_file,
    vh_file,
    patch_bounds=None
):

    # --------------------------------------------------
    # Read cluster raster
    # --------------------------------------------------

    with rasterio.open(cluster_file) as src:
        clusters = src.read(1)

    # --------------------------------------------------
    # Read SAR changes
    # --------------------------------------------------

    with rasterio.open(vv_file) as src:
        vv = src.read(1)

    with rasterio.open(vh_file) as src:
        vh = src.read(1)

    # --------------------------------------------------
    # Extract matching optical patches
    # --------------------------------------------------

    if patch_bounds is not None:

        with rasterio.open(ndvi_file) as src:

            window = rasterio.windows.from_bounds(
                *patch_bounds,
                transform=src.transform
            )

            ndvi = src.read(1, window=window)

        with rasterio.open(ndbi_file) as src:

            ndbi = src.read(1, window=window)

    else:

        with rasterio.open(ndvi_file) as src:
            ndvi = src.read(1)

        with rasterio.open(ndbi_file) as src:
            ndbi = src.read(1)

    # --------------------------------------------------
    # Check dimensions
    # --------------------------------------------------

    if not (
        clusters.shape ==
        ndvi.shape ==
        ndbi.shape ==
        vv.shape ==
        vh.shape
    ):
        raise ValueError(
            "Cluster, NDVI, NDBI, VV and VH rasters "
            "must have matching dimensions."
        )

    # --------------------------------------------------
    # Analyze clusters
    # --------------------------------------------------

    region_ids = np.unique(clusters)
    region_ids = region_ids[region_ids != 0]

    results = []

    for region_id in region_ids:

        mask = clusters == region_id

        pixels = int(mask.sum())

        area_m2 = pixels * 100
        area_ha = area_m2 / 10000

        ndvi_values = ndvi[mask]
        ndbi_values = ndbi[mask]
        vv_values = vv[mask]
        vh_values = vh[mask]

        result = {
            "id": int(region_id),
            "pixels": pixels,
            "area_ha": float(area_ha),
            "ndvi": float(np.mean(ndvi_values)),
            "ndbi": float(np.mean(ndbi_values)),
            "vv": float(np.mean(vv_values)),
            "vh": float(np.mean(vh_values))
        }

        results.append(result)

    # Largest clusters first
    results.sort(
        key=lambda x: x["pixels"],
        reverse=True
    )

    return {
        "total_regions": len(results),
        "regions": results
    }