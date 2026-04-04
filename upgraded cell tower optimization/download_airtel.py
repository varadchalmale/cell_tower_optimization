"""
Download Airtel 4G coverage map for Nagpur district via ArcGIS REST API.

Uses the full district bounding box (~78.25°E to 79.66°E, 20.58°N to 21.72°N)
rather than just central Nagpur.
"""

import requests
import io
import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from PIL import Image


def download_airtel_coverage(out_path):
    print("  Fetching Airtel 4G coverage for Nagpur district...")

    # Full district bounding box (from GADM Nagpur boundary)
    xmin, ymin = 78.25, 20.58
    xmax, ymax = 79.66, 21.72

    url = "https://digi-api.airtel.in/arcgis/rest/services/AirtelGIS/Coverage4g/MapServer/export"
    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": "1000,1000",
        "format": "png",
        "transparent": "true",
        "f": "image",
    }

    try:
        response = requests.get(url, params=params, verify=False, timeout=30)
        response.raise_for_status()

        img = Image.open(io.BytesIO(response.content)).convert("RGBA")
        img_array = np.array(img)
        mask = (img_array[:, :, 3] > 0).astype(np.uint8)

        transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, 1000, 1000)
        results = [
            {"properties": {"coverage": v}, "geometry": s}
            for s, v in shapes(mask, mask=mask, transform=transform)
            if v == 1
        ]

        if not results:
            print("  No coverage features extracted. Using district-wide fallback.")
            results = [{"properties": {"coverage": 1}, "geometry": {
                "type": "Polygon",
                "coordinates": [[[xmin, ymin], [xmax, ymin], [xmax, ymax],
                                 [xmin, ymax], [xmin, ymin]]]
            }}]

        gdf = gpd.GeoDataFrame.from_features(results, crs="EPSG:4326")
        gdf = gdf.to_crs(epsg=32644)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        gdf.to_file(out_path)
        print(f"  Saved Airtel coverage shapefile to {out_path}")
        return True

    except Exception as e:
        print(f"  Error downloading Airtel coverage: {e}")
        return False


if __name__ == "__main__":
    download_airtel_coverage("Data/raw/airtel_coverage.shp")
