"""
Limitation 1 Fix: Download real WorldPop 2020 population raster for Nagpur district.

WorldPop provides free, open 100m resolution population estimates for India.
This script downloads the India raster and clips it to Nagpur district boundary,
replacing the synthetic nagpur_population.tif used in the pipeline.

Usage:
    python scripts/download_worldpop.py

Output:
    data/processed/Population/nagpur_population.tif
"""

import os
import sys
import requests
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import mapping

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# WorldPop 2020 India constrained 100m population raster (open access)
WORLDPOP_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/"
    "2020/maxar_v1/IND/ind_ppp_2020_constrained.tif"
)

RAW_POP_PATH    = os.path.join(PROJECT_ROOT, 'data', 'raw', 'Population', 'ind_ppp_2020_constrained.tif')
OUTPUT_PATH     = os.path.join(PROJECT_ROOT, 'data', 'processed', 'Population', 'nagpur_population.tif')
BOUNDARY_PATH   = os.path.join(PROJECT_ROOT, 'data', 'processed', 'Boundaries', 'nagpur_boundary.gpkg')

# Nagpur district bounding box (WGS84) — used for partial download fallback
NAGPUR_BBOX = (78.0, 20.5, 79.8, 21.9)   # (west, south, east, north)


def download_file(url: str, dest: str, chunk_mb: int = 8) -> None:
    """Stream-download a file with progress display."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {url} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"  {pct:.1f}%  ({downloaded // 1_000_000} / {total // 1_000_000} MB)", end='\r')
    print(f"\nSaved to {dest}")


def clip_to_nagpur(src_path: str, boundary_path: str, out_path: str) -> None:
    """Clip the India-wide raster to Nagpur district and save."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(boundary_path):
        boundary = gpd.read_file(boundary_path)
    else:
        # Fallback: download Nagpur boundary from OSM
        print("Boundary file not found — fetching from OSM...")
        import osmnx as ox
        boundary = ox.geocode_to_gdf("Nagpur district, Maharashtra, India")
        os.makedirs(os.path.dirname(boundary_path), exist_ok=True)
        boundary.to_file(boundary_path, driver='GPKG')

    with rasterio.open(src_path) as src:
        # Reproject boundary to match raster CRS
        boundary_wgs = boundary.to_crs(src.crs)
        shapes = [mapping(geom) for geom in boundary_wgs.geometry]

        out_image, out_transform = rio_mask(src, shapes, crop=True, nodata=0)
        out_meta = src.meta.copy()
        out_meta.update({
            'height':    out_image.shape[1],
            'width':     out_image.shape[2],
            'transform': out_transform,
            'nodata':    0,
        })

    # Clamp negative values (WorldPop uses -99999 as nodata in some tiles)
    out_image = np.maximum(out_image, 0).astype(np.float32)

    with rasterio.open(out_path, 'w', **out_meta) as dst:
        dst.write(out_image)

    total_pop = int(out_image.sum())
    print(f"Clipped population raster saved: {out_path}")
    print(f"Nagpur district total population estimate: {total_pop:,}")


def main():
    # Step 1: Download India raster if not already present
    if not os.path.exists(RAW_POP_PATH):
        try:
            download_file(WORLDPOP_URL, RAW_POP_PATH)
        except Exception as e:
            print(f"ERROR: Could not download WorldPop raster: {e}")
            print("Please download manually from:")
            print(f"  {WORLDPOP_URL}")
            print(f"And place it at: {RAW_POP_PATH}")
            sys.exit(1)
    else:
        print(f"WorldPop raster already present: {RAW_POP_PATH}")

    # Step 2: Clip to Nagpur district
    print("Clipping to Nagpur district boundary...")
    clip_to_nagpur(RAW_POP_PATH, BOUNDARY_PATH, OUTPUT_PATH)
    print("\nDone. The pipeline will now use real WorldPop 2020 population data.")


if __name__ == '__main__':
    main()
