"""
Limitation 5 Fix: Download real SRTM 30m DEM tiles for Nagpur district.

NASA SRTM (Shuttle Radar Topography Mission) provides free 30m terrain elevation
data globally. This script downloads the tiles covering Nagpur district and places
them in data/raw/DEM/ so that the existing mosaic_dem_tiles() function in
feature_engineering.py picks them up automatically.

Usage:
    pip install elevation
    python scripts/download_srtm_dem.py

Output:
    data/raw/DEM/nagpur_srtm.tif   (SRTM3 = ~90m, or SRTM1 = 30m where available)
"""

import os
import sys
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DEM_RAW_DIR  = os.path.join(PROJECT_ROOT, 'data', 'raw', 'DEM')
OUTPUT_PATH  = os.path.join(DEM_RAW_DIR, 'nagpur_srtm.tif')

# Nagpur district bounding box (WGS84): west, south, east, north
BOUNDS = (78.0, 20.5, 79.8, 21.9)


def check_elevation_installed() -> bool:
    try:
        import elevation  # noqa: F401
        return True
    except ImportError:
        return False


def download_via_elevation_lib() -> None:
    """Use the `elevation` package to fetch SRTM data."""
    import elevation
    os.makedirs(DEM_RAW_DIR, exist_ok=True)
    print("Downloading SRTM DEM for Nagpur district via `elevation` library...")
    print(f"Bounding box: {BOUNDS}")
    elevation.clip(bounds=BOUNDS, output=OUTPUT_PATH, product='SRTM3')
    elevation.clean()
    print(f"DEM saved to: {OUTPUT_PATH}")


def download_via_direct_tiles() -> None:
    """
    Fallback: download individual SRTM HGT tiles directly from NASA/CGIAR mirrors.
    Nagpur district spans tiles: N20E078, N20E079, N21E078, N21E079
    """
    import requests

    # CGIAR-CSI SRTM v4.1 tiles (free, no login required)
    BASE_URL = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/"
    # Tile naming: srtm_XX_YY.zip  (5-degree tiles)
    # Nagpur falls in CGIAR tile: srtm_55_08
    tiles = ['srtm_55_08.zip']

    os.makedirs(DEM_RAW_DIR, exist_ok=True)

    for tile in tiles:
        url = BASE_URL + tile
        dest_zip = os.path.join(DEM_RAW_DIR, tile)
        print(f"Downloading {url} ...")
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest_zip, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                        f.write(chunk)
            print(f"Saved {dest_zip}")
            # Unzip
            import zipfile
            with zipfile.ZipFile(dest_zip, 'r') as z:
                z.extractall(DEM_RAW_DIR)
                print(f"Extracted: {z.namelist()}")
        except Exception as e:
            print(f"WARNING: Could not download {tile}: {e}")

    # Also try individual 1-degree HGT tiles from NASA EarthData (no login for SRTMGL3)
    hgt_base = "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL3.003/2000.02.11/"
    hgt_tiles = ['N20E078.SRTMGL3.hgt.zip', 'N20E079.SRTMGL3.hgt.zip',
                 'N21E078.SRTMGL3.hgt.zip', 'N21E079.SRTMGL3.hgt.zip']
    print("\nAttempting NASA EarthData tiles (may require Earthdata account)...")
    for tile in hgt_tiles:
        url = hgt_base + tile
        dest_zip = os.path.join(DEM_RAW_DIR, tile)
        try:
            with requests.get(url, stream=True, timeout=30, allow_redirects=True) as r:
                if r.status_code == 200:
                    with open(dest_zip, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):
                            f.write(chunk)
                    import zipfile
                    with zipfile.ZipFile(dest_zip, 'r') as z:
                        z.extractall(DEM_RAW_DIR)
                    print(f"  Downloaded: {tile}")
                else:
                    print(f"  Skipped {tile} (HTTP {r.status_code})")
        except Exception as e:
            print(f"  Could not download {tile}: {e}")


def main():
    os.makedirs(DEM_RAW_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        print(f"DEM already exists: {OUTPUT_PATH}")
        print("Delete it to re-download.")
        return

    if check_elevation_installed():
        try:
            download_via_elevation_lib()
            return
        except Exception as e:
            print(f"elevation library failed: {e}")
            print("Falling back to direct tile download...")
    else:
        print("`elevation` package not installed. Install it with:")
        print("    pip install elevation")
        print("Falling back to direct tile download...\n")

    download_via_direct_tiles()

    tif_files = [f for f in os.listdir(DEM_RAW_DIR) if f.endswith(('.tif', '.hgt'))]
    if tif_files:
        print(f"\nDEM files in {DEM_RAW_DIR}: {tif_files}")
        print("The existing mosaic_dem_tiles() function will merge these automatically.")
    else:
        print("\nWARNING: No DEM files downloaded successfully.")
        print("Please download SRTM tiles manually from https://srtm.csi.cgiar.org/")
        print(f"and place .tif or .hgt files in: {DEM_RAW_DIR}")


if __name__ == '__main__':
    main()
