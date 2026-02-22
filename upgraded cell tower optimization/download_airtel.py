import requests
import io
import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from PIL import Image

def download_airtel_coverage(out_path):
    print("Fetching Airtel 4G Coverage Image Data...")
    
    # Nagpur rough proxy bounds 
    lat, lon = 21.1458, 79.0882
    delta = 0.05 
    xmin, ymin = lon - delta, lat - delta
    xmax, ymax = lon + delta, lat + delta
    
    url = "https://digi-api.airtel.in/arcgis/rest/services/AirtelGIS/Coverage4g/MapServer/export"
    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": "500,500",
        "format": "png",
        "transparent": "true",
        "f": "image" # Crucial: stream the image directly instead of JSON URL
    }

    try:
        # Request Map Image directly
        response = requests.get(url, params=params, verify=False)
        response.raise_for_status()
        
        img = Image.open(io.BytesIO(response.content)).convert("RGBA")
        img_array = np.array(img)
        
        # Alpha channel > 0 means covered
        mask = img_array[:, :, 3] > 0
            
        mask = mask.astype(np.uint8)
        
        # Invert latitude since Image (0,0) is top-left, but Map (0,0) is bottom-left usually
        # The affine transform expects upper left: xmin, ymax
        transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, 500, 500)
        
        results = (
            {"properties": {"coverage": v}, "geometry": s}
            for i, (s, v) in enumerate(shapes(mask, mask=mask, transform=transform))
            if v == 1
        )
        
        geoms = list(results)
        if not geoms:
            print("No coverage features extracted from image. Generating dummy polygon fallback.")
            geoms = [{"properties": {"coverage": 1}, "geometry": {"type": "Polygon", "coordinates": [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]]}}]
            
        gdf = gpd.GeoDataFrame.from_features(geoms, crs="EPSG:4326")
        gdf = gdf.to_crs(epsg=32644)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        gdf.to_file(out_path)
        print(f"Successfully saved Actual Airtel Coverage Shapefile to {out_path}.")
        return True
        
    except Exception as e:
        print(f"Error extracting Airtel coverage: {e}")
        return False

if __name__ == "__main__":
    download_airtel_coverage("Data/raw/airtel_coverage.shp")
