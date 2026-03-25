import geopandas as gpd
import rasterio
from rasterio.mask import mask
import os
import matplotlib.pyplot as plt
from rasterio.plot import show
import contextily as cx
import osmnx as ox
import time
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

print("--- Starting Phase 1: Data Preprocessing ---")

class Config:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw')
    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
    BOUNDARY_SHP_FILENAME = 'gadm41_IND_2.shp'
    DISTRICT_NAME_COLUMN = 'NAME_2'
    NAGPUR_NAME_IN_SHP = 'Nagpur'
    MAX_RETRIES = 3 

def setup_directories():
    os.makedirs(os.path.join(Config.PROCESSED_DATA_PATH, 'DEM'), exist_ok=True)
    os.makedirs(os.path.join(Config.PROCESSED_DATA_PATH, 'Population'), exist_ok=True)
    os.makedirs(os.path.join(Config.PROCESSED_DATA_PATH, 'OSM'), exist_ok=True)
    os.makedirs(os.path.join(Config.PROCESSED_DATA_PATH, 'Boundaries'), exist_ok=True)
    print("Project paths set up.")

def process_boundary():
    print("Loading Nagpur district boundary...")
    try:
        boundary_shp_path = os.path.join(Config.RAW_DATA_PATH, 'Boundaries', Config.BOUNDARY_SHP_FILENAME)
        boundary_gdf = gpd.read_file(boundary_shp_path)
        nagpur_boundary = boundary_gdf[boundary_gdf[Config.DISTRICT_NAME_COLUMN] == Config.NAGPUR_NAME_IN_SHP].copy()
        if nagpur_boundary.empty:
            print(f"FATAL ERROR: Nagpur district not found.")
            return None
        print("Nagpur boundary loaded successfully.")
        nagpur_boundary = nagpur_boundary.to_crs(epsg=4326)
        processed_boundary_path = os.path.join(Config.PROCESSED_DATA_PATH, 'Boundaries', 'nagpur_boundary.gpkg')
        nagpur_boundary.to_file(processed_boundary_path, driver='GPKG')
        print(f"Processed Nagpur boundary saved to {processed_boundary_path}")
        return nagpur_boundary
    except Exception as e:
        print(f"FATAL ERROR loading boundary shapefile: {e}")
        return None


def process_rasters(nagpur_boundary):
    rasters_to_process = [('DEM', 'DEM', 'nagpur_dem.tif'), ('Population', 'Population', 'nagpur_population.tif')]
    for name, folder, out_filename in rasters_to_process:
        print(f"Processing {name} data...")
        raw_folder_path = os.path.join(Config.RAW_DATA_PATH, folder)
        try:
            raster_file = [os.path.join(raw_folder_path, f) for f in os.listdir(raw_folder_path) if f.endswith(('.tif', '.hgt'))][0]
        except IndexError:
            print(f"ERROR: No .tif or .hgt file found. Skipping.")
            continue
        with rasterio.open(raster_file) as src:
            nagpur_boundary_reprojected = nagpur_boundary.to_crs(src.crs)
            try:
                out_image, out_transform = mask(src, nagpur_boundary_reprojected.geometry, crop=True)
                out_meta = src.meta.copy()
                out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
                output_raster_path = os.path.join(Config.PROCESSED_DATA_PATH, folder, out_filename)
                with rasterio.open(output_raster_path, "w", **out_meta) as dest:
                    dest.write(out_image)
                print(f"Clipped {name} data saved to {output_raster_path}")
            except ValueError as e:
                print(f"Clipping failed for {name}. Error: {e}")

def process_osm_data(nagpur_boundary):
    
    nagpur_polygon = nagpur_boundary.geometry.union_all()

   
    def fetch_osm_features(polygon, tags, feature_name):
        for attempt in range(Config.MAX_RETRIES):
            try:
                print(f"Extracting OSM {feature_name}... (Attempt {attempt + 1}/{Config.MAX_RETRIES})")
                features = ox.features_from_polygon(polygon, tags)
                print(f"Successfully extracted {feature_name}.")
                return features
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {feature_name}: {e}")
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(5)  # Wait 5 seconds before retrying
                else:
                    print(f"Could not extract OSM {feature_name} after {Config.MAX_RETRIES} attempts.")
                    return None
    
   
    buildings = fetch_osm_features(nagpur_polygon, {'building': True}, "building footprints")
    if buildings is not None and not buildings.empty:
        buildings_path = os.path.join(Config.PROCESSED_DATA_PATH, 'OSM', 'nagpur_osm_buildings.gpkg')
        buildings.to_file(buildings_path, driver='GPKG')
        print(f"OSM Buildings for Nagpur saved to {buildings_path}")
    
  
    poi_tags = {"amenity": True, "shop": True, "tourism": True, "leisure": True, "office": True}
    pois = fetch_osm_features(nagpur_polygon, poi_tags, "POIs")
    if pois is not None and not pois.empty:
        original_poi_count = len(pois)
        pois_points_only = pois[pois.geometry.geom_type == 'Point'].copy()
        print(f"Filtered POIs: Kept {len(pois_points_only)} Point features out of {original_poi_count} total.")
        if not pois_points_only.empty:
            pois_path = os.path.join(Config.PROCESSED_DATA_PATH, 'OSM', 'nagpur_osm_pois.gpkg')
            pois_points_only.to_file(pois_path, driver='GPKG')
            print(f"OSM POIs for Nagpur saved to {pois_path}")


def create_verification_plot(nagpur_boundary):
    print("\nPreprocessing complete. Generating a final verification plot...")
    fig, ax = plt.subplots(figsize=(15, 15))
    nagpur_boundary_map_crs = nagpur_boundary.to_crs(epsg=3857) 

    try:
        pop_path = os.path.join(Config.PROCESSED_DATA_PATH, 'Population', 'nagpur_population.tif')
        with rasterio.open(pop_path) as src:
            show(src, ax=ax, cmap='YlOrRd', alpha=0.6, zorder=2)
        print("Population layer added to plot.")
    except Exception:
        print("Could not plot population data.")
    
    
    try:
        pois_path = os.path.join(Config.PROCESSED_DATA_PATH, 'OSM', 'nagpur_osm_pois.gpkg')
        pois_gdf = gpd.read_file(pois_path)
        pois_gdf.to_crs(epsg=3857).plot(ax=ax, color='blue', marker='.', markersize=10, alpha=0.7, zorder=3)
        print("POIs layer added to plot.")
    except Exception:
        print("Could not plot POI data.")
    
    
    nagpur_boundary_map_crs.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=3, zorder=4)

    
    minx, miny, maxx, maxy = nagpur_boundary_map_crs.total_bounds
    ax.set_xlim(minx - 500, maxx + 500) 
    ax.set_ylim(miny - 500, maxy + 500)

    print("Adding basemap...")
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, zorder=1)
    
    ax.set_title('Nagpur District: Processed Data Layers', fontsize=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    
    legend_elements = [
        Line2D([0], [0], color='black', lw=3, label='Nagpur Boundary'),
        Line2D([0], [0], marker='o', color='w', label='Points of Interest (POIs)', markerfacecolor='blue', markersize=8),
        Patch(facecolor='yellow', edgecolor='r', alpha=0.6, label='High Population Density')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.PROJECT_ROOT, 'results', 'verification_map.png'), dpi=300)
    print("\nVerification map saved to 'results/verification_map.png'")
    plt.show()


if __name__ == "__main__":
    setup_directories()
    nagpur_boundary_gdf = process_boundary()
    if nagpur_boundary_gdf is not None:
        process_rasters(nagpur_boundary_gdf)
        process_osm_data(nagpur_boundary_gdf)
    
        create_verification_plot(nagpur_boundary_gdf)
    
    print("\n--- Phase 1: Data Preprocessing FINISHED ---")