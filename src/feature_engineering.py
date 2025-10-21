import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import rasterio.warp
from rasterio.merge import merge
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import osmnx as ox
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import PathPatch

print("--- Starting Phase 2: Feature Engineering ---")

class Config:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw')
    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
    RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results', 'phase_2')
    GRID_RESOLUTION_METERS = 200

def setup_directories():
    os.makedirs(Config.RESULTS_PATH, exist_ok=True)
    print("Feature engineering paths set up.")

def mosaic_dem_tiles():
    dem_mosaic_path = os.path.join(Config.PROCESSED_DATA_PATH, 'DEM', 'nagpur_dem_mosaic.tif')
    if os.path.exists(dem_mosaic_path):
        print("DEM mosaic already exists. Skipping creation.")
        return dem_mosaic_path
    raw_dem_folder = os.path.join(Config.RAW_DATA_PATH, 'DEM')
    dem_files_to_mosaic = [os.path.join(raw_dem_folder, f) for f in os.listdir(raw_dem_folder) if f.endswith(('.tif', '.hgt'))]
    if not dem_files_to_mosaic: return None
    print(f"Found {len(dem_files_to_mosaic)} DEM tiles. Mosaicking...")
    src_files = [rasterio.open(fp) for fp in dem_files_to_mosaic]
    mosaic, out_trans = merge(src_files)
    out_meta = src_files[0].meta.copy()
    out_meta.update({"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans})
    with rasterio.open(dem_mosaic_path, "w", **out_meta) as dest: dest.write(mosaic)
    for src in src_files: src.close()
    print(f"DEM mosaic saved to {dem_mosaic_path}")
    return dem_mosaic_path


def load_data(dem_mosaic_path):
    print("Loading processed data...")
    try:
        district_path = os.path.join(Config.PROCESSED_DATA_PATH, 'Boundaries', 'nagpur_boundary.gpkg')
        buildings_path = os.path.join(Config.PROCESSED_DATA_PATH, 'OSM', 'nagpur_osm_buildings.gpkg')
        pois_path = os.path.join(Config.PROCESSED_DATA_PATH, 'OSM', 'nagpur_osm_pois.gpkg')
        pop_path = os.path.join(Config.PROCESSED_DATA_PATH, 'Population', 'nagpur_population.tif')
        print("Fetching Nagpur city boundary from OpenStreetMap...")
        city_gdf = ox.geocode_to_gdf("Nagpur, Maharashtra, India")
        district_gdf = gpd.read_file(district_path)
        buildings_gdf = gpd.read_file(buildings_path)
        pois_gdf = gpd.read_file(pois_path) if os.path.exists(pois_path) else None
        pop_raster = rasterio.open(pop_path)
        dem_raster = rasterio.open(dem_mosaic_path)
        print("Data loaded successfully.")
        return district_gdf, city_gdf, buildings_gdf, pois_gdf, pop_raster, dem_raster
    except Exception as e:
        print(f"FATAL ERROR loading data: {e}.")
        return None


def create_analysis_grid(boundary_gdf):
    boundary_utm = boundary_gdf.to_crs(epsg=32644)
    xmin, ymin, xmax, ymax = boundary_utm.total_bounds
    width = int((xmax - xmin) / Config.GRID_RESOLUTION_METERS)
    height = int((ymax - ymin) / Config.GRID_RESOLUTION_METERS)
    transform = rasterio.transform.from_origin(xmin, ymax, Config.GRID_RESOLUTION_METERS, Config.GRID_RESOLUTION_METERS)
    crs = 'EPSG:32644'
    return width, height, transform, crs

def engineer_features(data, grid_params):
    district_gdf, city_gdf, buildings_gdf, pois_gdf, pop_raster, dem_raster = data
    width, height, transform, crs = grid_params

    pop_aligned = np.zeros((height, width), dtype=np.float32); dem_aligned = np.zeros((height, width), dtype=np.float32)
    for r, arr in [(pop_raster, pop_aligned), (dem_raster, dem_aligned)]:
        rasterio.warp.reproject(source=rasterio.band(r, 1), destination=arr, src_transform=r.transform, src_crs=r.crs, dst_transform=transform, dst_crs=crs, resampling=rasterio.warp.Resampling.bilinear)
    pop_aligned[pop_aligned < 0] = 0; gy, gx = np.gradient(dem_aligned, Config.GRID_RESOLUTION_METERS)
    slope_deg = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2))); buildings_utm = buildings_gdf.to_crs(crs=crs)
    clutter_grid = rasterize(shapes=buildings_utm.geometry, out_shape=(height, width), transform=transform, fill=0, all_touched=True, dtype=np.uint8)
    clutter_density = gaussian_filter(clutter_grid.astype(float), sigma=2); pop_norm = (pop_aligned - pop_aligned.min()) / (pop_aligned.max() - pop_aligned.min() + 1e-9)
    if pois_gdf is not None:
        pois_utm = pois_gdf.to_crs(crs=crs); poi_counts = np.zeros((height, width), dtype=np.float32)
        if not pois_utm.empty:
            coords = [(g.x, g.y) for g in pois_utm.geometry if not g.is_empty]; rows, cols = rasterio.transform.rowcol(transform, [c[0] for c in coords], [c[1] for c in coords])
            for r, c in zip(rows, cols):
                if 0 <= r < height and 0 <= c < width: poi_counts[r, c] += 1
        poi_density = gaussian_filter(poi_counts, sigma=5); poi_norm = (poi_density - poi_density.min()) / (poi_density.max() - poi_density.min() + 1e-9)
        demand_score = (0.7 * pop_norm) + (0.3 * poi_norm)
    else: demand_score = pop_norm
    slope_norm = (slope_deg - slope_deg.min()) / (slope_deg.max() - slope_deg.min() + 1e-9); clutter_norm = (clutter_density - clutter_density.min()) / (clutter_density.max() - clutter_density.min() + 1e-9)
    cost_surface = (0.4 * slope_norm) + (0.6 * clutter_norm)
    return {'population': pop_aligned, 'elevation': dem_aligned, 'slope': slope_deg, 'clutter': clutter_density, 'demand_score': demand_score, 'cost_surface': cost_surface}


def save_and_visualize(features, district_gdf, city_gdf, transform, crs):
    print("Saving all processed feature layers..."); [
        rasterio.open(os.path.join(Config.PROCESSED_DATA_PATH, f'feature_{name}.tif'), 'w', driver='GTiff', height=array.shape[0], width=array.shape[1], count=1, dtype=array.dtype, crs=crs, transform=transform).write(array, 1)
        for name, array in features.items()]
    print("All feature layers saved.")
    print("Generating robust multi-panel verification plot...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 16)); fig.suptitle('Phase 2: Engineered Feature Layers for Nagpur', fontsize=20)
    grid_extent = [transform.c, transform.c + transform.a * features['population'].shape[1], transform.f + transform.e * features['population'].shape[0], transform.f]
    plot_data = [(axes[0, 0], features['population'], 'Population Density', 'viridis'), (axes[0, 1], features['slope'], 'Terrain Slope (Degrees)', 'magma'),
                 (axes[1, 0], features['demand_score'], 'Final Demand Score', 'inferno'), (axes[1, 1], features['cost_surface'], 'Deployment Cost Surface', 'cividis')]
    
    district_utm = district_gdf.to_crs(crs); clip_polygon = district_utm.geometry.unary_union
    
    vertices = np.array(clip_polygon.exterior.coords.xy).T
    path = Path(vertices)
    
    for ax, data, title, cmap in plot_data:
       
        clip_patch = PathPatch(path, transform=ax.transData, facecolor='none', edgecolor='none')
        ax.add_patch(clip_patch)
        
        im = ax.imshow(data, cmap=cmap, extent=grid_extent, origin='upper')
        im.set_clip_path(clip_patch)
        
        ax.set_title(title); ax.set_aspect('equal', 'box'); ax.axis('off')
        fig.colorbar(im, ax=ax, shrink=0.7)

    district_utm.plot(ax=axes[0, 1], facecolor='none', edgecolor='yellow', linewidth=2); district_utm.plot(ax=axes[1, 1], facecolor='none', edgecolor='yellow', linewidth=2)
    if city_gdf is not None:
        city_utm = city_gdf.to_crs(crs)
        city_utm.plot(ax=axes[0, 0], facecolor='none', edgecolor='cyan', linewidth=2); city_utm.plot(ax=axes[1, 0], facecolor='none', edgecolor='cyan', linewidth=2)

    legend_elements = [Line2D([0], [0], color='yellow', lw=2, label='Nagpur District Boundary'), Line2D([0], [0], color='cyan', lw=2, label='Nagpur City Boundary')]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96]); plt.savefig(os.path.join(Config.RESULTS_PATH, 'engineered_features_dashboard.png')); plt.show()


if __name__ == "__main__":
    setup_directories()
    mosaic_path = mosaic_dem_tiles()
    if mosaic_path:
        loaded_data = load_data(mosaic_path)
        if loaded_data:
            district_boundary, city_boundary, buildings, pois, pop, dem = loaded_data
            grid_parameters = create_analysis_grid(district_boundary)
            engineered_features = engineer_features(loaded_data, grid_parameters)
            save_and_visualize(engineered_features, district_boundary, city_boundary, grid_parameters[2], grid_parameters[3])
    print("\n--- Phase 2: Feature Engineering FINISHED ---")