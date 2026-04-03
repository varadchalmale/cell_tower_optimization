import folium
import rasterio
import numpy as np
import os
import json
import branca.colormap as cm_branca
import matplotlib.cm as cm_mpl
import geopandas as gpd
import pyproj
from folium.raster_layers import ImageOverlay
from src.config import Config
from src.propagation_models import PropagationModel

print("--- Starting Upgraded Interactive Map Generation ---")

# ---------------------------------------------------------------------------
# Compute maximum coverage radius using COST-231 Hata model
# RSRP = Tx_power - PathLoss  →  solve for d when RSRP = RSRP_MIN_DBM
# ---------------------------------------------------------------------------
def _cost231_coverage_radius_m():
    """Return edge-of-coverage radius (m) where RSRP >= Config.RSRP_MIN_DBM."""
    max_loss = Config.TRANSMIT_POWER_DBM - Config.RSRP_MIN_DBM  # dB budget
    f = Config.FREQUENCY_MHZ
    h_te = Config.TOWER_HEIGHT_M
    h_re = Config.USER_HEIGHT_M
    a_hre = (1.1 * np.log10(f) - 0.7) * h_re - (1.56 * np.log10(f) - 0.8)
    cm = 3.0  # urban correction (worst case)
    intercept = 46.3 + 33.9 * np.log10(f) - 13.82 * np.log10(h_te) - a_hre + cm
    slope = 44.9 - 6.55 * np.log10(h_te)
    log10_d = (max_loss - intercept) / slope
    d_km = float(np.clip(10 ** log10_d, 0.2, 5.0))
    return d_km * 1000.0  # metres

coverage_radius_m = _cost231_coverage_radius_m()
print(f"Coverage radius (COST-231 Hata, RSRP >= {Config.RSRP_MIN_DBM} dBm): {coverage_radius_m:.0f} m")

# Paths
solution_path = os.path.join(Config.RESULTS_PATH, 'best_solution_upgraded.json')
demand_raster_path = os.path.join(Config.PROCESSED_DATA_PATH, 'feature_demand_score.tif')
boundary_path = os.path.join(Config.PROCESSED_DATA_PATH, 'Boundaries', 'nagpur_boundary.gpkg')
output_map_path = os.path.join(Config.RESULTS_PATH, 'upgraded_interactive_map.html')

# Load Data
try:
    with open(solution_path, 'r') as f:
        best_solution = json.load(f)
    demand_raster = rasterio.open(demand_raster_path)
    boundary_gdf = gpd.read_file(boundary_path)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Setup Map
center_point = boundary_gdf.to_crs(epsg=4326).unary_union.centroid
m = folium.Map(location=[center_point.y, center_point.x], zoom_start=10, tiles="CartoDB dark_matter")

# Demand Heatmap
demand_grid = demand_raster.read(1)
xmin, ymin, xmax, ymax = demand_raster.bounds
transformer = pyproj.Transformer.from_crs(demand_raster.crs, "EPSG:4326", always_xy=True)
lon_min, lat_min = transformer.transform(xmin, ymin)
lon_max, lat_max = transformer.transform(xmax, ymax)
map_bounds = [[lat_min, lon_min], [lat_max, lon_max]]

cmap = cm_mpl.get_cmap('magma')
norm_demand = (demand_grid - np.min(demand_grid)) / (np.max(demand_grid) - np.min(demand_grid) + 1e-9)
image_data = (cmap(norm_demand) * 255).astype(np.uint8)

ImageOverlay(image=image_data, bounds=map_bounds, opacity=0.4, name='Demand Heatmap').add_to(m)

# Add Towers and their RSRP-computed coverage circles
tower_group = folium.FeatureGroup(name='Upgraded Tower Placement')
coverage_group = folium.FeatureGroup(name='Coverage Area (COST-231 Hata)')

for i, (y, x) in enumerate(best_solution):
    lon, lat = transformer.transform(*demand_raster.transform * (x + 0.5, y + 0.5))
    folium.Marker(
        location=[lat, lon],
        popup=(
            f"<b>Tower #{i+1}</b><br>"
            f"Lat: {lat:.4f}, Lon: {lon:.4f}<br>"
            f"Type: 5G/LTE Macrocell<br>"
            f"Coverage radius: {coverage_radius_m:.0f} m"
        ),
        icon=folium.Icon(color='red', icon='tower-cell', prefix='fa')
    ).add_to(tower_group)

    # Coverage boundary computed from propagation model, not a fixed estimate
    folium.Circle(
        radius=coverage_radius_m,
        location=[lat, lon],
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.08,
        weight=1
    ).add_to(coverage_group)

tower_group.add_to(m)
coverage_group.add_to(m)

# Boundary
folium.GeoJson(
    boundary_gdf.to_json(),
    style_function=lambda x: {'color': 'cyan', 'weight': 1, 'fillOpacity': 0},
    name='Nagpur District Boundary'
).add_to(m)

folium.LayerControl().add_to(m)
m.save(output_map_path)
print(f"Upgraded interactive map saved to {output_map_path}")
