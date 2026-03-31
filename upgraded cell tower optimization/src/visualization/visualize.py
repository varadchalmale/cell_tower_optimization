import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os
import folium
from branca.colormap import LinearColormap
import numpy as np


def _coverage_radius_m(config):
    """
    Compute the maximum coverage radius (metres) where RSRP = rsrp_min_dbm,
    using COST-231 Hata with the full EIRP budget from config.

    Derivation:
        EIRP     = Tx_power + Antenna_gain - Cable_loss        (dBm)
        RSRP     = EIRP - PathLoss
        PathLoss = EIRP - RSRP_min   at edge-of-coverage
        COST-231: L = intercept + slope * log10(d)
        → d = 10^( (max_path_loss - intercept) / slope )
    """
    rf         = config['rf_params']
    f_mhz      = rf['frequency_mhz']
    h_te       = rf['antenna_height_m']
    h_re       = rf.get('receiver_height_m', 1.5)
    tx_dbm     = rf['transmit_power_dbm']
    ant_gain   = rf.get('antenna_gain_dbi', 18)   # 65-deg sector panel
    cable_loss = rf.get('cable_loss_db',    2)     # feeder + connectors
    rsrp_min   = config.get('thresholds', {}).get('rsrp_min_dbm', -95)

    eirp_dbm  = tx_dbm + ant_gain - cable_loss           # Effective radiated power
    max_loss  = eirp_dbm - rsrp_min                      # Maximum tolerable path loss

    # COST-231 Hata (medium-small city form)
    a_hre     = (1.1 * np.log10(f_mhz) - 0.7) * h_re - (1.56 * np.log10(f_mhz) - 0.8)
    cm        = 3.0   # urban correction
    intercept = 46.3 + 33.9 * np.log10(f_mhz) - 13.82 * np.log10(h_te) - a_hre + cm
    slope     = 44.9 - 6.55 * np.log10(h_te)

    log10_d   = (max_loss - intercept) / slope
    d_km      = float(np.clip(10 ** log10_d, 0.1, 15.0))  # physical range 0.1–15 km
    return d_km * 1000.0  # metres

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.output_dir = self.config['paths']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_heatmap(self, grid_df, col, title, cmap='hot', vmin=None, vmax=None):
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(grid_df['x'], grid_df['y'], c=grid_df[col], cmap=cmap, vmin=vmin, vmax=vmax, s=5)
        plt.colorbar(scatter, label=col)
        plt.title(title)
        plt.axis('equal')
        plt.savefig(f"{self.output_dir}/{col}_heatmap.png", dpi=300)
        plt.close()

    def plot_pareto_front(self, res):
        """Plot the Pareto front from NSGA-II."""
        plt.figure(figsize=(10, 8))
        F = res.F
        # Objectives are minimized, flip signs for coverage and capacity (which were maximized)
        obj1_cov = -F[:, 0]
        obj2_cap = -F[:, 1]
        obj3_cost = F[:, 2]

        scatter = plt.scatter(obj1_cov, obj2_cap, c=obj3_cost, cmap='viridis', s=50)
        plt.colorbar(scatter, label='Deployment Cost Index (Lower is Better)')
        plt.xlabel('Satisfied Demand (Weighted Coverage)')
        plt.ylabel('Network Capacity (Sum Throughput Mbps)')
        plt.title('NSGA-II Pareto Frontier')
        plt.savefig(f"{self.output_dir}/pareto_frontier.png", dpi=300)
        plt.close()

    def plot_candidates_and_selected(self, candidates_df, selected_df, background_df, col_bg='predicted_traffic_mbps'):
        """Plot all candidates, highlighted selected towers, over a background (e.g. demand)."""
        plt.figure(figsize=(12, 10))
        # Background
        plt.scatter(background_df['x'], background_df['y'], c=background_df[col_bg], cmap='Blues', alpha=0.5, s=2)
        plt.colorbar(label='Traffic Demand (Mbps)')
        
        # All candidates
        plt.scatter(candidates_df['x'], candidates_df['y'], c='gray', alpha=0.5, s=15, label='Candidates')
        
        # Selected towers
        plt.scatter(selected_df['x'], selected_df['y'], c='red', marker='^', s=100, label='Selected Towers')
        
        plt.legend()
        plt.title('Candidate vs Selected Cell Towers Location')
        plt.axis('equal')
        plt.savefig(f"{self.output_dir}/sites_map.png", dpi=300)
        plt.close()

    def generate_html_map(self, selected_towers, grid_gdf, boundary_geom):
        """Generate interactive Folium map."""
        # Convert boundary coordinate
        boundary_gdf = gpd.GeoSeries([boundary_geom], crs=self.config['project']['crs']).to_crs("EPSG:4326")

        # Compute coverage radius from COST-231 Hata propagation model
        coverage_radius_m = _coverage_radius_m(self.config)
        print(f"  Coverage radius (COST-231, RSRP >= {self.config.get('thresholds', {}).get('rsrp_min_dbm', -110)} dBm): "
              f"{coverage_radius_m:.0f} m")

        # Center of Map
        m = folium.Map(location=[boundary_gdf[0].centroid.y, boundary_gdf[0].centroid.x], zoom_start=11)

        # Plot Nagpur boundary
        folium.GeoJson(
            boundary_gdf.__geo_interface__,
            style_function=lambda x: {'color': 'blue', 'fillOpacity': 0.05, 'weight': 2},
            name="Target Boundary"
        ).add_to(m)

        # We need WGS84 coordinates for folium
        selected_gdf = gpd.GeoDataFrame(
            selected_towers,
            geometry=gpd.points_from_xy(selected_towers['x'], selected_towers['y']),
            crs=self.config['project']['crs']
        )
        selected_wgs = selected_gdf.to_crs("EPSG:4326")

        for i, row in selected_wgs.iterrows():
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=(
                    f"Tower ID: {i}<br>"
                    f"Traffic: {row.get('predicted_traffic_mbps', 0):.2f} Mbps<br>"
                    f"Coverage radius: {coverage_radius_m:.0f} m (COST-231 Hata)"
                ),
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)

            # Coverage area derived from the COST-231 Hata propagation model
            folium.Circle(
                radius=coverage_radius_m,
                location=[row.geometry.y, row.geometry.x],
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.1
            ).add_to(m)

        m.save(f"{self.output_dir}/interactive_towers.html")
