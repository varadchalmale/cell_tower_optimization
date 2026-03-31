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

    def generate_html_map(self, all_towers_dict, grid_gdf, boundary_geom,
                          airtel_poly=None, coverage_stats=None):
        """
        Generate interactive multi-tier Folium map.

        Parameters
        ----------
        all_towers_dict : dict  {'macro': df, 'micro': df, 'small_cell': df}
                          OR a plain DataFrame (backward-compatible — treated as macro)
        grid_gdf        : planning grid DataFrame (unused for display, kept for signature compat)
        boundary_geom   : shapely Polygon of district boundary (projected CRS)
        airtel_poly     : optional shapely Polygon of Airtel coverage (projected CRS)
        coverage_stats  : optional dict from MultiTierOptimizer.compute_multi_tier_coverage
        """
        import pandas as pd
        from src.models.propagation import PropagationModel

        # Backward compatibility: plain DataFrame → wrap as macro-only dict
        if isinstance(all_towers_dict, pd.DataFrame):
            all_towers_dict = {'macro': all_towers_dict}

        crs = self.config['project']['crs']
        freq_mhz = self.config['rf_params']['frequency_mhz']
        rsrp_min = self.config.get('thresholds', {}).get('rsrp_min_dbm', -95)
        tower_specs = self.config.get('tower_types', {})

        # Per-tier visual config
        TIER_CFG = {
            'macro': {
                'color':        'red',
                'circle_color': '#cc0000',
                'fill_opacity': 0.07,
                'layer_name':   'Macro Towers (35 m · 3,454 m radius)',
                'marker_html':  '<div style="color:red;font-size:20px;font-weight:bold;'
                                'text-shadow:1px 1px 2px #000;">▲</div>',
                'popup_extra':  'Type: Ground-Based Tower (GBT/GBM)<br>Height: 35 m<br>EIRP: 62 dBm',
            },
            'micro': {
                'color':        'orange',
                'circle_color': '#ff8800',
                'fill_opacity': 0.12,
                'layer_name':   'Micro Cells (12 m · 600 m radius)',
                'marker_html':  '<div style="color:orange;font-size:14px;font-weight:bold;'
                                'text-shadow:1px 1px 2px #000;">●</div>',
                'popup_extra':  'Type: Street-Level Micro Cell<br>Height: 12 m<br>EIRP: 41 dBm',
            },
            'small_cell': {
                'color':        '#ddaa00',
                'circle_color': '#ddaa00',
                'fill_opacity': 0.18,
                'layer_name':   'Small Cells (6 m · 150 m radius)',
                'marker_html':  '<div style="color:#ddaa00;font-size:10px;font-weight:bold;'
                                'text-shadow:1px 1px 2px #000;">◆</div>',
                'popup_extra':  'Type: Lamppost / Rooftop Small Cell<br>Height: 6 m<br>EIRP: 28 dBm',
            },
        }

        # Map centre from boundary
        boundary_gdf = gpd.GeoSeries([boundary_geom], crs=crs).to_crs("EPSG:4326")
        centre = [boundary_gdf[0].centroid.y, boundary_gdf[0].centroid.x]
        m = folium.Map(location=centre, zoom_start=9, tiles='CartoDB positron')

        # ── District boundary layer ────────────────────────────────────────
        folium.GeoJson(
            boundary_gdf.__geo_interface__,
            style_function=lambda x: {
                'color': '#003366', 'fillOpacity': 0.03, 'weight': 2.5
            },
            name='Nagpur District Boundary',
        ).add_to(m)

        # ── Airtel coverage overlay (dashed blue) ──────────────────────────
        if airtel_poly is not None:
            try:
                airtel_gdf = gpd.GeoSeries([airtel_poly], crs=crs).to_crs("EPSG:4326")
                folium.GeoJson(
                    airtel_gdf.__geo_interface__,
                    style_function=lambda x: {
                        'color': '#1E90FF', 'fillColor': '#1E90FF',
                        'fillOpacity': 0.12, 'weight': 2.5, 'dashArray': '8,4'
                    },
                    name='Airtel Coverage (Real-World Reference)',
                    tooltip='Airtel real-world coverage area',
                ).add_to(m)
            except Exception as e:
                print(f"  Warning: could not render Airtel polygon ({e})")

        # ── Per-tier tower layers ──────────────────────────────────────────
        for tier in ['macro', 'micro', 'small_cell']:
            towers_df = all_towers_dict.get(tier, None)
            if towers_df is None or len(towers_df) == 0:
                continue

            cfg  = TIER_CFG.get(tier, TIER_CFG['macro'])
            spec = tower_specs.get(tier, {})
            r_m  = PropagationModel.coverage_radius_m(spec, freq_mhz, rsrp_min) if spec else 3454.0

            towers_gdf = gpd.GeoDataFrame(
                towers_df,
                geometry=gpd.points_from_xy(towers_df['x'], towers_df['y']),
                crs=crs
            ).to_crs("EPSG:4326")

            marker_group  = folium.FeatureGroup(name=cfg['layer_name'],          show=True)
            coverage_group = folium.FeatureGroup(name=cfg['layer_name'] + ' — Coverage Circles',
                                                  show=(tier == 'macro'))

            for i, row in towers_gdf.iterrows():
                lat, lon = row.geometry.y, row.geometry.x
                demand = row.get('predicted_traffic_mbps', row.get('demand_served', 0))
                popup_html = (
                    f"<b>{cfg['popup_extra'].split('<br>')[0].replace('Type: ','')}</b><br>"
                    f"{cfg['popup_extra'].replace(cfg['popup_extra'].split('<br>')[0]+'<br>','')}<br>"
                    f"Coverage radius: <b>{r_m:.0f} m</b> (COST-231 Hata)<br>"
                    f"Demand served: <b>{float(demand):.1f} Mbps</b><br>"
                    f"RSRP threshold: {rsrp_min} dBm"
                )
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_html, max_width=260),
                    icon=folium.DivIcon(
                        icon_size=(24, 24),
                        icon_anchor=(12, 12),
                        html=cfg['marker_html'],
                    ),
                ).add_to(marker_group)

                folium.Circle(
                    radius=r_m,
                    location=[lat, lon],
                    color=cfg['circle_color'],
                    fill=True,
                    fill_color=cfg['circle_color'],
                    fill_opacity=cfg['fill_opacity'],
                    weight=1,
                ).add_to(coverage_group)

            marker_group.add_to(m)
            coverage_group.add_to(m)

        # ── Comparison legend (top-right corner) ──────────────────────────
        if coverage_stats:
            agg = coverage_stats['aggregate']
            cmp = coverage_stats['comparison']
            per = coverage_stats['per_tier']

            macro_n = per.get('macro',      {}).get('n_towers', 0)
            micro_n = per.get('micro',      {}).get('n_towers', 0)
            small_n = per.get('small_cell', {}).get('n_towers', 0)

            legend_html = f"""
            <div style="position:fixed;top:12px;right:12px;z-index:9999;
                        background:white;border:2px solid #555;border-radius:8px;
                        padding:12px 16px;font-family:Arial,sans-serif;font-size:12px;
                        box-shadow:3px 3px 8px rgba(0,0,0,0.25);min-width:230px;">
              <b style="font-size:13px;">AI vs Airtel — Tower Count</b>
              <hr style="margin:6px 0;">
              <table style="width:100%;border-collapse:collapse;">
                <tr><td style="color:#888;">Tier</td>
                    <td style="color:#888;text-align:right;">AI</td>
                    <td style="color:#888;text-align:right;">Airtel</td></tr>
                <tr><td><span style="color:red;">▲</span> Macro</td>
                    <td style="text-align:right;"><b>{macro_n}</b></td>
                    <td style="text-align:right;color:#888;">~1,200</td></tr>
                <tr><td><span style="color:orange;">●</span> Micro</td>
                    <td style="text-align:right;"><b>{micro_n}</b></td>
                    <td style="text-align:right;color:#888;">—</td></tr>
                <tr><td><span style="color:#ddaa00;">◆</span> Small Cell</td>
                    <td style="text-align:right;"><b>{small_n}</b></td>
                    <td style="text-align:right;color:#888;">—</td></tr>
                <tr style="border-top:1px solid #ccc;">
                  <td><b>Total</b></td>
                  <td style="text-align:right;"><b>{agg['n_towers_total']}</b></td>
                  <td style="text-align:right;color:#888;"><b>~{cmp['airtel_total']:,}</b></td>
                </tr>
              </table>
              <hr style="margin:6px 0;">
              <div style="color:#006600;font-weight:bold;">
                ✓ {cmp['reduction_pct']:.0f}% fewer towers than Airtel
              </div>
              <div style="margin-top:4px;">
                Pop. coverage: <b>{agg['pop_pct']:.1f}%</b> &nbsp;|&nbsp;
                Area: <b>{agg['area_pct']:.1f}%</b>
              </div>
              <hr style="margin:6px 0;">
              <div style="color:#555;font-size:10px;">
                RF: COST-231 Hata · RSRP ≥ {rsrp_min} dBm<br>
                <span style="color:#1E90FF;">━ ━</span> Airtel real coverage reference
              </div>
            </div>"""
            m.get_root().html.add_child(folium.Element(legend_html))

        folium.LayerControl(collapsed=False).add_to(m)
        out_path = f"{self.output_dir}/interactive_towers.html"
        m.save(out_path)
        print(f"  Interactive map saved → {out_path}")
