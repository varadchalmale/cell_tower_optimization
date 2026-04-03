"""
Visualisation module for cell tower optimisation.

Generates:
  - Traffic demand heatmap
  - Pareto frontier plot
  - Candidate vs selected towers map
  - Interactive Folium HTML map with actual coverage radii
  - Sensitivity analysis plot (coverage vs tower count)
  - Baseline comparison bar chart
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import folium
import numpy as np
from src.models.propagation import compute_coverage_radius_km



def _coverage_radius_m(config):
    """
    Compute the maximum coverage radius (metres) at which RSRP equals the
    minimum usable threshold, using the COST-231 Hata model.

    Derivation:
        RSRP = Tx_power - PathLoss
        PathLoss = Tx_power - RSRP_min   (at edge-of-coverage)
        COST-231: L = A + B * log10(d)  →  d = 10^((L - A) / B)
    """
    rf = config['rf_params']
    f_mhz = rf['frequency_mhz']
    h_te  = rf['antenna_height_m']
    h_re  = rf.get('receiver_height_m', 1.5)
    tx_dbm = rf['transmit_power_dbm']
    rsrp_min_dbm = config.get('thresholds', {}).get('rsrp_min_dbm', -110)

    max_loss = tx_dbm - rsrp_min_dbm  # dB

    # a(h_re) correction
    a_hre = (1.1 * np.log10(f_mhz) - 0.7) * h_re - (1.56 * np.log10(f_mhz) - 0.8)
    # Cm environment correction (urban = 3 dB)
    cm = 3.0
    # Intercept and slope of COST-231 Hata
    intercept = 46.3 + 33.9 * np.log10(f_mhz) - 13.82 * np.log10(h_te) - a_hre + cm
    slope = 44.9 - 6.55 * np.log10(h_te)

    log10_d = (max_loss - intercept) / slope
    d_km = 10 ** log10_d
    # Clamp to sensible urban range (0.2 km – 5 km)
    d_km = float(np.clip(d_km, 0.2, 5.0))
    return d_km * 1000.0  # metres

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.output_dir = config["paths"]["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def plot_heatmap(self, grid_df, col, title, cmap="hot"):
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(grid_df["x"], grid_df["y"], c=grid_df[col],
                         cmap=cmap, s=3, edgecolors="none")
        plt.colorbar(sc, label=col)
        plt.title(title)
        plt.xlabel("Easting (m)")
        plt.ylabel("Northing (m)")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{col}_heatmap.png", dpi=200)
        plt.close()

    # ------------------------------------------------------------------
    def plot_pareto_front(self, res):
        plt.figure(figsize=(10, 8))
        F = res.F
        obj1 = -F[:, 0]   # coverage fraction (maximised)
        obj2 = -F[:, 1]   # capacity (maximised)
        obj3 = F[:, 2]    # cost (minimised)
        sc = plt.scatter(obj1 * 100, obj2, c=obj3, cmap="viridis", s=60,
                         edgecolors="k", linewidths=0.3)
        plt.colorbar(sc, label="Deployment Cost Index")
        plt.xlabel("Population Coverage (%)")
        plt.ylabel("Network Capacity (Sum Throughput Mbps)")
        plt.title("NSGA-II Pareto Frontier")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pareto_frontier.png", dpi=200)
        plt.close()

    # ------------------------------------------------------------------
    def plot_candidates_and_selected(self, candidates_df, selected_df,
                                     background_df, col_bg="predicted_traffic_mbps"):
        plt.figure(figsize=(12, 10))
        plt.scatter(background_df["x"], background_df["y"],
                    c=background_df[col_bg], cmap="Blues", alpha=0.4, s=2)
        plt.colorbar(label="Traffic Demand (Mbps)")
        plt.scatter(candidates_df["x"], candidates_df["y"],
                    c="gray", alpha=0.3, s=8, label="Candidates")

        # Separate macro vs gap-fill
        if "tower_source" in selected_df.columns:
            macro = selected_df[selected_df["tower_source"] == "nsga2_macro"]
            gap = selected_df[selected_df["tower_source"] == "gap_fill"]
            plt.scatter(macro["x"], macro["y"], c="red", marker="^",
                        s=90, label=f"NSGA-II Macro ({len(macro)})", zorder=5)
            if len(gap) > 0:
                plt.scatter(gap["x"], gap["y"], c="blue", marker="s",
                            s=50, label=f"Gap-fill ({len(gap)})", zorder=5)
        else:
            plt.scatter(selected_df["x"], selected_df["y"], c="red",
                        marker="^", s=90, label="Selected Towers", zorder=5)

        plt.legend(fontsize=9)
        plt.title("Candidate vs Selected Cell Tower Locations")
        plt.xlabel("Easting (m)")
        plt.ylabel("Northing (m)")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/sites_map.png", dpi=200)
        plt.close()

    # ------------------------------------------------------------------
    def generate_html_map(self, selected_towers, grid_gdf, boundary_geom):
        """Interactive Folium map with actual coverage radii and tower-type colours."""
        boundary_wgs = gpd.GeoSeries([boundary_geom],
                                     crs=self.config["project"]["crs"]).to_crs("EPSG:4326")
        centre = boundary_wgs[0].centroid
        m = folium.Map(location=[centre.y, centre.x], zoom_start=10)

        # District boundary
        folium.GeoJson(
            boundary_wgs.__geo_interface__,
            style_function=lambda x: {"color": "blue", "fillOpacity": 0.03, "weight": 2},
            name="Nagpur District Boundary",
        ).add_to(m)

        # Compute actual coverage radii per environment
        r_urban = compute_coverage_radius_km(self.config, "urban") * 1000
        r_suburban = compute_coverage_radius_km(self.config, "suburban") * 1000
        r_rural = compute_coverage_radius_km(self.config, "rural") * 1000

        # Convert towers to WGS84
        towers_gdf = gpd.GeoDataFrame(
            selected_towers,
            geometry=gpd.points_from_xy(selected_towers["x"], selected_towers["y"]),
            crs=self.config["project"]["crs"],
        ).to_crs("EPSG:4326")

        # Feature groups for layer control
        macro_fg = folium.FeatureGroup(name="NSGA-II Macro Towers")
        gap_fg = folium.FeatureGroup(name="Gap-fill Towers")
        cov_fg = folium.FeatureGroup(name="Coverage Circles", show=True)

        for i, row in towers_gdf.iterrows():
            loc = [row.geometry.y, row.geometry.x]
            source = row.get("tower_source", "nsga2_macro")
            is_macro = source == "nsga2_macro"

            colour = "red" if is_macro else "blue"
            icon = folium.Icon(color=colour, icon="tower-broadcast", prefix="fa")
            popup = (f"<b>{'Macro' if is_macro else 'Gap-fill'} Tower</b><br>"
                     f"Traffic: {row.get('predicted_traffic_mbps', 0):.1f} Mbps<br>"
                     f"Pop: {row.get('population', 0):.0f}")

            marker = folium.Marker(location=loc, popup=popup, icon=icon)
            (macro_fg if is_macro else gap_fg).add_child(marker)

            # Coverage circle with actual radius
            env = row.get("urban_class", "rural")
            radius = {"urban": r_urban, "suburban": r_suburban}.get(env, r_rural)
            circle = folium.Circle(
                location=loc, radius=radius,
                color=colour, fill=True, fill_color=colour, fill_opacity=0.06,
                weight=1,
            )
            cov_fg.add_child(circle)

        macro_fg.add_to(m)
        gap_fg.add_to(m)
        cov_fg.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        m.save(f"{self.output_dir}/interactive_towers.html")

    # ------------------------------------------------------------------
    def plot_sensitivity(self, tower_counts, coverage_pcts):
        """Coverage % vs tower count curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(tower_counts, coverage_pcts, "o-", color="darkblue", linewidth=2)
        plt.axhline(100, color="green", linestyle="--", alpha=0.5, label="100% target")
        plt.xlabel("Number of Towers")
        plt.ylabel("Population Coverage (%)")
        plt.title("Sensitivity Analysis: Towers vs Coverage")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/sensitivity_analysis.png", dpi=200)
        plt.close()

    # ------------------------------------------------------------------
    def plot_baseline_comparison(self, results_dict, ai_towers, ai_coverage):
        """Bar chart comparing AI vs baselines."""
        names = list(results_dict.keys()) + ["NSGA-II + Gap-fill"]
        values = list(results_dict.values()) + [ai_coverage]
        colours = ["#cccccc"] * len(results_dict) + ["#e74c3c"]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, values, color=colours, edgecolor="black", linewidth=0.5)
        plt.axhline(100, color="green", linestyle="--", alpha=0.5)
        plt.ylabel("Population Coverage (%)")
        plt.title(f"Baseline Comparison (all with {ai_towers} towers)")
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", fontsize=9)
        plt.ylim(0, 115)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/baseline_comparison.png", dpi=200)
        plt.close()
