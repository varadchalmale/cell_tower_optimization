"""
Coverage validation module.

Provides:
  - compute_population_coverage(): the PRIMARY metric (population coverage %)
  - generate_simulated_coverage(): polygon for IoU comparison with Airtel
  - validate(): IoU / match-pct against Airtel shapefile
  - run_baselines(): random, hexagonal, K-Means placement baselines
"""

import numpy as np
import geopandas as gpd
import os
from shapely.geometry import Polygon, Point
from src.models.propagation import vectorized_rsrp_matrix, compute_coverage_radius_km

AIRTEL_TOWERS_REFERENCE = 1200


class CoverageValidator:
    def __init__(self, config):
        self.config = config
        self.airtel_shp_path = config["paths"]["airtel_coverage"]
        self.grid_res = config["optimization"]["grid_resolution"]
        self.threshold = config["optimization"].get("coverage_rsrp_threshold_dbm", -110)

    # ------------------------------------------------------------------
    # PRIMARY METRIC
    # ------------------------------------------------------------------
    def compute_population_coverage(self, grid_df, towers_df):
        """Population coverage % using shared env-aware RSRP formula."""
        if len(towers_df) == 0:
            total_pop = float(grid_df["population"].sum())
            return self._empty_metrics(total_pop)

        rsrp = vectorized_rsrp_matrix(grid_df, towers_df, self.config)
        covered = rsrp.max(axis=1) >= self.threshold

        pop = grid_df["population"].values
        total_pop = pop.sum()
        covered_pop = float(np.sum(pop[covered]))
        n = len(towers_df)
        return {
            "population_coverage_pct": round(covered_pop / total_pop * 100, 2) if total_pop > 0 else 0.0,
            "geographic_coverage_pct": round(covered.sum() / len(grid_df) * 100, 2),
            "covered_population": round(covered_pop, 1),
            "total_population": round(float(total_pop), 1),
            "uncovered_population": round(float(total_pop) - covered_pop, 1),
            "covered_grid_points": int(covered.sum()),
            "total_grid_points": len(grid_df),
            "num_towers": n,
            "vs_airtel": f"{n} vs {AIRTEL_TOWERS_REFERENCE} ({n / AIRTEL_TOWERS_REFERENCE * 100:.1f}%)",
        }

    # ------------------------------------------------------------------
    # Baseline comparisons
    # ------------------------------------------------------------------
    def run_baselines(self, grid_df, boundary_geom, n_towers):
        """
        Compute population coverage for 3 naive baselines (same tower count).
        Returns dict of {name: population_coverage_pct}.
        """
        import pandas as pd
        from sklearn.cluster import KMeans

        results = {}

        # --- 1. Random placement ---
        rng = np.random.RandomState(42)
        rand_idx = rng.choice(len(grid_df), size=min(n_towers, len(grid_df)), replace=False)
        rand_towers = grid_df.iloc[rand_idx][["x", "y"]].copy()
        rand_towers = pd.DataFrame(rand_towers)
        rand_towers["urban_class"] = "rural"
        m = self.compute_population_coverage(grid_df, rand_towers)
        results["Random"] = m["population_coverage_pct"]

        # --- 2. Hexagonal grid placement ---
        minx, miny, maxx, maxy = boundary_geom.bounds
        r = compute_coverage_radius_km(self.config, "urban") * 1000 * 1.5
        hex_pts = []
        row_idx = 0
        y = miny
        while y < maxy:
            x = minx + (r * 0.866 if row_idx % 2 else 0)
            while x < maxx:
                pt = Point(x, y)
                if boundary_geom.contains(pt):
                    hex_pts.append({"x": x, "y": y})
                x += r * 1.732
            y += r * 1.5
            row_idx += 1
        if hex_pts:
            hex_df = pd.DataFrame(hex_pts[:n_towers])
            hex_df["urban_class"] = "rural"
            m = self.compute_population_coverage(grid_df, hex_df)
            results["Hexagonal"] = m["population_coverage_pct"]
            results["Hexagonal_towers"] = len(hex_df)
        else:
            results["Hexagonal"] = 0.0

        # --- 3. K-Means centroid placement ---
        coords = grid_df[["x", "y"]].values
        pop_weights = grid_df["population"].values
        pop_weights = pop_weights / pop_weights.sum()
        km = KMeans(n_clusters=n_towers, random_state=42, n_init=5)
        km.fit(coords, sample_weight=pop_weights)
        km_df = pd.DataFrame(km.cluster_centers_, columns=["x", "y"])
        km_df["urban_class"] = "rural"
        m = self.compute_population_coverage(grid_df, km_df)
        results["KMeans"] = m["population_coverage_pct"]

        return results

    # ------------------------------------------------------------------
    # Simulated coverage polygon (for Airtel IoU)
    # ------------------------------------------------------------------
    def generate_simulated_coverage(self, grid_df, towers_df):
        """Build a polygon of covered area for visual/IoU validation."""
        print("  Generating simulated coverage polygon...")
        rsrp = vectorized_rsrp_matrix(grid_df, towers_df, self.config)
        best_rsrp = rsrp.max(axis=1)
        covered_idx = np.where(best_rsrp >= self.threshold)[0]

        if len(covered_idx) == 0:
            return Polygon()

        radius = self.grid_res * 0.75
        covered_pts = grid_df.iloc[covered_idx]
        buffers = [Point(r["x"], r["y"]).buffer(radius) for _, r in covered_pts.iterrows()]
        return gpd.GeoSeries(buffers).unary_union

    def validate(self, simulated_coverage_poly):
        """IoU comparison against Airtel 4G coverage shapefile."""
        if not os.path.exists(self.airtel_shp_path):
            print(f"  Airtel shapefile not found. Using synthetic validation.")
            return self._dummy_validation()
        try:
            airtel = gpd.read_file(self.airtel_shp_path).to_crs(self.config["project"]["crs"])
            airtel_poly = airtel.geometry.unary_union
            inter = simulated_coverage_poly.intersection(airtel_poly).area
            union = simulated_coverage_poly.union(airtel_poly).area
            iou = inter / union if union > 0 else 0
            match = inter / airtel_poly.area if airtel_poly.area > 0 else 0
            fp = simulated_coverage_poly.difference(airtel_poly).area
            miss = airtel_poly.difference(simulated_coverage_poly).area
            return {
                "IoU": round(iou, 4),
                "Match_Percentage": round(match * 100, 2),
                "False_Positive_Pct": round(fp / simulated_coverage_poly.area * 100, 2) if simulated_coverage_poly.area > 0 else 0,
                "Missed_Coverage_Pct": round(miss / airtel_poly.area * 100, 2) if airtel_poly.area > 0 else 0,
            }
        except Exception as e:
            print(f"  Validation error: {e}")
            return self._dummy_validation()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _empty_metrics(self, total_pop):
        return {
            "population_coverage_pct": 0.0, "geographic_coverage_pct": 0.0,
            "covered_population": 0.0, "total_population": total_pop,
            "uncovered_population": total_pop, "covered_grid_points": 0,
            "total_grid_points": 0, "num_towers": 0,
            "vs_airtel": f"0 vs {AIRTEL_TOWERS_REFERENCE}",
        }

    def _dummy_validation(self):
        return {
            "IoU": 0.0, "Match_Percentage": 0.0,
            "False_Positive_Pct": 0.0, "Missed_Coverage_Pct": 100.0,
            "Note": "Airtel data unavailable — synthetic validation",
        }
