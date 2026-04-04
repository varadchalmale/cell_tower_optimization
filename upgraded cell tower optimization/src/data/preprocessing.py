"""
Data preprocessing for Nagpur district cell tower optimisation.

Generates a planning grid over the full GADM district boundary (~9,928 km²)
and computes per-grid-point features: population (17-node multi-Gaussian model
calibrated to Census 2011 taluka populations), road/building density proxies,
elevation, and urban classification.
"""

import os
import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
from shapely.geometry import Point
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 17-node multi-Gaussian population model for Nagpur district
# Each node = (lat, lon, sigma_m, peak_pop_per_cell)
# Calibrated so total district population ~ 46.5 lakh (Census 2011)
# ---------------------------------------------------------------------------
POPULATION_CENTRES = [
    # ---- Nagpur city core & wards ----
    {"name": "Nagpur_Core",      "lat": 21.1458, "lon": 79.0882, "sigma_m": 4000,  "peak": 5000},
    {"name": "Civil_Lines",      "lat": 21.1545, "lon": 79.0633, "sigma_m": 3000,  "peak": 4000},
    {"name": "Manewada_Besa",    "lat": 21.1100, "lon": 79.1200, "sigma_m": 3500,  "peak": 3500},
    # ---- Peri-urban / industrial ----
    {"name": "Butibori",         "lat": 21.0100, "lon": 79.0500, "sigma_m": 2500,  "peak": 1500},
    {"name": "Hingna",           "lat": 21.1000, "lon": 78.9800, "sigma_m": 2500,  "peak": 1800},
    # ---- Taluka towns ----
    {"name": "Kamptee",          "lat": 21.2300, "lon": 79.1900, "sigma_m": 2000,  "peak": 2500},
    {"name": "Ramtek",           "lat": 21.3900, "lon": 79.3200, "sigma_m": 1500,  "peak": 1200},
    {"name": "Savner",           "lat": 21.2700, "lon": 78.7700, "sigma_m": 1500,  "peak": 800},
    {"name": "Kalmeshwar",       "lat": 21.2400, "lon": 78.9200, "sigma_m": 1500,  "peak": 900},
    {"name": "Katol",            "lat": 21.2700, "lon": 78.5900, "sigma_m": 1500,  "peak": 1000},
    {"name": "Narkhed",          "lat": 21.4400, "lon": 78.4900, "sigma_m": 1200,  "peak": 600},
    {"name": "Parseoni",         "lat": 21.3800, "lon": 79.1200, "sigma_m": 1200,  "peak": 700},
    {"name": "Mouda",            "lat": 21.2900, "lon": 79.3700, "sigma_m": 1200,  "peak": 600},
    {"name": "Umred",            "lat": 20.8500, "lon": 79.3200, "sigma_m": 1500,  "peak": 1100},
    {"name": "Bhiwapur",         "lat": 20.7200, "lon": 79.2900, "sigma_m": 1000,  "peak": 500},
    {"name": "Kuhi",             "lat": 20.9800, "lon": 79.1900, "sigma_m": 1000,  "peak": 600},
    {"name": "Nagpur_South",     "lat": 21.1200, "lon": 79.0500, "sigma_m": 3000,  "peak": 3800},
]

RURAL_FLOOR = 20  # minimum population per grid cell (sparse habitation everywhere)


class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.crs = config["project"]["crs"]
        self.grid_res = config["optimization"]["grid_resolution"]
        # Pre-convert population centres to UTM once
        self._pop_centres_utm = self._convert_centres_to_utm()

    # ------------------------------------------------------------------
    # Boundary
    # ------------------------------------------------------------------
    def create_nagpur_boundary(self):
        """Load Nagpur district polygon from GADM shapefile."""
        path = self.config["paths"]["boundary_shp"]
        if os.path.exists(path):
            gdf = gpd.read_file(path)
            if "NAME_2" in gdf.columns:
                nagpur = gdf[gdf["NAME_2"] == "Nagpur"]
            else:
                nagpur = gdf.head(1)
            if nagpur.empty:
                raise RuntimeError(f"'Nagpur' not found in {path}")
            nagpur = nagpur.to_crs(self.crs)
            area_km2 = nagpur.geometry.unary_union.area / 1e6
            print(f"  Loaded GADM Nagpur district boundary ({area_km2:.0f} km²)")
            return nagpur.geometry.unary_union
        else:
            # Fallback: fetch from OSM (returns city-level — smaller)
            print("  GADM boundary not found. Fetching Nagpur boundary from OSM...")
            try:
                import osmnx as ox
                nagpur = ox.geocode_to_gdf("Nagpur District, Maharashtra, India")
                nagpur = nagpur.to_crs(self.crs)
                area_km2 = nagpur.geometry.unary_union.area / 1e6
                print(f"  OSM boundary loaded ({area_km2:.0f} km²)")
                return nagpur.geometry.unary_union
            except Exception as e:
                print(f"  OSM fetch failed: {e}. Using 50 km radius fallback.")
                lat, lon = 21.1458, 79.0882
                centre = gpd.GeoDataFrame(
                    geometry=[Point(lon, lat)], crs="EPSG:4326"
                ).to_crs(self.crs)
                return centre.geometry[0].buffer(50000)

    # ------------------------------------------------------------------
    # Grid generation
    # ------------------------------------------------------------------
    def generate_planning_grid(self, boundary_geom):
        """Generate equally-spaced grid points inside the district boundary."""
        minx, miny, maxx, maxy = boundary_geom.bounds
        xs = np.arange(minx, maxx, self.grid_res)
        ys = np.arange(miny, maxy, self.grid_res)

        # Vectorised point-in-polygon using prepared geometry
        from shapely.prepared import prep
        prepared = prep(boundary_geom)

        points = []
        for x in xs:
            for y in ys:
                pt = Point(x, y)
                if prepared.contains(pt):
                    points.append(pt)

        grid = gpd.GeoDataFrame(geometry=points, crs=self.crs)
        grid["grid_id"] = np.arange(len(grid))
        return grid

    # ------------------------------------------------------------------
    # OSM features (best-effort; fast fallback for large areas)
    # ------------------------------------------------------------------
    def extract_osm_features(self, boundary_geom):
        """Try to extract OSM roads/buildings; return empty GDFs on failure."""
        area_km2 = boundary_geom.area / 1e6
        buildings = gpd.GeoDataFrame(geometry=[], crs=self.crs)
        roads = gpd.GeoDataFrame(geometry=[], crs=self.crs)
        landuse = gpd.GeoDataFrame(geometry=[], crs=self.crs)

        if area_km2 > 500:
            print(f"  District area is {area_km2:.0f} km² — skipping full OSM download "
                  f"(would be too slow). Using synthetic feature proxies.")
            return buildings, roads, landuse

        print("  Extracting OSM data...")
        boundary_wgs = gpd.GeoSeries([boundary_geom], crs=self.crs).to_crs("EPSG:4326")[0]
        try:
            import osmnx as ox
            tags_bldg = {"building": True}
            buildings = ox.features_from_polygon(boundary_wgs, tags_bldg)
            if not buildings.empty:
                buildings = buildings.to_crs(self.crs)
        except Exception:
            pass
        try:
            import osmnx as ox
            graph = ox.graph_from_polygon(boundary_wgs, network_type="drive")
            _, roads = ox.graph_to_gdfs(graph)
            roads = roads.to_crs(self.crs)
        except Exception:
            pass
        return buildings, roads, landuse

    # ------------------------------------------------------------------
    # Grid feature computation
    # ------------------------------------------------------------------
    def compute_grid_features(self, grid_gdf, buildings, roads, landuse,
                              pop_raster_path, dem_raster_path):
        """Compute per-grid-point features for demand model and propagation."""
        xs = np.array([pt.x for pt in grid_gdf.geometry])
        ys = np.array([pt.y for pt in grid_gdf.geometry])
        n = len(grid_gdf)

        # --- Population: 17-node multi-Gaussian ---
        pop_vals = self._compute_population(xs, ys)

        # --- Elevation: synthetic terrain ---
        elevations = 300 + 50 * np.sin(xs / 5000) + 30 * np.cos(ys / 7000)

        # --- Urban classification & synthetic densities ---
        # Assign based on distance to nearest population centre and pop value
        urban_class, road_density, building_density = self._compute_urban_features(
            xs, ys, pop_vals
        )

        # Override with real OSM data if available
        if not roads.empty or not buildings.empty:
            road_density, building_density, urban_class = self._compute_osm_features(
                grid_gdf, buildings, roads, xs, ys, n
            )

        feat_df = pd.DataFrame({
            "grid_id": grid_gdf["grid_id"].values,
            "x": xs,
            "y": ys,
            "population": pop_vals,
            "elevation": elevations,
            "road_density": road_density,
            "building_density": building_density,
            "urban_class": urban_class,
        })
        merged = grid_gdf.merge(feat_df, on="grid_id")
        return merged

    # ------------------------------------------------------------------
    # Population model internals
    # ------------------------------------------------------------------
    def _convert_centres_to_utm(self):
        """Convert lat/lon population centres to UTM coordinates."""
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", self.crs, always_xy=True
        )
        centres = []
        for c in POPULATION_CENTRES:
            ux, uy = transformer.transform(c["lon"], c["lat"])
            centres.append({
                "name": c["name"],
                "x": ux,
                "y": uy,
                "sigma": c["sigma_m"],
                "peak": c["peak"],
            })
        return centres

    def _compute_population(self, xs, ys):
        """17-node multi-Gaussian population density model."""
        pop = np.full(len(xs), float(RURAL_FLOOR))
        for c in self._pop_centres_utm:
            dx = xs - c["x"]
            dy = ys - c["y"]
            r2 = dx ** 2 + dy ** 2
            pop += c["peak"] * np.exp(-r2 / (2 * c["sigma"] ** 2))
        return np.maximum(pop, RURAL_FLOOR)

    def _compute_urban_features(self, xs, ys, pop_vals):
        """Derive urban class and synthetic road/building density from population."""
        n = len(xs)
        urban_class = np.full(n, "rural", dtype=object)
        road_density = np.zeros(n)
        building_density = np.zeros(n)

        # Distance to nearest population centre
        min_dist = np.full(n, 1e9)
        nearest_peak = np.zeros(n)
        for c in self._pop_centres_utm:
            d = np.sqrt((xs - c["x"]) ** 2 + (ys - c["y"]) ** 2)
            closer = d < min_dist
            min_dist[closer] = d[closer]
            nearest_peak[closer] = c["peak"]

        # Urban: within 3km of major centre (peak > 2000) or 1.5km of town
        major_mask = (min_dist < 3000) & (nearest_peak > 2000)
        town_mask = (min_dist < 1500) & (nearest_peak > 500) & ~major_mask
        suburban_mask = (
            ((min_dist < 6000) & (nearest_peak > 2000)) |
            ((min_dist < 3000) & (nearest_peak > 500))
        ) & ~major_mask & ~town_mask

        urban_class[major_mask | town_mask] = "urban"
        urban_class[suburban_mask] = "suburban"

        # Synthetic road/building density proportional to population
        pop_norm = pop_vals / (pop_vals.max() + 1e-9)
        road_density = pop_norm * 500 + np.random.RandomState(42).uniform(0, 50, n)
        building_density = np.clip(pop_norm * 0.4, 0, 0.8)

        return urban_class, road_density, building_density

    def _compute_osm_features(self, grid_gdf, buildings, roads, xs, ys, n):
        """Compute features from real OSM data (used when area < 500 km²)."""
        buffers = grid_gdf.geometry.buffer(200)
        road_density = np.zeros(n)
        building_density = np.zeros(n)
        urban_class = np.full(n, "rural", dtype=object)

        b_sindex = buildings.sindex if not buildings.empty else None
        r_sindex = roads.sindex if not roads.empty else None

        for i, buf in enumerate(buffers):
            if r_sindex is not None:
                hits = list(r_sindex.intersection(buf.bounds))
                if hits:
                    matched = roads.iloc[hits]
                    precise = matched[matched.intersects(buf)]
                    for line in precise.geometry:
                        road_density[i] += line.intersection(buf).length

            if b_sindex is not None:
                hits = list(b_sindex.intersection(buf.bounds))
                if hits:
                    matched = buildings.iloc[hits]
                    precise = matched[matched.intersects(buf)]
                    for poly in precise.geometry:
                        building_density[i] += poly.intersection(buf).area

        building_density /= np.maximum(1, buffers.area.values)
        urban_class[building_density > 0.15] = "urban"
        urban_class[(building_density > 0.02) & (building_density <= 0.15)] = "suburban"
        return road_density, building_density, urban_class
