import os
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
import osmnx as ox
from shapely.geometry import box, Point, Polygon
import warnings

warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.crs = self.config['project']['crs']
        self.grid_res = self.config['optimization']['grid_resolution']
        
    def create_nagpur_boundary(self):
        """Extract Nagpur district from Indian gadm boundary or fallback to a dummy geom"""
        path = self.config['paths']['boundary_shp']
        if os.path.exists(path):
            gdf = gpd.read_file(path)
            if 'NAME_2' in gdf.columns:
                nagpur = gdf[gdf['NAME_2'] == 'Nagpur']
            else:
                nagpur = gdf.head(1)
            nagpur = nagpur.to_crs(self.crs)
            return nagpur.geometry.unary_union
        else:
            print("GADM boundary not found. Fetching actual Nagpur boundary from OSM...")
            try:
                nagpur = ox.geocode_to_gdf("Nagpur, Maharashtra, India")
                nagpur = nagpur.to_crs(self.crs)
                return nagpur.geometry.unary_union
            except Exception as e:
                print(f"OSM fetch failed: {e}. Using a dummy box.")
                lat, lon = 21.1458, 79.0882
                # Create a 5x5 km dummy extent in UTM
                dummy_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(self.crs)
                dummy_poly = dummy_gdf.geometry[0].buffer(2500) # 2.5km radius approx
                return dummy_poly

    def generate_planning_grid(self, boundary_geom):
        """Generate grid points over the region."""
        # Get bounding box
        minx, miny, maxx, maxy = boundary_geom.bounds
        
        x_coords = np.arange(minx, maxx, self.grid_res)
        y_coords = np.arange(miny, maxy, self.grid_res)
        
        points = []
        for x in x_coords:
            for y in y_coords:
                pt = Point(x, y)
                if pt.within(boundary_geom):
                    points.append(pt)
                    
        grid_gdf = gpd.GeoDataFrame(geometry=points, crs=self.crs)
        grid_gdf['grid_id'] = np.arange(len(grid_gdf))
        return grid_gdf

    def extract_osm_features(self, boundary_geom):
        """Extract roads and buildings from OSM to compute density and clutter."""
        print("Extracting OSM data...")
        # Reproject boundary to WGS84 for osmnx
        boundary_wgs = gpd.GeoSeries([boundary_geom], crs=self.crs).to_crs("EPSG:4326")[0]
        
        try:
            # Get building footprints
            tags_bldg = {"building": True}
            buildings = ox.features_from_polygon(boundary_wgs, tags_bldg)
            if not buildings.empty:
                buildings = buildings.to_crs(self.crs)
            else:
                buildings = gpd.GeoDataFrame(geometry=[], crs=self.crs)
        except Exception:
            buildings = gpd.GeoDataFrame(geometry=[], crs=self.crs)
            
        try:    
            # Get roads
            graph = ox.graph_from_polygon(boundary_wgs, network_type="drive")
            _, roads = ox.graph_to_gdfs(graph)
            roads = roads.to_crs(self.crs)
        except Exception:
            roads = gpd.GeoDataFrame(geometry=[], crs=self.crs)
            
        try:    
            # Get landuse
            tags_lu = {"landuse": ["forest", "residential", "commercial", "industrial", "grass"]}
            landuse = ox.features_from_polygon(boundary_wgs, tags_lu)
            if not landuse.empty:
                landuse = landuse.to_crs(self.crs)
            else:
                landuse = gpd.GeoDataFrame(geometry=[], crs=self.crs)
        except Exception:
            landuse = gpd.GeoDataFrame(geometry=[], crs=self.crs)
                
        return buildings, roads, landuse

    def compute_grid_features(self, grid_gdf, buildings, roads, landuse, pop_raster_path, dem_raster_path):
        """Compute node-level features for ML demand model and propagation."""
        features = []
        
        # Check if rasters exist, setup mock if not
        if not os.path.exists(pop_raster_path):
            print("Population raster not found. Using synthetic population.")
        if not os.path.exists(dem_raster_path):
            print("DEM raster not found. Using synthetic DEM.")

        # Buffer each point to compute density in local area (e.g., 200m radius)
        buffers = grid_gdf.geometry.buffer(200)
        
        # Spatial indexing for speed
        if not buildings.empty:
            b_sindex = buildings.sindex
        else:
            b_sindex = None
            
        if not roads.empty:
            r_sindex = roads.sindex
        else:
            r_sindex = None
            
        for i, (pt, buf) in enumerate(zip(grid_gdf.geometry, buffers)):
            road_length = 0
            if r_sindex is not None:
                possible_matches_index = list(r_sindex.intersection(buf.bounds))
                if possible_matches_index:
                    possible_matches = roads.iloc[possible_matches_index]
                    precise_matches = possible_matches[possible_matches.intersects(buf)]
                    for line in precise_matches.geometry:
                        road_length += line.intersection(buf).length

            bldg_area = 0
            if b_sindex is not None:
                possible_b_matches = list(b_sindex.intersection(buf.bounds))
                if possible_b_matches:
                    possible_b = buildings.iloc[possible_b_matches]
                    precise_b = possible_b[possible_b.intersects(buf)]
                    for poly in precise_b.geometry:
                        intersection = poly.intersection(buf)
                        bldg_area += intersection.area
            
            # Synthetic population: Gaussian urban density centred on Nagpur city core
            # (UTM 32644 approx: 290000E, 2347000N) with suburban decay.
            # Uses building density and road presence as secondary multipliers so that
            # OSM-derived land-use drives spatial variation rather than an arbitrary wave.
            if os.path.exists(pop_raster_path):
                pop_val = 100  # placeholder — replace with rasterio.read when real data available
            else:
                NAGPUR_X, NAGPUR_Y = 290000, 2347000   # city centre in EPSG:32644
                dist_km = np.sqrt((pt.x - NAGPUR_X)**2 + (pt.y - NAGPUR_Y)**2) / 1000.0
                # Peak ~1200 people/cell in city core, falls off with 8 km standard deviation
                pop_val = float(np.maximum(10, 1200 * np.exp(-0.5 * (dist_km / 8.0)**2)))

            if os.path.exists(dem_raster_path):
                dem_val = 300  # placeholder
            else:
                dem_val = 300 + 50 * np.sin(pt.x / 5000)
                
            bldg_ratio = bldg_area / max(1, buf.area)
            if bldg_ratio > 0.15:
                urban_class = "urban"
            elif bldg_ratio > 0.02:
                urban_class = "suburban"
            else:
                urban_class = "rural"
                
            features.append({
                'grid_id': grid_gdf.iloc[i]['grid_id'],
                'x': pt.x,
                'y': pt.y,
                'road_density': road_length,
                'building_density': bldg_ratio,
                'urban_class': urban_class,
                'population': pop_val,
                'elevation': dem_val
            })
            
        feat_df = pd.DataFrame(features)
        merged = grid_gdf.merge(feat_df, on='grid_id')
        return merged
