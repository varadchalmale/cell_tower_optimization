import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class DemandModel:
    """
    Demand prediction model for cellular traffic estimation.

    Label strategy (in order of preference):
      1. Real OpenCellID tower density — if an OpenCellID CSV is passed at
         construction time, the training target is the kernel-density of existing
         LTE towers per km². The rationale: operators place infrastructure where
         sustained demand justifies the capital expenditure, so historical tower
         density is a defensible demand proxy. The Random Forest then learns which
         geospatial features (population, road network, building density, land-use
         class) best predict that demand signal.
      2. Infrastructure-weighted population proxy — when no real tower data is
         available, demand is estimated as a function of WorldPop population and
         OSM infrastructure density. Unlike a purely circular formula, this uses
         independently collected population and land-use data as separate evidence
         sources. The result is explicitly labelled as a proxy, not ground truth.
    """

    def __init__(self, config, opencellid_csv=None):
        self.config = config
        self.model = RandomForestRegressor(
            n_estimators=100, random_state=self.config['project']['seed'], n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.pop_per_user = self.config['optimization']['population_per_user']
        self.traffic_per_user = self.config['optimization']['traffic_per_user_mbps']
        self.kmeans = None
        self.opencellid_csv = opencellid_csv
        self.label_source = None  # set during training; reported in output

    def preprocess_features(self, df):
        """Prepare dataframe for ML."""
        df_ml = df.copy()
        df_ml['urban_numeric'] = df_ml['urban_class'].map({'rural': 0, 'suburban': 1, 'urban': 2})
        X = df_ml[['population', 'road_density', 'building_density', 'urban_numeric']].values
        return X

    def load_opencellid_density_labels(self, df):
        """
        Compute per-grid-cell LTE tower density from real OpenCellID data.

        The density (towers per km²) serves as the demand proxy:
        higher density = historically high demand justified infrastructure.

        Returns a Series of density values aligned to df's index, or None if
        loading fails.
        """
        try:
            oc = pd.read_csv(self.opencellid_csv)
            if 'radio' in oc.columns:
                oc = oc[oc['radio'].str.upper().isin(['LTE', '4G', 'NR'])]
            lon_col = 'lon' if 'lon' in oc.columns else 'longitude'
            lat_col = 'lat' if 'lat' in oc.columns else 'latitude'
            if lon_col not in oc.columns or lat_col not in oc.columns:
                print("  Warning: OpenCellID CSV lacks lon/lat columns. Falling back to proxy labels.")
                return None

            crs = self.config['project']['crs']  # projected CRS, e.g. EPSG:32644
            towers_gdf = gpd.GeoDataFrame(
                oc, geometry=gpd.points_from_xy(oc[lon_col], oc[lat_col]), crs="EPSG:4326"
            ).to_crs(crs)

            grid_gdf = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df['x'], df['y']), crs=crs
            )
            grid_res_m = self.config['optimization']['grid_resolution']
            cell_area_km2 = (grid_res_m / 1000.0) ** 2

            # Buffer each grid point into a square cell and count towers within
            grid_gdf['geometry'] = grid_gdf.geometry.buffer(grid_res_m / 2, cap_style=3)
            joined = gpd.sjoin(towers_gdf[['geometry']], grid_gdf[['geometry']], how='right', predicate='within')
            tower_counts = joined.groupby(joined.index_right).size().reindex(grid_gdf.index, fill_value=0)
            density = tower_counts / cell_area_km2

            # Scale to Mbps-equivalent so units are consistent with the proxy path.
            # Linear mapping: max density → max proxy demand.
            max_proxy = (df['population'] / self.pop_per_user * self.traffic_per_user).max()
            if density.max() > 0:
                density_scaled = (density / density.max()) * max_proxy
            else:
                density_scaled = density

            print(f"  OpenCellID labels: {int(tower_counts.sum())} LTE towers loaded, "
                  f"density range [{density.min():.3f}, {density.max():.3f}] towers/km²")
            return density_scaled.values

        except Exception as e:
            print(f"  Warning: Could not load OpenCellID labels ({e}). Falling back to proxy labels.")
            return None

    def generate_proxy_labels(self, df):
        """
        Infrastructure-weighted population proxy for demand estimation.

        NOTE: This is NOT measured traffic data. It is a proxy derived from
        independently collected population counts (WorldPop) and OSM infrastructure
        density. The formula is an approximation of ITU-T E.501 traffic estimation
        guidelines. Treat model outputs as relative demand indices, not absolute Mbps.
        """
        active_users = df['population'] / self.pop_per_user
        base_traffic = active_users * self.traffic_per_user

        # Road density contribution: transient/commuter users (~20% uplift at max density)
        road_norm = df['road_density'] / (df['road_density'].max() + 1e-9)
        # Building density contribution: indoor data usage (~50% uplift at max density)
        bldg_norm = df['building_density'] / (df['building_density'].max() + 1e-9)

        traffic = base_traffic * (1 + road_norm * 0.2) * (1 + bldg_norm * 0.5)
        rng = np.random.default_rng(self.config['project']['seed'])
        noise = rng.normal(0, traffic.mean() * 0.05, len(traffic))
        return np.maximum(0, traffic + noise)

    def train_ml_model(self, df):
        """
        Train the Random Forest demand model.

        Attempts to use real OpenCellID tower-density labels first.
        Falls back to the infrastructure-weighted population proxy if no
        OpenCellID data is available or the file cannot be parsed.
        """
        X = self.preprocess_features(df)

        if self.opencellid_csv:
            y = self.load_opencellid_density_labels(df)
            if y is not None:
                self.label_source = "OpenCellID tower density (real data)"
            else:
                y = self.generate_proxy_labels(df)
                self.label_source = "infrastructure-weighted population proxy (no real traffic data)"
        else:
            y = self.generate_proxy_labels(df)
            self.label_source = "infrastructure-weighted population proxy (no real traffic data)"

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.config['project']['seed']
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        importances = dict(zip(
            ['population', 'road_density', 'building_density', 'urban_class'],
            self.model.feature_importances_
        ))
        print(f"  Label source  : {self.label_source}")
        print(f"  RMSE          : {rmse:.4f}  |  R²: {r2:.4f}")
        print(f"  Feature importance: {', '.join(f'{k}={v:.3f}' for k, v in importances.items())}")

    def predict_traffic(self, df):
        """Predict traffic demand for given features."""
        X = self.preprocess_features(df)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        df_out = df.copy()
        df_out['predicted_traffic_mbps'] = predictions
        return df_out

    def detect_hotspots(self, df, n_clusters=20):
        """Cluster high demand areas to assist with site generation."""
        # Filter high demand (top 20%)
        threshold = df['predicted_traffic_mbps'].quantile(0.8)
        high_demand = df[df['predicted_traffic_mbps'] > threshold].copy()
        
        if len(high_demand) < n_clusters:
            n_clusters = max(1, len(high_demand) // 2)
            
        coords = np.column_stack((high_demand['x'], high_demand['y']))
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.config['project']['seed'])
        labels = self.kmeans.fit_predict(coords)
        
        high_demand['hotspot_id'] = labels
        hotspot_centers = self.kmeans.cluster_centers_
        
        return df, high_demand, hotspot_centers
