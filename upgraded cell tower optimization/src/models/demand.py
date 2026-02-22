import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class DemandModel:
    def __init__(self, config):
        self.config = config
        self.model = RandomForestRegressor(n_estimators=100, random_state=self.config['project']['seed'])
        self.scaler = StandardScaler()
        self.pop_per_user = self.config['optimization']['population_per_user']
        self.traffic_per_user = self.config['optimization']['traffic_per_user_mbps']
        self.kmeans = None

    def preprocess_features(self, df):
        """Prepare dataframe for ML."""
        # Convert urban_class categorical to one-hot or numerical
        df_ml = df.copy()
        df_ml['urban_numeric'] = df_ml['urban_class'].map({'rural': 0, 'suburban': 1, 'urban': 2})
        
        X = df_ml[['population', 'road_density', 'building_density', 'urban_numeric']].values
        return X

    def generate_synthetic_labels(self, df):
        """Generate synthetic traffic demand labels if real ones aren't available."""
        # Realistic traffic roughly scales with users + density
        active_users = df['population'] / self.pop_per_user
        base_traffic = active_users * self.traffic_per_user
        
        # Roads and buildings add to traffic due to transient users / indoor data usage
        transient_multiplier = 1 + (df['road_density'] / (df['road_density'].max() + 1e-9)) * 0.5
        indoor_multiplier = 1 + (df['building_density'] / (df['building_density'].max() + 1e-9)) * 1.5
        
        traffic = base_traffic * transient_multiplier * indoor_multiplier
        # Add some noise
        noise = np.random.normal(0, traffic.mean() * 0.1, len(traffic))
        return np.maximum(0, traffic + noise)

    def train_ml_model(self, df):
        """Train the demand forecasting AI model."""
        X = self.preprocess_features(df)
        # Using synthetic labels for training if real labels are absent
        y = self.generate_synthetic_labels(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=self.config['project']['seed'])
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Demand Model trained successfully. RMSE: {rmse:.2f} Mbps")

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
