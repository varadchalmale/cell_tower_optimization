import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os
from src.config import Config

class MLDemandModel:
    def __init__(self):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=Config.RANDOM_SEED))
        ])
        self.model_path = os.path.join(Config.MODELS_PATH, 'demand_model.pkl')

    def prepare_features(self, population, clutter, poi_density=None):
        """
        Convert spatial grids to feature matrix.
        """
        h, w = population.shape
        pop_flat = population.flatten()
        clutter_flat = clutter.flatten()
        
        if poi_density is not None:
            poi_flat = poi_density.flatten()
            X = np.stack([pop_flat, clutter_flat, poi_flat], axis=1)
        else:
            X = np.stack([pop_flat, clutter_flat], axis=1)
            
        return X

    def train(self, population, clutter, target_demand, poi_density=None):
        print("Training ML Demand Prediction Model...")
        X = self.prepare_features(population, clutter, poi_density)
        y = target_demand.flatten()
        
        # Train-test split
        indices = np.where(y > 0)[0] # Focus on non-zero demand for training
        if len(indices) > 50000:
            indices = np.random.choice(indices, 50000, replace=False)
            
        X_train, X_test, y_train, y_test = train_test_split(
            X[indices], y[indices], test_size=0.2, random_state=Config.RANDOM_SEED
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluation
        from sklearn.metrics import mean_squared_error, r2_score
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Training Complete. RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        # Feature Importance
        importances = self.model.named_steps['rf'].feature_importances_
        feature_names = ['population', 'clutter']
        if poi_density is not None:
            feature_names.append('poi_density')
            
        print("\n--- Feature Importance ---")
        for name, imp in zip(feature_names, importances):
            print(f"{name:<15}: {imp:.4f}")
            
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")
        return {'rmse': rmse, 'r2': r2, 'importances': dict(zip(feature_names, importances))}

    def predict(self, population, clutter, poi_density=None):
        if not os.path.exists(self.model_path):
            print("Model not found. Please train first or fallback to engineered score.")
            return None
            
        print("Inference: Predicting spatial demand using ML model...")
        X = self.prepare_features(population, clutter, poi_density)
        self.model = joblib.load(self.model_path)
        prediction = self.model.predict(X)
        return prediction.reshape(population.shape)
