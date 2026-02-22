import numpy as np
import pandas as pd
from shapely.geometry import Point

class CandidateSiteGenerator:
    def __init__(self, config):
        self.config = config
        self.min_dist = self.config['optimization']['min_tower_distance']

    def generate_candidates(self, grid_df, num_candidates=500):
        """Filter grid points to find feasible candidate sites for towers."""
        # Rules:
        # 1. High demand: Top 40% of traffic
        # 2. Near roads: Road density > 0
        # 3. Valid land use/building density: avoid > 0.8 (too dense to build)
        # 4. Filter by minimum distance

        threshold = grid_df['predicted_traffic_mbps'].quantile(0.6)
        feasible = grid_df[
            (grid_df['predicted_traffic_mbps'] > threshold) &
            (grid_df['road_density'] > 0) &
            (grid_df['building_density'] < 0.8)
        ].copy()

        # Sort by demand to prioritize high traffic areas
        feasible = feasible.sort_values(by='predicted_traffic_mbps', ascending=False)
        
        candidates = []
        for _, row in feasible.iterrows():
            pt = Point(row['x'], row['y'])
            # Check min distance against already selected candidates
            too_close = False
            for c in candidates:
                if pt.distance(Point(c['x'], c['y'])) < self.min_dist:
                    too_close = True
                    break
            
            if not too_close:
                candidates.append(row)
            
            if len(candidates) >= num_candidates:
                break

        # If we didn't get enough candidates, relax constraints (fallback)
        if len(candidates) < num_candidates:
            print(f"Warning: Only found {len(candidates)} candidates following rules. Relaxing constraints.")
            fallback = grid_df.sort_values(by='predicted_traffic_mbps', ascending=False)
            for _, row in fallback.iterrows():
                pt = Point(row['x'], row['y'])
                too_close = False
                for c in candidates:
                    if pt.distance(Point(c['x'], c['y'])) < (self.min_dist / 2): # Relaxed distance
                        too_close = True
                        break
                
                if not too_close and row['grid_id'] not in [c['grid_id'] for c in candidates]:
                    candidates.append(row)

                if len(candidates) >= num_candidates:
                    break

        return pd.DataFrame(candidates).reset_index(drop=True)
