"""
3-tier candidate site generation for full-district coverage.

Tier 1 (40%): Urban demand anchors — high traffic, road-adjacent, buildable
Tier 2 (35%): Suburban bridging — medium demand, relaxed density constraint
Tier 3 (25%): Rural coverage — any populated point, sorted by population
"""

import numpy as np
import pandas as pd


class CandidateSiteGenerator:
    def __init__(self, config):
        self.config = config
        self.min_dist = config["optimization"]["min_tower_distance"]

    def generate_candidates(self, grid_df, num_candidates=None):
        if num_candidates is None:
            num_candidates = self.config["optimization"].get("candidate_pool_size", 800)

        t1 = int(num_candidates * 0.40)
        t2 = int(num_candidates * 0.35)
        t3 = num_candidates - t1 - t2

        q60 = grid_df["predicted_traffic_mbps"].quantile(0.60)
        q30 = grid_df["predicted_traffic_mbps"].quantile(0.30)

        tier1 = grid_df[
            (grid_df["predicted_traffic_mbps"] > q60) &
            (grid_df["road_density"] > 0) &
            (grid_df["building_density"] < 0.8)
        ].sort_values("predicted_traffic_mbps", ascending=False)

        tier2 = grid_df[
            (grid_df["predicted_traffic_mbps"] > q30) &
            (grid_df["predicted_traffic_mbps"] <= q60) &
            (grid_df["building_density"] < 0.9)
        ].sort_values("predicted_traffic_mbps", ascending=False)

        tier3 = grid_df[
            (grid_df["population"] > 0) &
            (grid_df["predicted_traffic_mbps"] <= q30)
        ].sort_values("population", ascending=False)

        candidates = []
        seen = set()
        counts = []
        min_d2 = self.min_dist ** 2

        for tier, limit in [(tier1, t1), (tier2, t2), (tier3, t3)]:
            added = 0
            for _, row in tier.iterrows():
                if row["grid_id"] in seen:
                    continue
                rx, ry = row["x"], row["y"]
                if any((rx - c["x"]) ** 2 + (ry - c["y"]) ** 2 < min_d2 for c in candidates):
                    continue
                candidates.append(row)
                seen.add(row["grid_id"])
                added += 1
                if added >= limit:
                    break
            counts.append(added)

        print(f"  Tier 1 (urban):    {counts[0]}")
        print(f"  Tier 2 (suburban): {counts[1]}")
        print(f"  Tier 3 (rural):    {counts[2]}")

        # Fallback: relax distance if under target
        if len(candidates) < num_candidates:
            half_d2 = (self.min_dist / 2) ** 2
            for _, row in grid_df.sort_values("population", ascending=False).iterrows():
                if row["grid_id"] in seen:
                    continue
                rx, ry = row["x"], row["y"]
                if any((rx - c["x"]) ** 2 + (ry - c["y"]) ** 2 < half_d2 for c in candidates):
                    continue
                candidates.append(row)
                seen.add(row["grid_id"])
                if len(candidates) >= num_candidates:
                    break

        return pd.DataFrame(candidates).reset_index(drop=True)

    def generate_coverage_gap_candidates(self, grid_df):
        """All populated grid points as gap-fill candidates (no filtering)."""
        return (grid_df[grid_df["population"] > 0]
                .sort_values("population", ascending=False)
                .reset_index(drop=True))
