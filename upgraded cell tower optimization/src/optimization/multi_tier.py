"""
Phase 2: Greedy maximum-marginal-gain gap-filling.

After NSGA-II places macro towers, this module iteratively adds the minimum
number of additional towers needed to bring population coverage to 100%.
Uses the shared vectorized_rsrp_matrix() to guarantee formula consistency.
"""

import numpy as np
import pandas as pd
from src.models.propagation import vectorized_rsrp_matrix


class GreedyGapFiller:

    def __init__(self, config):
        self.config = config
        self.threshold = config["optimization"].get("coverage_rsrp_threshold_dbm", -110)
        self.max_towers = config.get("gap_filling", {}).get("max_additional_towers", 80)
        self.stop_frac = config.get("gap_filling", {}).get("min_uncovered_pop_fraction", 0.001)

    def _coverage_mask(self, grid_df, towers_df):
        if len(towers_df) == 0:
            return np.zeros(len(grid_df), dtype=bool)
        rsrp = vectorized_rsrp_matrix(grid_df, towers_df, self.config)
        return rsrp.max(axis=1) >= self.threshold

    def fill_gaps(self, nsga2_towers_df, grid_df, gap_candidates_df):
        pop = grid_df["population"].values.astype(np.float64)
        pop_total = pop.sum()
        if pop_total == 0:
            return pd.DataFrame(), 1.0

        covered = self._coverage_mask(grid_df, nsga2_towers_df)
        init_pct = np.sum(pop * covered) / pop_total * 100
        print(f"  Coverage entering gap-fill: {init_pct:.1f}%")

        if np.sum(pop * (~covered)) / pop_total < self.stop_frac:
            return pd.DataFrame(), np.sum(pop * covered) / pop_total

        # Precompute coverage footprint matrix for all gap candidates
        print(f"  Precomputing gap-fill RSRP ({len(grid_df)} grid x "
              f"{len(gap_candidates_df)} candidates)...")
        rsrp_mat = vectorized_rsrp_matrix(grid_df, gap_candidates_df, self.config)
        cov_mat = rsrp_mat >= self.threshold
        del rsrp_mat

        placed = set()
        gap_towers = []

        for _ in range(self.max_towers):
            uncov_pop = pop * (~covered)
            if uncov_pop.sum() / pop_total < self.stop_frac:
                break

            gains = cov_mat.T @ uncov_pop
            for idx in placed:
                gains[idx] = -1.0

            best = int(np.argmax(gains))
            if gains[best] <= 0:
                print("  No candidate improves coverage. Stopping.")
                break

            placed.add(best)
            covered = covered | cov_mat[:, best]
            row = gap_candidates_df.iloc[best].copy()
            row["tower_source"] = "gap_fill"
            gap_towers.append(row)

            pct = np.sum(pop * covered) / pop_total * 100
            print(f"  Gap-fill [{len(gap_towers):3d}]: +{gains[best]:8.0f} pop -> {pct:.2f}%")

        final = np.sum(pop * covered) / pop_total
        gap_df = pd.DataFrame(gap_towers).reset_index(drop=True) if gap_towers else pd.DataFrame()
        return gap_df, final
