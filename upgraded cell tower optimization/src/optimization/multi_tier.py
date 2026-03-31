"""
multi_tier.py — Hierarchical gap-filling optimizer for Micro and Small Cell tiers.

Strategy
--------
Stage 1 (existing): NSGA-II places N_macro Macro towers — globally optimal
                     multi-objective placement across the full district.
Stage 2 (this file): Greedy Micro gap-fill — targets grid points inside the
                     Airtel coverage polygon that are NOT reached by macros.
Stage 3 (this file): Greedy Small Cell hotspot-fill — targets the top-demand
                     grid points still uncovered after macro + micro.

Why greedy (not NSGA-II) for stages 2 & 3?
- The objective for gap-fill is single: maximise demand covered in a specific
  sub-region. Under sub-modular coverage, greedy max-demand-first is optimal
  (Nemhauser et al. 1978) and runs in < 5 seconds for this grid size.
- Running NSGA-II for all three tiers would triple precomputation time and
  violate the 15-minute pipeline budget.

Core algorithm (_greedy_fill):
    while budget and uncovered points remain:
        1. Pick grid point with highest demand  (O(N) lookup, shrinks each iter)
        2. Place tower there
        3. Mark all points within radius_m as covered  (vectorized NumPy)
        4. Remove covered points from the working set
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from src.models.propagation import PropagationModel


class MultiTierOptimizer:

    def __init__(self, config: dict):
        self.config      = config
        self.freq_mhz    = config['rf_params']['frequency_mhz']
        self.rsrp_min    = config['thresholds']['rsrp_min_dbm']
        self.crs         = config['project']['crs']
        self.tower_specs = config['tower_types']
        self.mt_cfg      = config.get('multi_tier', {})

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _radius(self, tier: str) -> float:
        """Return COST-231 coverage radius (m) for a tier, computed at runtime."""
        return PropagationModel.coverage_radius_m(
            self.tower_specs[tier], self.freq_mhz, self.rsrp_min
        )

    def _covered_mask(self, grid_xy: np.ndarray, tower_xy: np.ndarray,
                      radius_m: float) -> np.ndarray:
        """Boolean mask (N,): True if grid point is within radius_m of any tower."""
        if len(tower_xy) == 0:
            return np.zeros(len(grid_xy), dtype=bool)
        d_x = grid_xy[:, 0:1] - tower_xy[:, 0]   # (N, M)
        d_y = grid_xy[:, 1:2] - tower_xy[:, 1]
        return (np.sqrt(d_x**2 + d_y**2) <= radius_m).any(axis=1)

    def _greedy_fill(self, uncovered: pd.DataFrame, demand_col: str,
                     radius_m: float, budget: int) -> pd.DataFrame:
        """
        Greedy max-demand tower placement on `uncovered` grid points.

        Returns DataFrame with columns:
            x, y, coverage_radius_m, demand_served, tower_index
        """
        remaining = uncovered.copy()
        placed    = []

        for i in range(budget):
            if remaining.empty:
                break

            # 1. Pick highest-demand point
            best_idx = remaining[demand_col].idxmax()
            tx, ty   = remaining.at[best_idx, 'x'], remaining.at[best_idx, 'y']

            # 2. Mark coverage
            dx   = remaining['x'].values - tx
            dy   = remaining['y'].values - ty
            mask = np.sqrt(dx**2 + dy**2) <= radius_m

            placed.append({
                'x':                tx,
                'y':                ty,
                'coverage_radius_m': radius_m,
                'demand_served':    remaining.loc[mask, demand_col].sum(),
                'tower_index':      i,
            })

            # 3. Remove covered points
            remaining = remaining[~mask].copy()

        return pd.DataFrame(placed)

    # ─── Public API ──────────────────────────────────────────────────────────

    def fill_micro_gaps(self, grid_df: pd.DataFrame,
                        macro_towers: pd.DataFrame,
                        airtel_poly,
                        n_micro: int = None) -> pd.DataFrame:
        """
        Stage 2: Place Micro towers inside the Airtel coverage polygon for areas
        not reached by Macro towers.

        Parameters
        ----------
        grid_df      : full planning grid (columns: x, y, population,
                       predicted_traffic_mbps, ...)
        macro_towers : NSGA-II output (columns: x, y, ...)
        airtel_poly  : shapely Polygon of Airtel coverage area (projected CRS)
        n_micro      : tower budget override; defaults to config deploy_budget

        Returns
        -------
        DataFrame with columns: x, y, tier, coverage_radius_m, demand_served
        """
        spec     = self.tower_specs['micro']
        radius_m = self._radius('micro')
        budget   = n_micro if n_micro is not None else spec['deploy_budget']

        # Step 1: restrict to points inside Airtel polygon
        pts = gpd.GeoDataFrame(
            grid_df,
            geometry=gpd.points_from_xy(grid_df['x'], grid_df['y']),
            crs=self.crs
        )
        in_airtel   = pts.geometry.within(airtel_poly)
        airtel_grid = grid_df[in_airtel.values].copy()

        # Step 2: remove points already covered by macros
        macro_r    = self._radius('macro')
        macro_xy   = macro_towers[['x', 'y']].values if len(macro_towers) > 0 else np.empty((0, 2))
        airtel_xy  = airtel_grid[['x', 'y']].values
        macro_cov  = self._covered_mask(airtel_xy, macro_xy, macro_r)
        uncovered  = airtel_grid[~macro_cov].copy()

        print(f"  [Micro] Inside Airtel polygon: {len(airtel_grid):,}  |  "
              f"Uncovered by macros: {len(uncovered):,}  |  Budget: {budget}  |  "
              f"Radius: {radius_m:.0f} m")

        if uncovered.empty:
            print("  [Micro] Macros cover entire Airtel polygon — no micro cells needed.")
            return pd.DataFrame(columns=['x', 'y', 'tier', 'coverage_radius_m', 'demand_served'])

        result = self._greedy_fill(uncovered, 'predicted_traffic_mbps', radius_m, budget)
        result['tier'] = 'micro'
        print(f"  [Micro] Placed {len(result)} micro towers.")
        return result

    def fill_small_cell_gaps(self, grid_df: pd.DataFrame,
                             macro_towers: pd.DataFrame,
                             micro_towers: pd.DataFrame,
                             n_small: int = None) -> pd.DataFrame:
        """
        Stage 3: Place Small Cells at high-demand hotspots still uncovered
        after Macro + Micro layers.

        Parameters
        ----------
        grid_df      : full planning grid
        macro_towers : DataFrame with x, y
        micro_towers : DataFrame with x, y (output of fill_micro_gaps)
        n_small      : tower budget override; defaults to config deploy_budget

        Returns
        -------
        DataFrame with columns: x, y, tier, coverage_radius_m, demand_served
        """
        spec     = self.tower_specs['small_cell']
        radius_m = self._radius('small_cell')
        budget   = n_small if n_small is not None else spec['deploy_budget']

        # Step 1: filter to high-demand points (top % threshold)
        pct       = self.mt_cfg.get('small_cell_demand_threshold_pct', 70)
        threshold = grid_df['predicted_traffic_mbps'].quantile(pct / 100.0)
        hotspots  = grid_df[grid_df['predicted_traffic_mbps'] >= threshold].copy()

        # Step 2: remove points already covered by macro or micro
        grid_xy  = hotspots[['x', 'y']].values
        macro_xy = macro_towers[['x', 'y']].values if len(macro_towers) > 0 else np.empty((0, 2))
        micro_xy = micro_towers[['x', 'y']].values if len(micro_towers) > 0 else np.empty((0, 2))

        macro_cov = self._covered_mask(grid_xy, macro_xy, self._radius('macro'))
        micro_cov = self._covered_mask(grid_xy, micro_xy, self._radius('micro')) if len(micro_xy) > 0 else np.zeros(len(grid_xy), dtype=bool)
        uncovered = hotspots[~(macro_cov | micro_cov)].copy()

        print(f"  [Small] High-demand pts (>{pct}th pct): {len(hotspots):,}  |  "
              f"Still uncovered: {len(uncovered):,}  |  Budget: {budget}  |  "
              f"Radius: {radius_m:.0f} m")

        if uncovered.empty:
            print("  [Small] All hotspots covered — no small cells needed.")
            return pd.DataFrame(columns=['x', 'y', 'tier', 'coverage_radius_m', 'demand_served'])

        result = self._greedy_fill(uncovered, 'predicted_traffic_mbps', radius_m, budget)
        result['tier'] = 'small_cell'
        print(f"  [Small] Placed {len(result)} small cells.")
        return result

    def compute_multi_tier_coverage(self, grid_df: pd.DataFrame,
                                    all_towers_dict: dict) -> dict:
        """
        Compute per-tier incremental and aggregate coverage statistics.

        Parameters
        ----------
        grid_df         : planning grid with 'population' and
                          'predicted_traffic_mbps' columns
        all_towers_dict : {'macro': df, 'micro': df, 'small_cell': df}

        Returns
        -------
        dict with keys:
            per_tier   — {tier: {n_towers, radius_m, area_pct, pop_pct,
                                  demand_pct, airtel_equivalent, display_name}}
            aggregate  — {n_towers_total, area_pct, pop_pct, demand_pct}
            comparison — {ai_total, airtel_total, reduction_pct}
        """
        grid_xy      = grid_df[['x', 'y']].values
        pop          = grid_df['population'].values
        demand       = grid_df['predicted_traffic_mbps'].values
        total_pop    = pop.sum() + 1e-9
        total_demand = demand.sum() + 1e-9
        n_grid       = len(grid_df)

        cumulative   = np.zeros(n_grid, dtype=bool)
        per_tier     = {}
        total_towers = 0

        for tier in ['macro', 'micro', 'small_cell']:
            df_t   = all_towers_dict.get(tier, pd.DataFrame())
            spec   = self.tower_specs.get(tier, {})
            radius = self._radius(tier) if spec else 0.0

            if len(df_t) > 0 and 'x' in df_t.columns:
                txy      = df_t[['x', 'y']].values
                tier_cov = self._covered_mask(grid_xy, txy, radius)
                incr     = tier_cov & ~cumulative
                cumulative |= tier_cov
            else:
                tier_cov = np.zeros(n_grid, dtype=bool)
                incr     = np.zeros(n_grid, dtype=bool)

            n_t = len(df_t) if len(df_t) > 0 else 0
            total_towers += n_t

            per_tier[tier] = {
                'n_towers':          n_t,
                'radius_m':          radius,
                'area_pct':          incr.mean() * 100,
                'pop_pct':           pop[incr].sum() / total_pop * 100,
                'demand_pct':        demand[incr].sum() / total_demand * 100,
                'airtel_equivalent': spec.get('airtel_equivalent', 0),
                'display_name':      spec.get('display_name', tier),
            }

        agg_area   = cumulative.mean() * 100
        agg_pop    = pop[cumulative].sum() / total_pop * 100
        agg_demand = demand[cumulative].sum() / total_demand * 100
        airtel_ref = self.mt_cfg.get('airtel_total_sites', 1200)
        reduction  = (airtel_ref - total_towers) / airtel_ref * 100 if airtel_ref > 0 else 0.0

        return {
            'per_tier':  per_tier,
            'aggregate': {
                'n_towers_total': total_towers,
                'area_pct':       agg_area,
                'pop_pct':        agg_pop,
                'demand_pct':     agg_demand,
            },
            'comparison': {
                'ai_total':      total_towers,
                'airtel_total':  airtel_ref,
                'reduction_pct': reduction,
            },
        }
