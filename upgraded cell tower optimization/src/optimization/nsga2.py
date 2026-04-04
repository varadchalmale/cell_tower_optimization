"""
NSGA-II multi-objective optimiser for macro tower placement.

Objectives:
  1. Maximise population coverage fraction (RSRP >= threshold)
  2. Maximise aggregate network capacity (Shannon throughput)
  3. Minimise deployment cost (road proximity + land cost proxy)
"""

import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from src.models.propagation import vectorized_rsrp_matrix


class CellTowerPlanningProblem(ElementwiseProblem):
    def __init__(self, candidates_df, grid_df, config):
        n_towers = config["optimization"]["num_towers"]
        super().__init__(
            n_var=n_towers,
            n_obj=3,
            n_ieq_constr=0,
            xl=0,
            xu=len(candidates_df) - 1,
        )
        self.candidates = candidates_df
        self.grid = grid_df
        self.config = config
        self.n_towers = n_towers
        self.rsrp_threshold = config["optimization"].get("coverage_rsrp_threshold_dbm", -110)
        self.pop_total = grid_df["population"].values.sum()

        print("  Precomputing RSRP matrix (env-aware, with fading margin)...")
        self.candidate_rsrp = vectorized_rsrp_matrix(grid_df, candidates_df, config)
        print(f"  Matrix shape: {self.candidate_rsrp.shape}  "
              f"({self.candidate_rsrp.nbytes / 1e6:.1f} MB)")

    def _evaluate(self, x, out, *args, **kwargs):
        selected = np.round(x).astype(int)
        if len(np.unique(selected)) < self.n_towers:
            out["F"] = [1e6, 1e6, 1e6]
            return

        rsrp_active = self.candidate_rsrp[:, selected]
        best_rsrp = np.max(rsrp_active, axis=1)

        # --- Objective 1: population coverage fraction ---
        covered = (best_rsrp >= self.rsrp_threshold).astype(np.float32)
        pop = self.grid["population"].values
        obj1 = -np.sum(pop * covered) / (self.pop_total + 1e-9)

        # --- Objective 2: aggregate capacity ---
        interference = np.sum(10 ** (rsrp_active / 10.0), axis=1) - 10 ** (best_rsrp / 10.0)
        noise = 10 ** (-104 / 10.0)
        sinr_lin = 10 ** (best_rsrp / 10.0) / (interference + noise + 1e-12)
        se = 0.6 * np.log2(1 + sinr_lin)
        throughput = se * self.config["rf_params"]["bandwidth_mhz"]
        obj2 = -np.sum(throughput)

        # --- Objective 3: deployment cost ---
        cands = self.candidates.iloc[selected]
        road_pen = 1.0 / (cands["road_density"].values + 0.1)
        land_pen = cands["building_density"].values * 10
        obj3 = np.sum(road_pen + land_pen)

        out["F"] = [obj1, obj2, obj3]


class MultiObjectiveOptimizer:
    def __init__(self, config):
        self.config = config

    def run_optimization(self, candidates_df, grid_df):
        cfg = self.config
        n_towers = cfg["optimization"]["num_towers"]
        print(f"  Starting NSGA-II: {len(candidates_df)} candidates -> {n_towers} towers")

        problem = CellTowerPlanningProblem(candidates_df, grid_df, cfg)

        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM

        algorithm = NSGA2(
            pop_size=cfg["nsga2"]["pop_size"],
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        res = minimize(
            problem, algorithm,
            ("n_gen", cfg["nsga2"]["n_gen"]),
            seed=cfg["project"]["seed"],
            verbose=True,
        )

        print(f"  Optimisation done. {len(res.F)} Pareto-optimal solutions.")

        # Weighted compromise selection
        F = res.F
        ideal = F.min(axis=0)
        nadir = F.max(axis=0)
        norm = (F - ideal) / (nadir - ideal + 1e-9)
        w = np.array([
            cfg["nsga2"].get("coverage_weight", 0.5),
            cfg["nsga2"].get("capacity_weight", 0.3),
            cfg["nsga2"].get("cost_weight", 0.2),
        ])
        best_idx = np.argmin(np.sum(norm * w, axis=1))

        indices = np.round(res.X[best_idx]).astype(int)
        best_towers = candidates_df.iloc[indices].copy()
        best_towers["tower_source"] = "nsga2_macro"

        phase1_cov = -F[best_idx, 0]
        print(f"  Phase 1 population coverage: {phase1_cov * 100:.1f}%")
        return res, best_towers
