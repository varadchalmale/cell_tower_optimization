import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from src.models.propagation import PropagationModel
from src.models.capacity import CapacityModel

class CellTowerPlanningProblem(ElementwiseProblem):
    def __init__(self, candidates_df, grid_df, config):
        n_towers = config['optimization']['num_towers']
        super().__init__(n_var=n_towers,
                         n_obj=3,       
                         n_ieq_constr=0,
                         xl=0,          
                         xu=len(candidates_df) - 1) 
        self.candidates = candidates_df
        self.grid = grid_df
        self.config = config
        self.prop_model = PropagationModel(config)
        self.cap_model = CapacityModel(config)
        self.n_towers = n_towers
        
        print("Precomputing pathloss matrix for candidates...")
        # Precompute path loss from every candidate to every grid point
        # rows: grid points, columns: candidates
        self.candidate_rsrp = np.zeros((len(grid_df), len(candidates_df)))
        
        fc         = self.config['rf_params']['frequency_mhz']
        tx_power   = self.config['rf_params']['transmit_power_dbm']
        ht         = self.config['rf_params']['antenna_height_m']
        hr         = self.config['rf_params']['receiver_height_m']
        # EIRP = Tx_power + Antenna_gain - Cable_loss
        ant_gain   = self.config['rf_params'].get('antenna_gain_dbi', 18)   # 65-deg sector panel
        cable_loss = self.config['rf_params'].get('cable_loss_db',    2)    # jumper + feeder

        # COST-231 Hata constants (medium-small city form; valid 150–2000 MHz)
        # a(h_re) receiver height correction
        a_hre = (1.1 * np.log10(fc) - 0.7) * hr - (1.56 * np.log10(fc) - 0.8)
        cm    = 3.0   # urban environment correction (dB)
        # Pre-compute distance-independent part of path loss
        pl_intercept = (46.3 + 33.9 * np.log10(fc)
                        - 13.82 * np.log10(ht) - a_hre + cm)
        pl_slope     = 44.9 - 6.55 * np.log10(ht)

        for i, row in candidates_df.iterrows():
            d_x   = grid_df['x'].values - row['x']
            d_y   = grid_df['y'].values - row['y']
            d_km  = np.maximum(np.sqrt(d_x**2 + d_y**2) / 1000.0, 0.001)

            pl    = pl_intercept + pl_slope * np.log10(d_km)
            pl    = np.maximum(38.0, pl)          # physical minimum path loss

            # RSRP = EIRP - PathLoss  (all in dBm / dB)
            self.candidate_rsrp[:, i] = tx_power + ant_gain - cable_loss - pl

    def _evaluate(self, x, out, *args, **kwargs):
        # Decode variables into candidate indices
        selected_indices = np.round(x).astype(int)
        
        # Penalize if repeated candidates selected
        unique_towers = np.unique(selected_indices)
        if len(unique_towers) < self.n_towers:
            out["F"] = [1e6, 1e6, 1e6]  
            return

        rsrp_active = self.candidate_rsrp[:, selected_indices]
        
        # Best server per grid point
        best_server_idx = np.argmax(rsrp_active, axis=1)
        best_rsrp = np.max(rsrp_active, axis=1)
        
        # Binary assignment matrix (grid_size x n_towers)
        assignment_matrix = np.zeros((len(self.grid), self.n_towers))
        assignment_matrix[np.arange(len(self.grid)), best_server_idx] = 1

        # Approximation of SINR
        # Noise floor: N = k·T·B  for 20 MHz LTE channel = -174 + 10·log10(20e6) + NF
        # Simplification: use standard -104 dBm (NF ≈ 7 dB, 20 MHz BW) unless overridden
        noise_floor_dbm = self.config.get('thresholds', {}).get('noise_floor_dbm', -104)
        interference_rsrp = np.sum(10**(rsrp_active/10.0), axis=1) - 10**(best_rsrp/10.0)
        noise = 10**(noise_floor_dbm / 10.0)
        sinr_lin = 10**(best_rsrp/10.0) / (interference_rsrp + noise + 1e-12)
        sinr_db = 10 * np.log10(sinr_lin + 1e-12)
        
        # Convert SINR to spectral efficiency and throughput
        se = 0.6 * np.log2(1 + sinr_lin)
        throughput = se * self.config['rf_params']['bandwidth_mhz']
        
        # Demand and load
        demand = self.grid['predicted_traffic_mbps'].values
        # Load of a cell = Sum_{users in cell} (demand / user_peak_throughput)
        cell_loads = np.zeros(self.n_towers)
        sat_demand = np.zeros(len(self.grid))
        
        for c in range(self.n_towers):
            mask = assignment_matrix[:, c] == 1
            if not np.any(mask):
                continue
            tput_c = np.maximum(1e-3, throughput[mask])
            req_fractions = demand[mask] / tput_c
            load = np.sum(req_fractions)
            cell_loads[c] = load
            
            if load <= 1.0:
                sat_demand[mask] = demand[mask]
            else:
                sat_demand[mask] = demand[mask] / load

        # Obj 1: Maximize satisfied demand weighted by population
        pop = self.grid['population'].values
        sat_ratio = sat_demand / (demand + 1e-9)
        obj1 = -np.sum(sat_ratio * pop)  # maximize -> negative

        # Obj 2: Maximize overall usable capacity
        obj2 = -np.sum(throughput) # maximize sum of user peak throughputs

        # Obj 3: Minimize deployment cost
        cands = self.candidates.iloc[selected_indices]
        road_penalties = 1.0 / (cands['road_density'].values + 0.1)
        land_penalties = cands['building_density'].values * 10
        obj3 = np.sum(road_penalties + land_penalties)
        
        out["F"] = [obj1, obj2, obj3]

class MultiObjectiveOptimizer:
    def __init__(self, config):
        self.config = config
        
    def run_optimization(self, candidates_df, grid_df):
        print(f"Starting NSGA-II optimization with {len(candidates_df)} candidates for {self.config['optimization']['num_towers']} towers...")
        
        problem = CellTowerPlanningProblem(candidates_df, grid_df, self.config)
        
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        
        algorithm = NSGA2(
            pop_size=self.config['nsga2']['pop_size'],
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        res = minimize(problem,
                       algorithm,
                       ("n_gen", self.config['nsga2']['n_gen']),
                       seed=self.config['project']['seed'],
                       verbose=True)
        
        print(f"Optimization finished. Found {len(res.F)} Pareto optimal solutions.")
        
        # Get the best compromise solution (e.g., minimum distance to ideal point)
        F = res.F
        ideal = F.min(axis=0)
        nadir = F.max(axis=0)
        normalized_F = (F - ideal) / (nadir - ideal + 1e-9)
        
        # Weighted sum of objectives (giving equal priority) to pick one solution
        weights = np.array([0.4, 0.4, 0.2]) # higher priority to coverage and capacity, less to cost
        weighted_sum = np.sum(normalized_F * weights, axis=1)
        best_idx = np.argmin(weighted_sum)
        
        best_solution_indices = np.round(res.X[best_idx]).astype(int)
        best_towers = candidates_df.iloc[best_solution_indices].copy()
        
        return res, best_towers
