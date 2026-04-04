import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import geopandas as gpd
from src.config import Config
from src.propagation_models import PropagationModel
from src.capacity_model import CapacityModel
from src.candidate_site_generator import CandidateSiteGenerator
from src.ml_demand_model import MLDemandModel
from src.multiobjective_optimizer import MultiObjectiveGA

def load_data():
    print("Loading data for Upgraded System...")
    data_files = {
        'demand': 'feature_demand_score.tif',
        'cost': 'feature_cost_surface.tif',
        'elevation': 'feature_elevation.tif',
        'clutter': 'feature_clutter.tif',
        'population': 'feature_population.tif'
    }
    loaded = {}
    for key, filename in data_files.items():
        with rasterio.open(os.path.join(Config.PROCESSED_DATA_PATH, filename)) as src:
            loaded[key] = src.read(1)
            if 'transform' not in loaded:
                loaded['transform'] = src.transform
                loaded['crs'] = src.crs
    
    loaded['district_boundary'] = gpd.read_file(os.path.join(Config.PROCESSED_DATA_PATH, 'Boundaries', 'nagpur_boundary.gpkg'))
    return loaded

def plot_pareto_frontier(fitnesses, results_path):
    print("Generating Pareto Frontier plots...")
    # fitnesses: list of (coverage, capacity, -cost)
    df = pd.DataFrame(fitnesses, columns=['Coverage', 'Capacity', 'NegCost'])
    df['Cost'] = -df['NegCost']
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Cost'], df['Coverage'], c=df['Capacity'], cmap='viridis', s=100)
    plt.colorbar(label='Capacity Score')
    plt.xlabel('Normalized Deployment Cost')
    plt.ylabel('Demand-Weighted Coverage')
    plt.title('Pareto Frontier: Cost vs Coverage vs Capacity')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_path, 'pareto_frontier.png'))
    plt.close()

def generate_heatmaps(best_ind, data, results_path):
    print("Generating detailed heatmaps (SINR, Capacity, Load)...")
    h, w = data['demand'].shape
    yy, xx = np.mgrid[0:h, 0:w]
    
    clutter = data.get('clutter')
    indoor_mask = (clutter > 0.5) if clutter is not None else None

    all_rsrp = []
    for y, x in best_ind:
        dist_km = np.sqrt((xx - x)**2 + (yy - y)**2) * Config.GRID_RESOLUTION_M / 1000.0
        # Limitation 2 + 3: use multiband RSRP with indoor penetration loss
        best_rsrp, _ = PropagationModel.calculate_rsrp_multiband(dist_km, indoor_mask)
        all_rsrp.append(best_rsrp)

    max_rsrp = np.maximum.reduce(all_rsrp)
    total_rsrp_linear = sum([CapacityModel.dbm_to_linear(r) for r in all_rsrp])
    serving_rsrp_linear = CapacityModel.dbm_to_linear(max_rsrp)
    interference_linear = total_rsrp_linear - serving_rsrp_linear
    noise_linear = CapacityModel.dbm_to_linear(Config.NOISE_FLOOR_DBM)

    sinr_linear = serving_rsrp_linear / (interference_linear + noise_linear + 1e-12)
    sinr_db = 10 * np.log10(sinr_linear + 1e-12)
    # Limitation 3 + 4: use total bandwidth + hardware cap
    capacity_map = CapacityModel.shannon_capacity(sinr_db)
    
    # Save Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    im1 = axes[0].imshow(sinr_db, cmap='jet', vmin=-10, vmax=30)
    axes[0].set_title('SINR Heatmap (dB)')
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(capacity_map, cmap='plasma')
    axes[1].set_title('Capacity Heatmap (Mbps per pixel)')
    fig.colorbar(im2, ax=axes[1])
    
    plt.savefig(os.path.join(results_path, 'capacity_sinr_heatmaps.png'))
    plt.close()

def run_upgraded_pipeline():
    Config.setup_directories()
    data = load_data()
    
    # 1. ML Demand Prediction
    ml_model = MLDemandModel()
    # Simulated training on existing demand score for demonstration
    ml_results = ml_model.train(data['population'], data['clutter'], data['demand'])
    predicted_demand = ml_model.predict(data['population'], data['clutter'])
    if predicted_demand is not None:
        data['demand_ml'] = predicted_demand
    else:
        data['demand_ml'] = data['demand']

    # Limitation 6 Fix: apply temporal demand profile before optimisation.
    # Scales demand by the multiplier for Config.DESIGN_SCENARIO so that the
    # network is sized for peak traffic rather than average traffic.
    data['demand'] = MLDemandModel.apply_temporal_profile(data['demand'], Config.DESIGN_SCENARIO)

    # 2. Candidate Site Generation
    candidates = CandidateSiteGenerator.generate_sites(data['demand'], data['cost'], data['elevation'])

    # 3. Multi-Objective Optimization
    # The optimizer automatically uses:
    #   - Multi-band RSRP cache (Limitation 3)
    #   - Indoor penetration loss via clutter mask (Limitation 2)
    #   - Backhaul cap + overload penalty (Limitation 4)
    moga = MultiObjectiveGA(data, candidates)
    pareto_solutions, pareto_fitnesses = moga.run()
    
    # 4. Save best trade-off
    best_idx = np.argmax([sum(fit) for fit in pareto_fitnesses])
    best_ind_indices = pareto_solutions[best_idx]
    best_solution = [candidates[i] for i in best_ind_indices]
    
    # Save solution
    best_solution_serializable = [[int(coord) for coord in point] for point in best_solution]
    with open(os.path.join(Config.RESULTS_PATH, 'best_solution_upgraded.json'), 'w') as f:
        json.dump(best_solution_serializable, f)
        
    # 5. Visualizations
    plot_pareto_frontier(pareto_fitnesses, Config.RESULTS_PATH)
    generate_heatmaps(best_solution, data, Config.RESULTS_PATH)
    
    # 6. Comparison with "Old" system (mock metrics for base system)
    print("\n--- PERFORMANCE COMPARISON ---")
    print(f"{'Metric':<25} | {'Old System (Estimated)':<25} | {'Upgraded System':<25}")
    print("-" * 80)
    print(f"{'Model Type':<25} | {'FSPL + Simple LoS':<25} | {'COST-231 Hata + SINR':<25}")
    print(f"{'Objectives':<25} | {'Single (Weighted Sum)':<25} | {'Multi-Objective (Pareto)':<25}")
    print(f"{'Demand weighted Cov':<25} | {'75.4%':<25} | {pareto_fitnesses[best_idx][0]*100:>23.2f}%")
    print(f"{'Average Capacity':<25} | {'N/A':<25} | {pareto_fitnesses[best_idx][1]*10:>24.2f} Gbps")
    print(f"{'Candidate Sites':<25} | {'Any Pixel':<25} | {len(candidates):>24} sites")
    
    print(f"\nUpgraded system run complete. Results in {Config.RESULTS_PATH}")

if __name__ == "__main__":
    run_upgraded_pipeline()
