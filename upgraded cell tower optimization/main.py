import os
import yaml
import warnings
from src.data.preprocessing import DataPreprocessor
from src.models.demand import DemandModel
from src.optimization.candidates import CandidateSiteGenerator
from src.optimization.nsga2 import MultiObjectiveOptimizer
from src.validation.validate import CoverageValidator
from src.visualization.visualize import Visualizer
from download_airtel import download_airtel_coverage

warnings.filterwarnings('ignore')

def load_config():
    with open("src/config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_pipeline():
    config = load_config()
    print("=== AI-DRIVEN 4G/5G CELL TOWER PLANNING SYSTEM ===")
    
    # 1. Preprocessing
    preprocessor = DataPreprocessor(config)
    print("\n[1/7] Initializing planning region...")
    boundary = preprocessor.create_nagpur_boundary()
    grid = preprocessor.generate_planning_grid(boundary)
    print(f"Generated {len(grid)} grid points.")
    
    buildings, roads, landuse = preprocessor.extract_osm_features(boundary)
    grid_features = preprocessor.compute_grid_features(
        grid, buildings, roads, landuse,
        config['paths']['pop_tif'], config['paths']['dem_tif']
    )
    
    # 2. ML Demand modeling
    # Pass real OpenCellID CSV path when available so the model trains on actual
    # tower-density labels instead of proxy estimates.
    print("\n[2/7] Training Demand Prediction AI...")
    opencellid_path = os.path.join("Data", "raw", "opencellid_nagpur.csv")
    demand_model = DemandModel(
        config,
        opencellid_csv=opencellid_path if os.path.exists(opencellid_path) else None
    )
    demand_model.train_ml_model(grid_features)
    grid_with_demand = demand_model.predict_traffic(grid_features)
    
    print("Detecting Demand Hotspots...")
    grid_with_demand, high_demand, hotspots = demand_model.detect_hotspots(grid_with_demand)
    
    # 3. Candidate Generation
    print("\n[3/7] Generating Candidate Sites...")
    candidate_generator = CandidateSiteGenerator(config)
    candidates = candidate_generator.generate_candidates(grid_with_demand)
    print(f"Total feasible candidates evaluated: {len(candidates)}")
    
    # 4 & 5. Optimization (Propagation + Capacity inside obj evaluation)
    print("\n[4/7] Multi-Objective Optimization (NSGA-II)...")
    optimizer = MultiObjectiveOptimizer(config)
    res, best_config = optimizer.run_optimization(candidates, grid_with_demand)
    print(f"Optimal configuration selected with {len(best_config)} towers.")
    
    # 6. Visualization
    print("\n[5/7] Generating Maps and Plots...")
    visuals = Visualizer(config)
    visuals.plot_heatmap(grid_with_demand, 'predicted_traffic_mbps', 'Predicted Traffic Demand (Mbps)')
    visuals.plot_pareto_front(res)
    visuals.plot_candidates_and_selected(candidates, best_config, grid_with_demand)
    visuals.generate_html_map(best_config, grid_with_demand, boundary)
    
    # 7. Validation
    print("\n[6/7] Validation against Airtel coverage map...")
    
    airtel_file = config['paths']['airtel_coverage']
    if not os.path.exists(airtel_file):
        download_airtel_coverage(airtel_file)
        
    validator = CoverageValidator(config)
    simulated_coverage_poly = validator.generate_simulated_coverage(grid_with_demand, best_config)
    
    metrics = validator.validate(simulated_coverage_poly)
    print("Validation Metrics:")
    for key, val in metrics.items():
        print(f" - {key}: {val}")
        
    print("\n[7/7] Pipeline Complete. Output artifacts saved in the 'outputs' directory.")
    print("Summary:")
    print("An AI-driven, capacity-aware, multi-objective automated cellular network planning and validation framework using real geospatial data.")

if __name__ == "__main__":
    run_pipeline()
