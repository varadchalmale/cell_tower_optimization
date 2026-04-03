"""
AI-Driven Cell Tower Optimisation — Full Pipeline

9-stage pipeline that proves 100% population coverage of Nagpur district
(~9,928 km²) can be achieved with far fewer towers than Airtel's ~1,200
real-world deployment, using NSGA-II multi-objective optimisation +
greedy gap-filling.

Includes: baseline comparisons (Random, Hexagonal, K-Means),
sensitivity analysis, and Airtel IoU validation.
"""

import os
import json
import yaml
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def load_config():
    with open("src/config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def run_pipeline():
    config = load_config()

    print("=" * 70)
    print("  AI-DRIVEN CELL TOWER PLANNING — NAGPUR DISTRICT")
    print("  Goal: 100% population coverage with minimum towers")
    print("  District: ~9,928 km² | Airtel reference: ~1,200 towers")
    print("=" * 70)

    from src.data.preprocessing import DataPreprocessor
    from src.models.demand import DemandModel
    from src.optimization.candidates import CandidateSiteGenerator
    from src.optimization.nsga2 import MultiObjectiveOptimizer
    from src.optimization.multi_tier import GreedyGapFiller
    from src.validation.validate import CoverageValidator
    from src.visualization.visualize import Visualizer
    from src.models.propagation import compute_coverage_radius_km

    # ==================================================================
    # [1/9] Data Preprocessing
    # ==================================================================
    print("\n[1/9] Loading Nagpur district boundary & generating grid...")
    preprocessor = DataPreprocessor(config)
    boundary = preprocessor.create_nagpur_boundary()
    grid = preprocessor.generate_planning_grid(boundary)
    area_km2 = boundary.area / 1e6
    print(f"  Grid: {len(grid)} points at {config['optimization']['grid_resolution']}m "
          f"over {area_km2:.0f} km²")

    buildings, roads, landuse = preprocessor.extract_osm_features(boundary)
    grid_features = preprocessor.compute_grid_features(
        grid, buildings, roads, landuse,
        config["paths"]["pop_tif"], config["paths"]["dem_tif"],
    )

    # Print coverage radii for reference
    for env in ("urban", "suburban", "rural"):
        r = compute_coverage_radius_km(config, env)
        print(f"  Coverage radius ({env}): {r:.2f} km")

    # ==================================================================
    # [2/9] ML Demand Prediction
    # ==================================================================
    print("\n[2/9] Training demand prediction model...")
    demand_model = DemandModel(config)
    demand_model.train_ml_model(grid_features)
    grid_with_demand = demand_model.predict_traffic(grid_features)
    grid_with_demand, high_demand, hotspots = demand_model.detect_hotspots(grid_with_demand)

    # ==================================================================
    # [3/9] 3-Tier Candidate Generation
    # ==================================================================
    pool_size = config["optimization"].get("candidate_pool_size", 800)
    print(f"\n[3/9] Generating {pool_size} candidate sites (urban/suburban/rural)...")
    cand_gen = CandidateSiteGenerator(config)
    candidates = cand_gen.generate_candidates(grid_with_demand, pool_size)
    print(f"  Total candidates: {len(candidates)}")

    # ==================================================================
    # [4/9] Phase 1 — NSGA-II Macro Placement
    # ==================================================================
    n_macro = config["optimization"]["num_towers"]
    print(f"\n[4/9] Phase 1 — NSGA-II optimising {n_macro} macro towers...")
    print("  Obj 1: Maximise population coverage fraction")
    print("  Obj 2: Maximise network capacity")
    print("  Obj 3: Minimise deployment cost")
    optimizer = MultiObjectiveOptimizer(config)
    res, nsga2_towers = optimizer.run_optimization(candidates, grid_with_demand)

    # ==================================================================
    # [5/9] Phase 1 Coverage Assessment
    # ==================================================================
    print("\n[5/9] Phase 1 coverage assessment...")
    validator = CoverageValidator(config)
    p1 = validator.compute_population_coverage(grid_with_demand, nsga2_towers)
    print(f"  Population coverage: {p1['population_coverage_pct']}%")
    print(f"  Covered: {p1['covered_population']:.0f} / {p1['total_population']:.0f}")

    # ==================================================================
    # [6/9] Phase 2 — Greedy Gap-Filling
    # ==================================================================
    gap_towers = pd.DataFrame()
    gap_enabled = config.get("gap_filling", {}).get("enabled", True)

    if gap_enabled and p1["population_coverage_pct"] < 99.9:
        remaining = 100.0 - p1["population_coverage_pct"]
        print(f"\n[6/9] Phase 2 — Gap-filling remaining {remaining:.1f}% population...")
        gap_filler = GreedyGapFiller(config)
        gap_cands = cand_gen.generate_coverage_gap_candidates(grid_with_demand)
        print(f"  Gap-fill candidate pool: {len(gap_cands)}")
        gap_towers, final_cov = gap_filler.fill_gaps(
            nsga2_towers, grid_with_demand, gap_cands,
        )
        print(f"  Added {len(gap_towers)} gap-fill towers -> {final_cov * 100:.2f}%")
    else:
        print(f"\n[6/9] Gap-filling skipped (coverage already {p1['population_coverage_pct']}%)")

    all_towers = (pd.concat([nsga2_towers, gap_towers], ignore_index=True)
                  if len(gap_towers) > 0 else nsga2_towers.copy())

    # ==================================================================
    # [7/9] Final Metrics + Baselines
    # ==================================================================
    print("\n[7/9] Final metrics & baseline comparisons...")
    final = validator.compute_population_coverage(grid_with_demand, all_towers)

    # Baselines with same tower count
    n_total = final["num_towers"]
    print(f"  Running baselines with {n_total} towers each...")
    baselines = validator.run_baselines(grid_with_demand, boundary, n_total)
    for name, pct in baselines.items():
        if not name.endswith("_towers"):
            print(f"    {name}: {pct:.1f}%")

    print("\n" + "=" * 70)
    print("  RESULTS: AI vs Baselines vs Airtel")
    print("=" * 70)
    print(f"  NSGA-II macro towers:     {len(nsga2_towers)}")
    print(f"  Gap-fill towers:          {len(gap_towers)}")
    print(f"  TOTAL AI towers:          {n_total}")
    print(f"  Population coverage:      {final['population_coverage_pct']}%")
    print(f"  Geographic coverage:      {final['geographic_coverage_pct']}%")
    print(f"  vs Airtel:                {final['vs_airtel']}")
    print(f"  ---")
    print(f"  Random baseline:          {baselines.get('Random', 0):.1f}%")
    print(f"  Hexagonal baseline:       {baselines.get('Hexagonal', 0):.1f}%")
    print(f"  K-Means baseline:         {baselines.get('KMeans', 0):.1f}%")
    print("=" * 70)

    # ==================================================================
    # [8/9] Airtel Validation + Sensitivity + Visualisation
    # ==================================================================
    print("\n[8/9] Visualisation & validation...")
    visuals = Visualizer(config)

    # Heatmaps
    visuals.plot_heatmap(grid_with_demand, "predicted_traffic_mbps",
                         "Predicted Traffic Demand (Mbps)")
    visuals.plot_heatmap(grid_with_demand, "population",
                         "Population Density (per grid cell)")

    # Pareto front
    visuals.plot_pareto_front(res)

    # Tower map
    visuals.plot_candidates_and_selected(candidates, all_towers, grid_with_demand)

    # Interactive HTML map
    visuals.generate_html_map(all_towers, grid_with_demand, boundary)

    # Baseline comparison chart
    visuals.plot_baseline_comparison(
        baselines, n_total, final["population_coverage_pct"],
    )

    # Sensitivity analysis: coverage vs tower count
    print("  Running sensitivity analysis...")
    sens_counts = []
    sens_covs = []
    sorted_towers = all_towers.copy()
    # Sort so macro towers come first (they cover more)
    if "tower_source" in sorted_towers.columns:
        sorted_towers = sorted_towers.sort_values(
            "tower_source", ascending=True,  # gap_fill after nsga2_macro
        ).reset_index(drop=True)
    total_n = len(sorted_towers)
    step_list = sorted(set([10, 20, 30, 50, 75, 100, 125, 150, total_n]))
    for k in step_list:
        if k > total_n:
            continue
        subset = sorted_towers.iloc[:k]
        m = validator.compute_population_coverage(grid_with_demand, subset)
        sens_counts.append(k)
        sens_covs.append(m["population_coverage_pct"])
        print(f"    {k} towers -> {m['population_coverage_pct']:.1f}%")
    visuals.plot_sensitivity(sens_counts, sens_covs)

    # Airtel IoU validation
    airtel_file = config["paths"]["airtel_coverage"]
    if not os.path.exists(airtel_file):
        try:
            from download_airtel import download_airtel_coverage
            download_airtel_coverage(airtel_file)
        except Exception as e:
            print(f"  Airtel download failed: {e}")

    sim_poly = validator.generate_simulated_coverage(grid_with_demand, all_towers)
    airtel_metrics = validator.validate(sim_poly)
    print("  Airtel IoU validation:")
    for k, v in airtel_metrics.items():
        print(f"    {k}: {v}")

    # ==================================================================
    # [9/9] Save outputs
    # ==================================================================
    print("\n[9/9] Saving outputs...")
    out_dir = config["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Coverage metrics JSON
    metrics_out = {
        "district_area_km2": round(area_km2, 1),
        "grid_points": len(grid_with_demand),
        "grid_resolution_m": config["optimization"]["grid_resolution"],
        "rf_frequency_mhz": config["rf_params"]["frequency_mhz"],
        "shadow_fading_margin_db": config["optimization"].get("shadow_fading_margin_db", 8),
        "rsrp_threshold_dbm": config["optimization"].get("coverage_rsrp_threshold_dbm", -110),
        "phase1_nsga2": p1,
        "final": final,
        "baselines": baselines,
        "airtel_iou": airtel_metrics,
        "sensitivity": {"tower_counts": sens_counts, "coverage_pcts": sens_covs},
        "nsga2_towers": len(nsga2_towers),
        "gap_fill_towers": len(gap_towers),
        "total_towers": n_total,
    }
    with open(os.path.join(out_dir, "coverage_metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    # Tower locations CSV
    all_towers.to_csv(os.path.join(out_dir, "tower_locations.csv"), index=False)

    print(f"\n  Outputs saved to {out_dir}/")
    print(f"    coverage_metrics.json")
    print(f"    tower_locations.csv")
    print(f"    interactive_towers.html")
    print(f"    pareto_frontier.png")
    print(f"    sites_map.png")
    print(f"    sensitivity_analysis.png")
    print(f"    baseline_comparison.png")
    print(f"    population_heatmap.png")
    print(f"    predicted_traffic_mbps_heatmap.png")
    print("\nDone.")


if __name__ == "__main__":
    run_pipeline()
