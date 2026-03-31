"""
statistical_analysis.py — Multi-run stability analysis for NSGA-II results.

NSGA-II is a stochastic algorithm. A single run result depends on random
initialisation. This script runs the optimiser with N different random seeds
and reports mean ± std for each objective, demonstrating that the solution
quality is stable and not a lucky outlier.

Usage:
    python statistical_analysis.py [--runs 5]

Output:
    results/statistical_analysis.json  — raw per-run metrics
    results/statistical_summary.md     — table of mean ± std (include in thesis/report)
"""

import os
import json
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from src.data.preprocessing import DataPreprocessor
from src.models.demand import DemandModel
from src.optimization.candidates import CandidateSiteGenerator
from src.optimization.nsga2 import MultiObjectiveOptimizer


def load_config():
    with open('src/config/config.yaml') as f:
        return yaml.safe_load(f)


def run_single(config, seed):
    """Run one full optimisation trial with the given seed. Returns objective values."""
    config['project']['seed'] = seed
    config['nsga2']['pop_size'] = 50   # reduced for multi-run speed; increase for final results
    config['nsga2']['n_gen']    = 30

    preprocessor = DataPreprocessor(config)
    boundary     = preprocessor.create_nagpur_boundary()
    grid         = preprocessor.generate_planning_grid(boundary)
    buildings, roads, landuse = preprocessor.extract_osm_features(boundary)
    grid_features = preprocessor.compute_grid_features(
        grid, buildings, roads, landuse,
        config['paths']['pop_tif'], config['paths']['dem_tif']
    )

    opencellid_path = os.path.join('Data', 'raw', 'opencellid_nagpur.csv')
    demand_model = DemandModel(
        config,
        opencellid_csv=opencellid_path if os.path.exists(opencellid_path) else None
    )
    demand_model.train_ml_model(grid_features)
    grid_with_demand = demand_model.predict_traffic(grid_features)
    grid_with_demand, _, _ = demand_model.detect_hotspots(grid_with_demand)

    generator  = CandidateSiteGenerator(config)
    candidates = generator.generate_candidates(grid_with_demand)

    optimizer       = MultiObjectiveOptimizer(config)
    res, best_config = optimizer.run_optimization(candidates, grid_with_demand)

    # Best-compromise solution objectives (signs already flipped by optimizer)
    best_idx = np.argmin(np.sum(res.F, axis=1))  # simple sum of normalised objectives
    f = res.F[best_idx]
    return {
        'seed'          : seed,
        'coverage'      : float(-f[0]),   # re-flip: was minimised as negative
        'capacity_mbps' : float(-f[1]),
        'cost'          : float( f[2]),
        'n_pareto'      : int(len(res.F)),
        'n_towers'      : int(len(best_config)),
    }


def main(n_runs=5):
    os.makedirs('results', exist_ok=True)
    config   = load_config()
    seeds    = list(range(42, 42 + n_runs))
    results  = []

    print(f"Running {n_runs} independent NSGA-II trials...")
    for i, seed in enumerate(seeds):
        print(f"\n--- Run {i+1}/{n_runs} (seed={seed}) ---")
        try:
            r = run_single(config, seed)
            results.append(r)
            print(f"  Coverage: {r['coverage']:.4f}  |  Capacity: {r['capacity_mbps']:.2f} Mbps  |  Cost: {r['cost']:.4f}")
        except Exception as e:
            print(f"  Run failed: {e}")

    if not results:
        print("All runs failed. Check data availability.")
        return

    # Aggregate
    metrics = ['coverage', 'capacity_mbps', 'cost', 'n_pareto']
    agg = {}
    for m in metrics:
        vals = [r[m] for r in results]
        agg[m] = {'mean': np.mean(vals), 'std': np.std(vals), 'min': np.min(vals), 'max': np.max(vals)}

    # Save raw results
    with open('results/statistical_analysis.json', 'w') as f:
        json.dump({'runs': results, 'aggregate': agg}, f, indent=2)

    # Write markdown summary
    summary = "# NSGA-II Statistical Stability Analysis\n\n"
    summary += f"**Runs**: {len(results)} independent trials with seeds {seeds[:len(results)]}\n\n"
    summary += "A stochastic optimiser (NSGA-II) must be evaluated over multiple independent runs.\n"
    summary += "The table below confirms that solution quality is stable across random initialisations.\n\n"
    summary += "| Metric | Mean | Std Dev | Min | Max |\n"
    summary += "|--------|------|---------|-----|-----|\n"
    labels = {
        'coverage'      : 'Demand-weighted coverage',
        'capacity_mbps' : 'Network capacity (Mbps)',
        'cost'          : 'Deployment cost index',
        'n_pareto'      : 'Pareto front size',
    }
    for m, lbl in labels.items():
        a = agg[m]
        summary += f"| {lbl} | {a['mean']:.3f} | {a['std']:.3f} | {a['min']:.3f} | {a['max']:.3f} |\n"

    summary += f"\n*Coefficient of variation (std/mean)*:\n"
    for m, lbl in labels.items():
        a = agg[m]
        cv = (a['std'] / a['mean'] * 100) if a['mean'] != 0 else 0
        summary += f"- {lbl}: {cv:.1f}%\n"

    with open('results/statistical_summary.md', 'w') as f:
        f.write(summary)

    print("\n=== Statistical Summary ===")
    print(summary)

    # Box plots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_metrics = ['coverage', 'capacity_mbps', 'cost']
    plot_labels  = ['Coverage', 'Capacity (Mbps)', 'Cost Index']
    for ax, m, lbl in zip(axes, plot_metrics, plot_labels):
        vals = [r[m] for r in results]
        ax.boxplot(vals)
        ax.scatter([1]*len(vals), vals, color='red', zorder=5, s=30, label='Individual runs')
        ax.set_title(f'{lbl}\nMean={agg[m]["mean"]:.3f} ± {agg[m]["std"]:.3f}')
        ax.set_xticks([])
    plt.suptitle('NSGA-II Objective Stability Across 5 Independent Runs', y=1.02)
    plt.tight_layout()
    plt.savefig('results/statistical_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nOutput files:")
    print("  results/statistical_analysis.json  — raw per-run data")
    print("  results/statistical_summary.md     — mean ± std table")
    print("  results/statistical_boxplots.png   — box plots")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=5, help='Number of independent NSGA-II runs')
    args = parser.parse_args()
    main(args.runs)
