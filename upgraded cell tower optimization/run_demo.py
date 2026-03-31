"""
run_demo.py — Single-file demo runner for the Cell Tower Optimization pipeline.

Runs end-to-end:
  1. Data preprocessing (GADM boundary + OSM features)
  2. ML demand prediction (OpenCellID labels if available, else proxy)
  3. Candidate site generation
  4. NSGA-II multi-objective optimization (macro towers)
  4.5 Multi-tier gap filling (micro + small cell)
  5. Visualization (heatmap, Pareto front, sites map, multi-tier Folium HTML)
  6. Airtel coverage validation (multi-tier)
  7. Results summary table saved to outputs/RESULTS.md
"""

import os, sys, time, warnings
os.environ["MPLBACKEND"] = "Agg"   # non-interactive backend — no display needed
warnings.filterwarnings('ignore')

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

start_total = time.time()

# ─── Setup ───────────────────────────────────────────────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import geopandas as gpd
from src.data.preprocessing import DataPreprocessor
from src.models.demand import DemandModel
from src.optimization.candidates import CandidateSiteGenerator
from src.optimization.nsga2 import MultiObjectiveOptimizer
from src.optimization.multi_tier import MultiTierOptimizer
from src.validation.validate import CoverageValidator
from src.visualization.visualize import Visualizer
from download_airtel import download_airtel_coverage

with open("src/config/config.yaml") as f:
    config = yaml.safe_load(f)

os.makedirs(config['paths']['output_dir'], exist_ok=True)
print("=" * 60)
print("  AI-Driven Cell Tower Placement — Demo Run")
print("=" * 60)
print(f"  Grid resolution : {config['optimization']['grid_resolution']} m")
print(f"  Target towers   : {config['optimization']['num_towers']}")
print(f"  NSGA-II         : pop={config['nsga2']['pop_size']}, gen={config['nsga2']['n_gen']}")
print("=" * 60)

# ─── Stage 1: Preprocessing ──────────────────────────────────────────────────
print("\n[1/7] Preprocessing: boundary + OSM features...")
t0 = time.time()
preprocessor = DataPreprocessor(config)
boundary      = preprocessor.create_nagpur_boundary()
grid          = preprocessor.generate_planning_grid(boundary)
print(f"  Grid points inside Nagpur boundary: {len(grid):,}")

buildings, roads, landuse = preprocessor.extract_osm_features(boundary)
print(f"  Buildings: {len(buildings):,}  |  Road segments: {len(roads):,}")

grid_features = preprocessor.compute_grid_features(
    grid, buildings, roads, landuse,
    config['paths']['pop_tif'], config['paths']['dem_tif']
)
print(f"  Stage 1 done in {time.time()-t0:.1f}s")

# ─── Stage 2: ML Demand Model ─────────────────────────────────────────────────
print("\n[2/7] Training demand prediction model...")
t0 = time.time()
opencellid_path = os.path.join("Data", "raw", "opencellid_nagpur.csv")
demand_model = DemandModel(
    config,
    opencellid_csv=opencellid_path if os.path.exists(opencellid_path) else None
)
demand_model.train_ml_model(grid_features)
grid_with_demand = demand_model.predict_traffic(grid_features)
grid_with_demand, high_demand, hotspots = demand_model.detect_hotspots(grid_with_demand)
print(f"  Hotspots found : {len(hotspots)}")
print(f"  Stage 2 done in {time.time()-t0:.1f}s")

# ─── Stage 3: Candidate Sites ─────────────────────────────────────────────────
print("\n[3/7] Generating candidate tower sites...")
t0 = time.time()
generator  = CandidateSiteGenerator(config)
candidates = generator.generate_candidates(grid_with_demand)
print(f"  Feasible candidate sites: {len(candidates)}")
print(f"  Stage 3 done in {time.time()-t0:.1f}s")

# ─── Stage 4: NSGA-II Optimisation ───────────────────────────────────────────
print("\n[4/7] NSGA-II multi-objective optimisation...")
t0 = time.time()
optimizer        = MultiObjectiveOptimizer(config)
res, best_towers = optimizer.run_optimization(candidates, grid_with_demand)
print(f"  Pareto-optimal solutions: {len(res.F)}")
print(f"  Stage 4 done in {time.time()-t0:.1f}s")

# ─── Stage 4.5: Multi-Tier Gap Filling ───────────────────────────────────────
print("\n[4.5/7] Multi-tier gap filling (micro + small cell)...")
t0 = time.time()

# Load Airtel coverage polygon (download if missing)
if not os.path.exists(config['paths']['airtel_coverage']):
    print("  Airtel shapefile not found — downloading...")
    download_airtel_coverage(config['paths']['airtel_coverage'])

crs = config['project']['crs']
try:
    airtel_gdf  = gpd.read_file(config['paths']['airtel_coverage']).to_crs(crs)
    airtel_poly = airtel_gdf.geometry.unary_union
    print(f"  Airtel coverage polygon loaded  ({airtel_poly.area/1e6:.0f} km²)")
except Exception as e:
    print(f"  Warning: could not load Airtel polygon ({e}) — micro fill skipped.")
    airtel_poly = boundary   # fallback: treat entire district as Airtel area

best_towers['tier'] = 'macro'
multi_opt    = MultiTierOptimizer(config)
micro_towers = multi_opt.fill_micro_gaps(grid_with_demand, best_towers, airtel_poly)
small_towers = multi_opt.fill_small_cell_gaps(grid_with_demand, best_towers, micro_towers)

all_towers_dict = {
    'macro':      best_towers,
    'micro':      micro_towers,
    'small_cell': small_towers,
}
coverage_stats = multi_opt.compute_multi_tier_coverage(grid_with_demand, all_towers_dict)
agg = coverage_stats['aggregate']
cmp = coverage_stats['comparison']
print(f"  Total AI towers : {agg['n_towers_total']}  "
      f"(macro={len(best_towers)}, micro={len(micro_towers)}, small={len(small_towers)})")
print(f"  vs Airtel       : ~{cmp['airtel_total']:,}  →  {cmp['reduction_pct']:.0f}% fewer towers")
print(f"  Pop coverage    : {agg['pop_pct']:.1f}%  |  Area: {agg['area_pct']:.1f}%")
print(f"  Stage 4.5 done in {time.time()-t0:.1f}s")

# ─── Stage 5: Visualisation ───────────────────────────────────────────────────
print("\n[5/7] Generating outputs...")
t0 = time.time()
visuals = Visualizer(config)
visuals.plot_heatmap(grid_with_demand, 'predicted_traffic_mbps',
                     'Predicted Traffic Demand (Mbps)', cmap='hot')
visuals.plot_pareto_front(res)
visuals.plot_candidates_and_selected(candidates, best_towers, grid_with_demand)
visuals.generate_html_map(all_towers_dict, grid_with_demand, boundary,
                           airtel_poly=airtel_poly, coverage_stats=coverage_stats)
print(f"  Stage 5 done in {time.time()-t0:.1f}s")

# ─── Stage 6: Airtel Validation ───────────────────────────────────────────────
print("\n[6/7] Validating against Airtel coverage map...")
t0 = time.time()
validator    = CoverageValidator(config)
sim_coverage = validator.generate_simulated_coverage_multi_tier(grid_with_demand, all_towers_dict)
val_metrics  = validator.validate_with_poly(sim_coverage, airtel_poly)
print(f"  Validation metrics: {val_metrics}")
print(f"  Stage 6 done in {time.time()-t0:.1f}s")

# ─── Stage 7: Results Summary ─────────────────────────────────────────────────
print("\n[7/7] Computing coverage statistics & writing results summary...")

# Coverage radius from COST-231 Hata
# EIRP = Tx_power + Antenna_gain - Cable_loss  (Airtel Band 3 macro: 46+18-2 = 62 dBm)
# max_path_loss = EIRP - RSRP_min               (62 - (-95) = 157 dB)
f_mhz      = config['rf_params']['frequency_mhz']
h_te       = config['rf_params']['antenna_height_m']
h_re       = config['rf_params']['receiver_height_m']
tx_dbm     = config['rf_params']['transmit_power_dbm']
ant_gain   = config['rf_params'].get('antenna_gain_dbi', 18)
cable_loss = config['rf_params'].get('cable_loss_db',    2)
rsrp_min   = config.get('thresholds', {}).get('rsrp_min_dbm', -95)

eirp_dbm  = tx_dbm + ant_gain - cable_loss
a_hre     = (1.1 * np.log10(f_mhz) - 0.7) * h_re - (1.56 * np.log10(f_mhz) - 0.8)
intercept = 46.3 + 33.9 * np.log10(f_mhz) - 13.82 * np.log10(h_te) - a_hre + 3.0
slope     = 44.9 - 6.55 * np.log10(h_te)
max_loss  = eirp_dbm - rsrp_min
d_km      = float(np.clip(10 ** ((max_loss - intercept) / slope), 0.1, 15.0))
radius_m  = d_km * 1000.0

print(f"  EIRP = {eirp_dbm:.0f} dBm  ({tx_dbm} Tx + {ant_gain} gain - {cable_loss} cable)  |  "
      f"RSRP threshold = {rsrp_min} dBm  |  Macro radius = {radius_m:.0f} m ({d_km:.2f} km)")

# Best Pareto objectives (un-flip signs)
F = res.F
best_idx = np.argmin(np.sum(
    (F - F.min(0)) / (F.max(0) - F.min(0) + 1e-9) * [0.4, 0.4, 0.2], axis=1
))
best_f = F[best_idx]
total_capacity_gbps = -best_f[1] / 1000.0
coverage_score      = -best_f[0]

elapsed = time.time() - start_total

per_tier = coverage_stats['per_tier']
macro_s  = per_tier.get('macro',      {})
micro_s  = per_tier.get('micro',      {})
small_s  = per_tier.get('small_cell', {})

summary_lines = [
    "# Cell Tower Optimization — Results Summary\n",
    f"_Generated in {elapsed/60:.1f} minutes | Grid: {config['optimization']['grid_resolution']} m | "
    f"Scope: Nagpur District (9,928 km²)_\n\n",
    "## Configuration\n",
    f"| Parameter | Value |\n|-----------|-------|\n",
    f"| Frequency | {f_mhz} MHz |\n",
    f"| Tx Power (macro) | {tx_dbm} dBm |\n",
    f"| Antenna Height (macro) | {h_te} m |\n",
    f"| RSRP Threshold | {rsrp_min} dBm |\n",
    f"| Macro Coverage Radius (COST-231 Hata) | **{radius_m:.0f} m ({d_km:.2f} km)** |\n\n",
    "## Multi-Tier Tower Deployment vs Airtel\n",
    "| Tier | AI Count | Airtel Equiv | Area Coverage | Pop Coverage | Demand Coverage |\n",
    "|------|----------|--------------|--------------|-------------|----------------|\n",
    f"| Macro Cell (35 m) | **{macro_s.get('n_towers',0)}** | ~1,200 | "
    f"{macro_s.get('area_pct',0):.1f}% | {macro_s.get('pop_pct',0):.1f}% | {macro_s.get('demand_pct',0):.1f}% |\n",
    f"| Micro Cell (12 m) | **{micro_s.get('n_towers',0)}** | — | "
    f"{micro_s.get('area_pct',0):.1f}% | {micro_s.get('pop_pct',0):.1f}% | {micro_s.get('demand_pct',0):.1f}% |\n",
    f"| Small Cell (6 m) | **{small_s.get('n_towers',0)}** | — | "
    f"{small_s.get('area_pct',0):.1f}% | {small_s.get('pop_pct',0):.1f}% | {small_s.get('demand_pct',0):.1f}% |\n",
    f"| **TOTAL** | **{agg['n_towers_total']}** | **~{cmp['airtel_total']:,}** | "
    f"**{agg['area_pct']:.1f}%** | **{agg['pop_pct']:.1f}%** | **{agg['demand_pct']:.1f}%** |\n\n",
    f"> **AI achieves {cmp['reduction_pct']:.0f}% fewer towers than Airtel** "
    f"({agg['n_towers_total']} vs ~{cmp['airtel_total']:,}) "
    f"while covering {agg['pop_pct']:.1f}% of the district population "
    f"at RSRP ≥ {rsrp_min} dBm (Airtel indoor planning threshold).\n\n",
    "## NSGA-II Macro Optimization Results\n",
    f"| Metric | AI Solution |\n|--------|-------------|\n",
    f"| Macro towers selected | **{len(best_towers)}** |\n",
    f"| Grid points analysed | {len(grid_with_demand):,} |\n",
    f"| Candidate sites evaluated | {len(candidates)} |\n",
    f"| Pareto-optimal configurations | {len(res.F)} |\n",
    f"| Estimated network capacity | **{total_capacity_gbps:.2f} Gbps** |\n\n",
    "## Validation vs Airtel Coverage Map\n",
    f"| Metric | Value |\n|--------|-------|\n",
]
for k, v in val_metrics.items():
    summary_lines.append(f"| {k} | {v} |\n")

summary_lines += [
    "\n## Label Source (demand model)\n",
    f"- {demand_model.label_source}\n\n",
    "## Output Files\n",
    "| File | Description |\n|------|-------------|\n",
    "| `outputs/interactive_towers.html` | Interactive Folium map — 4 toggleable layers |\n",
    "| `outputs/pareto_frontier.png` | NSGA-II Pareto trade-off chart |\n",
    "| `outputs/predicted_traffic_mbps_heatmap.png` | Spatial demand forecast |\n",
    "| `outputs/sites_map.png` | Candidate vs selected macro tower locations |\n",
]

summary_text = "".join(summary_lines)
with open(os.path.join(config['paths']['output_dir'], "RESULTS.md"), "w") as f:
    f.write(summary_text)

# Print table to console
print("\n" + "=" * 65)
print("  RESULTS SUMMARY — Nagpur District (9,928 km²)")
print("=" * 65)
print(f"  Macro coverage radius (COST-231 Hata) : {radius_m:.0f} m  ({d_km:.2f} km)")
print(f"  {'Tier':<18} {'AI':>6} {'Airtel':>8} {'Area%':>7} {'Pop%':>7}")
print(f"  {'-'*18} {'-'*6} {'-'*8} {'-'*7} {'-'*7}")
print(f"  {'Macro (35 m)':<18} {macro_s.get('n_towers',0):>6} {'~1,200':>8} "
      f"{macro_s.get('area_pct',0):>6.1f}% {macro_s.get('pop_pct',0):>6.1f}%")
print(f"  {'Micro (12 m)':<18} {micro_s.get('n_towers',0):>6} {'—':>8} "
      f"{micro_s.get('area_pct',0):>6.1f}% {micro_s.get('pop_pct',0):>6.1f}%")
print(f"  {'Small Cell (6 m)':<18} {small_s.get('n_towers',0):>6} {'—':>8} "
      f"{small_s.get('area_pct',0):>6.1f}% {small_s.get('pop_pct',0):>6.1f}%")
print(f"  {'TOTAL':<18} {agg['n_towers_total']:>6} {'~1,200':>8} "
      f"{agg['area_pct']:>6.1f}% {agg['pop_pct']:>6.1f}%")
print("=" * 65)
print(f"  AI uses {cmp['reduction_pct']:.0f}% FEWER towers than Airtel  "
      f"({agg['n_towers_total']} vs ~{cmp['airtel_total']:,})")
print(f"  Estimated network capacity : {total_capacity_gbps:.3f} Gbps")
print(f"  Pareto solutions found     : {len(res.F)}")
print("-" * 65)
print("  Airtel IoU validation           :", val_metrics.get('IoU', 'N/A'))
print("  Airtel match %                  :", val_metrics.get('Match_Percentage', 'N/A'))
print("-" * 65)
print(f"  Label source: {demand_model.label_source}")
print("=" * 65)
print(f"\n  Total runtime: {elapsed/60:.1f} minutes")
print(f"\n  Outputs saved to: {os.path.abspath(config['paths']['output_dir'])}/")
print("    - interactive_towers.html  (open in browser — 4 toggleable layers)")
print("    - pareto_frontier.png")
print("    - predicted_traffic_mbps_heatmap.png")
print("    - sites_map.png")
print("    - RESULTS.md")
