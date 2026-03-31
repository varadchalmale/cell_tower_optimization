"""
holdout_test.py — Hold-out Tower Prediction Validation
=======================================================

Methodology
-----------
1. Load real OpenCellID LTE towers for Nagpur (or synthetic demand-weighted towers
   if no real data is available — documented clearly).
2. Randomly split towers 80 / 20 (train / holdout) using a fixed seed.
3. Train the demand model using *only* the 80 % training towers (replicating what
   the system would know in a greenfield deployment scenario where 20 % of the
   network has not yet been built).
4. Run NSGA-II optimization on the demand surface derived from training towers.
5. For each predicted tower location, find the nearest real holdout tower and
   record the distance.
6. Report:
   - Mean / median nearest-neighbour distance (lower = better prediction)
   - % of predicted towers within 500 m of a real holdout tower
   - % of predicted towers within 1 km of a real holdout tower

Why this matters
----------------
If the optimizer is just memorising a formula it won't generalise: its placements
will not correlate with where real-world operators have deployed infrastructure.
A mean distance below ~1.5 km (half the 3.45 km coverage radius) indicates the
system is capturing meaningful demand signals.

Usage
-----
    cd "upgraded cell tower optimization"
    python validation/holdout_test.py

Optional:
    OPENCELLID_CSV=Data/raw/opencellid_nagpur.csv python validation/holdout_test.py
"""

import os, sys, warnings
os.environ["MPLBACKEND"] = "Agg"
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment

# ── Path setup ────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

from src.data.preprocessing import DataPreprocessor
from src.models.demand import DemandModel
from src.optimization.candidates import CandidateSiteGenerator
from src.optimization.nsga2 import MultiObjectiveOptimizer

RESULTS_DIR = os.path.join(script_dir, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
with open(os.path.join(project_dir, 'src', 'config', 'config.yaml')) as f:
    config = yaml.safe_load(f)

# Speed up for validation: use smaller NSGA-II params
config['nsga2']['pop_size'] = 30
config['nsga2']['n_gen']    = 25
config['optimization']['num_towers'] = 30   # predict 30 towers to match holdout size

os.chdir(project_dir)

print("=" * 65)
print("  VALIDATION 1: Hold-out Tower Prediction Test")
print("=" * 65)

# ─── Stage 1: Load / generate real tower positions ─────────────────────────
opencellid_path = os.environ.get(
    'OPENCELLID_CSV',
    os.path.join(project_dir, 'Data', 'raw', 'opencellid_nagpur.csv')
)

crs = config['project']['crs']

print("\n[1/5] Loading tower ground-truth data...")
real_towers_utm = None   # will be set below

if os.path.exists(opencellid_path):
    try:
        oc = pd.read_csv(opencellid_path)
        if 'radio' in oc.columns:
            oc = oc[oc['radio'].str.upper().isin(['LTE', '4G', 'NR'])]
        lon_col = 'lon'  if 'lon'  in oc.columns else 'longitude'
        lat_col = 'lat'  if 'lat'  in oc.columns else 'latitude'
        oc_gdf = gpd.GeoDataFrame(
            oc, geometry=gpd.points_from_xy(oc[lon_col], oc[lat_col]), crs="EPSG:4326"
        ).to_crs(crs)
        real_towers_utm = np.column_stack([
            oc_gdf.geometry.x.values,
            oc_gdf.geometry.y.values
        ])
        print(f"  Loaded {len(real_towers_utm):,} real LTE towers from OpenCellID.")
        data_source = "OpenCellID (real data)"
    except Exception as e:
        print(f"  Warning: Could not load OpenCellID ({e}). Generating synthetic towers.")

if real_towers_utm is None:
    print("  OpenCellID CSV not found — generating SYNTHETIC demand-weighted towers.")
    print("  NOTE: These are NOT real towers. Results are indicative only.")
    data_source = "SYNTHETIC demand-weighted (no real data)"

    # Build grid first, then generate synthetic towers from demand
    preprocessor = DataPreprocessor(config)
    boundary = preprocessor.create_nagpur_boundary()
    grid     = preprocessor.generate_planning_grid(boundary)
    buildings, roads, landuse = preprocessor.extract_osm_features(boundary)
    grid_features = preprocessor.compute_grid_features(
        grid, buildings, roads, landuse,
        config['paths']['pop_tif'], config['paths']['dem_tif']
    )
    demand_model_full = DemandModel(config)
    demand_model_full.train_ml_model(grid_features)
    grid_demand = demand_model_full.predict_traffic(grid_features)

    # Sample synthetic towers proportional to demand
    rng = np.random.default_rng(config['project']['seed'])
    weights = grid_demand['predicted_traffic_mbps'].values
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()
    n_synthetic = 150  # synthetic "real" network
    chosen_idx = rng.choice(len(grid_demand), size=n_synthetic, replace=False, p=weights)
    real_towers_utm = grid_demand[['x', 'y']].values[chosen_idx]
    print(f"  Generated {len(real_towers_utm)} synthetic reference towers from demand distribution.")

# ─── Stage 2: 80 / 20 train-holdout split ─────────────────────────────────
print("\n[2/5] Splitting towers 80/20 (train / holdout)...")
rng_split = np.random.default_rng(config['project']['seed'] + 1)
n_total   = len(real_towers_utm)
indices   = np.arange(n_total)
rng_split.shuffle(indices)

split = int(0.8 * n_total)
train_idx   = indices[:split]
holdout_idx = indices[split:]

train_towers   = real_towers_utm[train_idx]
holdout_towers = real_towers_utm[holdout_idx]

print(f"  Total towers : {n_total}")
print(f"  Training set : {len(train_towers)} towers (80 %)")
print(f"  Holdout set  : {len(holdout_towers)} towers (20 %)  ← hidden from optimizer")

# ─── Stage 3: Build demand surface from training towers only ───────────────
print("\n[3/5] Building demand surface from training towers only...")

# Save training-towers-only CSV so DemandModel can read it
train_df = pd.DataFrame(train_towers, columns=['x_utm', 'y_utm'])
# Convert back to WGS84 for the CSV (DemandModel expects lon/lat)
train_gdf = gpd.GeoDataFrame(
    train_df,
    geometry=gpd.points_from_xy(train_df['x_utm'], train_df['y_utm']),
    crs=crs
).to_crs("EPSG:4326")
train_df['lon'] = train_gdf.geometry.x.values
train_df['lat'] = train_gdf.geometry.y.values
train_df['radio'] = 'LTE'

train_csv_path = os.path.join(RESULTS_DIR, '_train_towers_tmp.csv')
train_df[['radio', 'lon', 'lat']].to_csv(train_csv_path, index=False)

# Run preprocessing (uses OSM cache if available)
preprocessor = DataPreprocessor(config)
boundary      = preprocessor.create_nagpur_boundary()
grid          = preprocessor.generate_planning_grid(boundary)
buildings, roads, landuse = preprocessor.extract_osm_features(boundary)
grid_features = preprocessor.compute_grid_features(
    grid, buildings, roads, landuse,
    config['paths']['pop_tif'], config['paths']['dem_tif']
)

demand_model = DemandModel(config, opencellid_csv=train_csv_path)
demand_model.train_ml_model(grid_features)
grid_demand  = demand_model.predict_traffic(grid_features)
grid_demand, _, _ = demand_model.detect_hotspots(grid_demand)

os.remove(train_csv_path)   # clean up temp file

# ─── Stage 4: NSGA-II optimization ────────────────────────────────────────
print("\n[4/5] Running NSGA-II on demand surface (without holdout towers)...")
generator       = CandidateSiteGenerator(config)
candidates      = generator.generate_candidates(grid_demand)
optimizer       = MultiObjectiveOptimizer(config)
res, best_towers = optimizer.run_optimization(candidates, grid_demand)

predicted_coords = best_towers[['x', 'y']].values
print(f"  Predicted towers : {len(predicted_coords)}")

# ─── Stage 5: Evaluation ──────────────────────────────────────────────────
print("\n[5/5] Measuring prediction accuracy against holdout towers...")

# For each predicted tower, find nearest holdout tower
holdout_tree = KDTree(holdout_towers)
dists_to_holdout, _ = holdout_tree.query(predicted_coords)

# For each holdout tower, find nearest predicted tower (reverse)
pred_tree = KDTree(predicted_coords)
dists_to_pred, _ = pred_tree.query(holdout_towers)

# Hungarian matching: minimum-cost bipartite assignment (as in literature)
n_p = len(predicted_coords)
n_h = len(holdout_towers)
n_match = min(n_p, n_h)
cost_matrix = np.linalg.norm(
    predicted_coords[:n_match, None, :] - holdout_towers[None, :n_match, :], axis=2
)
row_ind, col_ind = linear_sum_assignment(cost_matrix)
hungarian_dists  = cost_matrix[row_ind, col_ind]

# Coverage radius from config (COST-231 Hata result)
rf        = config['rf_params']
f_mhz     = rf['frequency_mhz'];   h_te = rf['antenna_height_m']
h_re      = rf.get('receiver_height_m', 1.5)
tx_dbm    = rf['transmit_power_dbm']
ant_gain  = rf.get('antenna_gain_dbi', 18)
cable_loss = rf.get('cable_loss_db', 2)
rsrp_min  = config.get('thresholds', {}).get('rsrp_min_dbm', -95)
eirp      = tx_dbm + ant_gain - cable_loss
a_hre     = (1.1*np.log10(f_mhz) - 0.7)*h_re - (1.56*np.log10(f_mhz) - 0.8)
intercept = 46.3 + 33.9*np.log10(f_mhz) - 13.82*np.log10(h_te) - a_hre + 3.0
slope     = 44.9 - 6.55*np.log10(h_te)
d_km      = float(np.clip(10**((eirp - rsrp_min - intercept)/slope), 0.1, 15.0))
radius_m  = d_km * 1000.0

within_500m = float(np.mean(dists_to_holdout <= 500)) * 100
within_1km  = float(np.mean(dists_to_holdout <= 1000)) * 100
within_radius = float(np.mean(dists_to_holdout <= radius_m)) * 100

results = {
    "Data source"              : data_source,
    "Training towers"          : len(train_towers),
    "Holdout towers (hidden)"  : len(holdout_towers),
    "Predicted towers"         : len(predicted_coords),
    "Mean dist → nearest holdout (m)"  : f"{dists_to_holdout.mean():.1f}",
    "Median dist → nearest holdout (m)": f"{np.median(dists_to_holdout):.1f}",
    "Max dist → nearest holdout (m)"   : f"{dists_to_holdout.max():.1f}",
    "% predicted within 500 m of real" : f"{within_500m:.1f}%",
    "% predicted within 1 km of real"  : f"{within_1km:.1f}%",
    f"% predicted within {radius_m:.0f}m (coverage radius)": f"{within_radius:.1f}%",
    "Hungarian match mean dist (m)"    : f"{hungarian_dists.mean():.1f}",
    "Hungarian match median dist (m)"  : f"{np.median(hungarian_dists):.1f}",
}

print("\n" + "=" * 65)
print("  HOLD-OUT TEST RESULTS")
print("=" * 65)
for k, v in results.items():
    print(f"  {k:<45} {v}")
print("=" * 65)

# ── Interpretation ────────────────────────────────────────────────────────
print("\n  Interpretation:")
mean_d = dists_to_holdout.mean()
half_r = radius_m / 2
if mean_d <= half_r:
    print(f"  ✓ Mean distance ({mean_d:.0f} m) < half the coverage radius ({half_r:.0f} m)")
    print("    → Predicted towers are within the same coverage cell as real towers.")
    print("    → The demand model is capturing meaningful spatial patterns.")
else:
    print(f"  ✗ Mean distance ({mean_d:.0f} m) > half the coverage radius ({half_r:.0f} m)")
    print("    → Predicted locations differ significantly from real network.")
    print("    → Consider: richer demand features (time-of-day, POI density).")

# ── Save figure ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: scatter map
ax = axes[0]
ax.scatter(train_towers[:, 0],   train_towers[:, 1],   c='steelblue', s=15,
           alpha=0.5, label=f'Training towers (n={len(train_towers)})')
ax.scatter(holdout_towers[:, 0], holdout_towers[:, 1], c='green', s=40,
           marker='*', label=f'Holdout towers (n={len(holdout_towers)})')
ax.scatter(predicted_coords[:, 0], predicted_coords[:, 1], c='red', s=60,
           marker='^', label=f'Predicted (NSGA-II) (n={len(predicted_coords)})')
ax.set_title('Tower Placement: Predicted vs Holdout')
ax.legend(fontsize=8)
ax.set_xlabel('Easting (m, EPSG:32644)')
ax.set_ylabel('Northing (m, EPSG:32644)')
ax.axis('equal')

# Right: CDF of distances
ax2 = axes[1]
sorted_dists = np.sort(dists_to_holdout)
cdf = np.arange(1, len(sorted_dists) + 1) / len(sorted_dists)
ax2.plot(sorted_dists / 1000, cdf * 100, color='red', linewidth=2,
         label='Predicted → nearest holdout')
sorted_rev = np.sort(dists_to_pred)
cdf_rev = np.arange(1, len(sorted_rev) + 1) / len(sorted_rev)
ax2.plot(sorted_rev / 1000, cdf_rev * 100, color='steelblue', linewidth=2,
         linestyle='--', label='Holdout → nearest predicted')
ax2.axvline(x=0.5,      color='gray', linestyle=':', label='500 m threshold')
ax2.axvline(x=1.0,      color='black', linestyle=':', label='1 km threshold')
ax2.axvline(x=d_km,     color='orange', linestyle='-.', label=f'Coverage radius ({d_km:.1f} km)')
ax2.set_xlabel('Distance to nearest tower (km)')
ax2.set_ylabel('Cumulative % of predicted towers')
ax2.set_title('CDF: Distance from Predicted to Nearest Real Tower')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, max(dists_to_holdout.max(), dists_to_pred.max()) / 1000 * 1.1)
ax2.set_ylim(0, 105)

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, 'holdout_test.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Figure saved → {fig_path}")

# Save CSV summary
summary_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
csv_path = os.path.join(RESULTS_DIR, 'holdout_test_results.csv')
summary_df.to_csv(csv_path, index=False)
print(f"  Results saved → {csv_path}")
print("\nDone.")
