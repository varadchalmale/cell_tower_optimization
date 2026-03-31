"""
census_crosscheck.py — Coverage vs Census Population Cross-check
================================================================

Methodology
-----------
Divides Nagpur district into a regular grid of "pseudo-wards" (proxy for
Census of India administrative wards, which require an official shapefile).
For each pseudo-ward:
  - Computes the WorldPop / proxy population fraction
  - Counts how many AI-optimized towers fall inside the ward
  - Measures the fraction of the ward covered by tower signals (COST-231 Hata radius)

Then computes:
  - Pearson correlation between ward population fraction and tower density
  - Scatter plot with regression line
  - Gini coefficient of tower distribution (0 = perfectly equal, 1 = all towers in one ward)

Interpretation
--------------
A correlation close to +1 means the optimizer places more towers in densely
populated wards — exactly what is expected from a demand-driven system.
A Gini coefficient near 0 means towers are distributed equitably.

If a real Census of India ward shapefile is available (districts/nagpur_wards.shp),
the script automatically uses it instead of the pseudo-ward grid.

Usage
-----
    cd "upgraded cell tower optimization"
    python validation/census_crosscheck.py
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
from scipy import stats
from shapely.geometry import box, Point

# ── Path setup ────────────────────────────────────────────────────────────────
script_dir  = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

from src.data.preprocessing import DataPreprocessor
from src.models.demand import DemandModel
from src.optimization.candidates import CandidateSiteGenerator
from src.optimization.nsga2 import MultiObjectiveOptimizer

RESULTS_DIR = os.path.join(script_dir, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

with open(os.path.join(project_dir, 'src', 'config', 'config.yaml')) as f:
    config = yaml.safe_load(f)

# Lighter NSGA-II for validation speed
config['nsga2']['pop_size'] = 30
config['nsga2']['n_gen']    = 25

os.chdir(project_dir)

print("=" * 65)
print("  VALIDATION 2: Coverage vs Census Population Cross-check")
print("=" * 65)

# ─── Stage 1: Build grid + demand surface ─────────────────────────────────
print("\n[1/4] Building demand surface...")
preprocessor = DataPreprocessor(config)
boundary     = preprocessor.create_nagpur_boundary()
grid         = preprocessor.generate_planning_grid(boundary)
buildings, roads, landuse = preprocessor.extract_osm_features(boundary)
grid_features = preprocessor.compute_grid_features(
    grid, buildings, roads, landuse,
    config['paths']['pop_tif'], config['paths']['dem_tif']
)

opencellid_path = os.path.join(project_dir, 'Data', 'raw', 'opencellid_nagpur.csv')
demand_model = DemandModel(
    config,
    opencellid_csv=opencellid_path if os.path.exists(opencellid_path) else None
)
demand_model.train_ml_model(grid_features)
grid_demand  = demand_model.predict_traffic(grid_features)
grid_demand, _, _ = demand_model.detect_hotspots(grid_demand)

# ─── Stage 2: Run optimizer ────────────────────────────────────────────────
print("\n[2/4] Running NSGA-II optimization...")
generator       = CandidateSiteGenerator(config)
candidates      = generator.generate_candidates(grid_demand)
optimizer       = MultiObjectiveOptimizer(config)
res, best_towers = optimizer.run_optimization(candidates, grid_demand)
print(f"  Selected {len(best_towers)} towers from {len(candidates)} candidates.")

# ─── Stage 3: Load or generate ward boundaries ───────────────────────────
print("\n[3/4] Setting up ward boundaries...")
crs = config['project']['crs']

real_ward_path = os.path.join(project_dir, 'Data', 'raw', 'nagpur_wards.shp')
osm_ward_path  = os.path.join(project_dir, 'Data', 'raw', 'nagpur_admin_wards.gpkg')

wards_gdf = None

for wpath in [real_ward_path, osm_ward_path]:
    if os.path.exists(wpath):
        try:
            wards_gdf = gpd.read_file(wpath).to_crs(crs)
            print(f"  Loaded real ward boundaries: {len(wards_gdf)} wards from {os.path.basename(wpath)}")
            ward_source = "Census of India ward shapefile"
            break
        except Exception as e:
            print(f"  Warning: Could not read {wpath}: {e}")

if wards_gdf is None:
    # Try fetching from OSM (admin_level=8 or 9 = municipal wards in India)
    print("  Attempting to fetch ward boundaries from OpenStreetMap (admin_level=10)...")
    try:
        import osmnx as ox
        boundary_wgs = gpd.GeoSeries([boundary], crs=crs).to_crs("EPSG:4326")[0]
        tags = {"admin_level": ["8", "9", "10"], "boundary": "administrative"}
        wards_osm = ox.features_from_polygon(boundary_wgs, tags)
        if len(wards_osm) > 0:
            wards_gdf = wards_osm[wards_osm.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
            wards_gdf = wards_gdf.to_crs(crs).reset_index(drop=True)
            wards_gdf['ward_id'] = np.arange(len(wards_gdf))
            print(f"  Fetched {len(wards_gdf)} administrative polygons from OSM.")
            ward_source = "OSM administrative boundaries"
        else:
            raise ValueError("No admin polygons returned")
    except Exception as e:
        print(f"  OSM ward fetch failed ({e}). Using pseudo-ward grid.")
        wards_gdf  = None
        ward_source = None

if wards_gdf is None or len(wards_gdf) < 5:
    # Fallback: divide the Nagpur boundary bounding box into an N×N grid of pseudo-wards
    N = 8  # 8×8 = 64 pseudo-wards
    minx, miny, maxx, maxy = boundary.bounds
    dx = (maxx - minx) / N
    dy = (maxy - miny) / N
    cells = []
    for i in range(N):
        for j in range(N):
            cell = box(minx + i*dx, miny + j*dy, minx + (i+1)*dx, miny + (j+1)*dy)
            inter = cell.intersection(boundary)
            if not inter.is_empty and inter.area > 0:
                cells.append({'geometry': inter, 'ward_id': len(cells)})
    wards_gdf    = gpd.GeoDataFrame(cells, crs=crs)
    ward_source  = f"Pseudo-ward grid ({N}×{N}) — proxy for Census wards"
    print(f"  Generated {len(wards_gdf)} pseudo-wards ({N}×{N} grid over Nagpur boundary).")
    print(f"  NOTE: These are spatial proxies, not official Census of India ward boundaries.")

wards_gdf = wards_gdf.reset_index(drop=True)
if 'ward_id' not in wards_gdf.columns:
    wards_gdf['ward_id'] = wards_gdf.index

# ─── Stage 4: Per-ward analysis ───────────────────────────────────────────
print("\n[4/4] Computing per-ward population and tower metrics...")

# Population per ward: sum grid points falling inside each ward
grid_gdf = gpd.GeoDataFrame(
    grid_demand,
    geometry=gpd.points_from_xy(grid_demand['x'], grid_demand['y']),
    crs=crs
)
joined_pop = gpd.sjoin(grid_gdf[['geometry', 'population', 'predicted_traffic_mbps']],
                       wards_gdf[['geometry', 'ward_id']],
                       how='left', predicate='within')
ward_pop = joined_pop.groupby('ward_id')['population'].sum()
ward_demand = joined_pop.groupby('ward_id')['predicted_traffic_mbps'].sum()

# Towers per ward: count AI towers inside each ward
tower_gdf = gpd.GeoDataFrame(
    best_towers,
    geometry=gpd.points_from_xy(best_towers['x'], best_towers['y']),
    crs=crs
)
joined_towers = gpd.sjoin(tower_gdf[['geometry']], wards_gdf[['geometry', 'ward_id']],
                           how='left', predicate='within')
ward_towers = joined_towers.groupby('ward_id').size().reindex(wards_gdf['ward_id'], fill_value=0)

# Coverage fraction per ward (% of grid points within COST-231 radius of any tower)
rf         = config['rf_params']
f_mhz      = rf['frequency_mhz'];  h_te = rf['antenna_height_m']
h_re       = rf.get('receiver_height_m', 1.5)
tx_dbm     = rf['transmit_power_dbm']
ant_gain   = rf.get('antenna_gain_dbi', 18)
cable_loss = rf.get('cable_loss_db', 2)
rsrp_min   = config.get('thresholds', {}).get('rsrp_min_dbm', -95)
eirp       = tx_dbm + ant_gain - cable_loss
a_hre_     = (1.1*np.log10(f_mhz) - 0.7)*h_re - (1.56*np.log10(f_mhz) - 0.8)
intercept_ = 46.3 + 33.9*np.log10(f_mhz) - 13.82*np.log10(h_te) - a_hre_ + 3.0
slope_     = 44.9 - 6.55*np.log10(h_te)
d_km_      = float(np.clip(10**((eirp - rsrp_min - intercept_) / slope_), 0.1, 15.0))
radius_m   = d_km_ * 1000.0

gx  = grid_demand[['x', 'y']].values
tx  = best_towers[['x', 'y']].values
covered = np.zeros(len(grid_demand), dtype=bool)
for tpt in tx:
    covered |= (np.sqrt(((gx - tpt)**2).sum(axis=1)) <= radius_m)
grid_demand['covered'] = covered

grid_gdf['covered'] = covered
joined_cov = gpd.sjoin(grid_gdf[['geometry', 'covered', 'population']],
                        wards_gdf[['geometry', 'ward_id']],
                        how='left', predicate='within')
ward_cov_pct = (joined_cov.groupby('ward_id')['covered'].mean() * 100).reindex(
    wards_gdf['ward_id'], fill_value=0
)

# ── Build summary dataframe ────────────────────────────────────────────────
ward_summary = pd.DataFrame({
    'ward_id'          : wards_gdf['ward_id'].values,
    'population'       : ward_pop.reindex(wards_gdf['ward_id'], fill_value=0).values,
    'demand_mbps'      : ward_demand.reindex(wards_gdf['ward_id'], fill_value=0).values,
    'n_towers'         : ward_towers.reindex(wards_gdf['ward_id'], fill_value=0).values,
    'coverage_pct'     : ward_cov_pct.reindex(wards_gdf['ward_id'], fill_value=0).values,
})

total_pop  = ward_summary['population'].sum()
ward_summary['pop_fraction_%'] = ward_summary['population'] / (total_pop + 1e-9) * 100
ward_summary['tower_fraction_%'] = ward_summary['n_towers'] / (ward_summary['n_towers'].sum() + 1e-9) * 100

# Filter to wards with at least some population (to avoid empty wards outside boundary)
ward_valid = ward_summary[ward_summary['population'] > 0].copy()

# Pearson correlation
r_towers_vs_pop, p_towers_vs_pop = stats.pearsonr(
    ward_valid['population'], ward_valid['n_towers']
)
r_cov_vs_pop, p_cov_vs_pop = stats.pearsonr(
    ward_valid['population'], ward_valid['coverage_pct']
)

# Gini coefficient for tower distribution
def gini(arr):
    a = np.sort(arr.astype(float))
    n = len(a)
    return (2 * np.sum((np.arange(1, n+1)) * a) - (n+1) * a.sum()) / (n * a.sum() + 1e-9)

tower_gini = gini(ward_valid['n_towers'].values)

print("\n" + "=" * 65)
print("  CENSUS CROSS-CHECK RESULTS")
print("=" * 65)
print(f"  Ward source           : {ward_source}")
print(f"  Number of wards       : {len(ward_valid)}")
print(f"  Total population      : {int(total_pop):,}")
print(f"  AI towers placed      : {int(ward_summary['n_towers'].sum())}")
print(f"")
print(f"  Pearson r (towers vs population) : {r_towers_vs_pop:.4f}  (p={p_towers_vs_pop:.4f})")
print(f"  Pearson r (coverage% vs population): {r_cov_vs_pop:.4f}  (p={p_cov_vs_pop:.4f})")
print(f"  Gini coefficient (tower distribution): {tower_gini:.4f}")
print(f"    (0 = perfectly equal,  1 = all towers in one ward)")
print("=" * 65)

# Interpretation
print("\n  Interpretation:")
if r_towers_vs_pop > 0.5:
    print(f"  ✓ r = {r_towers_vs_pop:.3f} → Strong positive correlation.")
    print("    More towers are placed in high-population wards — demand-driven.")
elif r_towers_vs_pop > 0.2:
    print(f"  ~ r = {r_towers_vs_pop:.3f} → Moderate positive correlation.")
    print("    Population is one factor, but not the sole driver (road/building density also matters).")
else:
    print(f"  ✗ r = {r_towers_vs_pop:.3f} → Weak correlation.")
    print("    Consider adding more population-weight to the demand model.")

if tower_gini < 0.4:
    print(f"  ✓ Gini = {tower_gini:.3f} → Towers distributed reasonably equitably across wards.")
else:
    print(f"  ! Gini = {tower_gini:.3f} → Tower distribution is concentrated; equity concern.")

# Top 5 and bottom 5 wards by population
print("\n  Top 5 wards by population:")
top5 = ward_valid.nlargest(5, 'population')[
    ['ward_id', 'population', 'n_towers', 'coverage_pct', 'pop_fraction_%']
]
print(top5.to_string(index=False))

print("\n  Bottom 5 wards by population:")
bot5 = ward_valid.nsmallest(5, 'population')[
    ['ward_id', 'population', 'n_towers', 'coverage_pct', 'pop_fraction_%']
]
print(bot5.to_string(index=False))

# ── Figures ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Scatter 1: towers vs population
ax = axes[0]
ax.scatter(ward_valid['population'] / 1000, ward_valid['n_towers'],
           c='steelblue', alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
z  = np.polyfit(ward_valid['population'], ward_valid['n_towers'], 1)
xr = np.linspace(ward_valid['population'].min(), ward_valid['population'].max(), 100)
ax.plot(xr / 1000, np.polyval(z, xr), 'r-', linewidth=2,
        label=f'r = {r_towers_vs_pop:.3f}')
ax.set_xlabel('Ward Population (thousands)')
ax.set_ylabel('Number of AI Towers')
ax.set_title('Tower Count vs Ward Population')
ax.legend()
ax.grid(True, alpha=0.3)

# Scatter 2: coverage% vs population
ax2 = axes[1]
ax2.scatter(ward_valid['population'] / 1000, ward_valid['coverage_pct'],
            c='darkorange', alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
z2  = np.polyfit(ward_valid['population'], ward_valid['coverage_pct'], 1)
ax2.plot(xr / 1000, np.polyval(z2, xr), 'r-', linewidth=2,
         label=f'r = {r_cov_vs_pop:.3f}')
ax2.set_xlabel('Ward Population (thousands)')
ax2.set_ylabel('AI Coverage (%)')
ax2.set_title('Coverage % vs Ward Population')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Lorenz curve for tower distribution
ax3 = axes[2]
sorted_pop    = ward_valid['population'].sort_values().values
sorted_towers = ward_valid.loc[ward_valid['population'].sort_values().index, 'n_towers'].values
cumulative_pop    = np.cumsum(sorted_pop)    / (sorted_pop.sum()    + 1e-9)
cumulative_towers = np.cumsum(sorted_towers) / (sorted_towers.sum() + 1e-9)
ax3.plot([0] + list(cumulative_pop), [0] + list(cumulative_towers),
         'steelblue', linewidth=2, label=f'Tower Lorenz (Gini={tower_gini:.3f})')
ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect equality')
ax3.fill_between([0] + list(cumulative_pop), [0] + list(cumulative_towers),
                 [0] + list(cumulative_pop), alpha=0.2, color='steelblue')
ax3.set_xlabel('Cumulative ward population fraction')
ax3.set_ylabel('Cumulative tower fraction')
ax3.set_title('Lorenz Curve: Tower Equity Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)

plt.suptitle(f'Census Cross-check | {ward_source}', fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, 'census_crosscheck.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Figure saved → {fig_path}")

# Save CSV
csv_path = os.path.join(RESULTS_DIR, 'census_crosscheck_results.csv')
ward_summary.to_csv(csv_path, index=False)
print(f"  Per-ward data saved → {csv_path}")

# Summary metrics CSV
summary_metrics = {
    'Metric': [
        'Ward source',
        'Number of wards (with population)',
        'Total population',
        'AI towers placed',
        'Pearson r (towers vs population)',
        'p-value (towers vs population)',
        'Pearson r (coverage% vs population)',
        'p-value (coverage% vs population)',
        'Gini coefficient (tower distribution)',
    ],
    'Value': [
        ward_source,
        len(ward_valid),
        f'{int(total_pop):,}',
        int(ward_summary['n_towers'].sum()),
        f'{r_towers_vs_pop:.4f}',
        f'{p_towers_vs_pop:.4f}',
        f'{r_cov_vs_pop:.4f}',
        f'{p_cov_vs_pop:.4f}',
        f'{tower_gini:.4f}',
    ]
}
pd.DataFrame(summary_metrics).to_csv(
    os.path.join(RESULTS_DIR, 'census_crosscheck_summary.csv'), index=False
)
print(f"  Summary metrics saved → {os.path.join(RESULTS_DIR, 'census_crosscheck_summary.csv')}")
print("\nDone.")
