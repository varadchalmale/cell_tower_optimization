# Methodology

## Problem Formulation

Cell tower placement is a **combinatorial optimisation** problem. Given a set of C candidate sites
and a target tower count N, we seek the subset S ⊂ C, |S| = N that simultaneously:

- **Maximises** demand-weighted population coverage
- **Maximises** aggregate network throughput
- **Minimises** deployment cost

This is a multi-objective combinatorial optimisation problem (NP-hard in general). The search space
has C-choose-N possible configurations — for C=500, N=50 this is ~10^93, making exhaustive search
infeasible. We use NSGA-II to explore the Pareto-optimal frontier efficiently.

---

## System Pipeline

### 1. Data Preprocessing

- **Boundary**: Nagpur district extracted from GADM v4.1 shapefile (India level-2 admin)
- **Grid**: Regular grid at 150 m spacing clipped to boundary (~30,000 cells)
- **OSM features**: Buildings, roads, land use fetched via `osmnx` with 3-retry backoff
- **Population**: WorldPop gridded population raster, clipped to boundary
- **Elevation**: SRTM 30 m DEM, used for slope and terrain correction
- **CRS**: All layers projected to EPSG:32644 (UTM Zone 44N) for metric distance calculations

### 2. Feature Engineering

Per grid cell, the following features are computed within a 200 m buffer:

| Feature | Computation |
|---------|------------|
| `population` | Raster pixel sum from WorldPop |
| `road_density` | Total road length (m) / buffer area (m²) |
| `building_density` | Building footprint area / buffer area |
| `urban_class` | Categorical: urban (>0.3), suburban (>0.05), rural |
| `elevation` | Mean DEM value |
| `slope` | Max gradient from DEM finite differences |

Demand score (used in candidate generation): `0.7 × population_norm + 0.3 × POI_density_norm`

Cost surface: `0.4 × slope_norm + 0.6 × building_density_norm`

### 3. Demand Prediction Model

**Algorithm**: Random Forest Regressor (sklearn), 100 trees, `random_state=42`

**Input features**: population, road_density, building_density, urban_class (encoded as 0/1/2)

**Label strategy** (in priority order):

1. **Real OpenCellID tower density** (preferred): For each grid cell, count LTE towers from
   OpenCellID within the cell area. Scale to Mbps-equivalent units. Rationale: operators
   historically deploy infrastructure proportional to sustained demand; tower density is therefore
   a valid demand proxy. The RF model then learns which spatial features predict that demand.

2. **Infrastructure-weighted population proxy** (fallback): `demand ≈ (population / N_users) ×
   Mbps_per_user × (1 + 0.2×road_factor) × (1 + 0.5×building_factor)`. This is an approximation
   of ITU-T E.501 traffic estimation guidelines. **This is not measured traffic data.** Outputs
   from this path should be interpreted as relative demand indices.

**Model evaluation**: 80/20 train-test split, RMSE and R² reported. Feature importances printed
to terminal. The model is a mapping from geospatial features → demand index; its accuracy is
bounded by the quality of the label source.

### 4. Candidate Site Generation

1. Score all grid cells: `score = 0.7 × demand_norm − 0.3 × cost_norm`
2. Select top-1000 by score
3. Apply greedy minimum-separation filter (≥300 m between any two candidates)
4. Returns ~200–500 feasible candidate sites

### 5. Multi-Objective Optimisation — NSGA-II

**Implementation**: `pymoo` library (Blank & Deb, 2020)

**Decision variables**: Binary selection vector of length C (indices into candidate pool)
mapping to a fixed-size N-tower configuration.

**Objectives** (all minimised internally, signs flipped for maximisation):

```
f1 = −Σ [ demand(i) × covered(i) ]       # maximise demand-weighted coverage
f2 = −Σ [ throughput_mbps(i) ]            # maximise total network capacity
f3 =  Σ [ cost_surface(tower_j) ]         # minimise deployment cost
```

**Fitness evaluation** per individual:
1. Compute RSRP matrix: `R[i,j] = Tx_power − PathLoss(distance(tower_j, pixel_i))`
2. Assign each pixel to strongest tower (max RSRP column)
3. Compute SINR: `SINR[i] = R[i, serving] / (Σ R[i, other] + N)` (linear scale)
4. Shannon capacity: `C[i] = 0.6 × BW × log2(1 + SINR[i])` (0.6 = spectral efficiency discount)
5. Cell load penalty: `penalty = Σ max(0, load_j − 1)²` for overloaded cells

**Algorithm parameters**: pop_size=100, n_gen=50, SBX crossover (η=15, p=0.9),
polynomial mutation (η=20). Best-compromise solution selected by minimum weighted Euclidean
distance from Utopian point.

**Why NSGA-II over alternatives**:
- Simulated annealing handles single objective only
- Weighted sum GA collapses Pareto diversity and requires arbitrary weights
- ILP (integer linear programming) is exact but scales poorly with non-linear propagation
- NSGA-II preserves Pareto diversity via crowding distance; well-validated in telecom planning
  literature (see Toril et al., 2010; Amaldi et al., 2003)

### 6. Radio Propagation: COST-231 Hata Model

The COST-231 Hata model (COST-231 Final Report, 1999) is valid for:
- Frequency: 150 MHz – 2000 MHz
- Distance: 1 km – 20 km
- Tx antenna height: 30 m – 200 m

**Path Loss (dB)**:

```
L = 46.3 + 33.9·log10(f) − 13.82·log10(h_te) − a(h_re)
      + (44.9 − 6.55·log10(h_te))·log10(d) + C_m

where:
  f     = frequency (MHz)
  h_te  = transmitter effective height (m)
  h_re  = receiver height (m)
  d     = distance (km)
  a(h_re) = (1.1·log10(f) − 0.7)·h_re − (1.56·log10(f) − 0.8)   [medium-small city]
  C_m   = 3 dB (dense urban), 0 dB (suburban)
```

**RSRP**: `RSRP (dBm) = Tx_power (dBm) − L (dB)`

**Coverage threshold**: RSRP ≥ −110 dBm (3GPP TS 36.133 minimum for LTE connectivity)

**Maximum coverage radius**: Computed analytically by inverting the Hata equation at max path loss.
This radius is used in all visualisations — not a fixed estimate.

### 7. Capacity Model

**SINR**:
```
SINR = P_s / (Σ P_i + N)

where:
  P_s = received power from serving (strongest) tower (linear, mW)
  P_i = received power from each interfering tower (linear, mW)
  N   = thermal noise = kTB = −104 dBm for 20 MHz channel
```

**Shannon capacity with spectral efficiency discount**:
```
C = η · B · log2(1 + SINR)

where:
  B = 20 MHz (channel bandwidth)
  η = 0.6 (empirical spectral efficiency factor accounting for overhead,
            coding rate limits, and HARQ retransmissions)
```

**Cell load**: `load_j = Σ demand(i assigned to j) / peak_throughput_j`

Cells with load > 1 are overloaded; penalised quadratically in objective f2.

### 8. Validation

Validation compares AI-predicted tower coordinates against real OpenCellID LTE towers within
the Nagpur bounding box.

| Metric | Method |
|--------|--------|
| Bipartite matching error | Hungarian algorithm (scipy.optimize.linear_sum_assignment) on pairwise distance matrix |
| Nearest-neighbour distance | KDTree (scipy.spatial) |
| Spatial density correlation | Pearson correlation of 2D histograms |
| Coverage IoU | Intersection-over-Union of simulated vs real network footprints |

**Critical requirement**: Real OpenCellID CSV must be provided. The system does **not** generate
mock ground-truth data — doing so would make the validation metrics meaningless.

---

## Assumptions and Limitations

1. **No real traffic measurements**: Demand labels are proxies (tower density or population-based).
   Treat demand predictions as relative orderings, not absolute Mbps values.

2. **Isotropic antennas**: The COST-231 model used here assumes omnidirectional antennas. Real macro
   cells use 3-sector directional antennas with 65°–90° horizontal beamwidth. Sector modelling would
   increase accuracy.

3. **Static demand**: The model predicts peak-hour demand. Time-varying load (e.g., day/night
   patterns) is not captured.

4. **No site acquisition constraints**: Zoning laws, building permits, right-of-way costs, and
   property access are not modelled.

5. **Flat terrain approximation for SINR**: COST-231 includes an empirical terrain correction term
   but does not model individual building blockage. The SINR matrix therefore slightly overestimates
   signal strength in dense urban canyons.

---

## References

1. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective
   genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

2. COST Action 231 (1999). *Digital Mobile Radio: COST 231 View on the Evolution Towards
   3rd Generation Systems*. European Commission, Brussels.

3. Rappaport, T. S. (2002). *Wireless Communications: Principles and Practice* (2nd ed.).
   Prentice Hall.

4. 3GPP TS 36.133 V17 (2020). *Requirements for Support of Radio Resource Management (LTE)*.
   3rd Generation Partnership Project.

5. Blank, J., & Deb, K. (2020). pymoo: Multi-objective optimization in Python. *IEEE Access*,
   8, 89497–89509.

6. Toril, M., Luna-Ramírez, S., & Wille, V. (2010). Optimization of handover parameters for
   traffic sharing in GERAN. *Wireless Personal Communications*, 52(3), 625–648.

7. Amaldi, E., Capone, A., & Malucelli, F. (2003). Planning UMTS base station location:
   Optimization models with power control and algorithms. *IEEE Transactions on Wireless
   Communications*, 2(5), 939–952.
