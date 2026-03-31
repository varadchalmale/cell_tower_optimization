# AI-Driven Cell Tower Placement Optimization

**Problem**: Telecom operators spend weeks and significant capital sending field engineers to survey
areas for new cell tower sites — a process that is slow, expensive, and dependent on individual
expertise.

**Solution**: An end-to-end AI system that ingests publicly available geospatial data (population,
terrain, land use, road networks) and automatically recommends optimal tower placements in minutes,
using Multi-Objective Genetic Algorithm optimization guided by a Random Forest demand model.

**Core result**: The system recommends fewer towers than are currently deployed in Nagpur while
maintaining comparable population coverage — demonstrating that systematic AI-driven planning can
reduce infrastructure cost without degrading service quality.

---

## Technical Approach

```
[OpenStreetMap / WorldPop / GADM / DEM]
              |
              v
    [Feature Engineering]         <- population, clutter, slope, POI density
              |
              v
   [Random Forest Demand Model]   <- trained on OpenCellID tower density (real) or
              |                      population+OSM proxy (clearly disclosed)
              v
  [Candidate Site Generation]     <- heuristic scoring + minimum separation filter
              |
              v
     [NSGA-II Optimizer]          <- 3 objectives: coverage, capacity, cost
              |                      produces Pareto frontier, not a single forced solution
              v
  [Folium Interactive Map]        <- coverage circles from COST-231 Hata model
              |
              v
  [Validation vs OpenCellID]      <- Hungarian bipartite matching, KDTree, IoU
```

### Algorithms & Models

| Component | Method | Why |
|-----------|--------|-----|
| Demand prediction | Random Forest (100 trees) | Handles non-linear spatial interactions between population, roads, and land use |
| Propagation | COST-231 Hata (urban) | Industry-standard ITU model for 1.8–2.1 GHz macro cells; more realistic than FSPL |
| Capacity | Shannon + SINR matrix | Accounts for inter-cell interference, not just binary coverage |
| Optimization | NSGA-II (pymoo) | Multi-objective; avoids arbitrary single-objective weighting |
| Validation | Hungarian algorithm + KDTree | 1-to-1 bipartite matching against real OpenCellID ground truth |

### Key Parameters (config.yaml)

| Parameter | Value | Source |
|-----------|-------|--------|
| Frequency | 2100 MHz | Standard 4G Band 1 |
| Tx Power | 43 dBm (20W) | Typical macro BTS |
| Antenna height | 30 m | Urban tower standard |
| RSRP threshold | -110 dBm | 3GPP TS 36.133 |
| Min tower spacing | 300 m | Co-channel interference constraint |

---

## Data Sources

| Data | Source | Purpose |
|------|--------|---------|
| Administrative boundary | GADM v4.1 | Nagpur district extent |
| Population density | WorldPop / synthetic fallback | Demand estimation |
| Elevation (DEM) | SRTM 30m | Terrain slope, propagation correction |
| Buildings, roads, POIs | OpenStreetMap (osmnx) | Land use features |
| LTE towers (validation) | OpenCellID (user-provided) | Ground-truth evaluation |
| Existing coverage | Airtel ArcGIS API | IoU coverage comparison |

**To run with real data**: Download the India OpenCellID CSV from https://opencellid.org,
filter for MCC=404, radio=LTE, and save to `Data/raw/opencellid_nagpur.csv`.

---

## Directory Structure

```
upgraded cell tower optimization/
├── main.py                     # Single entry point
├── run_pipeline.sh             # Bash automation
├── requirements.txt            # Dependencies
├── src/
│   ├── config/config.yaml      # All parameters — change here, not in code
│   ├── data/preprocessing.py   # GADM, OSM, raster ingestion
│   ├── models/
│   │   ├── demand.py           # RF demand model (OpenCellID labels preferred)
│   │   ├── propagation.py      # COST-231 Hata + 3GPP UMa
│   │   └── capacity.py         # SINR / Shannon capacity / cell load
│   ├── optimization/
│   │   ├── candidates.py       # Heuristic candidate filtering
│   │   └── nsga2.py            # NSGA-II multi-objective optimizer (pymoo)
│   ├── validation/validate.py  # IoU vs Airtel coverage map
│   └── visualization/
│       └── visualize.py        # Heatmaps, Pareto plot, Folium map
└── Data/raw/                   # Place input rasters and shapefiles here
```

---

## Setup & Running

```bash
# 1. Install dependencies (Python 3.10 recommended)
pip install -r requirements.txt

# 2. (Optional but recommended) Add real OpenCellID data
#    Download from opencellid.org → filter MCC=404, LTE → save as:
cp your_download.csv Data/raw/opencellid_nagpur.csv

# 3. Run the full pipeline
python main.py

# OR use the shell script (handles venv activation)
chmod +x run_pipeline.sh && ./run_pipeline.sh
```

For an interactive walkthrough, open `../demo.ipynb` (runs in ~10 minutes).

---

## Outputs

All artifacts are saved to `outputs/`:

| File | Description |
|------|-------------|
| `interactive_towers.html` | Folium map — demand heatmap + tower markers with COST-231 coverage radius |
| `pareto_frontier.png` | Trade-off surface: coverage vs capacity vs deployment cost |
| `predicted_traffic_mbps_heatmap.png` | Spatial demand forecast |
| `sites_map.png` | Candidate vs selected towers overlaid on demand map |

---

## Limitations & Honest Assumptions

- **Demand labels**: When real traffic measurements are unavailable, the model uses OpenCellID
  tower density as a proxy (higher existing tower density → historically higher demand). If
  OpenCellID data is absent, it falls back to a population+OSM infrastructure proxy. Neither is
  measured traffic data; both are reasonable engineering approximations.
- **Propagation**: COST-231 Hata is a statistical model valid for 150 MHz – 2 GHz, 1–20 km range.
  It does not model individual building reflections or small-scale fading.
- **Interference**: Inter-cell interference is simplified to a linear sum of received powers. Full
  MIMO beam management is not modeled.
- **Zoning / ROW**: Real-world site acquisition constraints (building permits, right-of-way,
  property access) are not included.

---

## References

- Deb, K. et al. (2002). "A Fast and Elitist Multi-Objective Genetic Algorithm: NSGA-II." *IEEE
  Transactions on Evolutionary Computation*, 6(2), 182–197.
- COST 231 Final Report (1999). "Digital Mobile Radio: COST 231 View on the Evolution Towards
  3rd Generation Systems." European Commission / COST Telecommunications.
- Rappaport, T.S. (2002). *Wireless Communications: Principles and Practice*, 2nd ed. Prentice Hall.
- 3GPP TS 36.133 (2020). "Requirements for support of radio resource management (LTE)."
- Blank, J. & Deb, K. (2020). "pymoo: Multi-Objective Optimization in Python." *IEEE Access*, 8,
  89497–89509.
