# AI-Driven 4G/5G Cell Tower Planning System

An AI-driven, capacity-aware, multi-objective automated cellular network planning and validation framework using real geospatial data.

## Project Overview

This is a complete, research-grade, academic cellular network planning tool built from scratch in Python. It evaluates real geospatial terrain, models user traffic demand dynamically with Machine Learning, and determines the most optimal locations for new cell sites using a Multi-Objective Genetic Algorithm (NSGA-II).

### Core Features:
1.  **Data Preprocessing**: Fetch, clip, and reproject DEM, Population Raster, and OSM geometry (Buildings, Roads, Land Usage).
2.  **Machine Learning Demand Model**: Trains a Random Forest on extracted grid density metrics to accurately forecast Mbps traffic expectations per cell block.
3.  **Radio Propagation**: Implements 3GPP UMa and COST-231 Hata pathloss models inclusive of terrain diffraction penalties.
4.  **Capacity Analysis**: Full end-to-end translation of received signal strength (RSRP) into SINR, Shannon spectral efficiency, physical throughput, and cell load balancing.
5.  **Multi-Objective Optimization**: Employs NSGA-II via `pymoo` to construct a Pareto frontier balancing:
    *   Maximum Demand Coverage (Population weighted)
    *   Maximum Network Capacity sum
    *   Minimum Deployment Expenditure (Land type & proximity to roads)
6.  **Scientific Validation**: Automates validation scoring (Intersection-Over-Union) against current existing Airtel coverages.

---

## Directory Structure

```text
Cell_tower_optimization_2/
├── requirements.txt            # All dependencies
├── run_pipeline.sh             # Bash script to run end-to-end
├── main.py                     # Entry point for the framework
├── src/
│   ├── config/
│   │   └── config.yaml         # Fully configurable params, no hardcoding
│   ├── data/
│   │   └── preprocessing.py    # GADM extraction, OSM fetched features
│   ├── models/
│   │   ├── demand.py           # Machine learning component and hotspots
│   │   ├── propagation.py      # COST-231 and 3GPP RF propagation physics
│   │   └── capacity.py         # SINR / Load mapping models
│   ├── optimization/
│   │   ├── candidates.py       # Heuristic-based valid tower selection
│   │   └── nsga2.py            # The Genetic Algorithm formulation
│   ├── validation/
│   │   └── validate.py         # True Positive / False Positive validation maps
│   └── visualization/
│       └── visualize.py        # Heatmap Generation, Folium interactive Maps
└── Data/
    └── raw/                    # Automatically generated. Place real geospatial assets here if available.
```

## Setup & Running

It is recommended to run this project inside a fresh virtual environment. The supplied shell script automates everything from dependency loading to module execution.

1. **Verify Configs**: In `src/config/config.yaml`, ensure tuning matches required memory and compute scopes. Defaults are configured for rapid testing over 50 towers in Nagpur.
2. **Execution**:
   ```bash
   chmod +x run_pipeline.sh
   ./run_pipeline.sh
   ```

*The system functions out-of-the-box. If the original raster layers (e.g. `gadm41_IND_2.shp`, `srtm_dem.tif`) are missing, the AI framework automatically fails over to synthesized proxy geo-bounds built using procedural math mimicking standard Indian districts.*

## Scientific Output & Artifacts
The framework outputs direct analytical artifacts to `outputs/`: 
- **`predicted_traffic_mbps_heatmap.png`**: Heatmap displaying dynamic spatial load forecasting.
- **`pareto_frontier.png`**: The trade-off locus between budget, coverage, and spectral load.
- **`sites_map.png`**: Global plot mapping chosen Pareto optimal towers vs candidate backdrop.
- **Folium Map (`interactive_towers.html`)**: Interactive web layout outlining geographic placement points.
