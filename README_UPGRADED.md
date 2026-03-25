# AI-Driven Upgraded Cell Tower Optimization System

This upgraded system transforms the basic cell tower placement tool into a telecom-grade planning engine.

## 🚀 Key Modernizations

1.  **Realistic Propagation**: Replaced simple FSPL with the **COST-231 Hata Model**, accounting for frequency (1.8GHz), tower/user heights, and environment types (Urban/Suburban).
2.  **Multi-Objective Optimization**: Switched from a single-weighted fitness to a **True Multi-Objective Genetic Algorithm**. It produces a **Pareto Frontier** balancing:
    -   Demand-Weighted Coverage
    -   Network Capacity (SINR-based)
    -   Deployment Cost
3.  **Capacity-Aware Planning**: Moves beyond simple "is it covered?" to "can it handle the load?". 
    -   Uses **SINR (Signal-to-Interference-plus-Noise Ratio)** to calculate Shannon capacity per pixel.
    -   Accounts for inter-cell interference from neighboring towers.
4.  **Feasibility Constraints**: Towers are only placed at **Candidate Sites** generated based on high-demand zones and low-clutter feasibility, rather than random pixels.
5.  **ML Traffic Prediction**: Integrates a **RandomForest ML model** to predict spatial demand from population and clutter features, moving away from purely heuristic scoring.
6.  **Performance & Scale**: Uses vectorized NumPy operations and pre-calculated candidacy pools for faster convergence.

## 📁 Mathematical Foundation

-   **Pathloss (COST-231 Hata)**: 
    $L = 46.3 + 33.9 \log(f) - 13.82 \log(h_{te}) - a(h_{re}) + (44.9 - 6.55 \log(h_{te})) \log(d) + C_m$
-   **SINR Calculation**:
    $SINR = \frac{P_{serving}}{\sum P_{interference} + N}$
-   **Spectral Efficiency**:
    $Capacity = B \cdot \log_2(1 + SINR)$

## 🛠 Run Instructions

### 1. Requirements
Ensure you have the updated dependencies installed:
```bash
pip install -r requirements_upgraded.txt
```

### 2. Prepare Data
Ensure you have run the original Phase 1 and Phase 2 scripts to generate the base feature layers:
```bash
python src/data_processing.py
python src/feature_engineering.py
```

### 3. Run Upgraded Optimization
Execute the new main entry point:
```bash
python src/main_upgraded.py
```

## 📊 Outputs
The system will generate in `results/upgraded/`:
-   `pareto_frontier.png`: Visualization of the trade-off space.
-   `capacity_sinr_heatmaps.png`: Realistic network quality maps.
-   `best_solution_upgraded.json`: Coordinates of the optimal tower configuration.
-   Metric comparisons (printed in terminal).
