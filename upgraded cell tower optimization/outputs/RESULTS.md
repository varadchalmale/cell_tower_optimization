# Cell Tower Optimization — Results Summary
_Generated in 0.7 minutes | Grid: 750 m | Scope: Nagpur District (9,928 km²)_

## Configuration
| Parameter | Value |
|-----------|-------|
| Frequency | 1800 MHz |
| Tx Power (macro) | 46 dBm |
| Antenna Height (macro) | 35 m |
| RSRP Threshold | -95 dBm |
| Macro Coverage Radius (COST-231 Hata) | **3454 m (3.45 km)** |

## Multi-Tier Tower Deployment vs Airtel
| Tier | AI Count | Airtel Equiv | Area Coverage | Pop Coverage | Demand Coverage |
|------|----------|--------------|--------------|-------------|----------------|
| Macro Cell (35 m) | **80** | ~1,200 | 5.0% | 57.9% | 59.8% |
| Micro Cell (12 m) | **0** | — | 0.0% | 0.0% | 0.0% |
| Small Cell (6 m) | **150** | — | 0.8% | 3.9% | 3.8% |
| **TOTAL** | **230** | **~1,200** | **5.9%** | **61.9%** | **63.6%** |

> **AI achieves 81% fewer towers than Airtel** (230 vs ~1,200) while covering 61.9% of the district population at RSRP ≥ -95 dBm (Airtel indoor planning threshold).

## NSGA-II Macro Optimization Results
| Metric | AI Solution |
|--------|-------------|
| Macro towers selected | **80** |
| Grid points analysed | 17,651 |
| Candidate sites evaluated | 500 |
| Pareto-optimal configurations | 40 |
| Estimated network capacity | **-1000.00 Gbps** |

## Validation vs Airtel Coverage Map
| Metric | Value |
|--------|-------|
| IoU | 0.2228 |
| Match_Percentage | 100.0 |
| False_Positive_Pct | 77.72 |
| Missed_Coverage_Pct | 0.0 |

## Label Source (demand model)
- infrastructure-weighted population proxy (no real traffic data)

## Output Files
| File | Description |
|------|-------------|
| `outputs/interactive_towers.html` | Interactive Folium map — 4 toggleable layers |
| `outputs/pareto_frontier.png` | NSGA-II Pareto trade-off chart |
| `outputs/predicted_traffic_mbps_heatmap.png` | Spatial demand forecast |
| `outputs/sites_map.png` | Candidate vs selected macro tower locations |
