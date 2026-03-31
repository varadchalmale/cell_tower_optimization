# Cell Tower Optimization — Results Summary
_Generated in 0.9 minutes | Grid: 500 m | Towers: 50_

## Configuration
| Parameter | Value |
|-----------|-------|
| Frequency | 1800 MHz |
| Tx Power | 46 dBm |
| Antenna Height | 35 m |
| RSRP Threshold | -95 dBm |
| Coverage Radius (COST-231 Hata) | **3454 m (3.45 km)** |

## Optimization Results
| Metric | AI Solution |
|--------|-------------|
| Number of towers | **50** |
| Grid points analysed | 39,691 |
| Candidate sites evaluated | 500 |
| Pareto-optimal configurations | 24 |
| Area coverage | **4.0%** |
| Population-weighted coverage | **54.1%** |
| Estimated network capacity | **64.20 Gbps** |

## Validation vs Airtel Coverage Map
| Metric | Value |
|--------|-------|
| IoU | 0.0561 |
| Match_Percentage | 24.13 |
| False_Positive_Pct | 93.18 |
| Missed_Coverage_Pct | 75.87 |

## Label Source (demand model)
- infrastructure-weighted population proxy (no real traffic data)

## Output Files
| File | Description |
|------|-------------|
| `outputs/interactive_towers.html` | Interactive Folium map — open in browser |
| `outputs/pareto_frontier.png` | NSGA-II Pareto trade-off chart |
| `outputs/predicted_traffic_mbps_heatmap.png` | Spatial demand forecast |
| `outputs/sites_map.png` | Candidate vs selected tower locations |
