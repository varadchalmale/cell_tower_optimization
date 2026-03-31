# Cell Tower Optimization — Results Summary
_Generated in 0.9 minutes | Grid: 500 m | Towers: 50_

## Configuration
| Parameter | Value |
|-----------|-------|
| Frequency | 2100 MHz |
| Tx Power | 43 dBm |
| Antenna Height | 30 m |
| RSRP Threshold | -110 dBm |
| Coverage Radius (COST-231 Hata) | **2126 m (2.13 km)** |

## Optimization Results
| Metric | AI Solution |
|--------|-------------|
| Number of towers | **50** |
| Grid points analysed | 39,691 |
| Candidate sites evaluated | 500 |
| Pareto-optimal configurations | 39 |
| Area coverage | **2.7%** |
| Population-weighted coverage | **41.0%** |
| Estimated network capacity | **52.07 Gbps** |

## Validation vs Airtel Coverage Map
| Metric | Value |
|--------|-------|
| IoU | 0.0798 |
| Match_Percentage | 44.09 |
| False_Positive_Pct | 91.12 |
| Missed_Coverage_Pct | 55.91 |

## Label Source (demand model)
- infrastructure-weighted population proxy (no real traffic data)

## Output Files
| File | Description |
|------|-------------|
| `outputs/interactive_towers.html` | Interactive Folium map — open in browser |
| `outputs/pareto_frontier.png` | NSGA-II Pareto trade-off chart |
| `outputs/predicted_traffic_mbps_heatmap.png` | Spatial demand forecast |
| `outputs/sites_map.png` | Candidate vs selected tower locations |
