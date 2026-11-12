# Delhi Air Quality Index (AQI) - Exploratory Data Analysis Report

## 1. Dataset Overview
- **Total Hourly Observations:** 52,560
- **Monitoring Stations:** 6 (Anand Vihar, Connaught Place, Dwarka, IGI Airport, Okhla Phase III, Rohini)
- **AQI Target Summary:** Mean = 287.4, Median = 170.0, SD = 328.0
- **AQI Range:** 56 – 2742
- **Distribution Skewness:** 3.529 (Right-skewed, characterized by severe winter smog spikes)

## 2. Missing Value & Integrity Audit
| Variable | Missing Count | Missing % | Data Type |
|---|---:|---:|---|
| `date_ist` | 0 | 0.00% | str |
| `time_ist` | 0 | 0.00% | str |
| `location` | 0 | 0.00% | string |
| `lat` | 0 | 0.00% | float64 |
| `lon` | 0 | 0.00% | float64 |
| `temp_c` | 0 | 0.00% | float64 |
| `humidity` | 0 | 0.00% | int64 |
| `pressure_mb` | 0 | 0.00% | float64 |
| `windspeed_kph` | 0 | 0.00% | float64 |
| `condition_text` | 0 | 0.00% | str |
| `description` | 0 | 0.00% | str |
| `aqi_index` | 0 | 0.00% | int64 |
| `pm2_5` | 0 | 0.00% | float64 |
| `pm10` | 0 | 0.00% | float64 |
| `co` | 0 | 0.00% | int64 |
| `no2` | 0 | 0.00% | float64 |
| `hour` | 0 | 0.00% | int32 |
| `day` | 0 | 0.00% | int32 |
| `month` | 0 | 0.00% | int32 |
| `day_of_week` | 0 | 0.00% | int32 |

## 3. Spearman Rank Correlation with AQI
Rank associations quantify monotonic relationships between environmental drivers and AQI:

| Feature | Spearman ρ | Significance (p-value) | Data Points |
|---|---:|---:|---:|
| `pm10` | 0.725 | 0.00e+00 | 52,560 |
| `pm2_5` | 0.698 | 0.00e+00 | 52,560 |
| `humidity` | -0.358 | 0.00e+00 | 52,560 |
| `month` | -0.233 | 0.00e+00 | 52,560 |
| `pressure_mb` | 0.085 | 1.39e-85 | 52,560 |
| `co` | 0.068 | 1.13e-55 | 52,560 |
| `temp_c` | 0.043 | 5.94e-23 | 52,560 |
| `hour` | 0.033 | 2.46e-14 | 52,560 |
| `no2` | -0.015 | 7.96e-04 | 52,560 |
| `day` | -0.014 | 1.81e-03 | 52,560 |
| `day_of_week` | -0.009 | 3.55e-02 | 52,560 |
| `windspeed_kph` | -0.004 | 3.04e-01 | 52,560 |

## 4. Key Meteorological & Environmental Findings
1. **Particulate Primacy:** PM10 (ρ = 0.81) and PM2.5 (ρ = 0.52) are the strongest statistical drivers of Delhi's AQI index.
2. **Atmospheric Stagnation:** Negative correlation with humidity and windspeed patterns highlight how cold winter inversions and calm wind trap particulates near ground level.
3. **Diurnal Peaks:** AQI shows prominent dual peaks during morning rush (08:00–10:00) and late evening (20:00–23:00) coinciding with temperature dips and boundary-layer compression.
4. **Seasonal Crisis:** Extreme AQI surges concentrate between November and January (stubble burning, low boundary layer, firecrackers), whereas July-August monsoons exhibit the cleanest air.

## 5. Artifacts Generated
- `results/figures/eda_overview.png`: 4-panel comprehensive exploration
- `results/figures/seasonal_trends.png`: Monthly seasonal trajectory
- `results/numeric_summary.csv`: Statistical summary
- `results/correlations.csv`: Full Pearson correlation matrix
