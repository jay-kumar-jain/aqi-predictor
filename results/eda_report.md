# AQI exploratory data analysis

- Rows: 52,560
- Locations: 6
- AQI mean / median / SD: 287.36 / 170.00 / 327.98
- AQI range: 56.00–2742.00
- AQI skewness: 3.529
- IQR rule flags: 8,586 target observations (flags are not automatic errors)

## Missingness

| Column | Missing count | Missing % | Type |
|---|---:|---:|---|
| date_ist | 0 | 0.000% | str |
| time_ist | 0 | 0.000% | str |
| location | 0 | 0.000% | string |
| lat | 0 | 0.000% | float64 |
| lon | 0 | 0.000% | float64 |
| temp_c | 0 | 0.000% | float64 |
| humidity | 0 | 0.000% | int64 |
| pressure_mb | 0 | 0.000% | float64 |
| windspeed_kph | 0 | 0.000% | float64 |
| condition_text | 0 | 0.000% | str |
| description | 0 | 0.000% | str |
| aqi_index | 0 | 0.000% | int64 |
| pm2_5 | 0 | 0.000% | float64 |
| pm10 | 0 | 0.000% | float64 |
| co | 0 | 0.000% | int64 |
| no2 | 0 | 0.000% | float64 |
| hour | 0 | 0.000% | int32 |
| day | 0 | 0.000% | int32 |
| month | 0 | 0.000% | int32 |
| day_of_week | 0 | 0.000% | int32 |

## AQI rank correlations

Spearman associations below describe monotonic association, not causation.

| Feature | Spearman ρ | p-value | n |
|---|---:|---:|---:|
| pm10 | 0.725 | 0 | 52,560 |
| pm2_5 | 0.698 | 0 | 52,560 |
| humidity | -0.358 | 0 | 52,560 |
| month | -0.233 | 0 | 52,560 |
| pressure_mb | 0.085 | 1.39e-85 | 52,560 |
| co | 0.068 | 1.13e-55 | 52,560 |
| temp_c | 0.043 | 5.94e-23 | 52,560 |
| hour | 0.033 | 2.46e-14 | 52,560 |
| no2 | -0.015 | 0.000796 | 52,560 |
| day | -0.014 | 0.00181 | 52,560 |
| day_of_week | -0.009 | 0.0355 | 52,560 |
| windspeed_kph | -0.004 | 0.304 | 52,560 |

## Files

See `figures/eda_overview.png`, `numeric_summary.csv`, `location_counts.csv`, and `correlations.csv`.

Because the data are hourly, random split/CV scores may be optimistic for forecasting future dates. Use a chronological holdout and time-series CV when the intended use is future prediction.
