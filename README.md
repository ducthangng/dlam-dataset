---
license: other
tags:
- time-series
- forecasting
- multivariate-time-series
- education
pretty_name: DLAM Time Series Forecasting Dataset 2026
---

# operations_forecasting_2026

Multivariate hourly forecasting for anonymized operations units.

## Target

Predict the future hourly operational load index for each series_id. Higher values indicate more operational pressure in that unit.

## Forecast Contract

- Frequency: `h`
- Series: `96`
- Timesteps per series: `4992`
- Target column: `target`
- Training history length used by the baseline templates: `168`
- Rollout block length: `24`
- Required prediction horizon: `validation: 336, test: 336`
- Primary metric: `WAPE`; lower is better.

Rollout block length. Submissions must still predict every row in the provided forecast_index_*.csv files.

## Files

- `train.csv`: public training data with targets.
- `validation_input.csv`: validation covariates and history without validation labels.
- `forecast_index_validation.csv`: exact validation timestamps to predict.
- `metadata.json`: machine-readable benchmark metadata mirrored in this card.

Validation labels and private test data are not included in this public dataset repository.

## Schema

- `train`: `series_id`, `timestamp`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_weekend`, `trend`, `workload_intensity`, `demand_forecast`, `staffing_forecast`, `upstream_quality_forecast`, `promotion_intensity`, `shock_risk`, `maintenance_known`, `unit_reliability_forecast`, `queue_pressure_forecast`, `network_pressure_forecast`, `event_load_forecast`, `service_irregularity_risk_forecast`, `throughput_disruption_risk_forecast`, `nominal_capacity`, `zone_sin`, `zone_cos`, `target`
- `prediction`: `series_id`, `timestamp`, `prediction`
- `labels`: `series_id`, `timestamp`, `target`

## Submission

Upload validation predictions through the course leaderboard Space. Prediction CSVs must use:

```csv
series_id,timestamp,prediction
```

The model name is entered in the leaderboard form and is not part of the CSV schema.

## Course Links

- Dataset repository: https://huggingface.co/datasets/AIML-TUDA/dlam-ts-project-data-2026
- Deadline: 04.09.2026, 23:59:59 CEST
