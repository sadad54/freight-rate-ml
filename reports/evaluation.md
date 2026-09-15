# Freight-rate evaluation

Internal temporal holdout; not external validation.csv target accuracy

Training: ['2025-01-01', '2025-08-31'] (38,477 rows). Holdout: ['2025-09-01', '2025-10-31'] (9,523 rows).

| Scenario | Model | MAE | RMSE | MAPE (%) |
|---|---|---:|---:|---:|
| full_features | linear_baseline | 142.76 | 637.53 | 8.28 |
| full_features | hist_gradient_boosting | 143.92 | 641.47 | 6.89 |
| december_feature_schema | linear_baseline | 164.33 | 639.35 | 10.03 |
| december_feature_schema | hist_gradient_boosting | 154.21 | 644.20 | 7.50 |

Reproduce: `python -m src.train`. Exact versions, data hash and values: [evaluation.json](evaluation.json).

The December scenario removes market signals and coordinates, then recovers coordinates from the training-only city lookup. It measures robustness to missing inputs, not December target accuracy.

Data provenance: supplied repository CSVs; the original provider and whether rates are simulated are not established here. Do not describe these as client or production results.
