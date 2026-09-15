"""Train the freight-rate model and save it for prediction.

Usage:
    python -m src.train
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import platform
import sklearn

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data import load_train_test
from src.evaluate import summarize
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_city_coords, build_features
from src.split import time_based_split

DATA_PATH = Path("data/train-test.csv")
MODEL_DIR = Path("models")

def _make_baseline()-> Pipeline:
    """Simple linear baseline -- gives an honest floor to beat.

    Unlike HistGradientBoostingRegressor, LinearRegression can't
    handle NaN, so numeric columns get median-imputed here only
    (the main model still sees the raw NaNs)."""
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ],
    )
    return Pipeline([("prep", preprocess), ("model", LinearRegression())])

def _make_main_model()-> HistGradientBoostingRegressor:
    """Gradient-boosted trees: handles missing values and mixed 
    categorical/numeric features natively, which matters given the
    December feature gap (see src/features.py)."""
    return HistGradientBoostingRegressor(
        categorical_features=CATEGORICAL_FEATURES,
        max_depth=8,
        learning_rate=0.06,
        max_iter=400,
        l2_regularization=0.1,
        random_state=42,
    )

def main() -> None:
    raw = load_train_test(DATA_PATH)
    train_raw, valid_raw = time_based_split(raw)
    city_coords = build_city_coords(train_raw)
    X_train = build_features(train_raw, city_coords)
    y_train = train_raw["posted_rate"]
    X_valid = build_features(valid_raw, city_coords)
    y_valid = valid_raw["posted_rate"]

    baseline = _make_baseline().fit(X_train, y_train)
    print("Baseline (linear, one-hot city):", summarize(y_valid.to_numpy(), baseline.predict(X_valid)))

    main_model = _make_main_model().fit(X_train, y_train)
    print("HistGradientBoostingRegressor:", summarize(y_valid.to_numpy(), main_model.predict(X_valid)))

    # Compare both models on exactly the same temporal holdout, including
    # the feature availability encountered by December inference.
    missing = valid_raw.drop(columns=["market_index", "quote_signal", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"])
    scenarios = {"full_features": X_valid, "december_feature_schema": build_features(missing, city_coords)}
    results = {name: {model_name: summarize(y_valid.to_numpy(), model.predict(features))
                     for model_name, model in [("linear_baseline", baseline), ("hist_gradient_boosting", main_model)]}
               for name, features in scenarios.items()}
    report = {
        "scope": "Internal temporal holdout; not external validation.csv target accuracy",
        "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "train_rows": len(train_raw), "holdout_rows": len(valid_raw),
        "train_dates": [str(train_raw.date.min().date()), str(train_raw.date.max().date())],
        "holdout_dates": [str(valid_raw.date.min().date()), str(valid_raw.date.max().date())],
        "features": list(X_train.columns),
        "preprocessing": "City lookup fitted on training partition only; final refit uses all labelled data",
        "seed": 42, "results": results,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Freight-rate evaluation", "", report["scope"], "",
             f"Training: {report['train_dates']} ({len(train_raw):,} rows). Holdout: {report['holdout_dates']} ({len(valid_raw):,} rows).", "",
             "| Scenario | Model | MAE | RMSE | MAPE (%) |", "|---|---|---:|---:|---:|"]
    for scenario, models in results.items():
        for name, metrics in models.items():
            lines.append(f"| {scenario} | {name} | {metrics['MAE']:.2f} | {metrics['RMSE']:.2f} | {metrics['MAPE_%']:.2f} |")
    lines += ["", "Reproduce: `python -m src.train`. Exact versions, data hash and values: [evaluation.json](evaluation.json).",
              "", "The December scenario removes market signals and coordinates, then recovers coordinates from the training-only city lookup. It measures robustness to missing inputs, not December target accuracy.",
              "", "Data provenance: supplied repository CSVs; the original provider and whether rates are simulated are not established here. Do not describe these as client or production results."]
    Path("reports/evaluation.md").write_text("\n".join(lines) + "\n")

    city_coords = build_city_coords(raw)
    #REFIT on ALL labeled data once architecture/hyperparameters are
    # chosen, so the deliverable predictions use every available row.
    X_full = build_features(raw, city_coords)
    y_full = raw["posted_rate"]
    final_model = _make_main_model().fit(X_full, y_full)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_DIR / "model.joblib")
    city_coords.to_csv(MODEL_DIR / "city_coords.csv")
    print(f"Saved model to {MODEL_DIR/'model.joblib'}")


if __name__ == "__main__":
    main()