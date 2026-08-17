"""Train the freight-rate model and save it for prediction.

Usage:
    python -m src.train
"""
from __future__ import annotations
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data import load_train_test
from src.evaluate import summarize
from src.features import CATEGORICAL_FEATURES, build_city_coords, build_features
from src.split import time_based_split

DATA_PATH = Path("data/train-test.csv")
MODEL_DIR = Path("models")

def _make_baseline()-> Pipeline:
    """Simple linear baseline -- gives an honest floot to beat."""
    preprocess = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"),CATEGORICAL_FEATURES)],
        remainder="passthrough",
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
    city_coords = build_city_coords(raw)

    train_raw, valid_raw = time_based_split(raw)
    X_train = build_features(train_raw, city_coords)
    y_train = train_raw["posted_rate"]
    X_valid = build_features(valid_raw, city_coords)
    y_valid = valid_raw["posted_rate"]

    baseline = _make_baseline().fit(X_train, y_train)
    print("Baseline (linear, one-hot city):", summarize(y_valid.to_numpy(), baseline.predict(X_valid)))

    main_model = _make_main_model().fit(X_train, y_train)
    print("HistGradientBoostingRegressor:", summarize(y_valid.to_numpy(), main_model.predict(X_valid)))

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