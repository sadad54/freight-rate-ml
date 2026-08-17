"""Produce validation_predictions.csv and december_predictions.csv.

Usage:
    python -m src.predict
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.data import load_december_inputs, load_validation
from src.features import build_features

MODEL_DIR = Path("models")
DATA_DIR = Path("data")


def _load_model_and_coords():
    model = joblib.load(MODEL_DIR / "model.joblib")
    city_coords = pd.read_csv(MODEL_DIR / "city_coords.csv", index_col="city")
    return model, city_coords


def predict_validation() -> None:
    model, city_coords = _load_model_and_coords()
    val = load_validation(DATA_DIR / "validation.csv")
    X = build_features(val, city_coords)
    preds = model.predict(X)

    template = pd.read_csv(DATA_DIR / "validation-predictions-template.csv")
    out = template[["load_id"]].merge(
        pd.DataFrame({"load_id": val["load_id"], "predicted_rate": preds}),
        on="load_id", how="left",
    )
    out.to_csv("validation_predictions.csv", index=False)
    print(f"Wrote validation predictions.csv ({len(out)} rows)")

def predict_december()-> None:
    model, city_coords = _load_model_and_coords()
    dec = load_december_inputs(DATA_DIR / "december-chart-inputs.csv")
    X = build_features(dec, city_coords)
    preds = model.predict(X)

    out=dec.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["predicted_rate"]= preds
    out.to_csv("december_predictions.csv", index=False)
    print("Wrote december_predictions.csv")

if __name__ == "__main__":
    predict_validation()
    predict_december()