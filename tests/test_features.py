import pandas as pd

from src.features import FEATURE_COLUMNS, build_city_coords, build_features


def _toy_df():
    return pd.DataFrame({
        "load_id": ["A", "B"],
        "pickup": ["Lexington", "Richmond"],
        "delivery": ["Fort Wayne", "Baltimore"],
        "pickup_lat": [36.99, 38.09],
        "pickup_lon": [-84.99, -76.78],
        "delivery_lat": [41.31, 38.16],
        "delivery_lon": [-85.36, -72.74],
        "distance": [360.0, 274.3],
        "equipment": ["Dry Van", "Dry Van"],
        "weight": [32000.0, 30658.0],
        "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        "market_index": [0.9, 0.97],
        "quote_signal": [2.3, 2.4],
        "posted_rate": [800.0, 645.4],
    })


def test_build_features_returns_expected_columns():
    df = _toy_df()
    coords = build_city_coords(df)
    features = build_features(df, coords)
    assert list(features.columns) == FEATURE_COLUMNS
    assert len(features) == len(df)


def test_build_features_fills_missing_coords_and_signals_for_december_like_input():
    df = _toy_df()
    coords = build_city_coords(df)
    december_like = df.drop(columns=[
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
        "market_index", "quote_signal", "posted_rate", "load_id",
    ])
    features = build_features(december_like, coords)
    assert features["pickup_lat"].notna().all()
    assert features["market_index"].isna().all()
