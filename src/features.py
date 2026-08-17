"""Feature engineering shared by validation and December predictions.
december-chart-inputs.csv provides only pickup, delivery, distance,
equipment, weight and date -- it has neither lat/lon nor the market_index / quote_signals that validation.csv carries.
Two options were considered:

1. Train two separate models (a "full" model for validation, a "reduced" model for December).
2. Train one model on the full feature set, and for December rows recover lat/lon from a city lookup table (every city name maps to exactly one coordinate pair in this dataset -- confirmed in eda.py) while leaving market_index/quote_signal as NaN, letting the model use the missing-value branch it already learns from 
the ~1% of training rows missing those same columns.

Option 2 was chosen: one model, one code pathm and it's a closer match to how a production system would behave if a market-data feed were occasionally unavailable at inference time.
"""
from __future__ import annotations
import pandas as pd

CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment"]
NUMERIC_FEATURES = [
    "distance", "weight", "pickup_lat", "pickup_lon",
    "delivery_lat", "delivery_lon", "market_index", "quote_signal", "day_of_week", "day_of_month", "day_of_year", "month", "is_weekend",
    "days_since_start",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TRAIN_START = pd.Timestamp("2025-01-01")

def build_city_coords(train_df: pd.DataFrame)-> pd.DataFrame:
    pickup = train_df[["pickup", "pickup_lat", "pickup_lon"]].rename(
        columns={"pickup":"city", "pickup_lat":"lat", "pickup_lon": "lon"}
    )
    delivery = train_df[["delivery", "delivery_lat", "delivery_lon"]].rename(
        columns={"delivery":"city", "delivery_lat": "lat", "delivery_lon":"lon"}
    )
    coords = pd.concat([pickup, delivery], ignore_index=True).drop_duplicates("city")
    return coords.set_index("city")[["lat", "lon"]]


def _add_date_features(df: pd.DataFrame)-> pd.DataFrame:
    df = df.copy()
    df["day_of_week"]=df["date"].dt.dayofweek
    df["day_of_month"]=df["date"].dt.day
    df["day_of_year"]=df["date"].dt.dayofyear
    df["month"]=df["date"].dt.month
    df["is_weekend"]=df["day_of_week"].isin([5,6]).astype(int)
    df["days_since_start"]=(df["date"] - TRAIN_START).dt.days
    return df

def _fill_missing_coords(df: pd.DataFrame, city_coords: pd.DataFrame) ->pd.DataFrame:
    df = df.copy()
    if "pickup_lat" not in df.columns:
        df["pickup_lat"] = df["pickup"].map(city_coords["lat"])
        df["pickup_lon"] =df["pickup"].map(city_coords["lon"])
        df["delivery_lat"] = df["delivery"].map(city_coords["lat"])
        df["delivery_lon"] =df["delivery"].map(city_coords["lon"])
    return df

def build_features(df:pd.DataFrame, city_coords: pd.DataFrame)-> pd.DataFrame:
    df = _fill_missing_coords(df, city_coords)
    df = _add_date_features(df)
    for col in ("market_index", "quote_signal"):
        if col not in df.columns:
            df[col] = float("nan")
    for col in CATEGORICAL_FEATURES:
        df[col]=df[col].astype("category")
    return df[FEATURE_COLUMNS]