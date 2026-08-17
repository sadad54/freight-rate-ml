"""Loading and light sanity-checking for the raw CSVs.

No imputation happens here for missing values: weight/market_index/
quote_signal NaNs are left as-is and handled downstream by the model
itself (HistGradientBoostingRegressor splits on "is this value
missing?" natively), which keeps a single code path consistent
between rows that happen to be missing a value in the training data
and December rows that are missing market_index/quote_signal
entirely.

Negative `weight` values ARE corrected here (not left alone): 292
rows in train-test.csv and 145 in validation.csv have negative
weight, symmetric with the positive range (-47,500 to 47,500 vs.
25,800-37,018 for the bulk of the data). That symmetry points to a
sign-flip data entry error rather than genuinely bad readings, so the
fix is `abs(weight)` rather than dropping or imputing those rows.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_train_test(path: str| Path)-> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = _fix_negative_weight(df)
    _check_no_negatives(df, ["distance", "market_index", "quote_signal", "posted_rate"])
    _check_unique_ids(df)
    return df

def load_validation(path: str| Path)-> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = _fix_negative_weight(df)
    _check_no_negatives(df, ["distance", "market_index", "quote_signal"])
    _check_unique_ids(df)
    return df

def load_december_inputs(path: str| Path)-> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return _fix_negative_weight(df)

def _fix_negative_weight(df: pd.DataFrame) -> pd.DataFrame:
    if "weight" in df.columns and (df["weight"] < 0).any():
        df = df.copy()
        df["weight"] = df["weight"].abs()
    return df

def _check_no_negatives(df: pd.DataFrame, columns: list[str])-> None:
    for col in columns:
        if col in df.columns and (df[col]<0).any():
            raise ValueError(f"Found negative values in {col!r}; these should never be negative.")

def _check_unique_ids(df: pd.DataFrame)-> None:
    if df["load_id"].duplicated().any():
        raise ValueError("duplicate load_id values found.")

def missing_value_report(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum().loc[lambda s: s> 0 ].sort_values(ascending=False)