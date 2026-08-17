"""Loading and light sanity-checking for the raw CSVs.

No imputation happens here, Missing weight/market_index/quote_signal
values are left as NaN and handled downstream by the model itself
(HistGradientBoostingRegressor splits on "is this value missing?"
natively), which keeps a single code path consistent between rows 
that happen to be missing a value in the training data and December 
rows that are missing market_index/quote_signal directly.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_train_test(path: str| Path)-> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    _check_no_negatives(df, ["distance", "weight", "market_index", "quote_signal", "posted_rate"])
    _check_unique_ids(df)
    return df

def load_validation(path: str| Path)-> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    _check_no_negatives(df, ["distance", "weight", "market_index", "quote_signal"])
    _check_unique_ids(df)
    return df

def load_december_inputs(path: str| Path)-> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])

def _check_no_negatives(df: pd.DataFrame, columns: list[str])-> None:
    for col in columns:
        if col in df.columns and (df[col]<0).any():
            raise ValueError(f"Found negative values in {col!r}; these should never be negative.")
        def _check_unique_ids(df: pd.DataFrame)-> None:
            if df["load_id"].duplicated().any():
                raise ValueError("duplicate load_id values found.")

def _check_unique_ids(df: pd.DataFrame)-> None:
    if df["load_id"].duplicated().any():
        raise ValueError("duplicate load_id values found.")

def missing_value_report(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum().loc[lambda s: s> 0 ].sort_values(ascending=False)