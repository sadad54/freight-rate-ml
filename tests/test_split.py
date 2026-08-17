import pandas as pd

from src.split import time_based_split


def test_time_based_split_has_no_date_overlap():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        "value": [1, 2, 3],
    })
    cutoff = pd.Timestamp("2025-01-02")
    train, valid = time_based_split(df, cutoff)
    assert train["date"].max() < cutoff
    assert valid["date"].min() >= cutoff
    assert len(train) + len(valid) == len(df)
