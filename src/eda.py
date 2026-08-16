"""Exploratory analysis for the freight-rate dataset.

Run once, look at the printed output and the saved figures, then move on to src/data.py and src/features.py - the findings here directly motivate decisions made in both.

Usage:
    python -m src.eda
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path("data")
FIG_DIR = Path("reports/figures")

def main()-> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train-test.csv", parse_dates=["date"])
    valid = pd.read_csv(DATA_DIR / "validation.csv", parse_dates=["date"])
    december = pd.read_csv(DATA_DIR / "december-chart-inputs.csv")

    print("train-test.csv:", train.shape, train["date"].min(), "to", train["date"].max())
    print("validation.csv:", valid.shape, valid["date"].min(), "to", valid["date"].max())
    print("\nMissing values (train): \n", train.isna().sum().loc[lambda s:s >0])
    print("\nMissing values (validation): \n", valid.isna().sum().loc[lambda s:s >0])

    print("\nposted_rate summary:\n", train["posted_rate"].describe())
    print("\nequipment_counts:\n", train["equipment"].value_counts())

    #Does market_index/quote_signal vary within a single date, or is it 
    #a shared daily macro number? If it varied only by date it could be
    #reconstructed for December from data alone -- it does not.
    per_date_uniques = train.groupby("date")[["market_index", "quote_signal"]].nunique()
    print("\nmedian unique market_index values per date:", per_date_uniques["market_index"].median())

    #Every city name should map to exactly one lat/lon pair --confirms
    # a lookup table can safely fill in December's missing coordinates.
    pickup_coords = train.groupby("pickup")[["pickup_lat","pickup_lon"]].nunique()
    print("\npickup cities with inconsistent coordinates:", (pickup_coords["pickup_lat"] > 1).sum())

    fig, ax = plt.subplots(figsize=(7, 5))
    for equip, group in train.groupby("equipment"):
        ax.scatter(group["distance"], group["posted_rate"], s=6, alpha=0.4, label=equip)
    ax.set_xlabel("distance (mi)")
    ax.set_ylabel("posted_rate ($)")
    ax.legend()
    ax.set_title("Rate vs. distance by equipment type")
    fig.savefig(FIG_DIR / "rate_vs_distance.png", dpi=150, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(train["quote_signal"], train["posted_rate"], s=6, alpha=0.3)
    ax.set_xlabel("quote_signal")
    ax.set_ylabel("posted_rate ($)")
    ax.set_title("Rate vs. quote_signal")
    fig.savefig(FIG_DIR / "rate_vs_quote_signal.png", dpi=150, bbox_inches="tight")

    print(f"\nFigures written to {FIG_DIR}/")

if __name__ == "__main__":
    main()