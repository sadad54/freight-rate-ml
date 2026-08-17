"""TIme-based train/v;aidation split.

The real task is to predict November/December from a model trained
on January-October -- a forecast into the future, not an interpolation
between random rows. A random split would let October rows sit in the
training fold used to "validate" on September, which overstates real
accuracy. Splitting by date instead makes the internal validation 
mimic the actual deployment scenario.
"""
import pandas as pd

DEFAULT_CUTOFF = pd.Timestamp("2025-09-01")

def time_based_split(df: pd.DataFrame, cutoff: pd.Timestamp =  DEFAULT_CUTOFF):
    train = df[df["date"] < cutoff].reset_index(drop=True)
    valid = df[df["date"]>=cutoff].reset_index(drop=True)
    return train, valid

