# Freight Rate Prediction

Predicts truckload freight rates (`posted_rate`) from load features:
pickup/delivery city, distance, equipment type, weight, date, and two
market signals (`market_index`, `quote_signal`).

## Setup
    python -m venv .venv
    .venv\Scripts\activate        # Windows
    pip install -r requirements-dev.txt

## Run
    python -m src.train           # trains model, saves to models/
    python -m src.predict         # writes validation_predictions.csv
                                   # and december_predictions.csv
    pytest                        # unit tests
    python score.py --predictions validation_predictions.csv \
                     --december-predictions december_predictions.csv


## Evaluation and output contract

See [the measured evaluation](reports/evaluation.md) and its machine-readable
[provenance](reports/evaluation.json). Training uses January–August 2025 and
holds out September–October (9,523 rows). City-coordinate lookup is built only
from training rows during evaluation. A second scenario removes market signals
and coordinates to test the December input schema.

Boosting improves MAPE (6.89% versus 8.28%) with full features, while the linear
baseline has slightly better MAE and RMSE. Missing-signal MAE improves from
164.33 to 154.21 with boosting. These are internal holdout results, not accuracy
on the unlabeled December or validation files. Dataset provider and real-world
provenance are not established; do not claim customer impact.

After evaluation, the selected pipeline is refitted on all labeled rows.
Prediction validates unique, non-null matching IDs, preserves template order,
and rejects non-finite or incorrectly sized outputs. The local run produced
12,000 validation rows. Nine tests cover features and output alignment.
On macOS/Linux activate with `source .venv/bin/activate`; use `python -m pytest -q`.
