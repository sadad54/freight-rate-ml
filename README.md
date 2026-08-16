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
