import numpy as np
import pandas as pd
import pytest
from src.predict import align_predictions
from src.features import build_city_coords, build_features
from test_features import _toy_df


def test_template_order_is_preserved():
    template = pd.DataFrame({"load_id": ["B", "A"]})
    source = pd.DataFrame({"load_id": ["A", "B"]})
    out = align_predictions(template, source, [10, 20])
    assert list(out.columns) == ["load_id", "predicted_rate"]
    assert out.predicted_rate.tolist() == [20, 10]


@pytest.mark.parametrize("ids,values", [(["A", "A"], [1, 2]), (["A", "C"], [1, 2]), (["A", "B"], [1, np.nan]), (["A", "B"], [1])])
def test_invalid_submission_is_rejected(ids, values):
    with pytest.raises(ValueError):
        align_predictions(pd.DataFrame({"load_id": ["A", "B"]}), pd.DataFrame({"load_id": ids}), values)


def test_unseen_city_and_missing_signals_are_left_missing():
    raw = _toy_df()
    coords = build_city_coords(raw.iloc[:1])
    unseen = raw.iloc[1:].drop(columns=["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "market_index", "quote_signal"])
    features = build_features(unseen, coords)
    assert features.pickup_lat.isna().all()
    assert features.market_index.isna().all()
    assert str(features.pickup.dtype) == "category"
