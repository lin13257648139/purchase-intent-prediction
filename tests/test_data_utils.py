import numpy as np
import pandas as pd

from data_utils import (
    clean_behavior_chunk,
    decay_weight,
    enrich_time_fields,
    merge_sum_frames,
)


def test_enrich_time_fields_and_decay_weight():
    frame = pd.DataFrame(
        {"time": pd.to_datetime(["2014-12-06 10:00:00", "2014-12-08 03:00:00"])}
    )
    enriched = enrich_time_fields(frame)
    assert enriched["day"].tolist() == [6, 8]
    assert enriched["hour"].tolist() == [10, 3]
    assert enriched["is_weekend"].tolist() == [1, 0]

    weights = decay_weight(np.array([5, 10]), max_day=10, half_life=5)
    np.testing.assert_allclose(weights, [0.5, 1.0])


def test_clean_behavior_chunk_filters_and_fills_category():
    chunk = pd.DataFrame(
        {
            "item_id": [1, 2, 3],
            "behavior_type": [1, 9, 4],
            "item_category": [np.nan, 20, np.nan],
        }
    )
    items = pd.DataFrame({"item_id": [1], "item_category": [10]})

    cleaned, stats = clean_behavior_chunk(chunk, items)

    assert cleaned["behavior_type"].tolist() == [1, 4]
    assert cleaned["item_category"].tolist() == [10, -1]
    assert stats == {
        "raw": 3,
        "invalid_behavior": 1,
        "missing_category": 2,
        "filled_category": 1,
        "valid": 2,
    }


def test_merge_sum_frames_combines_chunk_aggregates():
    first = pd.DataFrame({"user_id": [1, 2], "views": [2, 3]})
    second = pd.DataFrame({"user_id": [1], "views": [4]})

    merged = merge_sum_frames([first, second], ["user_id"]).set_index("user_id")

    assert merged.loc[1, "views"] == 6
    assert merged.loc[2, "views"] == 3
