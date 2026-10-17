import numpy as np
import pytest

from model_utils import equal_weight_soft_vote, evaluate_binary


def test_equal_weight_soft_vote_averages_probabilities():
    result = equal_weight_soft_vote(
        {"model_a": [0.1, 0.8], "model_b": [0.3, 0.6]}
    )
    np.testing.assert_allclose(result, [0.2, 0.7])


def test_equal_weight_soft_vote_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        equal_weight_soft_vote([[0.1], [0.2, 0.3]])


def test_evaluate_binary_returns_expected_metrics():
    result = evaluate_binary("demo", [0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert result["name"] == "demo"
    assert result["auc"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)
