"""Dependency-light helpers shared by model training and tests."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary(name, y_true, y_prob, threshold=0.5):
    """Evaluate binary probabilities with a fixed decision threshold."""
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")

    y_pred = (y_prob >= threshold).astype(int)
    return {
        "name": name,
        "auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def equal_weight_soft_vote(predictions):
    """Average probability vectors from one or more fitted models."""
    if isinstance(predictions, Mapping):
        vectors = list(predictions.values())
    else:
        vectors = list(predictions)
    if not vectors:
        raise ValueError("at least one prediction vector is required")

    arrays = [np.asarray(vector, dtype=float).reshape(-1) for vector in vectors]
    expected_shape = arrays[0].shape
    if any(array.shape != expected_shape for array in arrays[1:]):
        raise ValueError("all prediction vectors must have the same shape")
    return np.mean(np.vstack(arrays), axis=0)
