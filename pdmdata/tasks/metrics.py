"""Shared regression and PHM scoring helpers for task views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from pdmdata.tasks.types import TabularSplit


def nasa_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Asymmetric PHM 2008 C-MAPSS score (lower is better)."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty")
    delta = pred - true
    score = np.where(
        delta >= 0,
        np.exp(delta / 10.0) - 1.0,
        np.exp(-delta / 13.0) - 1.0,
    )
    return float(np.sum(score))


def regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> dict[str, float]:
    """Return MAE, RMSE, and the NASA PHM score."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty")
    residual = pred - true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    return {
        "mae": mae,
        "rmse": rmse,
        "nasa_score": nasa_score(true, pred),
        "n": float(true.size),
    }


def align_predictions(
    units: Sequence[int | str],
    predictions: Sequence[float] | Mapping,
) -> np.ndarray:
    """Align sequence or entity-keyed predictions to ``units`` order."""
    if isinstance(predictions, Mapping):
        try:
            return np.asarray(
                [predictions[unit] for unit in units], dtype=float
            )
        except KeyError as error:
            raise KeyError(
                f"Missing prediction for test entity {error.args[0]}"
            ) from error
    pred = np.asarray(predictions, dtype=float)
    if pred.shape != (len(units),):
        raise ValueError(
            f"Expected {len(units)} predictions for test rows, "
            f"got {pred.size}"
        )
    return pred


def score_split(
    split: TabularSplit,
    predictions: Sequence[float] | Mapping[int | str, float],
) -> dict[str, float]:
    """Score row-aligned predictions against ``split``'s target column.

    A mapping keyed by entity id reuses one value for every row of that
    entity. A sequence must follow ``split`` row order.
    """
    entities = split.groups().to_list()
    actual = split.y().to_numpy()
    predicted = align_predictions(entities, predictions)
    return regression_metrics(actual, predicted)
