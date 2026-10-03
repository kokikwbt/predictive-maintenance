"""Shared regression and PHM scoring helpers for task views."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


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
) -> Dict[str, float]:
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
