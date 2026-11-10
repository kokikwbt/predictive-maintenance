"""Backward-compatible re-exports of the IDA 2024 challenge cost helpers."""

from __future__ import annotations

from pdmdata.tasks.tte.scania_x_labels import (
    CLASSES,
    COST_MATRIX,
    challenge_cost,
    cost_table,
)

__all__ = [
    "CLASSES",
    "COST_MATRIX",
    "challenge_cost",
    "cost_table",
]
