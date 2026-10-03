"""Shared Polars-tabular containers for task views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import polars as pl


@dataclass(frozen=True)
class TabularSplit:
    """One split of a model-ready Polars table.

    ``frame`` keeps entity, time, features, and optional target columns.
    Use ``X()`` / ``y()`` / ``groups()`` at the model boundary; prefer Polars
    until that point.
    """

    frame: pl.DataFrame
    feature_columns: Tuple[str, ...]
    target_column: Optional[str]
    entity_column: str
    time_column: str

    def X(self) -> pl.DataFrame:
        """Return feature columns only."""
        return self.frame.select(list(self.feature_columns))

    def y(self) -> pl.Series:
        """Return the target column."""
        if self.target_column is None:
            raise ValueError("This split has no target column")
        return self.frame.get_column(self.target_column)

    def groups(self) -> pl.Series:
        """Return entity identifiers aligned with rows."""
        return self.frame.get_column(self.entity_column)

    def to_numpy(self) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Return ``(X, y, groups)`` NumPy arrays for estimators.

        ``y`` is ``None`` when the split has no target column.
        """
        features = self.X().to_numpy()
        groups = self.groups().to_numpy()
        if self.target_column is None:
            return features, None, groups
        return features, self.y().to_numpy(), groups
