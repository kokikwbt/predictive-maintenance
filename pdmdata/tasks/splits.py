"""Task-agnostic TTE column vocabulary and entity-split helpers.

This module lives outside ``pdmdata.tasks.tte`` so RUL views can import it
without initializing the TTE package, whose N-CMAPSS and XJTU-SY views import
the RUL modules in turn.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

# Generic TTE vocabulary defaults. C-MAPSS frames use
# CMAPSS_ENTITY_COLUMN / CMAPSS_TIME_COLUMN instead; adapters may keep
# native entity/time names while still exposing the target / censoring
# columns below.
ENTITY_COLUMN = "entity"
TIME_COLUMN = "time"
TIME_TO_EVENT_COLUMN = "time_to_event"
EVENT_OBSERVED_COLUMN = "event_observed"

TARGET_COLUMN = TIME_TO_EVENT_COLUMN


def _as_entity(value: object) -> int | str:
    """Normalize an entity id to ``int`` or ``str``."""
    if isinstance(value, bool) or value is None:
        raise ValueError("entity ids must be int or str")
    if isinstance(value, str):
        if not value:
            raise ValueError("entity ids must be non-empty strings")
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise ValueError("entity ids must be int or str")


def split_entities(
    entities: Sequence[int | str],
    *,
    validation_fraction: float = 0.2,
    random_state: int = 0,
    validation_entities: Sequence[int | str] | None = None,
) -> tuple[tuple[int | str, ...], tuple[int | str, ...]]:
    """Split entity IDs into train and validation groups.

    Entity ids may be integers (C-MAPSS / N-CMAPSS units) or strings
    (XJTU-SY bearings). When ``validation_entities`` is given it is used as
    the hold-out set and must be a subset of ``entities``. Otherwise entities
    are shuffled with ``random_state`` and the last fraction is held out. A
    zero fraction keeps every entity in train and returns an empty validation
    tuple.
    """
    unique = tuple(sorted({_as_entity(value) for value in entities}))
    if not unique:
        raise ValueError("entities must be non-empty")
    if validation_entities is not None:
        holdout = tuple(
            sorted({_as_entity(value) for value in validation_entities})
        )
        unknown = set(holdout) - set(unique)
        if unknown:
            raise ValueError(
                "validation_entities not present in entities: "
                f"{sorted(unknown)}"
            )
        if set(holdout) == set(unique):
            raise ValueError(
                "validation_entities must leave at least one training entity"
            )
        train = tuple(value for value in unique if value not in set(holdout))
        return train, holdout
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must satisfy 0 <= f < 1")
    if validation_fraction == 0.0:
        return unique, ()
    order = list(unique)
    rng = np.random.default_rng(random_state)
    rng.shuffle(order)
    holdout_count = max(1, int(round(len(order) * validation_fraction)))
    if holdout_count >= len(order):
        holdout_count = len(order) - 1
    # A single entity cannot be held out. ``order[-0:]`` would otherwise
    # select every id because ``-0`` is ``0``.
    if holdout_count == 0:
        return unique, ()
    holdout = tuple(sorted(order[-holdout_count:]))
    train = tuple(sorted(order[:-holdout_count]))
    return train, holdout


def filter_entities(
    frame: pl.DataFrame,
    entities: Sequence[int | str],
    entity_column: str,
) -> pl.DataFrame:
    """Return rows whose entity ID is in ``entities``, preserving order."""
    if not entities:
        return frame.clear()
    return frame.filter(
        pl.col(entity_column).is_in([_as_entity(value) for value in entities])
    )
