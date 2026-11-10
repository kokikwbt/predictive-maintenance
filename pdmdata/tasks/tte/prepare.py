"""TTE task preparation entry points and backward-compatible re-exports."""

from __future__ import annotations

from typing import Any

from pdmdata.tasks.splits import (
    ENTITY_COLUMN,
    EVENT_OBSERVED_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    TIME_TO_EVENT_COLUMN,
    filter_entities,
    split_entities,
)
from pdmdata.tasks.tte.backblaze import prepare_backblaze
from pdmdata.tasks.tte.care import prepare_care
from pdmdata.tasks.tte.cmapss import (
    CmapssTteBundle,
    eval_cmapss,
    prepare_cmapss,
)
from pdmdata.tasks.tte.mapm import prepare_mapm
from pdmdata.tasks.tte.ncmapss import prepare_ncmapss
from pdmdata.tasks.tte.scania_x import prepare_scania_x
from pdmdata.tasks.tte.xjtu_sy import prepare_xjtu_sy

_PREPARE = {
    "backblaze": prepare_backblaze,
    "care": prepare_care,
    "cmapss": prepare_cmapss,
    "mapm": prepare_mapm,
    "ncmapss": prepare_ncmapss,
    "scania_x": prepare_scania_x,
    "xjtu_sy": prepare_xjtu_sy,
}


def supported_tte_datasets() -> list[str]:
    """Return dataset IDs with an implemented TTE prepare helper."""
    return sorted(_PREPARE)


def prepare_tte(dataset_id: str, **options: Any) -> Any:
    """Dispatch to the dataset-specific TTE prepare helper."""
    try:
        prepare = _PREPARE[dataset_id]
    except KeyError as error:
        supported = ", ".join(supported_tte_datasets())
        raise KeyError(
            f"TTE preparation is not supported for {dataset_id!r}. "
            f"Supported datasets: {supported}"
        ) from error
    try:
        return prepare(**options)
    except TypeError as error:
        raise TypeError(
            f"Invalid options for {dataset_id!r}: {error}"
        ) from error


__all__ = [
    "CmapssTteBundle",
    "ENTITY_COLUMN",
    "EVENT_OBSERVED_COLUMN",
    "TARGET_COLUMN",
    "TIME_COLUMN",
    "TIME_TO_EVENT_COLUMN",
    "eval_cmapss",
    "filter_entities",
    "prepare_cmapss",
    "prepare_tte",
    "split_entities",
    "supported_tte_datasets",
]
