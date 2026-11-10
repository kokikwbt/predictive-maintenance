"""RUL task preparation entry points and backward-compatible re-exports."""

from __future__ import annotations

from typing import Any

from pdmdata.tasks.rul.cmapss import (
    ENTITY_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    CmapssRulBundle,
    apply_rul_cap,
    eval_cmapss,
    prepare_cmapss,
)
from pdmdata.tasks.rul.ncmapss import (
    NcmapssRulBundle,
    eval_ncmapss,
    prepare_ncmapss,
)
from pdmdata.tasks.rul.xjtu_sy import (
    XjtuSyRulBundle,
    eval_xjtu_sy,
    prepare_xjtu_sy,
)

_PREPARE = {
    "cmapss": prepare_cmapss,
    "ncmapss": prepare_ncmapss,
    "xjtu_sy": prepare_xjtu_sy,
}


def supported_rul_datasets() -> list[str]:
    """Return dataset IDs with an implemented RUL prepare helper."""
    return sorted(_PREPARE)


def prepare_rul(dataset_id: str, **options: Any) -> Any:
    """Dispatch to the dataset-specific RUL prepare helper."""
    try:
        prepare = _PREPARE[dataset_id]
    except KeyError as error:
        supported = ", ".join(supported_rul_datasets())
        raise KeyError(
            f"RUL preparation is not supported for {dataset_id!r}. "
            f"Supported datasets: {supported}"
        ) from error
    try:
        return prepare(**options)
    except TypeError as error:
        raise TypeError(
            f"Invalid options for {dataset_id!r}: {error}"
        ) from error


__all__ = [
    "CmapssRulBundle",
    "ENTITY_COLUMN",
    "NcmapssRulBundle",
    "TARGET_COLUMN",
    "TIME_COLUMN",
    "XjtuSyRulBundle",
    "apply_rul_cap",
    "eval_cmapss",
    "eval_ncmapss",
    "eval_xjtu_sy",
    "prepare_cmapss",
    "prepare_ncmapss",
    "prepare_rul",
    "prepare_xjtu_sy",
    "supported_rul_datasets",
]
