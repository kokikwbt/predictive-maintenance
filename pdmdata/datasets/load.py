"""Dispatch the common load and verify APIs to dataset-owned modules."""

from importlib import import_module
from typing import Any

import polars as pl

_LOADER_MODULES = {
    "alpi": "pdmdata.datasets.alpi.loader",
    "backblaze": "pdmdata.datasets.backblaze.loader",
    "care": "pdmdata.datasets.care.loader",
    "cmapss": "pdmdata.datasets.cmapss.loader",
    "cwru": "pdmdata.datasets.cwru.loader",
    "gdd": "pdmdata.datasets.gdd.loader",
    "gfd": "pdmdata.datasets.gfd.loader",
    "hydsys": "pdmdata.datasets.hydsys.loader",
    "ims": "pdmdata.datasets.ims.loader",
    "mapm": "pdmdata.datasets.mapm.loader",
    "metropt2": "pdmdata.datasets.metropt2.loader",
    "ncmapss": "pdmdata.datasets.ncmapss.loader",
    "oyicd": "pdmdata.datasets.oyicd.loader",
    "ppd": "pdmdata.datasets.ppd.loader",
    "scania_x": "pdmdata.datasets.scania_x.loader",
    "xjtu_sy": "pdmdata.datasets.xjtu_sy.loader",
}

_VERIFIER_MODULES = {
    "alpi": "pdmdata.datasets.alpi.validation",
    "backblaze": "pdmdata.datasets.backblaze.validation",
    "care": "pdmdata.datasets.care.validation",
    "cmapss": "pdmdata.datasets.cmapss.validation",
    "cwru": "pdmdata.datasets.cwru.validation",
    "ims": "pdmdata.datasets.ims.validation",
}


def load(dataset_id: str, **options: Any) -> pl.DataFrame | pl.LazyFrame:
    """Load local data through its dataset adapter; never download missing data."""
    try:
        module_name = _LOADER_MODULES[dataset_id]
    except KeyError:
        raise KeyError(f"Unknown dataset: {dataset_id!r}") from None
    adapter = import_module(module_name).load
    try:
        return adapter(**options)
    except TypeError as error:
        raise TypeError(f"Invalid options for {dataset_id!r}: {error}") from error


def supported_verifications() -> list[str]:
    """Return dataset IDs that provide an offline verification routine."""
    return sorted(_VERIFIER_MODULES)


def verify(dataset_id: str, **options: Any) -> pl.DataFrame | dict[str, Any]:
    """Run offline source verification for a supported dataset."""
    if dataset_id not in _LOADER_MODULES:
        raise KeyError(f"Unknown dataset: {dataset_id!r}")
    try:
        module_name = _VERIFIER_MODULES[dataset_id]
    except KeyError:
        supported = ", ".join(supported_verifications())
        raise ValueError(
            f"Offline verification is not supported for {dataset_id!r}. "
            f"Supported datasets: {supported}"
        ) from None
    verifier = import_module(module_name).verify
    try:
        return verifier(**options)
    except TypeError as error:
        raise TypeError(f"Invalid options for {dataset_id!r}: {error}") from error
