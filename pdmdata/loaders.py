"""Dispatch the common load API to adapters owned by each dataset."""

from importlib import import_module
from typing import Any

import polars as pl


_LOADER_MODULES = {
    "backblaze": "pdmdata.backblaze.loader",
    "care": "pdmdata.care.loader",
    "cmapss": "pdmdata.cmapss.loader",
    "gdd": "pdmdata.gdd.loader",
    "gfd": "pdmdata.gfd.loader",
    "hydsys": "pdmdata.hydsys.loader",
    "ims": "pdmdata.ims.loader",
    "mapm": "pdmdata.mapm.loader",
    "metropt2": "pdmdata.metropt2.loader",
    "oyicd": "pdmdata.oyicd.loader",
    "ppd": "pdmdata.ppd.loader",
}


def load(dataset_id: str, **options: Any) -> pl.DataFrame | pl.LazyFrame:
    """Load local data through its dataset adapter; never download missing data."""
    try:
        module_name = _LOADER_MODULES[dataset_id]
    except KeyError:
        raise KeyError("Unknown dataset: {!r}".format(dataset_id)) from None
    adapter = import_module(module_name).load
    try:
        return adapter(**options)
    except TypeError as error:
        raise TypeError("Invalid options for {!r}: {}".format(dataset_id, error)) from error
