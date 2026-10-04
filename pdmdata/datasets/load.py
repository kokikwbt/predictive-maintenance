"""Dispatch the common load API to adapters owned by each dataset."""

from importlib import import_module
from typing import Any

import polars as pl


_LOADER_MODULES = {
    "backblaze": "pdmdata.datasets.backblaze.loader",
    "care": "pdmdata.datasets.care.loader",
    "cmapss": "pdmdata.datasets.cmapss.loader",
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
