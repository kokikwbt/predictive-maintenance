"""Remaining useful life (RUL) task package."""

from pdmdata.tasks.rul.prepare import (
    ENTITY_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    CmapssRulBundle,
    apply_rul_cap,
    eval_test as eval_cmapss,
    nasa_score,
    prepare_cmapss,
    regression_metrics,
)
from pdmdata.tasks.rul.ncmapss import (
    NcmapssRulBundle,
    eval_test as eval_ncmapss,
    prepare_ncmapss,
)
from pdmdata.tasks.rul.xjtu_sy import (
    XjtuSyRulBundle,
    eval_test as eval_xjtu_sy,
    prepare_xjtu_sy,
)

# Backward-compatible alias: C-MAPSS is the historical default.
eval_test = eval_cmapss

__all__ = [
    "CmapssRulBundle",
    "NcmapssRulBundle",
    "XjtuSyRulBundle",
    "ENTITY_COLUMN",
    "TARGET_COLUMN",
    "TIME_COLUMN",
    "apply_rul_cap",
    "eval_cmapss",
    "eval_ncmapss",
    "eval_test",
    "eval_xjtu_sy",
    "nasa_score",
    "prepare_cmapss",
    "prepare_ncmapss",
    "prepare_xjtu_sy",
    "regression_metrics",
]
