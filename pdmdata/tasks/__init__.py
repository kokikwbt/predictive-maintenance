"""Predictive-maintenance task taxonomy and experiment views."""

from pdmdata.tasks import rul, tte
from pdmdata.tasks.rul import (
    CmapssRulBundle,
    NcmapssRulBundle,
    XjtuSyRulBundle,
    eval_cmapss as eval_cmapss_rul,
    eval_ncmapss as eval_ncmapss_rul,
    eval_xjtu_sy as eval_xjtu_sy_rul,
    prepare_cmapss as prepare_cmapss_rul,
    prepare_ncmapss as prepare_ncmapss_rul,
    prepare_xjtu_sy as prepare_xjtu_sy_rul,
)
from pdmdata.tasks.taxonomy import SUPPORT_LEVELS, TASK_BY_ID, TASKS
from pdmdata.tasks.tte import (
    CmapssTteBundle,
    NcmapssTteBundle,
    XjtuSyTteBundle,
    eval_cmapss as eval_cmapss_tte,
    eval_ncmapss as eval_ncmapss_tte,
    eval_xjtu_sy as eval_xjtu_sy_tte,
    prepare_cmapss as prepare_cmapss_tte,
    prepare_ncmapss as prepare_ncmapss_tte,
    prepare_xjtu_sy as prepare_xjtu_sy_tte,
)
from pdmdata.tasks.types import TabularSplit

__all__ = [
    "SUPPORT_LEVELS",
    "TASK_BY_ID",
    "TASKS",
    "TabularSplit",
    "CmapssRulBundle",
    "CmapssTteBundle",
    "NcmapssRulBundle",
    "NcmapssTteBundle",
    "XjtuSyRulBundle",
    "XjtuSyTteBundle",
    "eval_cmapss_rul",
    "eval_cmapss_tte",
    "eval_ncmapss_rul",
    "eval_ncmapss_tte",
    "eval_xjtu_sy_rul",
    "eval_xjtu_sy_tte",
    "prepare_cmapss_rul",
    "prepare_cmapss_tte",
    "prepare_ncmapss_rul",
    "prepare_ncmapss_tte",
    "prepare_xjtu_sy_rul",
    "prepare_xjtu_sy_tte",
    "rul",
    "tte",
]
