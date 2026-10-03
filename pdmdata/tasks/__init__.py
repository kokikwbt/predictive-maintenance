"""Predictive-maintenance task taxonomy and experiment views."""

from pdmdata.tasks import rul, tte
from pdmdata.tasks.metrics import nasa_score, regression_metrics
from pdmdata.tasks.rul import (
    CmapssRulBundle,
    evaluate_test_predictions as evaluate_rul_test_predictions,
    prepare_cmapss as prepare_cmapss_rul,
)
from pdmdata.tasks.taxonomy import SUPPORT_LEVELS, TASK_BY_ID, TASKS
from pdmdata.tasks.tte import (
    CmapssTteBundle,
    evaluate_test_predictions as evaluate_tte_test_predictions,
    prepare_cmapss as prepare_cmapss_tte,
)
from pdmdata.tasks.types import TabularSplit

# Backward-compatible alias used by the RUL notebook and earlier imports.
prepare_cmapss = prepare_cmapss_rul
evaluate_test_predictions = evaluate_rul_test_predictions

__all__ = [
    "SUPPORT_LEVELS",
    "TASK_BY_ID",
    "TASKS",
    "TabularSplit",
    "CmapssRulBundle",
    "CmapssTteBundle",
    "evaluate_rul_test_predictions",
    "evaluate_test_predictions",
    "evaluate_tte_test_predictions",
    "nasa_score",
    "prepare_cmapss",
    "prepare_cmapss_rul",
    "prepare_cmapss_tte",
    "regression_metrics",
    "rul",
    "tte",
]
