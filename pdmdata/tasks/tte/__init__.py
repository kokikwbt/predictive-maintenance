"""Time-to-event (TTE) task package."""

from pdmdata.tasks.tte.prepare import (
    ENTITY_COLUMN,
    EVENT_OBSERVED_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    TIME_TO_EVENT_COLUMN,
    CmapssTteBundle,
    eval_test as eval_cmapss,
    filter_entities,
    prepare_cmapss,
    split_entities,
)

from pdmdata.tasks.tte.sequences import (
    CmapssSequenceSpec,
    build_sequence_samples,
    common_windows,
    complete_sequences,
    export_cmapss_sequences,
    export_derived_windows,
    prepare_cmapss_sequences,
    sequence_provenance,
)

from pdmdata.tasks.tte.higs import HigsSpec, higs_corrupt

from pdmdata.tasks.tte.ncmapss import (
    NcmapssTteBundle,
    eval_test as eval_ncmapss,
    prepare_ncmapss,
)
from pdmdata.tasks.tte.xjtu_sy import (
    XjtuSyTteBundle,
    eval_test as eval_xjtu_sy,
    prepare_xjtu_sy,
)

# Backward-compatible alias: C-MAPSS is the historical default.
eval_test = eval_cmapss

__all__ = [
    "HigsSpec", "higs_corrupt",
    "CmapssSequenceSpec",
    "build_sequence_samples",
    "common_windows",
    "complete_sequences",
    "export_cmapss_sequences",
    "export_derived_windows",
    "prepare_cmapss_sequences",
    "sequence_provenance",
    "CmapssTteBundle",
    "NcmapssTteBundle",
    "XjtuSyTteBundle",
    "ENTITY_COLUMN",
    "EVENT_OBSERVED_COLUMN",
    "TARGET_COLUMN",
    "TIME_COLUMN",
    "TIME_TO_EVENT_COLUMN",
    "eval_cmapss",
    "eval_ncmapss",
    "eval_test",
    "eval_xjtu_sy",
    "filter_entities",
    "prepare_cmapss",
    "prepare_ncmapss",
    "prepare_xjtu_sy",
    "split_entities",
]
