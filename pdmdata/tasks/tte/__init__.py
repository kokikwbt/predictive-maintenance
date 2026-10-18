"""Time-to-event (TTE) task package."""

from pdmdata.tasks.tte.prepare import (
    ENTITY_COLUMN,
    EVENT_OBSERVED_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    TIME_TO_EVENT_COLUMN,
    CmapssTteBundle,
    evaluate_test_predictions,
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
    evaluate_test_predictions as evaluate_ncmapss_test_predictions,
    prepare_ncmapss,
)

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
    "ENTITY_COLUMN",
    "EVENT_OBSERVED_COLUMN",
    "TARGET_COLUMN",
    "TIME_COLUMN",
    "TIME_TO_EVENT_COLUMN",
    "evaluate_test_predictions",
    "evaluate_ncmapss_test_predictions",
    "filter_entities",
    "prepare_cmapss",
    "prepare_ncmapss",
    "split_entities",
]
