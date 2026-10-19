"""N-CMAPSS TTE task-view regressions."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from support_ncmapss import write_rich_fixture

from pdmdata.datasets.ncmapss import loader as ncmapss_loader
from pdmdata.tasks.tte import (
    EVENT_OBSERVED_COLUMN,
    TIME_TO_EVENT_COLUMN,
    eval_ncmapss,
    prepare_ncmapss,
)


class TteNcmapssTest(TestCase):
    def test_prepare_entity_table_and_eval(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "N-CMAPSS_DS01-005.h5"
            write_rich_fixture(path)
            with patch.object(ncmapss_loader, "_path", return_value=path):
                bundle = prepare_ncmapss(
                    "ds01",
                    time_cap=2,
                    validation_units=[4],
                    drop_constant_features=True,
                )
                self.assertNotIn("RUL", bundle.train.frame.columns)
                self.assertEqual(
                    bundle.train.frame[TIME_TO_EVENT_COLUMN].max(), 2
                )
                self.assertTrue(
                    bundle.train.frame[EVENT_OBSERVED_COLUMN].all()
                )
                entities = bundle.entity_table()
                self.assertEqual(
                    set(entities["cohort"].to_list()), {"dev", "test"}
                )
                self.assertFalse(
                    entities.filter(entities["cohort"] == "test")[
                        EVENT_OBSERVED_COLUMN
                    ].any()
                )
                eval_split = bundle.test_eval_split()
                self.assertEqual(eval_split.y().to_list(), [7, 4])
                self.assertFalse(
                    eval_split.frame[EVENT_OBSERVED_COLUMN].any()
                )
                metrics = eval_ncmapss(
                    bundle, {10: 7, 11: 4}
                )
                self.assertEqual(metrics["mae"], 0.0)
