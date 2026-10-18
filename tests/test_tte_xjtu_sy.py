"""XJTU-SY TTE task-view regressions."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from pdmdata.datasets.xjtu_sy import loader as xjtu_loader
from pdmdata.tasks.tte import (
    EVENT_OBSERVED_COLUMN,
    TIME_TO_EVENT_COLUMN,
    evaluate_xjtu_sy_test_predictions,
    prepare_xjtu_sy,
)
from support_xjtu_sy import write_bearing


class TteXjtuSyTest(TestCase):
    def test_prepare_entity_table_and_eval(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, n in (
                ("Bearing1_1", 4),
                ("Bearing1_2", 3),
                ("Bearing1_3", 5),
            ):
                write_bearing(root, name, n, samples=8)
            with patch.object(
                xjtu_loader, "get_settings"
            ) as settings, patch.object(
                xjtu_loader, "SAMPLES_PER_SNAPSHOT", 8
            ):
                settings.return_value.data_root = root
                bundle = prepare_xjtu_sy(
                    bearings=["Bearing1_1", "Bearing1_2", "Bearing1_3"],
                    test_bearings=["Bearing1_3"],
                    validation_fraction=0,
                    time_cap=2,
                )
                self.assertNotIn("RUL", bundle.train.frame.columns)
                self.assertEqual(
                    bundle.train.frame[TIME_TO_EVENT_COLUMN].max(), 2
                )
                self.assertTrue(
                    bundle.train.frame[EVENT_OBSERVED_COLUMN].all()
                )
                entities = bundle.entity_table()
                self.assertEqual(entities.height, 3)
                self.assertTrue(entities[EVENT_OBSERVED_COLUMN].all())
                self.assertEqual(
                    set(entities["cohort"].to_list()), {"train", "test"}
                )
                eval_split = bundle.test_eval_split()
                self.assertEqual(
                    eval_split.y().to_list(), [2, 2, 2, 1, 0]
                )
                metrics = evaluate_xjtu_sy_test_predictions(
                    bundle, [2, 2, 2, 1, 0]
                )
                self.assertEqual(metrics["mae"], 0.0)
