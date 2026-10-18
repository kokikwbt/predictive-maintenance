"""XJTU-SY RUL task-view regressions."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import polars as pl

from pdmdata.datasets.xjtu_sy import loader as xjtu_loader
from pdmdata.tasks.rul import (
    eval_xjtu_sy,
    prepare_xjtu_sy,
)
from pdmdata.tasks.tte import EVENT_OBSERVED_COLUMN
from support_xjtu_sy import write_bearing


class RulXjtuSyTest(TestCase):
    def test_prepare_holdouts_and_full_trajectory_eval(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, n in (
                ("Bearing1_1", 4),
                ("Bearing1_2", 3),
                ("Bearing1_3", 5),
                ("Bearing2_1", 3),
            ):
                write_bearing(root, name, n, samples=8)
            with patch.object(
                xjtu_loader, "get_settings"
            ) as settings, patch.object(
                xjtu_loader, "SAMPLES_PER_SNAPSHOT", 8
            ):
                settings.return_value.data_root = root
                bundle = prepare_xjtu_sy(
                    bearings=[
                        "Bearing1_1",
                        "Bearing1_2",
                        "Bearing1_3",
                        "Bearing2_1",
                    ],
                    test_bearings=["Bearing2_1"],
                    validation_bearings=["Bearing1_3"],
                    rul_cap=2,
                    drop_constant_features=False,
                )
                self.assertEqual(
                    bundle.train.groups().unique().sort().to_list(),
                    ["Bearing1_1", "Bearing1_2"],
                )
                self.assertEqual(
                    bundle.validation.groups().unique().to_list(),
                    ["Bearing1_3"],
                )
                self.assertEqual(
                    bundle.test.groups().unique().to_list(),
                    ["Bearing2_1"],
                )
                self.assertTrue(
                    bundle.train.frame[EVENT_OBSERVED_COLUMN].all()
                )
                self.assertEqual(
                    bundle.train.frame.filter(
                        pl.col("bearing") == "Bearing1_1"
                    )["RUL"].to_list(),
                    [2, 2, 1, 0],
                )
                self.assertIsNone(bundle.test.target_column)
                eval_split = bundle.test_eval_split()
                self.assertEqual(eval_split.frame.height, 3)
                self.assertEqual(eval_split.y().to_list(), [2, 1, 0])
                metrics = eval_xjtu_sy(
                    bundle, [2, 1, 0]
                )
                self.assertEqual(metrics["mae"], 0.0)
                mapped = eval_xjtu_sy(
                    bundle, {"Bearing2_1": 1.0}
                )
                self.assertAlmostEqual(mapped["mae"], 2 / 3)
