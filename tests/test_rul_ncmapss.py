"""N-CMAPSS RUL task-view regressions."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import polars as pl

from pdmdata.datasets.ncmapss import loader as ncmapss_loader
from pdmdata.tasks.rul import (
    eval_ncmapss,
    prepare_ncmapss,
)
from pdmdata.tasks.tte import EVENT_OBSERVED_COLUMN, TIME_TO_EVENT_COLUMN
from support_ncmapss import write_rich_fixture, write_tiny_fixture


class RulNcmapssTest(TestCase):
    def test_prepare_caps_splits_and_official_eval(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "N-CMAPSS_DS01-005.h5"
            write_rich_fixture(path)
            with patch.object(ncmapss_loader, "_path", return_value=path):
                bundle = prepare_ncmapss(
                    "ds01",
                    rul_cap=2,
                    validation_units=[4],
                    drop_constant_features=True,
                )
                self.assertEqual(bundle.variant, "ds01")
                self.assertEqual(
                    bundle.train.groups().unique().sort().to_list(),
                    [1, 2, 3],
                )
                self.assertEqual(
                    bundle.validation.groups().unique().to_list(), [4]
                )
                self.assertEqual(
                    bundle.train.frame.filter(pl.col("unit_number") == 1)[
                        "RUL"
                    ].to_list(),
                    [2, 2, 1, 0],
                )
                self.assertTrue(
                    bundle.train.frame[EVENT_OBSERVED_COLUMN].all()
                )
                self.assertEqual(
                    bundle.train.frame[TIME_TO_EVENT_COLUMN].to_list(),
                    bundle.train.y().to_list(),
                )
                self.assertEqual(
                    bundle.feature_columns,
                    tuple(f"s{i}" for i in range(14)),
                )
                self.assertIsNone(bundle.test.target_column)
                self.assertEqual(
                    bundle.test_end_rul["RUL"].to_list(), [7, 4]
                )
                eval_split = bundle.test_eval_split()
                self.assertEqual(eval_split.y().to_list(), [7, 4])
                metrics = eval_ncmapss(
                    bundle, [7, 4]
                )
                self.assertEqual(metrics["mae"], 0.0)
                summary = bundle.summary()
                self.assertEqual(
                    summary["split"].to_list(),
                    ["train", "validation", "test"],
                )

    def test_tiny_fixture_prepare(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.h5"
            write_tiny_fixture(path)
            with patch.object(ncmapss_loader, "_path", return_value=path):
                bundle = prepare_ncmapss(
                    "ds01",
                    validation_fraction=0,
                    drop_constant_features=True,
                )
                self.assertEqual(bundle.train.groups().n_unique(), 2)
                self.assertIsNone(bundle.validation)
