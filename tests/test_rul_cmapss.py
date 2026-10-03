"""C-MAPSS RUL task-view regressions."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import polars as pl

from pdmdata.datasets.cmapss import loader as cmapss_loader
from pdmdata.tasks import rul
from pdmdata.tasks.rul import (
    evaluate_test_predictions,
    nasa_score,
    prepare_cmapss,
    regression_metrics,
)
from pdmdata.tasks.tte import EVENT_OBSERVED_COLUMN, TIME_TO_EVENT_COLUMN


def _write_cmapss_fixture(root: Path) -> None:
    def rows(lengths, *, vary: bool):
        lines = []
        for unit, length in enumerate(lengths, 1):
            for cycle in range(1, length + 1):
                ops = [0.1, 0.2, 0.3]
                if vary:
                    sensors = [float(cycle + index) for index in range(21)]
                else:
                    sensors = [0.5] * 21
                values = [unit, cycle, *ops, *sensors]
                lines.append("  ".join(map(str, values)) + "  \n")
        return "".join(lines)

    (root / "train_FD001.txt").write_text(
        rows([3, 4, 5, 6], vary=True)
    )
    (root / "test_FD001.txt").write_text(rows([2, 3], vary=True))
    (root / "RUL_FD001.txt").write_text("10\n20\n")


class RulCmapssTest(TestCase):
    def test_prepare_caps_splits_and_official_eval(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_cmapss_fixture(root)
            with patch.object(
                cmapss_loader,
                "find_raw_file",
                side_effect=lambda _, name: root / name,
            ):
                bundle = prepare_cmapss(
                    "FD001",
                    rul_cap=2,
                    validation_units=[4],
                    drop_constant_features=True,
                )

                self.assertEqual(bundle.subset, "FD001")
                self.assertEqual(bundle.rul_cap, 2)
                self.assertEqual(
                    bundle.train.frame["unit_number"]
                    .unique()
                    .sort()
                    .to_list(),
                    [1, 2, 3],
                )
                self.assertIsNotNone(bundle.validation)
                self.assertEqual(
                    bundle.validation.groups().unique().sort().to_list(),
                    [4],
                )
                self.assertEqual(
                    bundle.train.frame.filter(pl.col("unit_number") == 1)[
                        "RUL"
                    ].to_list(),
                    [2, 1, 0],
                )
                self.assertEqual(
                    bundle.train.frame.filter(pl.col("unit_number") == 3)[
                        "RUL"
                    ].to_list(),
                    [2, 2, 2, 1, 0],
                )
                self.assertTrue(
                    bundle.train.frame[EVENT_OBSERVED_COLUMN].all()
                )
                self.assertEqual(
                    bundle.train.frame[TIME_TO_EVENT_COLUMN].to_list(),
                    bundle.train.y().to_list(),
                )
                # Operating settings are constant in the fixture; sensors vary.
                self.assertEqual(
                    bundle.feature_columns,
                    tuple(f"sensor_{index}" for index in range(1, 22)),
                )
                self.assertIsNone(bundle.test.target_column)
                self.assertEqual(
                    list(bundle.test.frame.columns),
                    ["unit_number", "cycle", *bundle.feature_columns],
                )
                eval_split = bundle.test_eval_split()
                self.assertEqual(eval_split.y().to_list(), [10, 20])
                self.assertEqual(
                    eval_split.frame["cycle"].to_list(),
                    [2, 3],
                )
                metrics = evaluate_test_predictions(bundle, [10, 20])
                self.assertEqual(metrics["mae"], 0.0)
                self.assertEqual(metrics["rmse"], 0.0)
                self.assertEqual(metrics["nasa_score"], 0.0)
                mapped = evaluate_test_predictions(
                    bundle, {1: 11.0, 2: 18.0}
                )
                self.assertAlmostEqual(mapped["mae"], 1.5)
                summary = bundle.summary()
                self.assertEqual(
                    summary["split"].to_list(),
                    ["train", "validation", "test"],
                )

    def test_validation_fraction_zero_keeps_all_train_units(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_cmapss_fixture(root)
            with patch.object(
                cmapss_loader,
                "find_raw_file",
                side_effect=lambda _, name: root / name,
            ):
                bundle = prepare_cmapss(
                    "FD001",
                    validation_fraction=0.0,
                )
                self.assertIsNone(bundle.validation)
                self.assertEqual(bundle.train.groups().n_unique(), 4)

    def test_numpy_boundary_and_metrics(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_cmapss_fixture(root)
            with patch.object(
                cmapss_loader,
                "find_raw_file",
                side_effect=lambda _, name: root / name,
            ):
                bundle = prepare_cmapss(
                    "FD001",
                    validation_fraction=0.0,
                )
                X, y, groups = bundle.train.to_numpy()
                self.assertEqual(X.shape[0], y.shape[0])
                self.assertEqual(X.shape[0], groups.shape[0])
                self.assertEqual(X.shape[1], len(bundle.feature_columns))
                X_test, y_test, _ = bundle.test.to_numpy()
                self.assertIsNone(y_test)
                self.assertEqual(X_test.shape[1], X.shape[1])

        self.assertGreater(nasa_score([10], [20]), nasa_score([10], [0]))
        metrics = regression_metrics([0, 10], [0, 10])
        self.assertEqual(metrics["mae"], 0.0)
        with self.assertRaises(ValueError):
            nasa_score([1], [1, 2])
        with self.assertRaises(ValueError):
            prepare_cmapss("FD005")

    def test_module_export(self):
        self.assertIs(rul.prepare_cmapss, prepare_cmapss)
