"""C-MAPSS time-to-event task-view regressions."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import polars as pl

from pdmdata.datasets.cmapss import loader as cmapss_loader
from pdmdata.tasks import tte
from pdmdata.tasks.tte import (
    EVENT_OBSERVED_COLUMN,
    TIME_TO_EVENT_COLUMN,
    evaluate_test_predictions,
    prepare_cmapss,
)


def _write_cmapss_fixture(root: Path) -> None:
    def rows(lengths):
        lines = []
        for unit, length in enumerate(lengths, 1):
            for cycle in range(1, length + 1):
                ops = [0.1, 0.2, 0.3]
                sensors = [float(cycle + index) for index in range(21)]
                values = [unit, cycle, *ops, *sensors]
                lines.append("  ".join(map(str, values)) + "  \n")
        return "".join(lines)

    (root / "train_FD001.txt").write_text(rows([3, 4, 5, 6]))
    (root / "test_FD001.txt").write_text(rows([2, 3]))
    (root / "RUL_FD001.txt").write_text("10\n20\n")


class TteCmapssTest(TestCase):
    def test_prepare_labels_censoring_and_entity_table(self):
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
                    time_cap=2,
                    validation_units=[4],
                    drop_constant_features=True,
                )
                self.assertEqual(bundle.subset, "FD001")
                self.assertEqual(bundle.time_cap, 2)
                self.assertEqual(
                    bundle.train.frame["unit_number"]
                    .unique()
                    .sort()
                    .to_list(),
                    [1, 2, 3],
                )
                self.assertEqual(
                    bundle.train.target_column,
                    TIME_TO_EVENT_COLUMN,
                )
                self.assertNotIn("RUL", bundle.train.frame.columns)
                self.assertEqual(
                    bundle.train.frame.filter(pl.col("unit_number") == 3)[
                        TIME_TO_EVENT_COLUMN
                    ].to_list(),
                    [2, 2, 2, 1, 0],
                )
                self.assertTrue(
                    bundle.train.frame[EVENT_OBSERVED_COLUMN].all()
                )
                eval_split = bundle.test_eval_split()
                self.assertEqual(
                    eval_split.y().to_list(),
                    [10, 20],
                )
                self.assertFalse(
                    eval_split.frame[EVENT_OBSERVED_COLUMN].any()
                )
                metrics = evaluate_test_predictions(bundle, [10, 20])
                self.assertEqual(metrics["mae"], 0.0)

                entities = bundle.entity_table()
                self.assertEqual(entities.height, 6)
                train_rows = entities.filter(pl.col("cohort") == "train")
                test_rows = entities.filter(pl.col("cohort") == "test")
                self.assertEqual(train_rows.height, 4)
                self.assertEqual(test_rows.height, 2)
                self.assertTrue(train_rows[EVENT_OBSERVED_COLUMN].all())
                self.assertFalse(test_rows[EVENT_OBSERVED_COLUMN].any())
                self.assertEqual(
                    test_rows["duration"].to_list(),
                    [2, 3],
                )
                # Censoring durations ignore official remaining-time offsets.
                self.assertEqual(
                    train_rows.sort("unit_number")["duration"].to_list(),
                    [3, 4, 5, 6],
                )

    def test_validation_fraction_zero(self):
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
                X, y, groups = bundle.train.to_numpy()
                self.assertEqual(X.shape[0], y.shape[0])
                self.assertEqual(groups.shape[0], X.shape[0])

    def test_module_export(self):
        self.assertIs(tte.prepare_cmapss, prepare_cmapss)
