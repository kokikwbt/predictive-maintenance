"""MAPM time-to-event bundle."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import polars as pl

import pdmdata
from pdmdata.tasks.tte import (
    EVENT_OBSERVED_COLUMN,
    TIME_TO_EVENT_COLUMN,
    eval_mapm,
    prepare_mapm,
)


def _write_tables(root: Path) -> None:
    directory = root / "mapm"
    directory.mkdir()
    (directory / "PdM_telemetry.csv").write_text(
        "datetime,machineID,volt,rotate,pressure,vibration\n"
        "2015-01-01 00:00:00,1,1,1,1,1\n"
        "2015-01-01 01:00:00,1,1,1,1,2\n"
        "2015-01-01 02:00:00,1,1,1,1,3\n"
        "2015-01-01 03:00:00,1,1,1,1,4\n"
        "2015-01-01 00:00:00,2,2,2,2,1\n"
        "2015-01-01 01:00:00,2,2,2,2,2\n"
    )
    (directory / "PdM_failures.csv").write_text(
        "datetime,machineID,failure\n"
        "2015-01-01 02:00:00,1,comp1\n"
    )
    (directory / "PdM_machines.csv").write_text(
        "machineID,model,age\n1,model1,4\n2,model2,7\n"
    )


class TteMapmTest(TestCase):
    def test_next_failure_and_following_censored_spell(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_tables(root)
            with patch(
                "pdmdata.io.paths.get_settings",
                return_value=pdmdata.Settings(data_root=root),
            ):
                bundle = prepare_mapm(
                    test_machines=[2],
                    validation_fraction=0,
                )
                self.assertIn("age", bundle.train.feature_columns)
                self.assertNotIn("model", bundle.train.frame.columns)
                train = bundle.train.frame.sort("datetime")
                self.assertEqual(
                    train[TIME_TO_EVENT_COLUMN].to_list(), [2.0, 1.0, 0.0, 0.0]
                )
                self.assertEqual(
                    train[EVENT_OBSERVED_COLUMN].to_list(),
                    [True, True, True, False],
                )
                spells = bundle.entity_table().filter(pl.col("machineID") == 1)
                self.assertEqual(spells["duration"].to_list(), [2.0, 0.0])
                self.assertEqual(
                    spells[EVENT_OBSERVED_COLUMN].to_list(), [True, False]
                )
                restored = bundle.test_eval_split()
                self.assertEqual(restored.y().to_list(), [1.0, 0.0])
                self.assertFalse(restored.frame[EVENT_OBSERVED_COLUMN].any())
                self.assertEqual(eval_mapm(bundle, [1.0, 0.0])["mae"], 0.0)
